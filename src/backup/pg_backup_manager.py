# -*- coding: utf-8 -*-
"""PostgreSQL 备份（容器内可运行，零镜像负担）。

使用 Python docker SDK 通过已挂载的 /var/run/docker.sock 进入 db 容器，
在其中执行 pg_dump 并把 dump 文件拉回宿主机备份目录，取代旧版依赖宿主
docker CLI 的方式，因此可以在 bot_app 容器内由调度器自动运行。

用法（在项目根目录执行）：
    python -m src.backup.pg_backup_manager

环境变量（自动从项目根目录的 .env 读取）：
    POSTGRES_DB          数据库名（默认 odysseia_db）
    POSTGRES_USER        用户名（默认 user）
    POSTGRES_PASSWORD    密码
    PG_CONTAINER_NAME    数据库容器名（默认 odysseia_pg_db）

恢复示例：
    docker exec -e PGPASSWORD=<password> -i odysseia_pg_db \
        pg_restore --clean --create --if-exists -U <user> -d <db> -h 127.0.0.1 -p 5432 < <dump_file>
"""

import os
import sys
import shutil
import tarfile
import logging
from datetime import datetime
from pathlib import Path

from src.runtime_env import load_project_dotenv

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKUP_DIR = BASE_DIR / "src" / "backup"

DUMP_MARKER = ".pgbak."
REMOTE_TMP = "/tmp/odysseia_pg_dump.dump"


def _get_db_container_name() -> str:
    return os.getenv("PG_CONTAINER_NAME", "odysseia_pg_db").strip() or "odysseia_pg_db"


def _get_db_container(client):
    name = _get_db_container_name()
    try:
        return client.containers.get(name)
    except Exception as e:
        log.error(f"获取数据库容器 '{name}' 失败（是否已 docker compose up？）: {e}")
        raise


def _run_pg_dump_in_container(container, db_user, db_name, db_password) -> bool:
    """在 db 容器内执行 pg_dump，输出到容器内临时文件。"""
    cmd = [
        "pg_dump", "-h", "127.0.0.1", "-p", "5432",
        "-U", db_user, "-d", db_name, "-Fc", "-f", REMOTE_TMP,
    ]
    env = {"PGPASSWORD": db_password} if db_password else None
    try:
        exit_code, output = container.exec_run(
            cmd,
            environment=env,
            stdout=True,
            stderr=True,
            stream=False,
        )
    except Exception as e:
        log.error(f"在容器内执行 pg_dump 失败: {e}", exc_info=True)
        raise

    if exit_code != 0:
        err = output.decode("utf-8", errors="replace").strip() if isinstance(output, bytes) else str(output)
        log.error(f"容器内 pg_dump 返回非零 (exit_code={exit_code}): {err}")
        return False
    return True


class _GeneratorReader:
    """把 docker get_archive 返回的字节生成器包装成支持 .read() 的文件对象，
    供 tarfile 以流式模式读取。"""

    def __init__(self, chunks):
        self._it = iter(chunks)
        self._buf = b""

    def read(self, size=-1):
        if size is None or size < 0:
            return self.readall()
        need = size
        parts = []
        if self._buf:
            take = self._buf[:need]
            self._buf = self._buf[len(take):]
            parts.append(take)
            need -= len(take)
        while need > 0:
            try:
                chunk = next(self._it)
            except StopIteration:
                break
            if len(chunk) > need:
                self._buf = chunk[need:]
                parts.append(chunk[:need])
                need = 0
            else:
                parts.append(chunk)
                need -= len(chunk)
        return b"".join(parts)

    def readall(self):
        parts = [self._buf] if self._buf else []
        self._buf = b""
        try:
            for chunk in self._it:
                parts.append(chunk)
        except StopIteration:
            pass
        return b"".join(parts)


def _pull_dump_file(container, backup_path) -> bool:
    """把容器内生成的 dump 通过 tar 流拉回到本地文件。"""
    try:
        tar_stream, _ = container.get_archive(REMOTE_TMP)
        reader = _GeneratorReader(tar_stream)
        with open(backup_path, "wb") as out_fh:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                member = tar.next()
                while member is not None:
                    if member.isreg():
                        src = tar.extractfile(member)
                        if src is not None:
                            shutil.copyfileobj(src, out_fh)
                    member = tar.next()
    except Exception as e:
        log.error(f"从容器拉取 dump 文件失败: {e}", exc_info=True)
        return False
    return True


def _cleanup_remote_tmp(container):
    try:
        container.exec_run(["rm", "-f", REMOTE_TMP])
    except Exception:
        pass


def backup_postgres() -> bool:
    """执行一次 PostgreSQL 备份，成功返回 True。"""
    load_project_dotenv(__file__, parents=2)

    db_name = os.getenv("POSTGRES_DB") or "odysseia_db"
    db_user = os.getenv("POSTGRES_USER") or "user"
    db_password = os.getenv("POSTGRES_PASSWORD", "")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    backup_filename = f"{db_name}{DUMP_MARKER}{date_str}.dump"
    backup_path = BACKUP_DIR / backup_filename

    log.info(f"开始备份 PostgreSQL 数据库 '{db_name}'...")
    log.info(f"目标文件: {backup_path}")

    try:
        import docker
    except ImportError as e:
        log.error(f"未安装 python docker 包（requirements.txt 应已包含 docker）: {e}")
        return False

    try:
        client = docker.from_env()
    except Exception as e:
        log.error(f"连接 Docker daemon 失败（容器内需挂载 /var/run/docker.sock）: {e}", exc_info=True)
        return False

    try:
        container = _get_db_container(client)
    except Exception:
        return False

    try:
        if not _run_pg_dump_in_container(container, db_user, db_name, db_password):
            _cleanup_remote_tmp(container)
            return False

        if not _pull_dump_file(container, backup_path):
            _cleanup_remote_tmp(container)
            try:
                backup_path.unlink()
            except Exception:
                pass
            return False
    finally:
        _cleanup_remote_tmp(container)

    if not backup_path.exists() or backup_path.stat().st_size == 0:
        log.error("备份文件为空，视为失败。")
        try:
            backup_path.unlink()
        except Exception:
            pass
        return False

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    log.info(f"备份成功: '{backup_path}' ({size_mb:.2f} MB)")
    log.info("PostgreSQL 备份完成。")
    log.info("恢复命令: docker exec -e PGPASSWORD=<password> -i odysseia_pg_db pg_restore "
             "--clean --create --if-exists "
             f"-U {db_user} -d {db_name} -h 127.0.0.1 -p 5432 < {backup_path.name}")
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sys.exit(0 if backup_postgres() else 1)