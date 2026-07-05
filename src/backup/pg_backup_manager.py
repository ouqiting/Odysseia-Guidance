# -*- coding: utf-8 -*-
"""PostgreSQL 手动备份脚本.

通过 `docker compose exec` 调用 db 容器内的 pg_dump，将主库导出为自定义格式
（-Fc，压缩、支持完整恢复）的 dump 文件。恢复示例：

    docker compose exec -T db pg_restore --clean --create --if-exists \
        -U <user> -d <db> -p 5432 < <dump_file>

用法（在项目根目录执行）：
    python -m src.backup.pg_backup_manager

环境变量（自动从项目根目录的 .env 读取）：
    POSTGRES_DB            数据库名（默认 odysseia_db）
    POSTGRES_USER          用户名（默认 user）
    POSTGRES_PASSWORD      密码
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from src.runtime_env import load_project_dotenv

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKUP_DIR = BASE_DIR / "src" / "backup"

DUMP_MARKER = ".pgbak."


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

    # 将密码通过 PGPASSWORD 传给 docker compose exec 进程，
    # 容器内的 pg_dump 会读取它，避免在命令行中明文出现。
    env = os.environ.copy()
    if db_password:
        env["PGPASSWORD"] = db_password

    log.info(f"开始备份 PostgreSQL 数据库 '{db_name}'...")
    log.info(f"目标文件: {backup_path}")

    # -Fc: 自定义格式（压缩，支持完整恢复）
    # 容器内部端口固定为 5432（见 docker-compose.yml），不用 DB_PORT。
    cmd = [
        "docker", "compose", "exec", "-T",
        "db",
        "pg_dump",
        "-U", db_user,
        "-d", db_name,
        "-p", "5432",
        "-Fc",
    ]

    try:
        with open(backup_path, "wb") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(BASE_DIR),
            )
    except FileNotFoundError:
        log.error("未找到 docker 命令，请确保 Docker 已安装并在 PATH 中。")
        return False
    except Exception as e:
        log.error(f"备份过程发生异常: {e}", exc_info=True)
        return False

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        log.error(f"pg_dump 失败 (返回码 {result.returncode}): {err}")
        try:
            backup_path.unlink()
        except Exception:
            pass
        return False

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    log.info(f"备份成功: '{backup_path}' ({size_mb:.2f} MB)")
    log.info("PostgreSQL 备份完成。")
    log.info("恢复命令: docker compose exec -T db pg_restore --clean --create --if-exists "
             f"-U {db_user} -d {db_name} -p 5432 < {backup_path.name}")
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sys.exit(0 if backup_postgres() else 1)
