# -*- coding: utf-8 -*-
"""每日备份编排：PostgreSQL + SQLite → 宿主机挂载目录 → 上传 Cloudflare R2。"""

import logging

from src.runtime_env import load_project_dotenv
from src.backup.backup_manager import backup_databases
from src.backup.pg_backup_manager import backup_postgres
from src.backup.cloud_upload import upload_backups, resolve_upload_source_dir

log = logging.getLogger(__name__)


def run_daily_backup():
    """每日 0 点执行：先 PG 备份，再 SQLite 备份与同步，最后上传 R2。"""
    # 每次运行前重新加载 .env，使 WebUI 修改的 R2/保留天数配置即时生效
    load_project_dotenv(__file__, parents=2)

    log.info("===== 开始每日数据库备份 =====")

    backup_ok = True

    try:
        if not backup_postgres():
            log.error("PostgreSQL 备份失败，继续执行剩余步骤。")
            backup_ok = False
    except Exception:
        log.error("PostgreSQL 备份异常", exc_info=True)
        backup_ok = False

    try:
        backup_databases()
    except Exception:
        log.error("SQLite 备份异常", exc_info=True)
        backup_ok = False

    # 仅当全部备份成功时才允许清理 R2 远端旧备份，避免误删
    try:
        upload_backups(resolve_upload_source_dir(), prune=backup_ok)
        if not backup_ok:
            log.warning("本次备份存在失败项，已跳过 R2 远端过期清理。")
    except Exception:
        log.error("上传备份到 R2 异常", exc_info=True)

    log.info("===== 每日数据库备份完成 =====")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_daily_backup()