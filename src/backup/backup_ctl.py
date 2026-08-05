# -*- coding: utf-8 -*-
"""备份控制器（供 WebUI 通过 docker exec 在 bot_app 容器内调用）。

用法（在项目根目录执行）：
    python -m src.backup.backup_ctl status          # 输出当前备份状态 JSON
    python -m src.backup.backup_ctl run             # 立即执行一次完整备份
    python -m src.backup.backup_ctl save_config     # 从环境变量 BACKUP_ENV_JSON 保存配置到 .env

安全说明：save_config 只接受白名单字段（R2_* 与备份相关），避免任意写入 .env。
"""

import json
import os
import sys
import logging

from src.runtime_env import load_project_dotenv, persist_env_updates, resolve_project_root

log = logging.getLogger(__name__)

CONFIG_KEYS = {
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_PREFIX",
    "R2_ENDPOINT",
    "BACKUP_KEEP_DAYS",
    "BACKUP_CRON",
}

PROJECT_ROOT = resolve_project_root(__file__, parents=2)


def _backup_dirs():
    mount = os.getenv("BACKUP_OUTPUT_MOUNT_PATH", "/backup-output")
    return [os.path.join(PROJECT_ROOT, "src", "backup"), mount]


def _list_files():
    results = []
    for d in _backup_dirs():
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if ".bak." not in name and ".pgbak." not in name:
                continue
            path = os.path.join(d, name)
            try:
                results.append(
                    {
                        "name": name,
                        "size": os.path.getsize(path),
                        "mtime": os.path.getmtime(path),
                        "dir": d,
                    }
                )
            except OSError:
                continue
    return results


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return value[:2] + "****" + value[-2:]


def _last_json_line(output: str) -> dict:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"status": "error", "message": output[:500]}


def cmd_status() -> dict:
    load_project_dotenv(__file__, parents=2)
    r2_configured = bool(
        os.getenv("R2_ACCOUNT_ID")
        and os.getenv("R2_ACCESS_KEY_ID")
        and os.getenv("R2_SECRET_ACCESS_KEY")
        and os.getenv("R2_BUCKET")
    )
    cfg = {
        "r2_configured": r2_configured,
        "r2_account": os.getenv("R2_ACCOUNT_ID", ""),
        "r2_bucket": os.getenv("R2_BUCKET", ""),
        "r2_prefix": os.getenv("R2_PREFIX", "odysseia/backup/"),
        "access_key": _mask_secret(os.getenv("R2_ACCESS_KEY_ID", "")),
        "secret": _mask_secret(os.getenv("R2_SECRET_ACCESS_KEY", "")),
        "keep_days": int(os.getenv("BACKUP_KEEP_DAYS", "3") or 3),
        "schedule": os.getenv("BACKUP_CRON", "0:0"),
    }
    return {"status": "ok", "config": cfg, "files": _list_files()}


def cmd_save_config() -> dict:
    payload_raw = os.getenv("BACKUP_ENV_JSON", "")
    if not payload_raw:
        return {"status": "error", "message": "缺少 BACKUP_ENV_JSON 环境变量"}
    try:
        payload = json.loads(payload_raw)
    except Exception as e:
        return {"status": "error", "message": f"解析配置失败: {e}"}

    updates = {}
    for key, value in payload.items():
        key = str(key).upper()
        if key not in CONFIG_KEYS:
            continue
        if value is None or str(value).strip() == "":
            continue
        updates[key] = str(value).strip()

    if not updates:
        return {"status": "error", "message": "没有可保存的字段"}

    env_path = os.path.join(PROJECT_ROOT, ".env")
    try:
        persist_env_updates(env_path, updates)
    except Exception as e:
        return {"status": "error", "message": f"写入 .env 失败: {e}"}
    return {
        "status": "ok",
        "message": f"已更新 {len(updates)} 项配置",
        "updated": sorted(updates),
    }


def cmd_run() -> dict:
    from src.backup.daily_backup import run_daily_backup

    run_daily_backup()
    return {"status": "ok", "message": "备份流程已执行完毕"}


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "status"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if verb == "run" else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if verb == "status":
        result = cmd_status()
    elif verb == "save_config":
        result = cmd_save_config()
    elif verb == "run":
        result = cmd_run()
    else:
        result = {"status": "error", "message": f"未知命令: {verb}"}
    print(json.dumps(result, ensure_ascii=False))