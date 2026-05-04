import logging
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import psutil
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

try:
    import docker
except ImportError:
    docker = None

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.runtime_env import load_project_dotenv

dotenv_path = load_project_dotenv(__file__, parents=1)

template_dir = current_dir
static_dir = os.path.join(template_dir, "root")
login_page_path = os.path.join(static_dir, "html", "login.html")
main_page_path = os.path.join(static_dir, "html", "main.html")

web_app = FastAPI()
web_app.mount("/static", StaticFiles(directory=static_dir), name="static")

logger = logging.getLogger("webui")
logger.setLevel(logging.DEBUG)
logger.propagate = False

multiline_token = os.getenv("WEBUI_ADMIN_TOKEN")
TOKEN_ARRAY = []
if multiline_token:
    TOKEN_ARRAY = [line.strip() for line in multiline_token.splitlines() if line.strip()]
else:
    logger.error("TOKEN Error: TOKEN未设置")

BOT_CONTAINER_NAME = os.getenv("BOT_CONTAINER_NAME", "Odysseia_Guidance")
WEB_CONTAINER_NAME = os.getenv("WEB_CONTAINER_NAME", "config_web")
WEBUI_REMEMBER_ME_MAX_AGE = int(
    os.getenv("WEBUI_REMEMBER_ME_MAX_AGE", str(60 * 60 * 24 * 30))
)
last_heartbeat_time = datetime.utcnow()
heartbeat_tolerance_seconds = 5.0
SYSTEM_STATS_HISTORY = deque(maxlen=1440)
BOT_LOG_TAIL_LINES = int(os.getenv("WEBUI_BOT_LOG_TAIL_LINES", "1000"))
WEBUI_LOG_TAIL_LINES = int(os.getenv("WEBUI_SERVER_LOG_TAIL_LINES", "1000"))
BOT_LOG_BUFFER = deque(maxlen=BOT_LOG_TAIL_LINES)
WEBUI_LOG_BUFFER = deque(maxlen=WEBUI_LOG_TAIL_LINES)
BOT_LOG_LAST_ID = 0
WEBUI_LOG_LAST_ID = 0
KNOWN_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
BOT_LOG_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\] "
    r"\[(?P<level>[A-Z]+)\] "
    r"\[(?P<logger>[^\]]+)\] "
    r"(?P<message>[\s\S]*)$"
)
WEBUI_LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?) - "
    r"(?P<logger>.+?) - "
    r"(?P<level>[A-Z]+) - "
    r"(?P<message>[\s\S]*)$"
)


def normalize_log_level(value: Any) -> str:
    level = str(value or "UNKNOWN").upper()
    return level if level in KNOWN_LOG_LEVELS else "UNKNOWN"


def stringify_log_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def build_raw_log_line(
    timestamp: str,
    level: str,
    logger_name: str,
    message: str,
) -> str:
    if timestamp and logger_name:
        return f"[{timestamp}] [{level}] [{logger_name}] {message}"
    return message


def build_structured_log_entry(
    timestamp: str,
    level: str,
    logger_name: str,
    message: str,
    raw: str,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "level": normalize_log_level(level),
        "logger": logger_name or "unknown",
        "message": message,
        "raw": raw or build_raw_log_line(timestamp, level, logger_name, message),
    }


def parse_raw_log_entry(raw: str, pattern: re.Pattern[str]) -> dict[str, Any]:
    match = pattern.match(raw)
    if not match:
        return build_structured_log_entry("", "UNKNOWN", "unknown", raw, raw)

    timestamp = stringify_log_value(match.group("timestamp"))
    level = normalize_log_level(match.group("level"))
    logger_name = stringify_log_value(match.group("logger"), "unknown")
    message = stringify_log_value(match.group("message"))
    return build_structured_log_entry(timestamp, level, logger_name, message, raw)


def normalize_bot_log_entry(log_entry: Any) -> dict[str, Any]:
    if isinstance(log_entry, dict):
        timestamp = stringify_log_value(log_entry.get("timestamp"))
        level = normalize_log_level(log_entry.get("level"))
        logger_name = stringify_log_value(
            log_entry.get("logger") or log_entry.get("name"),
            "unknown",
        )
        message = stringify_log_value(
            log_entry.get("message"),
            stringify_log_value(log_entry.get("raw")),
        )
        raw = stringify_log_value(
            log_entry.get("raw"),
            build_raw_log_line(timestamp, level, logger_name, message),
        )
        return build_structured_log_entry(timestamp, level, logger_name, message, raw)

    return parse_raw_log_entry(stringify_log_value(log_entry), BOT_LOG_PATTERN)


def normalize_webui_log_entry(log_entry: Any) -> dict[str, Any]:
    if isinstance(log_entry, dict):
        timestamp = stringify_log_value(log_entry.get("timestamp"))
        level = normalize_log_level(log_entry.get("level"))
        logger_name = stringify_log_value(
            log_entry.get("logger") or log_entry.get("name"),
            "unknown",
        )
        message = stringify_log_value(
            log_entry.get("message"),
            stringify_log_value(log_entry.get("raw")),
        )
        raw = stringify_log_value(
            log_entry.get("raw"),
            build_raw_log_line(timestamp, level, logger_name, message),
        )
        return build_structured_log_entry(timestamp, level, logger_name, message, raw)

    return parse_raw_log_entry(stringify_log_value(log_entry), WEBUI_LOG_PATTERN)


def build_structured_entry_from_record(
    record: logging.LogRecord,
    raw_message: str,
) -> dict[str, Any]:
    message = record.getMessage()
    exc_text = getattr(record, "exc_text", None)
    if exc_text:
        message = f"{message}\n{exc_text}"
    elif record.exc_info:
        formatter = logging.Formatter()
        message = f"{message}\n{formatter.formatException(record.exc_info)}"

    if record.stack_info:
        formatter = logging.Formatter()
        message = f"{message}\n{formatter.formatStack(record.stack_info)}"

    timestamp = datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds")
    return build_structured_log_entry(
        timestamp,
        record.levelname,
        record.name,
        message,
        raw_message,
    )


def append_bot_log_entry(log_entry: Any):
    global BOT_LOG_LAST_ID
    BOT_LOG_LAST_ID += 1
    structured_entry = normalize_bot_log_entry(log_entry)
    structured_entry["id"] = BOT_LOG_LAST_ID
    BOT_LOG_BUFFER.append(structured_entry)


def append_webui_log_entry(log_entry: Any):
    global WEBUI_LOG_LAST_ID
    WEBUI_LOG_LAST_ID += 1
    structured_entry = normalize_webui_log_entry(log_entry)
    structured_entry["id"] = WEBUI_LOG_LAST_ID
    WEBUI_LOG_BUFFER.append(structured_entry)


def build_log_payload(
    buffer: deque,
    tail_lines: int,
    after_id: int | None = None,
):
    records = list(buffer)
    last_id = records[-1]["id"] if records else 0
    oldest_id = records[0]["id"] if records else 0
    reset_required = False

    if after_id and records:
        if after_id < oldest_id:
            selected_records = records[-tail_lines:]
            reset_required = True
        else:
            selected_records = [record for record in records if record["id"] > after_id]
    else:
        selected_records = records[-tail_lines:]

    messages = [record.get("raw", record.get("message", "")) for record in selected_records]
    content = "\n".join(messages)
    if content:
        content += "\n"

    return {
        "logs": content,
        "entries": selected_records,
        "last_id": last_id,
        "reset_required": reset_required,
        "tail_lines": tail_lines,
    }


def file_response_no_cache(path: str):
    return FileResponse(path, headers={"Cache-Control": "no-store"})


class DequeLogHandler(logging.Handler):
    def __init__(self, target_buffer: deque[str]):
        super().__init__()
        self.target_buffer = target_buffer

    def emit(self, record):
        try:
            message = self.format(record)
            if self.target_buffer is WEBUI_LOG_BUFFER:
                append_webui_log_entry(build_structured_entry_from_record(record, message))
            else:
                self.target_buffer.append(message)
        except Exception:
            self.handleError(record)


stream_handler = DequeLogHandler(WEBUI_LOG_BUFFER)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
if not logger.handlers:
    logger.addHandler(stream_handler)


def is_logged_in(request: Request) -> bool:
    token = request.cookies.get("webui_token", "")
    return token in TOKEN_ARRAY

def unauthorized_response(api: bool):
    if api:
        return JSONResponse(
            {"status": "error", "message": "Authentication required"},
            status_code=401,
        )
    return RedirectResponse(url="/", status_code=302)


def collect_system_stats():
    while True:
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/").percent
            net_io = psutil.net_io_counters()

            SYSTEM_STATS_HISTORY.append(
                {
                    "timestamp": datetime.utcnow(),
                    "cpu_usage": cpu,
                    "ram_usage_percent": mem.percent,
                    "ram_usage_mb": mem.used / (1024**2),
                    "disk_usage_percent": disk,
                    "net_sent": net_io.bytes_sent,
                    "net_recv": net_io.bytes_recv,
                }
            )
        except Exception as exc:
            logger.error(f"Error collecting system stats: {exc}")
        time.sleep(60)


@web_app.post("/api/log")
async def receive_log(request: Request):
    global last_heartbeat_time
    last_heartbeat_time = datetime.utcnow()

    data = await request.json()
    if data:
        log_entries = data.get("entries")
        if log_entries is None:
            log_entries = data.get("logs")
    else:
        log_entries = None

    if log_entries:
        try:
            for log_entry in log_entries:
                append_bot_log_entry(log_entry)
        except Exception as exc:
            logger.error(f"Failed to write logs: {exc}")
            return PlainTextResponse("Error writing logs", status_code=500)

    return PlainTextResponse("OK")


@web_app.get("/")
def config_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/main", status_code=302)
    return file_response_no_cache(login_page_path)


@web_app.post("/login")
async def login(request: Request):
    data = await request.json()
    token = data.get("token")
    remember_token = bool(data.get("rememberToken"))
    if token in TOKEN_ARRAY:
        response = JSONResponse({"message": "登录成功"})
        cookie_kwargs = {
            "key": "webui_token",
            "value": token,
            "httponly": True,
            "samesite": "lax",
            "path": "/",
        }
        if remember_token:
            cookie_kwargs["max_age"] = WEBUI_REMEMBER_ME_MAX_AGE
        response.set_cookie(**cookie_kwargs)
        return response
    return JSONResponse({"message": "无效的用户名或密码"}, status_code=401)


@web_app.get("/main")
def main_page(request: Request):
    if not is_logged_in(request):
        return unauthorized_response(api=False)
    return file_response_no_cache(main_page_path)


@web_app.get("/logout")
def logout(request: Request):
    response = RedirectResponse(url="/?logged_out=1", status_code=302)
    response.delete_cookie("webui_token", path="/")
    return response


@web_app.get("/api/status")
def get_status():
    global last_heartbeat_time
    time_diff = (datetime.utcnow() - last_heartbeat_time).total_seconds()
    status = "RUNNING" if time_diff < heartbeat_tolerance_seconds else "DOWN"
    return JSONResponse({"status": status})


@web_app.post("/api/bot/restart")
def restart_bot(request: Request):
    if not is_logged_in(request):
        return unauthorized_response(api=True)
    try:
        if docker is None:
            raise RuntimeError("Python package 'docker' is not installed.")
        client = docker.from_env()
        bot_container = client.containers.get(BOT_CONTAINER_NAME)
        bot_container.restart()
        return JSONResponse(
            {
                "status": "success",
                "message": f"Container '{BOT_CONTAINER_NAME}' is restarting.",
            }
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )


@web_app.post("/api/bot/shutdown")
def shutdown_bot(request: Request):
    if not is_logged_in(request):
        return unauthorized_response(api=True)
    try:
        if docker is None:
            raise RuntimeError("Python package 'docker' is not installed.")
        client = docker.from_env()
        bot_container = client.containers.get(BOT_CONTAINER_NAME)
        bot_container.stop()
        return JSONResponse(
            {
                "status": "success",
                "message": f"Container '{BOT_CONTAINER_NAME}' is stopping.",
            }
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )


@web_app.post("/api/web/restart")
def restart_web(request: Request):
    if not is_logged_in(request):
        return unauthorized_response(api=True)
    try:
        if docker is None:
            raise RuntimeError("Python package 'docker' is not installed.")
        client = docker.from_env()
        web_container = client.containers.get(WEB_CONTAINER_NAME)
        web_container.restart()
        return JSONResponse(
            {
                "status": "success",
                "message": f"Container '{WEB_CONTAINER_NAME}' is restarting.",
            }
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )


@web_app.get("/api/logs")
def get_logs(request: Request, date: str | None = None, after_id: int | None = None):
    if not is_logged_in(request):
        return unauthorized_response(api=True)
    current_date = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        payload = build_log_payload(BOT_LOG_BUFFER, BOT_LOG_TAIL_LINES, after_id)
        payload["date"] = current_date
        return JSONResponse(payload)
    except Exception as exc:
        logger.error(f"Error in get_logs: {exc}")
        return JSONResponse(
            {"status": "error", "message": "Internal server error"},
            status_code=500,
        )


@web_app.get("/api/webui-logs")
def get_webui_logs(request: Request, after_id: int | None = None):
    if not is_logged_in(request):
        return unauthorized_response(api=True)
    try:
        return JSONResponse(
            build_log_payload(WEBUI_LOG_BUFFER, WEBUI_LOG_TAIL_LINES, after_id)
        )
    except Exception as exc:
        logger.error(f"Error fetching webui logs from stream: {exc}")
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )


@web_app.get("/api/system-info")
def system_info(request: Request):
    if not is_logged_in(request):
        return unauthorized_response(api=True)
    try:
        cpu_usage_current = psutil.cpu_percent(interval=0.1)
        mem_current = psutil.virtual_memory()
        ram_usage_current_percent = mem_current.percent
        ram_usage_current_mb = mem_current.used / (1024**2)
        ram_total_mb = mem_current.total / (1024**2)
        disk_usage_details = psutil.disk_usage("/")
        net_io_current = psutil.net_io_counters()

        now = datetime.utcnow()
        past_24_hours = now - timedelta(hours=24)
        relevant_stats = [
            stat
            for stat in list(SYSTEM_STATS_HISTORY)
            if stat["timestamp"] > past_24_hours
        ]

        if relevant_stats:
            cpu_values = [stat["cpu_usage"] for stat in relevant_stats]
            ram_percent_values = [stat["ram_usage_percent"] for stat in relevant_stats]
            ram_mb_values = [stat["ram_usage_mb"] for stat in relevant_stats]
            cpu_avg_24h = sum(cpu_values) / len(cpu_values)
            cpu_peak_24h = max(cpu_values)
            ram_avg_24h_percent = sum(ram_percent_values) / len(ram_percent_values)
            ram_peak_24h_percent = max(ram_percent_values)
            ram_avg_24h_mb = sum(ram_mb_values) / len(ram_mb_values)
            ram_peak_24h_mb = max(ram_mb_values)
        else:
            cpu_avg_24h = cpu_peak_24h = 0
            ram_avg_24h_percent = ram_peak_24h_percent = 0
            ram_avg_24h_mb = ram_peak_24h_mb = 0

        return JSONResponse(
            {
                "current": {
                    "cpu_usage": cpu_usage_current,
                    "ram_usage_percent": ram_usage_current_percent,
                    "ram_usage_mb": ram_usage_current_mb,
                    "ram_total_mb": ram_total_mb,
                    "disk_usage": {
                        "total": f"{(disk_usage_details.total / (1024**3)):.2f} GB",
                        "used": f"{(disk_usage_details.used / (1024**3)):.2f} GB",
                        "free": f"{(disk_usage_details.free / (1024**3)):.2f} GB",
                        "percent": disk_usage_details.percent,
                    },
                    "net_io": {
                        "bytes_sent": net_io_current.bytes_sent,
                        "bytes_recv": net_io_current.bytes_recv,
                    },
                },
                "stats_24h": {
                    "cpu_avg": cpu_avg_24h,
                    "cpu_peak": cpu_peak_24h,
                    "ram_avg_percent": ram_avg_24h_percent,
                    "ram_peak_percent": ram_peak_24h_percent,
                    "ram_avg_mb": ram_avg_24h_mb,
                    "ram_peak_mb": ram_peak_24h_mb,
                    "sample_count": len(relevant_stats),
                },
            }
        )
    except Exception as exc:
        logger.error(f"Error in system_info: {exc}")
        return JSONResponse(
            {"status": "error", "message": "Internal server error"},
            status_code=500,
        )


logger.info("Initializing and starting background threads.")
stats_thread = threading.Thread(target=collect_system_stats, daemon=True)
stats_thread.start()
