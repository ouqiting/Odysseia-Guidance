import importlib
import sys
import threading
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self):
        return None


def _load_web_main_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_bytes: int = 1024 * 1024,
    backup_count: int = 20,
):
    monkeypatch.setenv("WEBUI_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("WEBUI_BOT_LOG_DIR", str(tmp_path / "bot-logs"))
    monkeypatch.setenv("WEBUI_BOT_LOG_MAX_BYTES", str(max_bytes))
    monkeypatch.setenv("WEBUI_BOT_LOG_BACKUP_COUNT", str(backup_count))
    import src.runtime_env as runtime_env

    monkeypatch.setattr(runtime_env, "load_project_dotenv", lambda *args, **kwargs: "")

    sys.modules.pop("web.web_main", None)
    original_thread = threading.Thread
    monkeypatch.setattr(threading, "Thread", _DummyThread)
    try:
        module = importlib.import_module("web.web_main")
        return importlib.reload(module)
    finally:
        setattr(threading, "Thread", original_thread)


def _authorized_client(module) -> TestClient:
    client = TestClient(module.web_app)
    client.cookies.set("webui_token", "test-token")
    return client


def _make_log_entry(index: int, message_size: int = 32) -> dict[str, str]:
    message = f"log-{index}-" + ("x" * message_size)
    return {
        "timestamp": f"2026-05-23T00:00:{index:02d}.000Z",
        "level": "INFO",
        "logger": "tests.bot",
        "message": message,
        "raw": f"[2026-05-23T00:00:{index:02d}.000Z] [INFO] [tests.bot] {message}",
    }


def _make_debug_log_entry(index: int) -> dict[str, str]:
    message = f"debug-{index}"
    return {
        "timestamp": f"2026-05-23T00:10:{index:02d}.000Z",
        "level": "DEBUG",
        "logger": "tests.bot",
        "message": message,
        "raw": f"[2026-05-23T00:10:{index:02d}.000Z] [DEBUG] [tests.bot] {message}",
    }


def test_get_logs_returns_latest_file_and_supports_history_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_web_main_module(monkeypatch, tmp_path, max_bytes=280, backup_count=5)
    client = _authorized_client(module)

    for index in range(1, 5):
        response = client.post("/api/log", json={"entries": [_make_log_entry(index, message_size=180)]})
        assert response.status_code == 200

    latest_response = client.get("/api/logs")
    assert latest_response.status_code == 200
    latest_payload = latest_response.json()

    assert latest_payload["current_file"] == "bot.log"
    assert latest_payload["next_before"] == "bot.log.1"
    assert latest_payload["has_more"] is True
    assert [entry["id"] for entry in latest_payload["entries"]] == [4]

    log_dir = tmp_path / "bot-logs"
    assert (log_dir / "bot.log").is_file()
    assert (log_dir / "bot.log.1").is_file()

    history_one = client.get("/api/logs", params={"before_file": "bot.log.1"})
    assert history_one.status_code == 200
    history_one_payload = history_one.json()
    assert history_one_payload["loaded_file"] == "bot.log.1"
    assert [entry["id"] for entry in history_one_payload["entries"]] == [3]
    assert history_one_payload["next_before"] == "bot.log.2"

    history_two = client.get("/api/logs", params={"before_file": "bot.log.2"})
    assert history_two.status_code == 200
    history_two_payload = history_two.json()
    assert [entry["id"] for entry in history_two_payload["entries"]] == [2]
    assert history_two_payload["next_before"] == "bot.log.3"

    history_three = client.get("/api/logs", params={"before_file": "bot.log.3"})
    assert history_three.status_code == 200
    history_three_payload = history_three.json()
    assert [entry["id"] for entry in history_three_payload["entries"]] == [1]
    assert history_three_payload["next_before"] is None
    assert history_three_payload["has_more"] is False


def test_get_logs_after_id_returns_only_new_entries_from_current_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_web_main_module(monkeypatch, tmp_path)
    client = _authorized_client(module)

    client.post("/api/log", json={"entries": [_make_log_entry(1), _make_log_entry(2)]})
    initial_payload = client.get("/api/logs").json()
    assert [entry["id"] for entry in initial_payload["entries"]] == [1, 2]

    client.post("/api/log", json={"entries": [_make_log_entry(3)]})
    incremental_response = client.get("/api/logs", params={"after_id": initial_payload["last_id"]})
    assert incremental_response.status_code == 200
    incremental_payload = incremental_response.json()

    assert [entry["id"] for entry in incremental_payload["entries"]] == [3]
    assert incremental_payload["reset_required"] is False
    assert incremental_payload["current_file"] == "bot.log"


def test_debug_entries_are_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_web_main_module(monkeypatch, tmp_path)
    client = _authorized_client(module)

    response = client.post(
        "/api/log",
        json={"entries": [_make_debug_log_entry(1), _make_log_entry(2)]},
    )
    assert response.status_code == 200

    payload = client.get("/api/logs").json()
    assert [entry["id"] for entry in payload["entries"]] == [1]
    assert [entry["level"] for entry in payload["entries"]] == ["INFO"]
    assert all("debug-1" not in entry["message"] for entry in payload["entries"])


def test_log_store_recovers_last_id_after_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_web_main_module(monkeypatch, tmp_path)
    client = _authorized_client(module)
    client.post("/api/log", json={"entries": [_make_log_entry(1), _make_log_entry(2)]})

    reloaded_module = _load_web_main_module(monkeypatch, tmp_path)
    reloaded_client = _authorized_client(reloaded_module)
    reloaded_client.post("/api/log", json={"entries": [_make_log_entry(3)]})

    payload = reloaded_client.get("/api/logs").json()
    assert [entry["id"] for entry in payload["entries"]] == [1, 2, 3]
    assert payload["last_id"] == 3


def test_get_logs_requires_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_web_main_module(monkeypatch, tmp_path)
    client = TestClient(module.web_app)

    response = client.get("/api/logs")
    assert response.status_code == 401
    assert response.json()["message"] == "Authentication required"
