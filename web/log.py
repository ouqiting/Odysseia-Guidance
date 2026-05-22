import json
import os
import threading
from pathlib import Path
from typing import Any


class BotLogStore:
    def __init__(self, log_dir: str, max_bytes: int, backup_count: int):
        self.log_dir = Path(log_dir)
        self.max_bytes = max(1, int(max_bytes))
        self.backup_count = max(1, int(backup_count))
        self.current_file_name = "bot.log"
        self._lock = threading.Lock()
        self._current_entries: list[dict[str, Any]] = []
        self._current_file_name = self.current_file_name
        self._last_id = 0
        self._rotation_epoch = 0
        self._ensure_directory()
        self._initialize_state()

    @property
    def current_file(self) -> str:
        return self._current_file_name

    @property
    def last_id(self) -> int:
        return self._last_id

    @property
    def rotation_epoch(self) -> int:
        return self._rotation_epoch

    def append_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        normalized_entry = dict(entry)
        with self._lock:
            encoded_entry = self._encode_entry(normalized_entry)
            if self._should_rotate_before_append(len(encoded_entry)):
                self._rotate_files()
                self._current_entries = []
                self._rotation_epoch += 1

            self._last_id += 1
            normalized_entry["id"] = self._last_id
            encoded_entry = self._encode_entry(normalized_entry)

            with self._current_file_path.open("ab") as handle:
                handle.write(encoded_entry)

            self._current_entries.append(normalized_entry)
            return dict(normalized_entry)

    def get_latest_payload(self, after_id: int | None = None) -> dict[str, Any]:
        with self._lock:
            reset_required = False
            if after_id is not None and self._current_entries:
                oldest_id = self._current_entries[0]["id"]
                if after_id < oldest_id:
                    selected_entries = list(self._current_entries)
                    reset_required = True
                else:
                    selected_entries = [
                        dict(entry) for entry in self._current_entries if entry["id"] > after_id
                    ]
            else:
                selected_entries = [dict(entry) for entry in self._current_entries]

            next_before = self._get_next_before_file_locked(self.current_file_name)
            return {
                "logs": "",
                "entries": selected_entries,
                "last_id": self._last_id,
                "reset_required": reset_required,
                "tail_lines": None,
                "current_file": self.current_file_name,
                "next_before": next_before,
                "has_more": next_before is not None,
                "rotation_epoch": self._rotation_epoch,
            }

    def get_history_payload(self, before_file: str) -> dict[str, Any]:
        normalized_name = self._normalize_history_file_name(before_file)
        with self._lock:
            if normalized_name == self.current_file_name:
                raise ValueError("before_file must reference a rotated history file")

            file_path = self.log_dir / normalized_name
            if not file_path.is_file():
                raise FileNotFoundError(normalized_name)

            next_before = self._get_next_before_file_locked(normalized_name)
            return {
                "logs": "",
                "entries": self._read_entries_from_file(file_path),
                "last_id": self._last_id,
                "reset_required": False,
                "tail_lines": None,
                "current_file": self.current_file_name,
                "loaded_file": normalized_name,
                "next_before": next_before,
                "has_more": next_before is not None,
                "rotation_epoch": self._rotation_epoch,
            }

    def _ensure_directory(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_state(self) -> None:
        current_path = self._current_file_path
        if current_path.is_file():
            self._current_entries = self._read_entries_from_file(current_path)

        self._last_id = self._detect_last_id()

    def _detect_last_id(self) -> int:
        history_files = self._list_log_files_desc()
        for file_path in history_files:
            last_id = self._read_last_id_from_file(file_path)
            if last_id > 0:
                return last_id
        return 0

    def _read_last_id_from_file(self, file_path: Path) -> int:
        try:
            with file_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                file_size = handle.tell()
                if file_size <= 0:
                    return 0

                chunk_size = min(4096, file_size)
                handle.seek(-chunk_size, os.SEEK_END)
                chunk = handle.read(chunk_size).decode("utf-8", errors="ignore")
        except OSError:
            return 0

        for line in reversed(chunk.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_id = entry.get("id")
            if isinstance(entry_id, int):
                return entry_id
        return 0

    @property
    def _current_file_path(self) -> Path:
        return self.log_dir / self.current_file_name

    def _should_rotate_before_append(self, entry_size_bytes: int) -> bool:
        current_path = self._current_file_path
        if not current_path.is_file():
            return False
        return current_path.stat().st_size + entry_size_bytes > self.max_bytes

    def _rotate_files(self) -> None:
        oldest_backup = self.log_dir / f"{self.current_file_name}.{self.backup_count}"
        if oldest_backup.exists():
            oldest_backup.unlink()

        for index in range(self.backup_count - 1, 0, -1):
            source = self.log_dir / f"{self.current_file_name}.{index}"
            if source.exists():
                source.rename(self.log_dir / f"{self.current_file_name}.{index + 1}")

        current_path = self._current_file_path
        if current_path.exists():
            current_path.rename(self.log_dir / f"{self.current_file_name}.1")

    def _list_log_files_desc(self) -> list[Path]:
        files: list[tuple[int, Path]] = []
        for path in self.log_dir.iterdir():
            if not path.is_file():
                continue
            if path.name == self.current_file_name:
                files.append((0, path))
                continue
            prefix = f"{self.current_file_name}."
            if not path.name.startswith(prefix):
                continue
            suffix = path.name[len(prefix) :]
            if suffix.isdigit():
                files.append((int(suffix), path))

        return [path for _, path in sorted(files, key=lambda item: item[0])]

    def _get_next_before_file_locked(self, reference_file_name: str) -> str | None:
        if reference_file_name == self.current_file_name:
            candidate = self.log_dir / f"{self.current_file_name}.1"
            return candidate.name if candidate.is_file() else None

        try:
            current_index = int(reference_file_name.rsplit(".", 1)[-1])
        except (ValueError, IndexError):
            return None

        next_candidate = self.log_dir / f"{self.current_file_name}.{current_index + 1}"
        return next_candidate.name if next_candidate.is_file() else None

    def _normalize_history_file_name(self, file_name: str) -> str:
        normalized = os.path.basename(file_name.strip())
        if not normalized.startswith(f"{self.current_file_name}."):
            raise ValueError("Invalid history file name")
        suffix = normalized[len(self.current_file_name) + 1 :]
        if not suffix.isdigit():
            raise ValueError("Invalid history file name")
        return normalized

    def _read_entries_from_file(self, file_path: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict):
                        entries.append(entry)
        except OSError:
            return []
        return entries

    def _encode_entry(self, entry: dict[str, Any]) -> bytes:
        return (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
