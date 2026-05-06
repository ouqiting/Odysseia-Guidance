# -*- coding: utf-8 -*-

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import DATA_DIR


class PixivStorage:
    def __init__(self, db_path: str | None = None, retention_days: int = 7):
        self.db_path = db_path or str(Path(DATA_DIR) / "pixiv_tool.db")
        self.retention_days = max(1, int(retention_days))

    async def init_async(self) -> None:
        await asyncio.to_thread(self._init_db)

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sent_illusts (
                    illust_id INTEGER PRIMARY KEY,
                    sent_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    async def mark_sent(self, illust_id: int) -> None:
        await asyncio.to_thread(self._mark_sent_sync, int(illust_id))

    def _mark_sent_sync(self, illust_id: int) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sent_illusts (illust_id, sent_at)
                VALUES (?, ?)
                ON CONFLICT(illust_id) DO UPDATE SET sent_at=excluded.sent_at
                """,
                (illust_id, timestamp),
            )
            conn.commit()

    async def prune_old_entries(self) -> None:
        await asyncio.to_thread(self._prune_old_entries_sync)

    def _prune_old_entries_sync(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM sent_illusts WHERE sent_at < ?",
                (cutoff.isoformat(),),
            )
            conn.commit()

    async def get_recent_sent_ids(self, limit: int = 100) -> set[int]:
        return await asyncio.to_thread(self._get_recent_sent_ids_sync, int(limit))

    def _get_recent_sent_ids_sync(self, limit: int) -> set[int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT illust_id
                FROM sent_illusts
                ORDER BY sent_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return {int(row[0]) for row in rows}
