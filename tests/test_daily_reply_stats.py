import os
import sys
import pytest
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.chat.utils.database import ChatDatabaseManager


pytestmark = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_get_total_reply_count_today_aggregates_daily_model_usage(monkeypatch):
    manager = ChatDatabaseManager(db_path=":memory:")
    manager._execute = AsyncMock(return_value={"total": 321})

    monkeypatch.setattr(
        "src.chat.utils.database.get_beijing_today_str", lambda: "2026-05-03"
    )

    result = await manager.get_total_reply_count_today()

    assert result == 321
    manager._execute.assert_awaited_once_with(
        manager._db_transaction,
        """
            SELECT COALESCE(SUM(usage_count), 0) AS total
            FROM daily_model_usage
            WHERE usage_date = ?
        """,
        ("2026-05-03",),
        fetch="one",
    )


@pytest.mark.asyncio
async def test_get_total_reply_count_yesterday_aggregates_daily_model_usage(
    monkeypatch,
):
    manager = ChatDatabaseManager(db_path=":memory:")
    manager._execute = AsyncMock(return_value={"total": 347})

    monkeypatch.setattr(
        "src.chat.utils.database.get_beijing_relative_date_str",
        lambda _: "2026-05-02",
    )

    result = await manager.get_total_reply_count_yesterday()

    assert result == 347
    manager._execute.assert_awaited_once_with(
        manager._db_transaction,
        """
            SELECT COALESCE(SUM(usage_count), 0) AS total
            FROM daily_model_usage
            WHERE usage_date = ?
        """,
        ("2026-05-02",),
        fetch="one",
    )
