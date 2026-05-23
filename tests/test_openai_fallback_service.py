import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.chat.services.openai_fallback_service import (
    OPENAI_FALLBACK_SECONDARY_MODEL_KEY,
    OPENAI_FALLBACK_TERTIARY_MODEL_KEY,
    OpenAIFallbackService,
)


class _FakeDBManager:
    def __init__(self):
        self.values = {}

    async def get_global_setting(self, key: str):
        return self.values.get(key)

    async def set_global_setting(self, key: str, value: str):
        self.values[key] = value


@pytest.mark.asyncio
async def test_build_channel_order_deduplicates_and_filters_unsupported():
    order = OpenAIFallbackService.build_channel_order(
        "custom",
        "deepseek-v4-flash",
        "custom",
    )
    assert order == ["custom", "deepseek-v4-flash"]

    unsupported_order = OpenAIFallbackService.build_channel_order(
        "gemini-2.5-flash",
        "custom",
        "kimi-k2.5",
    )
    assert unsupported_order == []


@pytest.mark.asyncio
async def test_mark_channel_failed_skips_it_for_rest_of_day(monkeypatch: pytest.MonkeyPatch):
    service = OpenAIFallbackService()
    fake_db = _FakeDBManager()
    service.db_manager = fake_db

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-18"),
    )

    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.5"

    state = await service.get_daily_state("custom")
    assert state.order == ["custom", "deepseek-v4-flash", "kimi-k2.5"]
    assert state.active_order == ["custom", "deepseek-v4-flash", "kimi-k2.5"]

    updated_state = await service.mark_channel_failed(
        primary_model="custom",
        channel_name="custom",
    )
    assert updated_state.failed_channels == ["custom"]
    assert updated_state.active_order == ["deepseek-v4-flash", "kimi-k2.5"]

    reloaded_state = await service.get_daily_state("custom")
    assert reloaded_state.failed_channels == ["custom"]
    assert reloaded_state.active_order == ["deepseek-v4-flash", "kimi-k2.5"]


@pytest.mark.asyncio
async def test_daily_state_resets_when_date_changes(monkeypatch: pytest.MonkeyPatch):
    service = OpenAIFallbackService()
    fake_db = _FakeDBManager()
    service.db_manager = fake_db

    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.5"

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-18"),
    )
    await service.mark_channel_failed(
        primary_model="custom",
        channel_name="custom",
    )

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-19"),
    )
    reset_state = await service.get_daily_state("custom")

    assert reset_state.date == "2026-05-19"
    assert reset_state.failed_channels == []
    assert reset_state.active_order == ["custom", "deepseek-v4-flash", "kimi-k2.5"]


@pytest.mark.asyncio
async def test_failure_state_is_in_memory_only_and_resets_after_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_db = _FakeDBManager()
    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.5"

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-18"),
    )

    first_service = OpenAIFallbackService()
    first_service.db_manager = fake_db
    await first_service.mark_channel_failed(
        primary_model="custom",
        channel_name="custom",
    )

    first_state = await first_service.get_daily_state("custom")
    assert first_state.failed_channels == ["custom"]

    restarted_service = OpenAIFallbackService()
    restarted_service.db_manager = fake_db
    restarted_state = await restarted_service.get_daily_state("custom")

    assert restarted_state.failed_channels == []
    assert restarted_state.active_order == ["custom", "deepseek-v4-flash", "kimi-k2.5"]
