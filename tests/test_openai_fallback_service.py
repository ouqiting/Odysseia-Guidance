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
async def test_build_channel_order_requires_full_three_channel_chain():
    order = OpenAIFallbackService.build_channel_order(
        "custom",
        "deepseek-v4-flash",
        "kimi-k2.6",
    )
    assert order == ["custom", "deepseek-v4-flash", "kimi-k2.6"]

    duplicated_order = OpenAIFallbackService.build_channel_order(
        "custom",
        "deepseek-v4-flash",
        "custom",
    )
    assert duplicated_order == []

    unsupported_order = OpenAIFallbackService.build_channel_order(
        "gemini-2.5-flash",
        "custom",
        "kimi-k2.6",
    )
    assert unsupported_order == []


def test_is_custom_preset_channel_recognizes_preset_form():
    assert OpenAIFallbackService.is_custom_preset_channel("custom-vercel-kimi") is True
    assert OpenAIFallbackService.is_custom_preset_channel("custom-kimchi白嫖") is True
    # 纯 custom 不算预设渠道（使用当前启用配置）
    assert OpenAIFallbackService.is_custom_preset_channel("custom") is False
    # 缺失预设名
    assert OpenAIFallbackService.is_custom_preset_channel("custom-") is False
    assert OpenAIFallbackService.is_custom_preset_channel("") is False
    assert OpenAIFallbackService.is_custom_preset_channel("deepseek-v4-flash") is False


def test_extract_custom_preset_name_splits_on_first_dash():
    assert (
        OpenAIFallbackService.extract_custom_preset_name("custom-vercel-kimi")
        == "vercel-kimi"
    )
    assert (
        OpenAIFallbackService.extract_custom_preset_name("custom-kimchi白嫖")
        == "kimchi白嫖"
    )
    assert OpenAIFallbackService.extract_custom_preset_name("custom") == ""
    assert OpenAIFallbackService.extract_custom_preset_name("deepseek-v4-flash") == ""


def test_is_supported_fallback_channel_accepts_custom_preset_for_secondary():
    # 基础模型仍受支持
    assert OpenAIFallbackService.is_supported_fallback_channel("deepseek-v4-flash") is True
    assert OpenAIFallbackService.is_supported_fallback_channel("kimi-k2.6") is True
    assert OpenAIFallbackService.is_supported_fallback_channel("custom") is True
    # custom-<preset> 仅对回退渠道（第 2 / 第 3）允许
    assert (
        OpenAIFallbackService.is_supported_fallback_channel("custom-vercel-kimi")
        is True
    )
    assert (
        OpenAIFallbackService.is_supported_fallback_channel("custom-kimchi白嫖")
        is True
    )
    # 无效预设名或其它模型
    assert OpenAIFallbackService.is_supported_fallback_channel("custom-") is False
    assert (
        OpenAIFallbackService.is_supported_fallback_channel("gemini-2.5-flash")
        is False
    )


def test_build_channel_order_accepts_custom_preset_secondary_and_tertiary():
    order = OpenAIFallbackService.build_channel_order(
        "custom",
        "custom-vercel-kimi",
        "custom-kimchi白嫖",
    )
    assert order == ["custom", "custom-vercel-kimi", "custom-kimchi白嫖"]

    # 主渠道仍必须是基础模型，不接受 custom-<preset>
    primary_preset_order = OpenAIFallbackService.build_channel_order(
        "custom-vercel-kimi",
        "deepseek-v4-flash",
        "kimi-k2.6",
    )
    assert primary_preset_order == []

    # 混合：custom + custom-<preset> + kimi
    mixed_order = OpenAIFallbackService.build_channel_order(
        "custom",
        "custom-vercel-kimi",
        "kimi-k2.6",
    )
    assert mixed_order == ["custom", "custom-vercel-kimi", "kimi-k2.6"]

    # custom 与 custom-<preset> 视为不同渠道；但第三渠道重复 custom 时
    # 仅剩 2 个不同渠道，不满足完整三渠道链要求
    dedup_order = OpenAIFallbackService.build_channel_order(
        "custom",
        "custom-vercel-kimi",
        "custom",
    )
    assert dedup_order == []

    # 缺失预设名（custom-）不视为有效渠道，导致回退链不完整
    invalid_order = OpenAIFallbackService.build_channel_order(
        "custom",
        "custom-",
        "kimi-k2.6",
    )
    assert invalid_order == []


@pytest.mark.asyncio
async def test_daily_state_tracks_custom_preset_channels(monkeypatch: pytest.MonkeyPatch):
    service = OpenAIFallbackService()
    fake_db = _FakeDBManager()
    service.db_manager = fake_db

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-18"),
    )

    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "custom-vercel-kimi"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "custom-kimchi白嫖"

    state = await service.get_daily_state("custom")
    assert state.order == ["custom", "custom-vercel-kimi", "custom-kimchi白嫖"]
    assert state.active_order == ["custom", "custom-vercel-kimi", "custom-kimchi白嫖"]

    # 失败一个预设渠道后应跳过它
    updated = await service.mark_channel_failed(
        primary_model="custom",
        channel_name="custom-vercel-kimi",
    )
    assert updated.failed_channels == ["custom-vercel-kimi"]
    assert updated.active_order == ["custom", "custom-kimchi白嫖"]

    # 重新加载仍保留失败状态
    reloaded = await service.get_daily_state("custom")
    assert reloaded.failed_channels == ["custom-vercel-kimi"]
    assert reloaded.active_order == ["custom", "custom-kimchi白嫖"]


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
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.6"

    state = await service.get_daily_state("custom")
    assert state.order == ["custom", "deepseek-v4-flash", "kimi-k2.6"]
    assert state.active_order == ["custom", "deepseek-v4-flash", "kimi-k2.6"]

    updated_state = await service.mark_channel_failed(
        primary_model="custom",
        channel_name="custom",
    )
    assert updated_state.failed_channels == ["custom"]
    assert updated_state.active_order == ["deepseek-v4-flash", "kimi-k2.6"]

    reloaded_state = await service.get_daily_state("custom")
    assert reloaded_state.failed_channels == ["custom"]
    assert reloaded_state.active_order == ["deepseek-v4-flash", "kimi-k2.6"]


@pytest.mark.asyncio
async def test_daily_state_resets_when_date_changes(monkeypatch: pytest.MonkeyPatch):
    service = OpenAIFallbackService()
    fake_db = _FakeDBManager()
    service.db_manager = fake_db

    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.6"

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
    assert reset_state.active_order == ["custom", "deepseek-v4-flash", "kimi-k2.6"]


@pytest.mark.asyncio
async def test_failure_state_is_in_memory_only_and_resets_after_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_db = _FakeDBManager()
    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.6"

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
    assert restarted_state.active_order == ["custom", "deepseek-v4-flash", "kimi-k2.6"]


@pytest.mark.asyncio
async def test_daily_state_restarts_from_first_channel_after_all_channels_failed(
    monkeypatch: pytest.MonkeyPatch,
):
    service = OpenAIFallbackService()
    fake_db = _FakeDBManager()
    service.db_manager = fake_db

    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = "kimi-k2.6"

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-18"),
    )

    await service.mark_channel_failed(primary_model="custom", channel_name="custom")
    await service.mark_channel_failed(
        primary_model="custom",
        channel_name="deepseek-v4-flash",
    )
    await service.mark_channel_failed(primary_model="custom", channel_name="kimi-k2.6")

    restarted_cycle_state = await service.get_daily_state("custom")

    assert restarted_cycle_state.failed_channels == []
    assert restarted_cycle_state.active_order == [
        "custom",
        "deepseek-v4-flash",
        "kimi-k2.6",
    ]


@pytest.mark.asyncio
async def test_get_daily_state_returns_empty_order_when_fallback_chain_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    service = OpenAIFallbackService()
    fake_db = _FakeDBManager()
    service.db_manager = fake_db

    monkeypatch.setattr(
        OpenAIFallbackService,
        "_get_today_str",
        staticmethod(lambda: "2026-05-18"),
    )

    fake_db.values[OPENAI_FALLBACK_SECONDARY_MODEL_KEY] = "deepseek-v4-flash"
    fake_db.values[OPENAI_FALLBACK_TERTIARY_MODEL_KEY] = ""

    state = await service.get_daily_state("custom")

    assert state.order == []
    assert state.failed_channels == []
    assert state.active_order == []
