import pathlib
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.chat.services.openai_fallback_service import OpenAIFallbackState

try:
    from src.chat.services.openai_service import (
        OpenAIChannelExecutionFailure,
        OpenAIChannelExecutionResult,
    )
    from src.chat.services.gemini_service import gemini_service
except ModuleNotFoundError as exc:
    if exc.name in {"pgvector", "asyncpg", "sqlalchemy"}:
        gemini_service = None
        OpenAIChannelExecutionFailure = None
        OpenAIChannelExecutionResult = None
    else:
        raise


@pytest.mark.asyncio
async def test_main_reply_uses_next_channel_after_same_day_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    if gemini_service is None:
        pytest.skip("optional database/vector dependencies are not installed.")
    fallback_states = [
        OpenAIFallbackState(
            date="2026-05-18",
            order=["custom", "deepseek-v4-flash", "kimi-k2.5"],
            failed_channels=[],
        ),
        OpenAIFallbackState(
            date="2026-05-18",
            order=["custom", "deepseek-v4-flash", "kimi-k2.5"],
            failed_channels=["custom"],
        ),
    ]

    async def _get_daily_state(primary_model: str):
        return fallback_states.pop(0)

    mark_channel_failed = AsyncMock(
        return_value=OpenAIFallbackState(
            date="2026-05-18",
            order=["custom", "deepseek-v4-flash", "kimi-k2.5"],
            failed_channels=["custom"],
        )
    )
    execute_channel_response = AsyncMock(
        side_effect=[
            OpenAIChannelExecutionFailure(
                channel_name="custom",
                user_message="custom failed",
                failure_kind="request_error",
                should_lock_channel=True,
            ),
            OpenAIChannelExecutionResult(
                channel_name="deepseek-v4-flash",
                response_text="fallback ok",
                used_model_name="deepseek-v4-flash",
            ),
            OpenAIChannelExecutionResult(
                channel_name="deepseek-v4-flash",
                response_text="second request ok",
                used_model_name="deepseek-v4-flash",
            ),
        ]
    )

    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.get_daily_state",
        _get_daily_state,
    )
    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.mark_channel_failed",
        mark_channel_failed,
    )
    monkeypatch.setattr(
        gemini_service,
        "consume_one_time_debug_base_url",
        lambda: None,
    )
    monkeypatch.setattr(
        gemini_service.openai_service,
        "execute_channel_response",
        execute_channel_response,
    )
    notify_alert = AsyncMock()
    monkeypatch.setattr(
        gemini_service.openai_service.kimi_model_client,
        "notify_alert",
        notify_alert,
    )

    first_result = await gemini_service.generate_response(
        user_id=1,
        guild_id=1,
        message="hello",
        channel=None,
        model_name="custom",
    )
    second_result = await gemini_service.generate_response(
        user_id=1,
        guild_id=1,
        message="hello again",
        channel=None,
        model_name="custom",
    )

    assert first_result == "fallback ok"
    assert second_result == "second request ok"
    assert [
        call.kwargs["model_name"] for call in execute_channel_response.await_args_list
    ] == [
        "custom",
        "deepseek-v4-flash",
        "deepseek-v4-flash",
    ]
    mark_channel_failed.assert_awaited_once_with(
        primary_model="custom",
        channel_name="custom",
    )
    notify_alert.assert_awaited_once()
    assert "失败渠道: custom" in notify_alert.await_args.args[0]
    assert "下一渠道: deepseek-v4-flash" in notify_alert.await_args.args[0]


@pytest.mark.asyncio
async def test_main_reply_continues_to_third_channel_after_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    if gemini_service is None:
        pytest.skip("optional database/vector dependencies are not installed.")

    fallback_state = OpenAIFallbackState(
        date="2026-05-20",
        order=["custom", "kimi-k2.5", "deepseek-v4-flash"],
        failed_channels=[],
    )

    async def _get_daily_state(primary_model: str):
        return fallback_state

    mark_channel_failed = AsyncMock(return_value=fallback_state)
    execute_channel_response = AsyncMock(
        side_effect=[
            OpenAIChannelExecutionFailure(
                channel_name="custom",
                user_message="custom failed",
                failure_kind="request_error",
                should_lock_channel=True,
            ),
            RuntimeError("kimi exploded unexpectedly"),
            OpenAIChannelExecutionResult(
                channel_name="deepseek-v4-flash",
                response_text="third channel ok",
                used_model_name="deepseek-v4-flash",
            ),
        ]
    )

    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.get_daily_state",
        _get_daily_state,
    )
    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.mark_channel_failed",
        mark_channel_failed,
    )
    monkeypatch.setattr(
        gemini_service,
        "consume_one_time_debug_base_url",
        lambda: None,
    )
    monkeypatch.setattr(
        gemini_service.openai_service,
        "execute_channel_response",
        execute_channel_response,
    )
    notify_alert = AsyncMock()
    monkeypatch.setattr(
        gemini_service.openai_service.kimi_model_client,
        "notify_alert",
        notify_alert,
    )

    result = await gemini_service.generate_response(
        user_id=1,
        guild_id=1,
        message="hello",
        channel=None,
        model_name="custom",
    )

    assert result == "third channel ok"
    assert [
        call.kwargs["model_name"] for call in execute_channel_response.await_args_list
    ] == ["custom", "kimi-k2.5", "deepseek-v4-flash"]
    assert mark_channel_failed.await_count == 2


@pytest.mark.asyncio
async def test_main_reply_does_not_fallback_on_non_locking_channel_error(
    monkeypatch: pytest.MonkeyPatch,
):
    if gemini_service is None:
        pytest.skip("optional database/vector dependencies are not installed.")

    fallback_state = OpenAIFallbackState(
        date="2026-05-23",
        order=["custom", "kimi-k2.5", "deepseek-v4-flash"],
        failed_channels=[],
    )

    async def _get_daily_state(primary_model: str):
        return fallback_state

    mark_channel_failed = AsyncMock(return_value=fallback_state)
    execute_channel_response = AsyncMock(
        side_effect=[
            OpenAIChannelExecutionFailure(
                channel_name="custom",
                user_message="哎呀，我好像陷入了一个复杂的思考循环里，换个话题聊聊吧！",
                failure_kind="max_calls_exceeded",
                should_lock_channel=False,
            ),
        ]
    )

    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.get_daily_state",
        _get_daily_state,
    )
    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.mark_channel_failed",
        mark_channel_failed,
    )
    monkeypatch.setattr(
        gemini_service,
        "consume_one_time_debug_base_url",
        lambda: None,
    )
    monkeypatch.setattr(
        gemini_service.openai_service,
        "execute_channel_response",
        execute_channel_response,
    )

    result = await gemini_service.generate_response(
        user_id=1,
        guild_id=1,
        message="hello",
        channel=None,
        model_name="custom",
    )

    assert result == "哎呀，我好像陷入了一个复杂的思考循环里，换个话题聊聊吧！"
    assert [
        call.kwargs["model_name"] for call in execute_channel_response.await_args_list
    ] == ["custom"]
    mark_channel_failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_reply_skips_fallback_flow_when_secondary_or_tertiary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    if gemini_service is None:
        pytest.skip("optional database/vector dependencies are not installed.")

    fallback_state = OpenAIFallbackState(
        date="2026-05-23",
        order=[],
        failed_channels=[],
    )

    async def _get_daily_state(primary_model: str):
        return fallback_state

    generate_response = AsyncMock(return_value="primary channel error surfaced")
    execute_channel_response = AsyncMock()

    monkeypatch.setattr(
        "src.chat.services.gemini_service.openai_fallback_service.get_daily_state",
        _get_daily_state,
    )
    monkeypatch.setattr(
        gemini_service,
        "consume_one_time_debug_base_url",
        lambda: None,
    )
    monkeypatch.setattr(
        gemini_service.openai_service,
        "generate_response",
        generate_response,
    )
    monkeypatch.setattr(
        gemini_service.openai_service,
        "execute_channel_response",
        execute_channel_response,
    )

    result = await gemini_service.generate_response(
        user_id=1,
        guild_id=1,
        message="hello",
        channel=None,
        model_name="custom",
    )

    assert result == "primary channel error surfaced"
    generate_response.assert_awaited_once()
    execute_channel_response.assert_not_awaited()
