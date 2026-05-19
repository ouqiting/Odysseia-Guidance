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
            order=["custom", "deepseek-chat", "kimi-k2.5"],
            failed_channels=[],
        ),
        OpenAIFallbackState(
            date="2026-05-18",
            order=["custom", "deepseek-chat", "kimi-k2.5"],
            failed_channels=["custom"],
        ),
    ]

    async def _get_daily_state(primary_model: str):
        return fallback_states.pop(0)

    mark_channel_failed = AsyncMock(
        return_value=OpenAIFallbackState(
            date="2026-05-18",
            order=["custom", "deepseek-chat", "kimi-k2.5"],
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
                channel_name="deepseek-chat",
                response_text="fallback ok",
                used_model_name="deepseek-chat",
            ),
            OpenAIChannelExecutionResult(
                channel_name="deepseek-chat",
                response_text="second request ok",
                used_model_name="deepseek-chat",
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
        "deepseek-chat",
        "deepseek-chat",
    ]
    mark_channel_failed.assert_awaited_once_with(
        primary_model="custom",
        channel_name="custom",
    )
    notify_alert.assert_awaited_once()
    assert "失败渠道: custom" in notify_alert.await_args.args[0]
    assert "下一渠道: deepseek-chat" in notify_alert.await_args.args[0]
