# -*- coding: utf-8 -*-

from types import SimpleNamespace
from unittest.mock import AsyncMock
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import src.chat.features.chat_settings.services.chat_settings_service as chat_settings_service_module


@pytest.mark.asyncio
async def test_full_context_logging_uses_existing_debug_flags_as_default(
    monkeypatch: pytest.MonkeyPatch,
):
    service = chat_settings_service_module.ChatSettingsService()
    service.db_manager = SimpleNamespace(get_global_setting=AsyncMock(return_value=None))

    monkeypatch.setitem(
        chat_settings_service_module.app_config.DEBUG_CONFIG,
        "LOG_FINAL_CONTEXT",
        False,
    )
    monkeypatch.setitem(
        chat_settings_service_module.app_config.DEBUG_CONFIG,
        "LOG_AI_FULL_CONTEXT",
        False,
    )
    monkeypatch.setitem(
        chat_settings_service_module.app_config.DEBUG_CONFIG,
        "LOG_DETAILED_GEMINI_PROCESS",
        True,
    )

    assert await service.get_full_context_logging_enabled() is True


@pytest.mark.asyncio
async def test_logging_settings_are_persisted_to_global_settings():
    set_global_setting = AsyncMock()
    service = chat_settings_service_module.ChatSettingsService()
    service.db_manager = SimpleNamespace(set_global_setting=set_global_setting)

    await service.set_full_context_logging_enabled(False)
    await service.set_final_reply_logging_enabled(True)

    assert set_global_setting.await_args_list[0].args == (
        chat_settings_service_module.FULL_CONTEXT_LOGGING_ENABLED_KEY,
        "false",
    )
    assert set_global_setting.await_args_list[1].args == (
        chat_settings_service_module.FINAL_REPLY_LOGGING_ENABLED_KEY,
        "true",
    )


def test_final_reply_logging_defaults_to_enabled_when_setting_missing():
    service = chat_settings_service_module.ChatSettingsService()
    service.db_manager = SimpleNamespace(get_global_setting_sync=lambda key: None)

    assert service.get_final_reply_logging_enabled_sync() is True
