# -*- coding: utf-8 -*-

"""自定义 Gemini 通道（CUSTOM_GEMINI_URL）的重试上限测试。

背景：自定义通道只有一把 Key，遇到 503 时原来的循环会无限重试并刷屏
（日志里会出现成百上千条 “可重试错误 (503)”）。这里验证重试轮数被限制住。
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import errors as genai_errors

from src.chat.config import chat_config as app_config
from src.chat.services.gemini_service import GeminiService

CUSTOM_CHANNEL_ENV = {
    "CUSTOM_GEMINI_URL": "https://gemini.example.com",
    "CUSTOM_GEMINI_API_KEY": "custom-channel-key",
}


def _build_service_with_client(client_instance: MagicMock) -> GeminiService:
    """在自定义通道配置下构造服务，并让 genai.Client 返回给定替身。"""
    with patch("google.genai.Client", return_value=client_instance):
        return GeminiService()


class TestCustomChannelRetryCap(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client_instance = MagicMock()
        self.client_instance.models.generate_content.side_effect = (
            genai_errors.ServerError(503, {"error": {"message": "upstream unavailable"}})
        )

        env_patcher = patch.dict(os.environ, CUSTOM_CHANNEL_ENV)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        client_patcher = patch("google.genai.Client", return_value=self.client_instance)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)

        self.service = GeminiService()
        # 只走自定义通道，不涉及官方 Key 轮换
        self.service.key_rotation_service = None

    def _patch_sleep(self):
        sleep_patcher = patch(
            "src.chat.services.gemini_service.asyncio.sleep", new=AsyncMock()
        )
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)
        return sleep_patcher

    async def test_503_retry_is_capped_by_max_cycles(self):
        self._patch_sleep()

        with patch.dict(
            app_config.API_RETRY_CONFIG, {"MAX_CYCLES_PER_CUSTOM_CHANNEL": 3}
        ):
            result = await self.service.generate_simple_response(
                "hello", {"temperature": 1.0}, model_name="gemini-3-flash-preview"
            )

        self.assertEqual(result, "服务连接失败，请稍后再试。")
        self.assertEqual(
            self.client_instance.models.generate_content.call_count,
            3,
            "自定义通道 503 重试次数应被 MAX_CYCLES_PER_CUSTOM_CHANNEL 限制",
        )

    async def test_max_cycles_is_configurable(self):
        self._patch_sleep()

        with patch.dict(
            app_config.API_RETRY_CONFIG, {"MAX_CYCLES_PER_CUSTOM_CHANNEL": 1}
        ):
            result = await self.service.generate_simple_response(
                "hello", {"temperature": 1.0}, model_name="gemini-3-flash-preview"
            )

        self.assertEqual(result, "服务连接失败，请稍后再试。")
        self.assertEqual(self.client_instance.models.generate_content.call_count, 1)

    async def test_success_after_one_retry_returns_text(self):
        self._patch_sleep()

        success_response = MagicMock()
        success_response.parts = [MagicMock(text="成功获得回复")]
        success_response.text = "成功获得回复"

        self.client_instance.models.generate_content.side_effect = [
            genai_errors.ServerError(503, {"error": {"message": "unavailable"}}),
            success_response,
        ]

        with patch.dict(
            app_config.API_RETRY_CONFIG, {"MAX_CYCLES_PER_CUSTOM_CHANNEL": 3}
        ):
            result = await self.service.generate_simple_response(
                "hello", {"temperature": 1.0}, model_name="gemini-3-flash-preview"
            )

        self.assertEqual(result, "成功获得回复")
        self.assertEqual(self.client_instance.models.generate_content.call_count, 2)

    def test_default_max_cycles_is_three(self):
        self.assertEqual(
            app_config.API_RETRY_CONFIG["MAX_CYCLES_PER_CUSTOM_CHANNEL"], 3
        )

    async def test_non_retryable_error_returns_single_attempt(self):
        """非可重试错误（例如 400）不应进入重试循环。"""
        self._patch_sleep()
        self.client_instance.models.generate_content.side_effect = (
            genai_errors.ClientError(400, {"error": {"message": "invalid argument"}})
        )

        result = await self.service.generate_simple_response(
            "hello", {"temperature": 1.0}, model_name="gemini-3-flash-preview"
        )

        # 自定义通道的非重试错误直接返回失败提示，不重试
        self.assertEqual(result, "服务连接失败，请稍后再试。")
        self.assertEqual(self.client_instance.models.generate_content.call_count, 1)


if __name__ == "__main__":
    unittest.main()
