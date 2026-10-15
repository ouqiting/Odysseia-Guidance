# -*- coding: utf-8 -*-

"""投喂文字阶段（图文评价）链路的单元测试。

覆盖点：
- provider 分发（gemini / openai）
- gemini 链路固定使用自定义端点，模型名可配置
- openai 链路 URL / Key / 模型名可配置，图片以 data URI 上传
- 两条链路共用图片压缩
- 上游失败时返回 None（不把错误文本当成评价）
- 可重试错误（503/429）有次数上限，整体受 120 秒超时限制
"""

import asyncio
import base64
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PIL import Image

from src.chat.config import chat_config as app_config
from src.chat.features.affection.service.feeding_text_service import (
    FeedingTextService,
    FeedingTextTimeoutError,
)

GEMINI_ENV = {
    "FEEDING_TEXT_PROVIDER": "gemini",
    "CUSTOM_GEMINI_URL": "https://gemini.example.com",
    "CUSTOM_GEMINI_API_KEY": "gemini-test-key",
    "FEEDING_GEMINI_MODEL": "gemini-3-flash-preview",
}

OPENAI_ENV = {
    "FEEDING_TEXT_PROVIDER": "openai",
    "FEEDING_OPENAI_URL": "https://feed.example.com/v1",
    "FEEDING_OPENAI_API_KEY": "sk-test",
    "FEEDING_OPENAI_MODEL": "gpt-4o-mini",
}


def _build_png(size: int = 8, noise: bool = False) -> bytes:
    if noise:
        image = Image.frombytes("RGB", (size, size), os.urandom(size * size * 3))
    else:
        image = Image.new("RGB", (size, size), (200, 120, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _openai_json_response(text: str, *, status_code: int = 200) -> SimpleNamespace:
    return _FakeResponse(
        status_code=status_code,
        json_payload={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        },
    )


class _FakeResponse:
    """最小化的 httpx.Response 替身（仅实现被测代码用到的接口）。"""

    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict | None = None,
        json_payload: object | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = (
            headers if headers is not None else {"content-type": "application/json"}
        )
        self._json_payload = json_payload

    def json(self):
        if self._json_payload is None:
            raise ValueError("no json payload")
        return self._json_payload


class _FakeGeminiClient:
    """最小化的 genai.Client 替身。"""

    def __init__(self, response: object) -> None:
        self.aio = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(return_value=response)
            ),
            aclose=AsyncMock(),
        )
        self.close = MagicMock()


class _FakeHttpClient:
    """最小化的 httpx.AsyncClient 替身，按顺序抛出或返回预置结果。"""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.post_calls = 0

    async def __aenter__(self) -> "_FakeHttpClient":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def post(self, *args, **kwargs):
        self.post_calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeUpstreamError(Exception):
    """模拟 google-genai 的 503 报错（状态码出现在错误文本里）。"""


def _patch_sleep():
    """让重试退避不真的等待。"""
    return patch(
        "src.chat.features.affection.service.feeding_text_service.asyncio.sleep",
        new=AsyncMock(),
    )


class TestFeedingTextHelpers(unittest.TestCase):
    def test_build_chat_completions_url(self):
        build_url = FeedingTextService._build_chat_completions_url

        self.assertEqual(
            build_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            build_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            build_url("https://proxy.example.com/v1/chat/completions"),
            "https://proxy.example.com/v1/chat/completions",
        )

    def test_build_image_data_uri(self):
        data_uri = FeedingTextService._build_image_data_uri(b"\x00\x01", "image/webp")

        self.assertTrue(data_uri.startswith("data:image/webp;base64,"))
        self.assertEqual(
            base64.b64decode(data_uri.split(",", 1)[1]),
            b"\x00\x01",
        )

    def test_build_image_data_uri_falls_back_to_png_mime(self):
        data_uri = FeedingTextService._build_image_data_uri(b"abc", "")

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_extract_response_text_from_json_body(self):
        response = _openai_json_response("评价<affection:+5;coins:+50>")

        self.assertEqual(
            FeedingTextService._extract_response_text(response),
            "评价<affection:+5;coins:+50>",
        )

    def test_extract_response_text_from_block_content(self):
        response = _FakeResponse(
            json_payload={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "你好"},
                                {"type": "text", "text": "呀"},
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(FeedingTextService._extract_response_text(response), "你好呀")

    def test_extract_response_text_from_sse_body(self):
        body = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"你"}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"好"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        response = _FakeResponse(
            text=body, headers={"content-type": "text/event-stream"}
        )

        self.assertEqual(FeedingTextService._extract_response_text(response), "你好")

    def test_extract_response_text_returns_none_without_choices(self):
        self.assertIsNone(
            FeedingTextService._extract_response_text(
                _FakeResponse(json_payload={"error": "boom"})
            )
        )
        self.assertIsNone(FeedingTextService._extract_response_text(_FakeResponse()))

    def test_normalize_feeding_text_provider(self):
        self.assertEqual(app_config.normalize_feeding_text_provider("OpenAI"), "openai")
        self.assertEqual(app_config.normalize_feeding_text_provider("gemini"), "gemini")
        self.assertEqual(
            app_config.normalize_feeding_text_provider("unknown"),
            app_config.FEEDING_TEXT_PROVIDER_GEMINI,
        )
        self.assertEqual(
            app_config.normalize_feeding_text_provider(""),
            app_config.FEEDING_TEXT_PROVIDER_GEMINI,
        )


class TestFeedingTextServiceGemini(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = FeedingTextService()
        self.png_bytes = _build_png()

    async def test_uses_custom_endpoint_and_configured_model(self):
        gemini_response = SimpleNamespace(
            parts=[object()], text="好吃<affection:+5;coins:+50>"
        )
        fake_client = _FakeGeminiClient(gemini_response)

        with patch.dict(os.environ, GEMINI_ENV):
            with patch(
                "src.chat.features.affection.service.feeding_text_service.genai.Client",
                return_value=fake_client,
            ) as client_factory:
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertEqual(result, "好吃<affection:+5;coins:+50>")
        self.assertEqual(client_factory.call_args.kwargs["api_key"], "gemini-test-key")
        self.assertEqual(
            client_factory.call_args.kwargs["http_options"].base_url,
            "https://gemini.example.com",
        )

        generate_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        self.assertEqual(generate_kwargs["model"], "gemini-3-flash-preview")
        self.assertEqual(generate_kwargs["contents"][0], "投喂提示词")
        inline_data = generate_kwargs["contents"][1].inline_data
        self.assertEqual(inline_data.mime_type, "image/png")
        self.assertEqual(bytes(inline_data.data), self.png_bytes)

    async def test_missing_custom_endpoint_config_returns_none(self):
        with patch.dict(
            os.environ,
            {**GEMINI_ENV, "CUSTOM_GEMINI_URL": "", "CUSTOM_GEMINI_API_KEY": ""},
        ):
            result = await self.service.generate_text_with_image(
                "投喂提示词", self.png_bytes, "image/png"
            )

        self.assertIsNone(result)

    async def test_safety_block_returns_in_character_reply(self):
        blocked_response = SimpleNamespace(
            parts=[],
            text=None,
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        )
        fake_client = _FakeGeminiClient(blocked_response)

        with patch.dict(os.environ, GEMINI_ENV):
            with patch(
                "src.chat.features.affection.service.feeding_text_service.genai.Client",
                return_value=fake_client,
            ):
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertEqual(result, "为啥要投喂色图啊喂")

    async def test_upstream_error_returns_none(self):
        fake_client = _FakeGeminiClient(None)
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with patch.dict(os.environ, GEMINI_ENV):
            with patch(
                "src.chat.features.affection.service.feeding_text_service.genai.Client",
                return_value=fake_client,
            ):
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertIsNone(result)


class TestFeedingTextServiceOpenAI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = FeedingTextService()
        self.png_bytes = _build_png()

    async def test_sends_prompt_and_image_to_configured_endpoint(self):
        captured: dict = {}
        response = _openai_json_response("还行吧<affection:+2;coins:+20>")

        async def fake_post(endpoint, headers, payload, timeout_seconds):
            captured.update(
                endpoint=endpoint,
                headers=headers,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            return response

        with patch.dict(os.environ, OPENAI_ENV):
            with patch.object(self.service, "_post_openai", side_effect=fake_post):
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertEqual(result, "还行吧<affection:+2;coins:+20>")
        self.assertEqual(
            captured["endpoint"], "https://feed.example.com/v1/chat/completions"
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")

        payload = captured["payload"]
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertFalse(payload["stream"])

        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "投喂提示词"})
        image_url = content[1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(
            base64.b64decode(image_url.split(",", 1)[1]), self.png_bytes
        )

    async def test_oversized_image_is_compressed_before_upload(self):
        big_png = _build_png(size=600, noise=True)
        captured: dict = {}

        async def fake_post(endpoint, headers, payload, timeout_seconds):
            captured["payload"] = payload
            return _openai_json_response("压缩后依旧好吃<affection:+1;coins:+10>")

        with patch.dict(os.environ, OPENAI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"TEXT_IMAGE_MAX_MB": 0.05}):
                with patch.object(self.service, "_post_openai", side_effect=fake_post):
                    result = await self.service.generate_text_with_image(
                        "投喂提示词", big_png, "image/png"
                    )

        self.assertIsNotNone(result)
        image_url = captured["payload"]["messages"][0]["content"][1]["image_url"]["url"]
        mime_type, encoded_image = image_url.split(";base64,")
        uploaded_bytes = base64.b64decode(encoded_image)

        self.assertLess(len(uploaded_bytes), len(big_png))
        self.assertLess(len(uploaded_bytes), int(0.05 * 1024 * 1024) * 2)
        self.assertIn(mime_type, {"data:image/jpeg", "data:image/webp"})

    async def test_image_within_limit_is_uploaded_unchanged(self):
        """未超限的图片保持原样，避免无意义的重编码。"""
        captured: dict = {}

        async def fake_post(endpoint, headers, payload, timeout_seconds):
            captured["payload"] = payload
            return _openai_json_response("好吃<affection:+1;coins:+10>")

        with patch.dict(os.environ, OPENAI_ENV):
            with patch.object(self.service, "_post_openai", side_effect=fake_post):
                await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        image_url = captured["payload"]["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))

    async def test_incomplete_config_returns_none(self):
        with patch.dict(
            os.environ,
            {
                **OPENAI_ENV,
                "FEEDING_OPENAI_URL": "",
                "FEEDING_OPENAI_API_KEY": "",
                "FEEDING_OPENAI_MODEL": "",
            },
        ):
            with patch.dict(
                app_config.FEEDING_CONFIG,
                {"OPENAI_URL": "", "OPENAI_API_KEY": "", "OPENAI_MODEL": ""},
            ):
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertIsNone(result)

    async def test_http_error_status_returns_none(self):
        with patch.dict(os.environ, OPENAI_ENV):
            with patch.object(
                self.service,
                "_post_openai",
                new=AsyncMock(
                    return_value=_FakeResponse(
                        status_code=401, text='{"error":"invalid api key"}'
                    )
                ),
            ):
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertIsNone(result)

    async def test_network_error_returns_none(self):
        with patch.dict(os.environ, OPENAI_ENV):
            with patch.object(
                self.service,
                "_post_openai",
                new=AsyncMock(side_effect=httpx.ConnectError("boom")),
            ):
                result = await self.service.generate_text_with_image(
                    "投喂提示词", self.png_bytes, "image/png"
                )

        self.assertIsNone(result)

    async def test_empty_image_returns_none(self):
        result = await self.service.generate_text_with_image(
            "投喂提示词", b"", "image/png"
        )

        self.assertIsNone(result)

    async def test_network_error_is_retried_once(self):
        success_response = _openai_json_response("重试成功")
        fake_client = _FakeHttpClient([httpx.ConnectError("first"), success_response])

        with patch("httpx.AsyncClient", lambda *args, **kwargs: fake_client):
            with patch(
                "src.chat.features.affection.service.feeding_text_service.asyncio.sleep",
                new=AsyncMock(),
            ):
                response = await self.service._post_openai(
                    "https://feed.example.com/v1/chat/completions",
                    {"Authorization": "Bearer sk-test"},
                    {"model": "gpt-4o-mini"},
                    10.0,
                )

        self.assertIs(response, success_response)
        self.assertEqual(fake_client.post_calls, 2)

    async def test_network_error_raises_after_final_attempt(self):
        fake_client = _FakeHttpClient(
            [httpx.ConnectError("first"), httpx.ConnectError("second")]
        )

        with patch("httpx.AsyncClient", lambda *args, **kwargs: fake_client):
            with patch(
                "src.chat.features.affection.service.feeding_text_service.asyncio.sleep",
                new=AsyncMock(),
            ):
                with self.assertRaises(httpx.RequestError):
                    await self.service._post_openai(
                        "https://feed.example.com/v1/chat/completions",
                        {"Authorization": "Bearer sk-test"},
                        {"model": "gpt-4o-mini"},
                        10.0,
                    )

        self.assertEqual(fake_client.post_calls, 2)


class TestFeedingTextServiceProviderRouting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = FeedingTextService()
        self.png_bytes = _build_png()

    async def test_env_provider_selects_openai(self):
        with patch.dict(os.environ, OPENAI_ENV):
            with patch.object(
                self.service, "_generate_with_openai", new=AsyncMock(return_value="o")
            ) as openai_call:
                with patch.object(
                    self.service,
                    "_generate_with_gemini",
                    new=AsyncMock(return_value="g"),
                ) as gemini_call:
                    result = await self.service.generate_text_with_image(
                        "投喂提示词", self.png_bytes, "image/png"
                    )

        self.assertEqual(result, "o")
        openai_call.assert_awaited_once()
        gemini_call.assert_not_awaited()

    async def test_env_provider_selects_gemini_by_default(self):
        with patch.dict(os.environ, {"FEEDING_TEXT_PROVIDER": "gemini"}):
            with patch.object(
                self.service, "_generate_with_openai", new=AsyncMock(return_value="o")
            ) as openai_call:
                with patch.object(
                    self.service,
                    "_generate_with_gemini",
                    new=AsyncMock(return_value="g"),
                ) as gemini_call:
                    result = await self.service.generate_text_with_image(
                        "投喂提示词", self.png_bytes, "image/png"
                    )

        self.assertEqual(result, "g")
        gemini_call.assert_awaited_once()
        openai_call.assert_not_awaited()

    async def test_invalid_provider_falls_back_to_gemini(self):
        with patch.dict(os.environ, {"FEEDING_TEXT_PROVIDER": "not-a-provider"}):
            with patch.object(
                self.service, "_generate_with_openai", new=AsyncMock(return_value="o")
            ) as openai_call:
                with patch.object(
                    self.service,
                    "_generate_with_gemini",
                    new=AsyncMock(return_value="g"),
                ) as gemini_call:
                    result = await self.service.generate_text_with_image(
                        "投喂提示词", self.png_bytes, "image/png"
                    )

        self.assertEqual(result, "g")
        gemini_call.assert_awaited_once()
        openai_call.assert_not_awaited()


class TestFeedingTextServiceRetryLimit(unittest.IsolatedAsyncioTestCase):
    """503/429 等可重试错误必须受次数上限约束，不能无限重试。"""

    def setUp(self):
        self.service = FeedingTextService()
        self.png_bytes = _build_png()

    def test_extract_status_code(self):
        extract = FeedingTextService._extract_status_code

        self.assertEqual(extract(_FakeUpstreamError("503 UNAVAILABLE. 上游不可用")), 503)
        self.assertEqual(
            extract(SimpleNamespace(status_code=429, __str__=lambda self: "429")), 429
        )
        self.assertEqual(
            extract(SimpleNamespace(code=502, __str__=lambda self: "502")), 502
        )
        self.assertIsNone(extract(RuntimeError("boom")))

    def test_is_retryable_status(self):
        is_retryable = FeedingTextService._is_retryable_status

        self.assertTrue(is_retryable(503))
        self.assertTrue(is_retryable(429))
        self.assertTrue(is_retryable(502))
        self.assertFalse(is_retryable(401))
        self.assertFalse(is_retryable(400))
        self.assertFalse(is_retryable(None))

    async def test_gemini_503_stops_after_max_attempts(self):
        fake_client = _FakeGeminiClient(None)
        generate_content = AsyncMock(
            side_effect=_FakeUpstreamError("503 UNAVAILABLE. 上游不可用")
        )
        fake_client.aio.models.generate_content = generate_content

        with patch.dict(os.environ, GEMINI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 3}):
                with patch(
                    "src.chat.features.affection.service.feeding_text_service.genai.Client",
                    return_value=fake_client,
                ):
                    with _patch_sleep() as sleep_mock:
                        result = await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertIsNone(result)
        self.assertEqual(generate_content.await_count, 3)
        self.assertEqual(sleep_mock.await_count, 2)

    async def test_gemini_503_then_success_returns_text(self):
        success_response = SimpleNamespace(
            parts=[object()], text="终于好了<affection:+1;coins:+10>"
        )
        fake_client = _FakeGeminiClient(None)
        generate_content = AsyncMock(
            side_effect=[
                _FakeUpstreamError("503 UNAVAILABLE. 上游不可用"),
                success_response,
            ]
        )
        fake_client.aio.models.generate_content = generate_content

        with patch.dict(os.environ, GEMINI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 3}):
                with patch(
                    "src.chat.features.affection.service.feeding_text_service.genai.Client",
                    return_value=fake_client,
                ):
                    with _patch_sleep():
                        result = await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertEqual(result, "终于好了<affection:+1;coins:+10>")
        self.assertEqual(generate_content.await_count, 2)

    async def test_gemini_non_retryable_error_is_not_retried(self):
        fake_client = _FakeGeminiClient(None)
        generate_content = AsyncMock(side_effect=RuntimeError("400 INVALID_ARGUMENT"))
        fake_client.aio.models.generate_content = generate_content

        with patch.dict(os.environ, GEMINI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 3}):
                with patch(
                    "src.chat.features.affection.service.feeding_text_service.genai.Client",
                    return_value=fake_client,
                ):
                    with _patch_sleep() as sleep_mock:
                        result = await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertIsNone(result)
        self.assertEqual(generate_content.await_count, 1)
        sleep_mock.assert_not_awaited()

    async def test_openai_503_stops_after_max_attempts(self):
        post = AsyncMock(
            return_value=_FakeResponse(status_code=503, text="upstream unavailable")
        )

        with patch.dict(os.environ, OPENAI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 2}):
                with patch.object(self.service, "_post_openai", post):
                    with _patch_sleep() as sleep_mock:
                        result = await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertIsNone(result)
        self.assertEqual(post.await_count, 2)
        self.assertEqual(sleep_mock.await_count, 1)

    async def test_openai_503_then_success_returns_text(self):
        post = AsyncMock(
            side_effect=[
                _FakeResponse(status_code=503, text="upstream unavailable"),
                _openai_json_response("恢复了<affection:+2;coins:+20>"),
            ]
        )

        with patch.dict(os.environ, OPENAI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 3}):
                with patch.object(self.service, "_post_openai", post):
                    with _patch_sleep():
                        result = await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertEqual(result, "恢复了<affection:+2;coins:+20>")
        self.assertEqual(post.await_count, 2)

    async def test_openai_non_retryable_status_is_not_retried(self):
        post = AsyncMock(
            return_value=_FakeResponse(status_code=401, text='{"error":"bad key"}')
        )

        with patch.dict(os.environ, OPENAI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 3}):
                with patch.object(self.service, "_post_openai", post):
                    with _patch_sleep() as sleep_mock:
                        result = await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertIsNone(result)
        self.assertEqual(post.await_count, 1)
        sleep_mock.assert_not_awaited()


class TestFeedingTextServiceTimeout(unittest.IsolatedAsyncioTestCase):
    """文字阶段必须有整体超时上限（默认 120 秒）。"""

    def setUp(self):
        self.service = FeedingTextService()
        self.png_bytes = _build_png()

    def test_timeout_reads_from_config(self):
        with patch.dict(app_config.FEEDING_CONFIG, {"TEXT_TIMEOUT": 42}):
            self.assertEqual(FeedingTextService._resolve_timeout_seconds(), 42.0)

    def test_max_attempts_reads_from_config(self):
        with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 5}):
            self.assertEqual(FeedingTextService._resolve_max_attempts(), 5)

        with patch.dict(app_config.FEEDING_CONFIG, {"MAX_RETRY_ATTEMPTS": 0}):
            self.assertEqual(FeedingTextService._resolve_max_attempts(), 1)

    async def test_timeout_raises_feeding_text_timeout_error(self):
        async def _hang_forever(*args, **kwargs):
            await asyncio.sleep(30)

        with patch.dict(os.environ, GEMINI_ENV):
            with patch.dict(app_config.FEEDING_CONFIG, {"TEXT_TIMEOUT": 0.05}):
                with patch.object(
                    self.service, "_generate_with_gemini", side_effect=_hang_forever
                ):
                    with self.assertRaises(FeedingTextTimeoutError) as raised:
                        await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )

        self.assertEqual(raised.exception.timeout_seconds, 0.05)
        self.assertIn("生成投喂文本超时", str(raised.exception))

    async def test_timeout_cancels_pending_retry_sleep(self):
        """超时应中断重试等待，而不是继续睡下去。"""
        fake_client = _FakeGeminiClient(None)
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=_FakeUpstreamError("503 UNAVAILABLE. 上游不可用")
        )

        with patch.dict(os.environ, GEMINI_ENV):
            with patch.dict(
                app_config.FEEDING_CONFIG,
                {"TEXT_TIMEOUT": 0.05, "MAX_RETRY_ATTEMPTS": 5},
            ):
                with patch(
                    "src.chat.features.affection.service.feeding_text_service.genai.Client",
                    return_value=fake_client,
                ):
                    with self.assertRaises(FeedingTextTimeoutError):
                        await self.service.generate_text_with_image(
                            "投喂提示词", self.png_bytes, "image/png"
                        )


if __name__ == "__main__":
    unittest.main()
