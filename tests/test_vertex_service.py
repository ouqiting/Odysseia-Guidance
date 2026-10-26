# -*- coding: utf-8 -*-

"""Google Vertex AI (快速模式 / Express Mode) 链路测试。

验证：
1. get_vertexai_base_url 端点计算逻辑（region 默认 global、us/eu、各区域分流、有无 project_id）；
2. 模型别名映射（vertex-gemini-4-argon -> gemini-4-argon, vertex-gemini-3.8-flash -> gemini-3.8-flash）；
3. VertexGeminiClient 的 URL 拼装、query 鉴权、payload 序列化与响应解析；
4. 思考链 (thought) 与工具调用 (functionCall) 在响应中的正确提取；
5. GeminiService 的 Vertex 路由分发与失败回退。
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# 将项目根目录添加到 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 为缺失三方库的极简环境提供基础 mock
if "httpx" not in sys.modules:
    mock_httpx = MagicMock()
    mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
    mock_httpx.RequestError = type("RequestError", (Exception,), {})
    sys.modules["httpx"] = mock_httpx

for mod in (
    "dotenv",
    "google",
    "google.genai",
    "google.genai.types",
    "google.genai.errors",
    "discord",
    "discord.ui",
    "discord.app_commands",
    "PIL",
    "sqlalchemy",
    "sqlalchemy.future",
    "sqlalchemy.orm",
    "sqlalchemy.ext.asyncio",
    "chromadb",
    "pgvector",
    "pgvector.sqlalchemy",
    "alembic",
    "requests",
    "aiohttp",
):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from src.config import AVAILABLE_AI_MODELS
from src.chat.config import chat_config as app_config
from src.chat.config.chat_config import get_vertexai_base_url
from src.chat.services.vertex_service import (
    VertexGeminiClient,
    VertexGenerateContentResponse,
    VertexPart,
    VertexFunctionCall,
)


class TestVertexBaseUrl(unittest.TestCase):
    """测试 Base URL 规则。"""

    def test_default_global_without_project(self):
        """未提供 project_id 时，无论 region 为何均使用全局默认端点。"""
        url = get_vertexai_base_url()
        self.assertEqual(url, "https://aiplatform.googleapis.com/v1")

        url = get_vertexai_base_url(region="us-central1", project_id="")
        self.assertEqual(url, "https://aiplatform.googleapis.com/v1")

        url = get_vertexai_base_url(region=None, project_id=None)
        self.assertEqual(url, "https://aiplatform.googleapis.com/v1")

    def test_with_project_global_region(self):
        """提供 project_id 且 region 为 global（或默认）。"""
        url = get_vertexai_base_url(region="global", project_id="my-gcp-project")
        self.assertEqual(
            url,
            "https://aiplatform.googleapis.com/v1/projects/my-gcp-project/locations/global",
        )

        # 留空默认为 global
        url = get_vertexai_base_url(region="", project_id="my-gcp-project")
        self.assertEqual(
            url,
            "https://aiplatform.googleapis.com/v1/projects/my-gcp-project/locations/global",
        )

    def test_with_project_us_and_eu_regions(self):
        """us 和 eu 特殊域名分支。"""
        url_us = get_vertexai_base_url(region="us", project_id="my-gcp-project")
        self.assertEqual(
            url_us,
            "https://aiplatform.us.rep.googleapis.com/v1/projects/my-gcp-project/locations/us",
        )

        url_eu = get_vertexai_base_url(region="EU", project_id="my-gcp-project")
        self.assertEqual(
            url_eu,
            "https://aiplatform.eu.rep.googleapis.com/v1/projects/my-gcp-project/locations/eu",
        )

    def test_with_project_regional_locations(self):
        """其他具体区域（如 us-central1, asia-northeast1 等）。"""
        url = get_vertexai_base_url(
            region="us-central1", project_id="my-gcp-project"
        )
        self.assertEqual(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-gcp-project/locations/us-central1",
        )

        url_asia = get_vertexai_base_url(
            region="asia-northeast1", project_id="my-gcp-project"
        )
        self.assertEqual(
            url_asia,
            "https://asia-northeast1-aiplatform.googleapis.com/v1/projects/my-gcp-project/locations/asia-northeast1",
        )


class TestVertexModelsAndConfig(unittest.TestCase):
    """测试模型定义与配置列表。"""

    def test_models_in_available_list(self):
        self.assertIn("vertex-gemini-4-argon", AVAILABLE_AI_MODELS)
        self.assertIn("vertex-gemini-3.8-flash", AVAILABLE_AI_MODELS)

    def test_models_mapping(self):
        self.assertEqual(
            app_config.VERTEX_GEMINI_MODELS.get("vertex-gemini-4-argon"),
            "gemini-4-argon",
        )
        self.assertEqual(
            app_config.VERTEX_GEMINI_MODELS.get("vertex-gemini-3.8-flash"),
            "gemini-3.8-flash",
        )

    def test_model_generation_config(self):
        argon_cfg = app_config.MODEL_GENERATION_CONFIG.get("vertex-gemini-4-argon")
        self.assertIsNotNone(argon_cfg)
        self.assertIn("thinking_config", argon_cfg)

        flash_cfg = app_config.MODEL_GENERATION_CONFIG.get("vertex-gemini-3.8-flash")
        self.assertIsNotNone(flash_cfg)
        self.assertIn("thinking_config", flash_cfg)


class TestVertexClient(unittest.IsolatedAsyncioTestCase):
    """测试 VertexGeminiClient 的请求与响应解析。"""

    def setUp(self):
        self.client = VertexGeminiClient(
            api_key="test-vertex-api-key",
            project_id="test-proj",
            region="global",
        )

    def test_serialization(self):
        # 验证 contents 序列化
        contents = [
            {"role": "user", "parts": [{"text": "你好"}]},
        ]
        serialized = self.client._serialize_contents(contents)
        self.assertEqual(len(serialized), 1)
        self.assertEqual(serialized[0]["role"], "user")
        self.assertEqual(serialized[0]["parts"][0]["text"], "你好")

        # 验证 config 序列化
        cfg = {
            "temperature": 0.8,
            "max_output_tokens": 2048,
            "thinking_config": {
                "thinking_level": "Medium",
                "include_thoughts": True,
            },
        }
        gen_cfg, _, _, _ = self.client._serialize_config(cfg)
        self.assertEqual(gen_cfg["temperature"], 0.8)
        self.assertEqual(gen_cfg["maxOutputTokens"], 2048)
        self.assertEqual(gen_cfg["thinkingConfig"]["thinkingLevel"], "Medium")
        self.assertTrue(gen_cfg["thinkingConfig"]["includeThoughts"])

    def test_response_parsing(self):
        mock_raw_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "我在思考中..."},
                            {"text": "你好！我是神所娘。"},
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "Shanghai"},
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 25,
                "totalTokenCount": 35,
            },
        }

        parsed = self.client._parse_response_json(mock_raw_resp)
        self.assertEqual(len(parsed.candidates), 1)
        # text 属性应过滤掉 thought，只返回正文
        self.assertEqual(parsed.text, "你好！我是神所娘。")
        # parts 包含全部 3 个 part
        self.assertEqual(len(parsed.parts), 3)
        # 能够提取到 function_call
        self.assertEqual(len(parsed.function_calls), 1)
        self.assertEqual(parsed.function_calls[0].name, "get_weather")
        self.assertEqual(parsed.function_calls[0].args["city"], "Shanghai")
        # 用量元数据
        self.assertEqual(parsed.usage_metadata.total_token_count, 35)

    async def test_generate_content_async_call(self):
        mock_raw_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "来自 Vertex 的回复"}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }

        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_raw_resp
        mock_http_response.raise_for_status = MagicMock()

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_http_response)
        mock_http_client.is_closed = False
        self.client._http_client = mock_http_client

        response = await self.client.generate_content_async(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        )

        self.assertEqual(response.text, "来自 Vertex 的回复")
        # 验证调用 URL 与 query 参数
        mock_http_client.post.assert_called_once()
        call_args, call_kwargs = mock_http_client.post.call_args
        self.assertIn("publishers/google/models/gemini-4-argon:generateContent", call_args[0])
        self.assertEqual(call_kwargs["params"], {"key": "test-vertex-api-key"})


class TestGeminiServiceVertexRouting(unittest.IsolatedAsyncioTestCase):
    """测试 GeminiService 对 Vertex 模型的识别、路由与回退。"""

    def setUp(self):
        # 延迟导入 GeminiService
        from src.chat.services.gemini_service import GeminiService
        self.service = GeminiService.__new__(GeminiService)
        self.service.default_model_name = "gemini-2.5-flash"
        self.service.key_rotation_service = MagicMock()
        self.service.consume_one_time_debug_base_url = MagicMock(return_value=None)
        self.service._apply_blacklist_notice = MagicMock(side_effect=lambda resp, active: resp)
        self.service._generate_with_vertex_api = AsyncMock(return_value="Vertex成功响应")
        self.service._generate_with_official_api = AsyncMock(return_value="官方API回退响应")

    async def test_route_to_vertex_for_argon(self):
        result = await self.service.generate_response(
            user_id=123,
            guild_id=456,
            message="你好",
            model_name="vertex-gemini-4-argon",
        )
        self.assertEqual(result, "Vertex成功响应")
        self.service._generate_with_vertex_api.assert_called_once()
        self.service._generate_with_official_api.assert_not_called()

    async def test_route_to_vertex_for_flash(self):
        result = await self.service.generate_response(
            user_id=123,
            guild_id=456,
            message="你好",
            model_name="vertex-gemini-3.8-flash",
        )
        self.assertEqual(result, "Vertex成功响应")
        self.service._generate_with_vertex_api.assert_called_once()
        self.service._generate_with_official_api.assert_not_called()

    async def test_vertex_failure_fallback_to_official(self):
        self.service._generate_with_vertex_api.side_effect = RuntimeError("Vertex上游超时")
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await self.service.generate_response(
                user_id=123,
                guild_id=456,
                message="你好",
                model_name="vertex-gemini-4-argon",
            )
        self.assertEqual(result, "官方API回退响应")
        self.assertEqual(self.service._generate_with_vertex_api.call_count, 2)
        self.service._generate_with_official_api.assert_called_once()
        # 确认回退时模型固定为 gemini-2.5-flash
        call_kwargs = self.service._generate_with_official_api.call_args[1]
        self.assertEqual(call_kwargs["model_name"], "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
