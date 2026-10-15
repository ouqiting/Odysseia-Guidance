# -*- coding: utf-8 -*-

"""投喂文字阶段（图文评价）的模型链路。

`/投喂` 的文字评价支持两种 provider，由 `FEEDING_TEXT_PROVIDER` 选择：

- ``gemini``（默认）：渠道固定走自定义 Gemini 端点
  （``CUSTOM_GEMINI_URL`` / ``CUSTOM_GEMINI_API_KEY``），模型名由 ``FEEDING_GEMINI_MODEL`` 配置。
- ``openai``：走自行配置的 OpenAI 兼容端点
  （``FEEDING_OPENAI_URL`` / ``FEEDING_OPENAI_API_KEY`` / ``FEEDING_OPENAI_MODEL``）。

两条链路都会：
1. 先把投喂图片压缩到 ``FEEDING_TEXT_IMAGE_MAX_MB`` 以内再上传；
2. 遇到可重试错误（503/429 等）最多尝试 ``FEEDING_MAX_RETRY_ATTEMPTS`` 次；
3. 整体受 ``FEEDING_TEXT_TIMEOUT``（默认 120 秒）限制，超时抛 ``FeedingTextTimeoutError``。
"""

import asyncio
import base64
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from google import genai
from google.genai import types

from src.chat.config import chat_config as app_config
from src.chat.services.openai_models.custom_model import CustomModelClient
from src.chat.utils.image_utils import (
    MAX_IMAGE_SIZE_BYTES,
    sanitize_image_to_size_limit,
)

log = logging.getLogger(__name__)

# 压缩后的兜底最低质量（与 image_utils 的默认值保持一致）
_MIN_COMPRESS_QUALITY = 50

# 网络层失败在单次请求内自动重试 1 次，与 OpenAIService 的重试策略保持一致
_MAX_SEND_ATTEMPTS = 2

# 关闭 Gemini client 的等待上限，避免拖住整体超时控制
_CLOSE_CLIENT_TIMEOUT_SECONDS = 5.0

# 上游可重试的状态码（503 服务不可用 / 429 限流 / 5xx 网关错误）
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# 安全策略拦截时的固定回应（沿用原 Gemini 链路的行为）
_SAFETY_BLOCKED_REPLY = "为啥要投喂色图啊喂"

_SAFETY_SETTINGS: List[types.SafetySetting] = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


class FeedingTextTimeoutError(Exception):
    """投喂文字阶段整体超时。"""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"生成投喂文本超时（超过 {timeout_seconds:g} 秒）")
        self.timeout_seconds = timeout_seconds


class FeedingTextRetryableError(Exception):
    """上游返回可重试错误（503/429 等），由上层按次数上限重试。"""

    def __init__(self, status_code: Optional[int], message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class FeedingTextService:
    """按 provider 分发投喂文字阶段的图文请求。"""

    async def generate_text_with_image(
        self, prompt: str, image_bytes: bytes, mime_type: str
    ) -> Optional[str]:
        """生成投喂评价文本。

        返回 None 表示本轮无法生成（配置缺失或上游失败），由调用方按失败处理；
        唯一例外是 Gemini 安全策略拦截，此时返回固定的角色内回应。
        超过 ``FEEDING_TEXT_TIMEOUT`` 时抛出 ``FeedingTextTimeoutError``。
        """
        if not image_bytes:
            log.warning("[FeedingText] 投喂图片为空，跳过文字阶段。")
            return None

        provider = app_config.get_feeding_text_provider()
        timeout_seconds = self._resolve_timeout_seconds()

        try:
            return await asyncio.wait_for(
                self._generate_with_retry(provider, prompt, image_bytes, mime_type),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            log.error(
                "[FeedingText] 生成投喂文本超时 | provider=%s | timeout=%ss",
                provider,
                timeout_seconds,
            )
            raise FeedingTextTimeoutError(timeout_seconds) from error

    async def _generate_with_retry(
        self, provider: str, prompt: str, image_bytes: bytes, mime_type: str
    ) -> Optional[str]:
        """压缩图片后调用上游；可重试错误按上限重试，其余错误直接放弃。"""
        compressed_bytes, compressed_mime_type = await self._compress_image(
            image_bytes, mime_type
        )

        max_attempts = self._resolve_max_attempts()
        for attempt in range(1, max_attempts + 1):
            try:
                if provider == app_config.FEEDING_TEXT_PROVIDER_OPENAI:
                    return await self._generate_with_openai(
                        prompt, compressed_bytes, compressed_mime_type
                    )
                return await self._generate_with_gemini(
                    prompt, compressed_bytes, compressed_mime_type
                )
            except FeedingTextRetryableError as error:
                if attempt >= max_attempts:
                    log.error(
                        "[FeedingText] 可重试错误已达最大尝试次数，放弃本次文字生成 | "
                        "provider=%s | attempts=%s | status=%s | error=%s",
                        provider,
                        max_attempts,
                        error.status_code,
                        error,
                    )
                    return None

                retry_delay_seconds = 2 ** (attempt - 1)
                log.warning(
                    "[FeedingText] 上游可重试错误，%.0f 秒后进行第 %s/%s 次尝试 | "
                    "provider=%s | status=%s | error=%s",
                    retry_delay_seconds,
                    attempt + 1,
                    max_attempts,
                    provider,
                    error.status_code,
                    error,
                )
                await asyncio.sleep(retry_delay_seconds)

        return None

    # --- 图片压缩（两条链路共用）---
    async def _compress_image(
        self, image_bytes: bytes, mime_type: str
    ) -> Tuple[bytes, str]:
        """压缩投喂图片供文字（视觉）模型使用；压缩失败时回退原图。"""
        original_size = len(image_bytes)
        target_mb = float(app_config.FEEDING_CONFIG.get("TEXT_IMAGE_MAX_MB", 7.0))
        target_size_bytes = int(max(target_mb, 0.1) * 1024 * 1024)

        try:
            compressed_bytes, compressed_mime_type = await asyncio.to_thread(
                sanitize_image_to_size_limit,
                image_bytes,
                target_size_bytes=target_size_bytes,
                max_image_size_bytes=max(target_size_bytes, MAX_IMAGE_SIZE_BYTES),
                min_quality=_MIN_COMPRESS_QUALITY,
            )
        except Exception as e:
            log.warning(
                "[FeedingText] 投喂图片压缩失败，回退使用原图 | size_kb=%.2f | error=%s",
                original_size / 1024,
                e,
            )
            return image_bytes, mime_type

        if not compressed_bytes:
            return image_bytes, mime_type

        if len(compressed_bytes) < original_size:
            log.info(
                "[FeedingText] 投喂图片已压缩 | %d -> %d bytes | mime=%s",
                original_size,
                len(compressed_bytes),
                compressed_mime_type,
            )
        return compressed_bytes, compressed_mime_type

    @staticmethod
    def _resolve_timeout_seconds() -> float:
        return float(app_config.FEEDING_CONFIG.get("TEXT_TIMEOUT", 120.0))

    @staticmethod
    def _resolve_max_attempts() -> int:
        return max(int(app_config.FEEDING_CONFIG.get("MAX_RETRY_ATTEMPTS", 3)), 1)

    # --- Gemini 链路：固定走自定义端点 ---
    @staticmethod
    def _resolve_gemini_config() -> Tuple[str, str, str]:
        """返回 (base_url, api_key, model_name)，端点固定复用 CUSTOM_GEMINI_*。"""
        base_url = str(os.getenv("CUSTOM_GEMINI_URL") or "").strip()
        api_key = str(os.getenv("CUSTOM_GEMINI_API_KEY") or "").strip()
        model_name = (
            str(os.getenv("FEEDING_GEMINI_MODEL") or "").strip()
            or str(app_config.FEEDING_CONFIG.get("GEMINI_MODEL") or "").strip()
            or app_config.FEEDING_DEFAULT_GEMINI_MODEL
        )
        return base_url, api_key, model_name

    async def _generate_with_gemini(
        self, prompt: str, image_bytes: bytes, mime_type: str
    ) -> Optional[str]:
        base_url, api_key, model_name = self._resolve_gemini_config()
        if not base_url or not api_key:
            log.error(
                "[FeedingText/Gemini] 缺少自定义 Gemini 端点配置，请检查 CUSTOM_GEMINI_URL 与 "
                "CUSTOM_GEMINI_API_KEY，或改用 FEEDING_TEXT_PROVIDER=openai。"
            )
            return None

        request_contents = [
            prompt,
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
        ]
        gen_config = types.GenerateContentConfig(
            temperature=float(app_config.FEEDING_CONFIG.get("TEMPERATURE", 1.1)),
            max_output_tokens=int(
                app_config.FEEDING_CONFIG.get("MAX_OUTPUT_TOKENS", 3000)
            ),
            safety_settings=_SAFETY_SETTINGS,
        )

        client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(base_url=base_url)
        )
        try:
            response = await client.aio.models.generate_content(
                model=model_name, contents=request_contents, config=gen_config
            )
        except Exception as e:
            status_code = self._extract_status_code(e)
            if self._is_retryable_status(status_code):
                raise FeedingTextRetryableError(status_code, str(e)) from e
            log.error(
                "[FeedingText/Gemini] 调用失败 | model=%s | base_url=%s | error=%s",
                model_name,
                base_url,
                e,
                exc_info=True,
            )
            return None
        finally:
            await self._close_gemini_client(client)

        if response.parts:
            response_text = (response.text or "").strip()
            if response_text:
                return response_text
            log.warning("[FeedingText/Gemini] 模型返回了空文本。")
            return None

        block_reason = getattr(
            getattr(response, "prompt_feedback", None), "block_reason", None
        )
        if block_reason:
            log.warning("[FeedingText/Gemini] 请求被安全策略拦截: %s", block_reason)
            return _SAFETY_BLOCKED_REPLY

        log.warning("[FeedingText/Gemini] 未获得有效回复。response=%s", response)
        return None

    @staticmethod
    async def _close_gemini_client(client: Any) -> None:
        async def _close() -> None:
            try:
                await client.aio.aclose()
            except Exception:
                log.debug("[FeedingText/Gemini] 关闭 async client 失败", exc_info=True)
            try:
                client.close()
            except Exception:
                log.debug("[FeedingText/Gemini] 关闭 sync client 失败", exc_info=True)

        try:
            await asyncio.wait_for(_close(), timeout=_CLOSE_CLIENT_TIMEOUT_SECONDS)
        except Exception:
            log.debug("[FeedingText/Gemini] 关闭 client 超时或失败", exc_info=True)

    # --- OpenAI 兼容链路：URL / Key / 模型名均可自行配置 ---
    @staticmethod
    def _resolve_openai_config() -> Dict[str, Any]:
        feeding_config = app_config.FEEDING_CONFIG

        def _resolve(env_name: str, config_key: str) -> str:
            return str(
                os.getenv(env_name) or feeding_config.get(config_key) or ""
            ).strip()

        return {
            "url": _resolve("FEEDING_OPENAI_URL", "OPENAI_URL"),
            "api_key": _resolve("FEEDING_OPENAI_API_KEY", "OPENAI_API_KEY"),
            "model": _resolve("FEEDING_OPENAI_MODEL", "OPENAI_MODEL"),
            "timeout": float(feeding_config.get("OPENAI_TIMEOUT", 120.0)),
            "stream": bool(feeding_config.get("OPENAI_STREAM", False)),
        }

    async def _generate_with_openai(
        self, prompt: str, image_bytes: bytes, mime_type: str
    ) -> Optional[str]:
        config = self._resolve_openai_config()
        if not config["url"] or not config["api_key"] or not config["model"]:
            log.error(
                "[FeedingText/OpenAI] 配置不完整，请检查 FEEDING_OPENAI_URL / "
                "FEEDING_OPENAI_API_KEY / FEEDING_OPENAI_MODEL。"
            )
            return None

        endpoint = self._build_chat_completions_url(config["url"])
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._build_image_data_uri(
                                    image_bytes, mime_type
                                )
                            },
                        },
                    ],
                }
            ],
            "stream": config["stream"],
            "temperature": float(app_config.FEEDING_CONFIG.get("TEMPERATURE", 1.1)),
            "max_tokens": int(
                app_config.FEEDING_CONFIG.get("MAX_OUTPUT_TOKENS", 3000)
            ),
        }
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._post_openai(
                endpoint, headers, payload, config["timeout"]
            )
        except httpx.RequestError as e:
            # 网络层失败已在 _post_openai 内部重试过，这里不再重试
            log.error(
                "[FeedingText/OpenAI] 网络请求失败 | url=%s | error=%s",
                endpoint,
                e,
                exc_info=True,
            )
            return None

        if response.status_code >= 400:
            if self._is_retryable_status(response.status_code):
                raise FeedingTextRetryableError(
                    response.status_code,
                    self._preview_text(response.text),
                )
            log.error(
                "[FeedingText/OpenAI] 上游返回错误 | status=%s | url=%s | model=%s | body=%s",
                response.status_code,
                endpoint,
                config["model"],
                self._preview_text(response.text),
            )
            return None

        response_text = self._extract_response_text(response)
        if not response_text:
            log.warning(
                "[FeedingText/OpenAI] 未能从响应中解析出文本 | status=%s | body=%s",
                response.status_code,
                self._preview_text(response.text),
            )
            return None
        return response_text

    @staticmethod
    def _extract_status_code(error: BaseException) -> Optional[int]:
        """从异常对象中尽量取出 HTTP 状态码。"""
        for attribute_name in ("status_code", "code"):
            raw_value = getattr(error, attribute_name, None)
            if isinstance(raw_value, int):
                return raw_value

        match = re.match(r"(\d{3})", str(error))
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _is_retryable_status(status_code: Optional[int]) -> bool:
        return status_code in _RETRYABLE_STATUS_CODES

    @staticmethod
    def _build_chat_completions_url(base_url: str) -> str:
        normalized = (base_url or "").strip().rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _build_image_data_uri(image_bytes: bytes, mime_type: str) -> str:
        normalized_mime_type = (mime_type or "").strip() or "image/png"
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{normalized_mime_type};base64,{encoded_image}"

    async def _post_openai(
        self,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        """发送 OpenAI 兼容请求；仅网络层失败时自动重试 1 次。"""
        last_error: Optional[httpx.RequestError] = None

        for attempt in range(_MAX_SEND_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    return await client.post(endpoint, headers=headers, json=payload)
            except httpx.RequestError as e:
                last_error = e
                if attempt < _MAX_SEND_ATTEMPTS - 1:
                    retry_delay_seconds = 2**attempt
                    log.warning(
                        "[FeedingText/OpenAI] 网络层请求失败，%.1f 秒后重试 | url=%s | error=%s",
                        retry_delay_seconds,
                        endpoint,
                        e,
                    )
                    await asyncio.sleep(retry_delay_seconds)

        if last_error is not None:
            raise last_error
        raise httpx.RequestError("投喂请求未被执行")

    @classmethod
    def _extract_response_text(cls, response: httpx.Response) -> Optional[str]:
        """从普通 JSON 或 SSE 响应中取出第一条候选文本。"""
        if CustomModelClient.is_sse_chat_completion_response(response):
            result = CustomModelClient.parse_streaming_chat_completion_body(
                response.text or ""
            )
        else:
            try:
                result = response.json()
            except ValueError:
                result = None

        if not isinstance(result, dict):
            return None

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None

        for block_key in ("message", "delta"):
            block = first_choice.get(block_key)
            if not isinstance(block, dict):
                continue
            text = cls._extract_text_from_content(block.get("content"))
            if text:
                return text
        return None

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_chunks: List[str] = []
            for block in content:
                if isinstance(block, str):
                    text_chunks.append(block)
                    continue
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    text_chunks.append(block["text"])
            return "".join(text_chunks).strip()

        return ""

    @staticmethod
    def _preview_text(text: str, max_length: int = 500) -> str:
        normalized = str(text or "")
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[:max_length]}...(truncated)"


feeding_text_service = FeedingTextService()
