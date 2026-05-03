# -*- coding: utf-8 -*-

import base64
import copy
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image

from src.chat.services.moonshot_vision_service import moonshot_vision_service

log = logging.getLogger(__name__)


class DeepSeekModelClient:
    """DeepSeek 通道客户端：负责配置管理、请求发送与 DeepSeek 专属内容构建。"""

    def __init__(self) -> None:
        self.base_url = os.getenv("DEEPSEEK_URL")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        if self.base_url and self.api_key:
            log.info("✅ [DeepSeekModelClient] 已加载 DeepSeek 配置。URL: %s", self.base_url)

    @staticmethod
    def _build_chat_completions_url(base_url: str) -> str:
        normalized = (base_url or "").rstrip("/")
        if not normalized.endswith("/chat/completions"):
            normalized += "/chat/completions"
        return normalized

    @staticmethod
    def _build_moonshot_image_payload_from_pil(image: Image.Image) -> Dict[str, Any]:
        """将 PIL 图片转换为 Moonshot 识别所需 payload。"""
        mime_type = "image/png"
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        image_bytes = buffered.getvalue()
        return {
            "type": "image",
            "mime_type": mime_type,
            "data_size": len(image_bytes),
            "data_preview": image_bytes.hex(),
        }

    @staticmethod
    def _is_supported_image_format(mime_type: str) -> bool:
        """检查是否为支持的图片格式（排除 gif 和视频）。"""
        if not mime_type:
            return False
        lower = mime_type.lower()
        if lower.startswith("video/"):
            return False
        if lower == "image/gif":
            return False
        return True

    def build_turn_content(self, parts: List[Any]) -> List[Dict[str, Any]]:
        """
        构建 DeepSeek 多模态单条消息 content。
        直接把图片作为 image_url(data URI) 发给模型，不做 OCR。
        仅支持静态图片，过滤 gif 和视频。
        """
        content_blocks: List[Dict[str, Any]] = []

        for part in parts or []:
            if hasattr(part, "thought") and getattr(part, "thought", False):
                continue

            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_value = part["text"].strip()
                if text_value:
                    content_blocks.append({"type": "text", "text": text_value})
                continue

            if isinstance(part, Image.Image):
                buffered = io.BytesIO()
                part.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                )
                continue

            if isinstance(part, dict) and part.get("type") == "image":
                mime_type = str(part.get("mime_type", "image/png"))

                if not self._is_supported_image_format(mime_type):
                    log.info("[DeepSeek] 跳过不支持的图片格式: %s", mime_type)
                    continue

                image_bytes: Optional[bytes] = None

                direct_bytes = part.get("data") or part.get("bytes")
                if isinstance(direct_bytes, (bytes, bytearray)):
                    image_bytes = bytes(direct_bytes)

                if image_bytes is None:
                    image_base64 = part.get("image_base64")
                    if isinstance(image_base64, str) and image_base64.strip():
                        try:
                            image_bytes = base64.b64decode(image_base64)
                        except Exception:
                            image_bytes = None

                if image_bytes is None:
                    data_preview = part.get("data_preview")
                    if isinstance(data_preview, str) and data_preview.strip():
                        try:
                            image_bytes = bytes.fromhex(data_preview.strip())
                        except Exception:
                            try:
                                image_bytes = base64.b64decode(data_preview.strip())
                            except Exception:
                                image_bytes = None

                if image_bytes:
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        }
                    )
                else:
                    content_blocks.append({"type": "text", "text": "（收到一张图片，但解析失败）"})
                continue

            fallback_text = str(part).strip()
            if fallback_text:
                content_blocks.append({"type": "text", "text": fallback_text})

        return content_blocks

    async def post_process_tool_response(self, raw_response: Any) -> Any:
        """
        DeepSeek 专属工具结果后处理：
        对 get_user_profile 返回的头像/横幅图片做识图摘要，并移除原始 base64。
        """
        clean_response = copy.deepcopy(raw_response)

        if not isinstance(clean_response, dict):
            return clean_response

        result_data = clean_response.get("result", {})
        if not isinstance(result_data, dict):
            return clean_response

        profile = result_data.get("profile", {})
        if not isinstance(profile, dict):
            return clean_response

        image_fields = [
            (
                "avatar_image_base64",
                "avatar_mime_type",
                "avatar",
                "请识别这张用户头像图片，简洁描述可见人物、风格、配色与关键元素。",
                "头像",
            ),
            (
                "banner_image_base64",
                "banner_mime_type",
                "banner",
                "请识别这张用户横幅图片，简洁描述画面内容、风格、配色与关键信息。",
                "横幅",
            ),
        ]

        for (
            image_key,
            mime_key,
            label,
            vision_prompt,
            label_cn,
        ) in image_fields:
            image_b64 = profile.get(image_key)
            if not (isinstance(image_b64, str) and image_b64.strip()):
                continue

            try:
                image_bytes = base64.b64decode(image_b64)
                image_mime_type = profile.get(mime_key, "image/png")
                if not isinstance(image_mime_type, str) or not image_mime_type:
                    image_mime_type = "image/png"

                image_payload = {
                    "type": "image",
                    "mime_type": image_mime_type,
                    "data_size": len(image_bytes),
                    "data_preview": image_bytes.hex(),
                }
                image_vision_text = await moonshot_vision_service.recognize_image(
                    image_payload,
                    prompt=vision_prompt,
                )
                profile[f"{label}_image_vision"] = image_vision_text
            except Exception as e:
                log.error(
                    "处理 get_user_profile %s识图失败: %s",
                    label_cn,
                    e,
                    exc_info=True,
                )
                profile[f"{label}_image_vision"] = f"（{label_cn}识图失败：处理异常）"
            finally:
                profile.pop(image_key, None)
                profile.setdefault(
                    f"{label}_note",
                    f"（{label_cn}原始图片数据已省略，已提供识图摘要）",
                )

        return clean_response

    def get_validation_error(self, model_name: str) -> Optional[str]:
        if self.base_url and self.api_key:
            return None

        log.warning("请求使用 %s 但未配置 DEEPSEEK_URL 或 DEEPSEEK_API_KEY。", model_name)
        return "DeepSeek 配置缺失，请检查环境变量。"

    async def send(
        self,
        http_client: httpx.AsyncClient,
        payload: Dict[str, Any],
        override_base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        api_url = (override_base_url or self.base_url or "").rstrip("/")
        api_key = self.api_key or ""

        if not api_url:
            raise ValueError("OpenAI 兼容通道 URL 配置缺失，请检查配置。")
        if not api_key:
            raise ValueError("OpenAI 兼容通道 API Key 配置缺失，请检查配置。")

        api_url = self._build_chat_completions_url(api_url)

        response = await http_client.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

        return {
            "response": response,
            "used_api_url": api_url,
            "used_slot_label": "deepseek",
            "used_slot_id": None,
            "used_key_tail": "N/A",
            "used_model_name": str(payload.get("model", "")),
            "skip_custom_site_for_this_turn": False,
        }
