# -*- coding: utf-8 -*-
import asyncio
import base64
import logging
import random
from pathlib import Path
from typing import Optional

import httpx

from src.chat.config.chat_config import GPT_IMAGE_CONFIG

log = logging.getLogger(__name__)


class GPTImageService:
    def __init__(self):
        self._api_key = GPT_IMAGE_CONFIG["API_KEY"]
        self._base_url = GPT_IMAGE_CONFIG["BASE_URL"]
        self._model = GPT_IMAGE_CONFIG["MODEL"]
        self._size = GPT_IMAGE_CONFIG["SIZE"]
        self._quality = GPT_IMAGE_CONFIG["QUALITY"]
        self._timeout = GPT_IMAGE_CONFIG["TIMEOUT"]
        self._reference_image_url = GPT_IMAGE_CONFIG.get("REFERENCE_IMAGE_URL", "")
        self._client: Optional[httpx.AsyncClient] = None
        self._reference_images: list[bytes] = []
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            if not self._api_key:
                log.warning("GPTImageService: 未配置 API key，服务已禁用")
                self._initialized = True
                return

            assets_dir = Path(__file__).resolve().parents[3] / "assets"
            for ref_file in sorted(assets_dir.glob("ref_*.png")):
                try:
                    data = ref_file.read_bytes()
                except OSError as exc:
                    log.warning(
                        f"GPTImageService: 读取参考图失败 {ref_file.name}: {exc}"
                    )
                    continue
                self._reference_images.append(data)

            if not self._reference_images and self._reference_image_url:
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(self._reference_image_url)
                        response.raise_for_status()
                        self._reference_images.append(response.content)
                except Exception as exc:
                    log.warning(f"GPTImageService: 下载远程参考图失败: {exc}")

            self._initialized = True
            log.info(
                "GPTImageService 初始化完成: refs=%s, model=%s, base_url=%s",
                len(self._reference_images),
                self._model,
                self._base_url,
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._client

    async def generate_feeding_image(
        self,
        feed_image_bytes: bytes,
        feed_mime_type: str,
    ) -> Optional[bytes]:
        if not self.is_available:
            return None

        await self._ensure_initialized()

        if self._reference_images:
            reference_bytes = random.choice(self._reference_images)
            return await self._try_edit_once(
                feed_image_bytes=feed_image_bytes,
                feed_mime_type=feed_mime_type,
                reference_bytes=reference_bytes,
            )

        return await self._try_generate_once()

    async def _try_edit_once(
        self,
        feed_image_bytes: bytes,
        feed_mime_type: str,
        reference_bytes: bytes,
    ) -> Optional[bytes]:
        client = self._get_client()
        prompt = (
            "参考第一张图的人物外貌、发型、服装和整体画风。"
            "让她正在开心地接住并吃掉第二张图里的内容。"
            "第二张图只用来决定被投喂物，不要照搬它的构图、背景、文字、水印或界面元素。"
            "自由构图，突出被投喂时的动作、表情和满足感，整体氛围温暖可爱。"
        )
        files = [
            ("image", ("reference.png", reference_bytes, "image/png")),
            ("image", ("feed.png", feed_image_bytes, feed_mime_type or "image/png")),
        ]
        data = {
            "model": self._model,
            "prompt": prompt,
            "n": "1",
            "size": self._size,
            "input_fidelity": "low",
            "response_format": "b64_json",
        }

        response = await client.post("/images/edits", files=files, data=data)
        response.raise_for_status()
        return self._extract_image(response.json())

    async def _try_generate_once(self) -> Optional[bytes]:
        client = self._get_client()
        prompt = (
            "1girl, solo, brown hair, braid, long hair, white shirt, brown vest, black necktie, "
            "一个可爱的动漫少女正在开心地被投喂，张嘴接住食物，表情满足，温暖柔和的插画风格。"
        )
        body = {
            "model": self._model,
            "prompt": prompt,
            "n": 1,
            "size": self._size,
            "quality": self._quality,
            "response_format": "b64_json",
        }

        response = await client.post("/images/generations", json=body)
        response.raise_for_status()
        return self._extract_image(response.json())

    def _extract_image(self, response_data: dict) -> Optional[bytes]:
        images = response_data.get("data") or []
        if not images:
            log.warning("GPTImageService: 响应中没有图片数据")
            return None

        image_data = images[0]
        b64_json = image_data.get("b64_json")
        if not b64_json:
            log.warning(
                "GPTImageService: 非预期响应格式 keys=%s",
                list(image_data.keys()),
            )
            return None

        return base64.b64decode(b64_json)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


gpt_image_service = GPTImageService()
