# -*- coding: utf-8 -*-
import asyncio
import base64
import io
import logging
import random
import time
from pathlib import Path
from typing import Optional

import httpx
import requests
from PIL import Image

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

    def _summarize_exception(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            body_preview = ""
            try:
                body_preview = response.text[:500]
            except Exception:
                body_preview = "<unavailable>"
            return (
                f"HTTPStatusError status={response.status_code} "
                f"url={response.request.method} {response.request.url} "
                f"body={body_preview}"
            )

        if isinstance(exc, httpx.RequestError):
            request = exc.request
            return (
                f"{exc.__class__.__name__} "
                f"url={request.method} {request.url} "
                f"detail={exc}"
            )

        return f"{exc.__class__.__name__}: {exc}"

    def _normalize_image_to_png(self, image_bytes: bytes, image_label: str) -> bytes:
        input_buffer = io.BytesIO(image_bytes)
        output_buffer = io.BytesIO()
        try:
            with Image.open(input_buffer) as image:
                normalized = image.convert("RGBA")
                normalized.save(output_buffer, format="PNG", optimize=True)
            return output_buffer.getvalue()
        except Exception as exc:
            log.error(
                "GPTImageService: 图片规范化失败 label=%s error=%s",
                image_label,
                self._summarize_exception(exc),
            )
            raise
        finally:
            input_buffer.close()
            output_buffer.close()

    async def generate_feeding_image(
        self,
        feed_image_bytes: bytes,
        feed_mime_type: str,
    ) -> Optional[bytes]:
        if not self.is_available:
            log.info("GPTImageService: 跳过投喂生图，原因=未配置 API key")
            return None

        await self._ensure_initialized()
        start_time = time.perf_counter()
        log.info(
            "GPTImageService: 开始投喂生图 mime=%s bytes=%s refs=%s model=%s",
            feed_mime_type,
            len(feed_image_bytes),
            len(self._reference_images),
            self._model,
        )

        if self._reference_images:
            reference_bytes = random.choice(self._reference_images)
            try:
                result = await self._try_edit_once(
                    feed_image_bytes=feed_image_bytes,
                    feed_mime_type=feed_mime_type,
                    reference_bytes=reference_bytes,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                log.error(
                    "GPTImageService: EDIT 生图失败 elapsed=%.2fs error=%s",
                    elapsed,
                    self._summarize_exception(exc),
                )
                raise

            elapsed = time.perf_counter() - start_time
            if result is None:
                log.warning(
                    "GPTImageService: EDIT 生图未返回图片数据 elapsed=%.2fs",
                    elapsed,
                )
            else:
                log.info(
                    "GPTImageService: EDIT 生图成功 elapsed=%.2fs output_bytes=%s",
                    elapsed,
                    len(result),
                )
            return result

        try:
            result = await self._try_generate_once()
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            log.error(
                "GPTImageService: GENERATE 生图失败 elapsed=%.2fs error=%s",
                elapsed,
                self._summarize_exception(exc),
            )
            raise

        elapsed = time.perf_counter() - start_time
        if result is None:
            log.warning(
                "GPTImageService: GENERATE 生图未返回图片数据 elapsed=%.2fs",
                elapsed,
            )
        else:
            log.info(
                "GPTImageService: GENERATE 生图成功 elapsed=%.2fs output_bytes=%s",
                elapsed,
                len(result),
            )
        return result

    async def _try_edit_once(
        self,
        feed_image_bytes: bytes,
        feed_mime_type: str,
        reference_bytes: bytes,
    ) -> Optional[bytes]:
        client = self._get_client()
        normalized_reference_bytes = self._normalize_image_to_png(
            reference_bytes, "reference"
        )
        normalized_feed_bytes = self._normalize_image_to_png(
            feed_image_bytes, "feed"
        )
        prompt = (
            "参考第一张图的人物外貌、发型、服装和整体画风。"
            "让她正在开心地接住并吃掉第二张图里的内容。"
            "第二张图只用来决定被投喂物，不要照搬它的构图、背景、文字、水印或界面元素。"
            "自由构图，突出被投喂时的动作、表情和满足感，整体氛围温暖可爱。"
        )
        files = [
            ("image", ("reference.png", normalized_reference_bytes, "image/png")),
            ("image", ("feed.png", normalized_feed_bytes, "image/png")),
        ]
        data = {
            "model": self._model,
            "prompt": prompt,
            "n": "1",
            "size": self._size,
            "input_fidelity": "low",
            "response_format": "b64_json",
        }

        prepared_request = requests.Request(
            method="POST",
            url=f"{self._base_url.rstrip('/')}/images/edits",
            files=files,
            data=data,
        ).prepare()
        body = prepared_request.body
        if not isinstance(body, (bytes, bytearray)):
            raise TypeError(
                f"Unexpected multipart body type: {type(body).__name__}"
            )

        headers = {
            "Content-Type": prepared_request.headers["Content-Type"],
            "Content-Length": str(len(body)),
        }
        response = await client.post("/images/edits", content=body, headers=headers)
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
