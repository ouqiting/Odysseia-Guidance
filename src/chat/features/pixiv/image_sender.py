# -*- coding: utf-8 -*-

import io
import logging
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import discord

from .config import PixivConfig
from .models import PixivImageResult

log = logging.getLogger(__name__)


class PixivMessageDeleteView(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=None)

    @discord.ui.button(label="删除", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def delete_message(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        del button

        message = getattr(interaction, "message", None)
        if message is None:
            try:
                await interaction.response.send_message("找不到可删除的消息。", ephemeral=True)
            except Exception:
                pass
            return

        try:
            await message.delete()
        except Exception as exc:
            log.warning("Pixiv 删除按钮删除消息失败: %s", exc, exc_info=True)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("删除失败，可能消息已经不存在了。", ephemeral=True)
                else:
                    await interaction.followup.send("删除失败，可能消息已经不存在了。", ephemeral=True)
            except Exception:
                pass


def get_proxied_image_url(original_url: str, config: PixivConfig) -> str:
    if not original_url:
        return original_url
    if not config.use_image_proxy:
        return original_url
    if "i.pximg.net" in original_url:
        return original_url.replace("i.pximg.net", config.image_proxy_host)
    return original_url


def extract_best_image_url(illust) -> str:
    image_urls = getattr(illust, "image_urls", None)
    if image_urls is not None:
        for key in ("large", "medium", "square_medium"):
            candidate = getattr(image_urls, key, None)
            if candidate:
                return str(candidate)

    meta_single_page = getattr(illust, "meta_single_page", None)
    if meta_single_page is not None:
        candidate = getattr(meta_single_page, "original_image_url", None)
        if not candidate and isinstance(meta_single_page, dict):
            candidate = meta_single_page.get("original_image_url")
        if candidate:
            return str(candidate)

    meta_pages = getattr(illust, "meta_pages", None) or []
    if meta_pages:
        first_page = meta_pages[0]
        if isinstance(first_page, dict):
            page_urls = first_page.get("image_urls", {})
            for key in ("large", "medium", "square_medium", "original"):
                candidate = page_urls.get(key)
                if candidate:
                    return str(candidate)

    return ""


def build_file_name(title: str, illust_id: int, image_url: str) -> str:
    safe_title = "".join(
        ch for ch in str(title or "pixiv") if ch.isalnum() or ch in {" ", "_", "-"}
    ).strip()
    if not safe_title:
        safe_title = "pixiv"
    ext = Path(urlparse(image_url).path).suffix or ".jpg"
    return f"{safe_title[:40]}_{illust_id}{ext}"


async def download_image_bytes(url: str, config: PixivConfig) -> tuple[bytes, str]:
    actual_url = get_proxied_image_url(url, config)
    headers = {
        "Referer": "https://www.pixiv.net/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    }

    timeout = aiohttp.ClientTimeout(total=45)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(actual_url, proxy=config.proxy or None) as response:
            response.raise_for_status()
            data = await response.read()
            mime_type = response.headers.get("Content-Type", "image/jpeg")
            return data, mime_type


async def send_illust_to_channel(
    channel: discord.abc.Messageable,
    image_result: PixivImageResult,
    config: PixivConfig,
) -> tuple[bool, str | None]:
    try:
        image_bytes, _mime_type = await download_image_bytes(image_result.image_url, config)
    except Exception as exc:
        log.error("Pixiv 图片下载失败: %s", exc, exc_info=True)
        return False, f"Pixiv 图片下载失败：{exc}"

    try:
        file = discord.File(
            fp=io.BytesIO(image_bytes),
            filename=image_result.file_name,
        )
        caption = image_result.caption[:1900]
        view = PixivMessageDeleteView()
        sent_message = await channel.send(content=caption, file=file, view=view)
        if hasattr(sent_message, "edit"):
            try:
                await sent_message.edit(suppress=True)
            except Exception as exc:
                log.warning("Pixiv 消息 suppress embeds 失败，保留原消息显示: %s", exc)
        return True, None
    except Exception as exc:
        log.error("Pixiv 图片发送失败: %s", exc, exc_info=True)
        return False, f"Pixiv 图片发送失败：{exc}"
