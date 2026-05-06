# -*- coding: utf-8 -*-

import asyncio
import logging
import socket
from importlib import import_module
from typing import Any

import requests

from .config import PixivConfig

log = logging.getLogger(__name__)


class PixivClientWrapper:
    def __init__(self, pixiv_config: PixivConfig):
        self.pixiv_config = pixiv_config
        self._refresh_task: asyncio.Task | None = None
        self._pixivpy = import_module("pixivpy3")

        if pixiv_config.proxy:
            self.client_api = self._pixivpy.AppPixivAPI(
                **pixiv_config.get_requests_kwargs()
            )
            log.info("Pixiv 客户端使用代理模式。")
        elif pixiv_config.api_proxy_host:
            self.client_api = self._pixivpy.AppPixivAPI()
            self.client_api.hosts = f"https://{pixiv_config.api_proxy_host}"
            log.info(
                "Pixiv 客户端使用 API 反代模式: %s",
                pixiv_config.api_proxy_host,
            )
        else:
            self.client_api = self._create_direct_client()

    def _create_direct_client(self) -> Any:
        try:
            socket.gethostbyname("oauth.secure.pixiv.net")
            requests.head(
                "https://oauth.secure.pixiv.net/",
                timeout=5,
                allow_redirects=False,
            )
            log.info("Pixiv 客户端使用标准直连模式。")
            return self._pixivpy.AppPixivAPI()
        except Exception:
            log.info("Pixiv 标准直连不可用，尝试 ByPassSniApi。")

        client_api = self._pixivpy.ByPassSniApi()
        hosts_result = self._require_appapi_hosts_with_cn_doh(client_api)
        if hosts_result:
            log.info("Pixiv 客户端使用 ByPassSniApi 模式: %s", hosts_result)
            return client_api

        log.warning("Pixiv 直连方案初始化失败，回退到标准 AppPixivAPI。")
        return self._pixivpy.AppPixivAPI()

    def _require_appapi_hosts_with_cn_doh(
        self,
        api: Any,
        hostname: str = "app-api.secure.pixiv.net",
        timeout: int = 10,
    ) -> str | bool:
        doh_urls = [
            "https://doh.pub/dns-query",
            "https://dns.alidns.com/dns-query",
            "https://1.0.0.1/dns-query",
            "https://1.1.1.1/dns-query",
            "https://doh.dns.sb/dns-query",
        ]
        headers = {"Accept": "application/dns-json"}
        params = {"name": hostname, "type": "A", "do": "false", "cd": "false"}

        for url in doh_urls:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )
                if response.status_code != 200:
                    continue
                data = response.json()
                answers = data.get("Answer") or []
                if not answers:
                    continue
                ip = answers[0].get("data")
                if ip:
                    api.hosts = f"https://{ip}"
                    return api.hosts
            except Exception:
                continue

        return False

    async def authenticate(self) -> bool:
        if not self.pixiv_config.refresh_token:
            log.warning("Pixiv 未配置 refresh token。")
            return False

        try:
            await asyncio.to_thread(
                self.client_api.auth,
                refresh_token=self.pixiv_config.refresh_token,
            )
            return True
        except Exception as exc:
            log.error("Pixiv 认证失败: %s", exc, exc_info=True)
            return False

    async def periodic_token_refresh(self) -> None:
        while True:
            try:
                wait_seconds = self.pixiv_config.refresh_interval_minutes * 60
                await asyncio.sleep(wait_seconds)
                if not self.pixiv_config.refresh_token:
                    continue
                self.client_api.auth(refresh_token=self.pixiv_config.refresh_token)
                log.info("Pixiv refresh token 保活成功。")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Pixiv refresh token 保活失败: %s", exc, exc_info=True)

    def start_refresh_task(self) -> asyncio.Task | None:
        if self.pixiv_config.refresh_interval_minutes <= 0:
            return None
        if self._refresh_task and not self._refresh_task.done():
            return self._refresh_task
        self._refresh_task = asyncio.create_task(self.periodic_token_refresh())
        return self._refresh_task

    async def call_pixiv_api(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
