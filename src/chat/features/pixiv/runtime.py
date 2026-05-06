# -*- coding: utf-8 -*-

import asyncio
import logging

from .client import PixivClientWrapper
from .config import PixivConfig
from .service import PixivService
from .storage import PixivStorage

log = logging.getLogger(__name__)

_runtime = None
_runtime_lock = asyncio.Lock()


class PixivRuntime:
    def __init__(self) -> None:
        self.config = PixivConfig.from_env()
        self.storage = PixivStorage(retention_days=self.config.random_dedupe_days)
        self.client_wrapper = PixivClientWrapper(self.config)
        self.service = PixivService(self.client_wrapper, self.config, self.storage)

    @classmethod
    async def create(cls) -> "PixivRuntime":
        runtime = cls()
        await runtime.initialize()
        return runtime

    async def initialize(self) -> None:
        await self.storage.init_async()
        self.client_wrapper.start_refresh_task()
        log.info("PixivRuntime 初始化完成。")

    async def random_by_tag(self, **kwargs):
        return await self.service.random_by_tag(**kwargs)

    async def random_ranking(self, **kwargs):
        return await self.service.random_ranking(**kwargs)

    async def random_illust(self, **kwargs):
        return await self.service.random_by_tag(**kwargs)

    async def mark_sent(self, illust_id: int) -> None:
        await self.service.mark_sent(illust_id)


async def get_runtime() -> PixivRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime

    async with _runtime_lock:
        if _runtime is None:
            _runtime = await PixivRuntime.create()
        return _runtime


def reset_runtime_for_tests() -> None:
    global _runtime
    _runtime = None
