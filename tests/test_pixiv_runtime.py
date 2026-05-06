import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from src.chat.features.pixiv import runtime as pixiv_runtime


@pytest.mark.asyncio
async def test_get_runtime_initializes_only_once(monkeypatch):
    pixiv_runtime.reset_runtime_for_tests()
    created = []

    class FakeRuntime:
        pass

    async def fake_create():
        created.append("created")
        return FakeRuntime()

    monkeypatch.setattr(pixiv_runtime.PixivRuntime, "create", staticmethod(fake_create))

    first = await pixiv_runtime.get_runtime()
    second = await pixiv_runtime.get_runtime()

    assert first is second
    assert created == ["created"]

    pixiv_runtime.reset_runtime_for_tests()

