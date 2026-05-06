import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from src.chat.features.pixiv.models import PixivImageResult, PixivToolResult
from src.chat.features.tools.functions import pixiv_tool as pixiv_tool_module


class FakeChannel:
    def __init__(self):
        self.sent_messages = []

    async def send(self, *args, **kwargs):
        self.sent_messages.append({"args": args, "kwargs": kwargs})


class FakeRuntime:
    def __init__(self):
        self.config = SimpleNamespace()
        self.marked = []

    async def random_by_tag(self, **kwargs):
        return PixivToolResult(
            True,
            f"已发送 {kwargs.get('count', 1)} 张 Pixiv 插画。",
            images=[
                PixivImageResult(
                    illust_id=123,
                    title="miku",
                    author="tester",
                    caption="caption-1",
                    image_url="https://example.com/test1.jpg",
                    file_name="test1.jpg",
                    tags=["miku"],
                )
            ][: kwargs.get("count", 1)],
        )

    async def random_ranking(self, **kwargs):
        count = kwargs.get("count", 1)
        images = [
            PixivImageResult(
                illust_id=123,
                title="rank1",
                author="tester",
                caption="caption-1",
                image_url="https://example.com/test1.jpg",
                file_name="test1.jpg",
                tags=["rank"],
            ),
            PixivImageResult(
                illust_id=456,
                title="rank2",
                author="tester",
                caption="caption-2",
                image_url="https://example.com/test2.jpg",
                file_name="test2.jpg",
                tags=["rank"],
            ),
        ][:count]
        return PixivToolResult(
            True,
            f"已从排行榜随机发送 {len(images)} 张 Pixiv 插画。",
            images=images,
        )

    async def random_illust(self, **kwargs):
        return await self.random_by_tag(**kwargs)

    async def mark_sent(self, illust_id: int):
        self.marked.append(illust_id)


@pytest.mark.asyncio
async def test_pixiv_tool_returns_error_without_channel():
    result = await pixiv_tool_module.pixiv_tool(
        action="random_by_tag",
        params={"tags": ["miku"], "count": 1},
    )

    assert "找不到有效的消息频道" in result


@pytest.mark.asyncio
async def test_pixiv_tool_sends_image_and_marks_sent(monkeypatch):
    runtime = FakeRuntime()
    channel = FakeChannel()

    async def fake_get_runtime():
        return runtime

    async def fake_send(channel_obj, image_result, config):
        await channel_obj.send(content=image_result.caption)
        return True, None

    monkeypatch.setattr(pixiv_tool_module.pixiv_runtime, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(
        pixiv_tool_module.pixiv_image_sender,
        "send_illust_to_channel",
        fake_send,
    )

    result = await pixiv_tool_module.pixiv_tool(
        action="random_by_tag",
        params={"tags": ["miku"], "count": 1},
        channel=channel,
    )

    assert result == "已发送 1 张 Pixiv 插画。"
    assert len(channel.sent_messages) == 1
    assert runtime.marked == [123]


@pytest.mark.asyncio
async def test_pixiv_tool_supports_multiple_images(monkeypatch):
    runtime = FakeRuntime()
    channel = FakeChannel()

    async def fake_get_runtime():
        return runtime

    async def fake_send(channel_obj, image_result, config):
        await channel_obj.send(content=image_result.caption)
        return True, None

    monkeypatch.setattr(pixiv_tool_module.pixiv_runtime, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(
        pixiv_tool_module.pixiv_image_sender,
        "send_illust_to_channel",
        fake_send,
    )

    result = await pixiv_tool_module.pixiv_tool(
        action="random_ranking",
        params={"count": 2, "mode": "safe"},
        channel=channel,
    )

    assert result == "已从排行榜随机发送 2 张 Pixiv 插画。"
    assert len(channel.sent_messages) == 2
    assert runtime.marked == [123, 456]


@pytest.mark.asyncio
async def test_pixiv_tool_returns_send_failure(monkeypatch):
    runtime = FakeRuntime()
    channel = FakeChannel()

    async def fake_get_runtime():
        return runtime

    async def fake_send(channel_obj, image_result, config):
        return False, "Pixiv 图片下载失败：boom"

    monkeypatch.setattr(pixiv_tool_module.pixiv_runtime, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(
        pixiv_tool_module.pixiv_image_sender,
        "send_illust_to_channel",
        fake_send,
    )

    result = await pixiv_tool_module.pixiv_tool(
        action="random_by_tag",
        params={"tags": ["miku"], "count": 1},
        channel=channel,
    )

    assert result == "Pixiv 图片下载失败：boom"


@pytest.mark.asyncio
async def test_pixiv_tool_returns_parse_error_for_invalid_params():
    channel = FakeChannel()

    result = await pixiv_tool_module.pixiv_tool(
        action="random_by_tag",
        params={"tags": ["miku"], "count": 6},
        channel=channel,
    )

    assert "参数解析失败" in result
