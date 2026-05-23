import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from src.chat.features.pixiv.config import PixivConfig
from src.chat.features.pixiv.image_sender import (
    PixivMessageDeleteView,
    get_proxied_image_url,
    send_illust_to_channel,
)
from src.chat.features.pixiv.models import PixivImageResult


class FakeSentMessage:
    def __init__(self):
        self.suppress_called = False

    async def edit(self, **kwargs):
        if kwargs.get("suppress") is True:
            self.suppress_called = True


class FakeChannel:
    def __init__(self):
        self.sent_message = FakeSentMessage()
        self.last_kwargs = None

    async def send(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return self.sent_message


def _config():
    return PixivConfig(
        refresh_token="token",
        proxy="",
        api_proxy_host="",
        image_proxy_host="i.pixiv.re",
        use_image_proxy=True,
        default_mode="safe",
        allow_ai_default=False,
        default_excluded_tags=[],
        refresh_interval_minutes=180,
        random_dedupe_days=7,
    )


def test_get_proxied_image_url_rewrites_old_proxy_path_for_yuki():
    config = _config()
    config.image_proxy_host = "i.yuki.sh"

    actual = get_proxied_image_url(
        "https://i.pixiv.re/c/600x1200_90/img-master/img/2026/05/23/test_p0_master1200.jpg",
        config,
    )

    assert actual == "https://i.yuki.sh/img-master/img/2026/05/23/test_p0_master1200.jpg"


@pytest.mark.asyncio
async def test_send_illust_to_channel_suppresses_embeds(monkeypatch):
    channel = FakeChannel()
    image_result = PixivImageResult(
        illust_id=1,
        title="test",
        author="tester",
        caption="caption with https://www.pixiv.net/artworks/1",
        image_url="https://example.com/1.jpg",
        file_name="1.jpg",
    )

    async def fake_download_image_bytes(url, config):
        return b"fake-image", "image/jpeg"

    monkeypatch.setattr(
        "src.chat.features.pixiv.image_sender.download_image_bytes",
        fake_download_image_bytes,
    )

    success, error = await send_illust_to_channel(channel, image_result, _config())

    assert success is True
    assert error is None
    assert channel.sent_message.suppress_called is True
    assert isinstance(channel.last_kwargs["view"], PixivMessageDeleteView)


@pytest.mark.asyncio
async def test_delete_view_deletes_message():
    view = PixivMessageDeleteView()

    class FakeMessage:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class FakeResponse:
        def is_done(self):
            return False

        async def send_message(self, *args, **kwargs):
            return None

    fake_message = FakeMessage()
    fake_interaction = SimpleNamespace(message=fake_message, response=FakeResponse())
    button = view.children[0]

    await button.callback(fake_interaction)

    assert fake_message.deleted is True
