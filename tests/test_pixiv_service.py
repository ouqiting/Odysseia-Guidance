import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from src.chat.features.pixiv.config import PixivConfig
from src.chat.features.pixiv.service import PixivService


def _illust(
    *,
    illust_id: int,
    title: str = "title",
    x_restrict: int = 0,
    ai_type: int = 0,
    tags=None,
):
    return SimpleNamespace(
        id=illust_id,
        title=title,
        x_restrict=x_restrict,
        illust_ai_type=ai_type,
        tags=tags or [],
        user=SimpleNamespace(name="tester"),
        image_urls=SimpleNamespace(large=f"https://i.pximg.net/img/{illust_id}.jpg"),
    )


class FakeStorage:
    def __init__(self, recent_ids=None):
        self.recent_ids = set(recent_ids or [])
        self.marked = []

    async def prune_old_entries(self):
        return None

    async def get_recent_sent_ids(self, limit=100):
        return set(self.recent_ids)

    async def mark_sent(self, illust_id: int):
        self.marked.append(illust_id)


class FakeClientWrapper:
    def __init__(self, *, authenticated=True):
        self.client_api = SimpleNamespace(
            search_illust=self._search_illust,
            illust_recommended=self._illust_recommended,
            illust_ranking=self._illust_ranking,
            parse_qs=lambda _url: None,
        )
        self.authenticated = authenticated
        self.search_result = SimpleNamespace(illusts=[])
        self.recommended_result = SimpleNamespace(illusts=[])
        self.ranking_result = SimpleNamespace(illusts=[])

    async def authenticate(self):
        return self.authenticated

    async def call_pixiv_api(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _search_illust(self, **kwargs):
        return self.search_result

    def _illust_recommended(self):
        return self.recommended_result

    def _illust_ranking(self, **kwargs):
        return self.ranking_result


def _config(refresh_token="token"):
    return PixivConfig(
        refresh_token=refresh_token,
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


@pytest.mark.asyncio
async def test_random_by_tag_with_tags_returns_filtered_random_item():
    wrapper = FakeClientWrapper()
    wrapper.search_result = SimpleNamespace(
        illusts=[
            _illust(illust_id=1, x_restrict=1, tags=[{"name": "R-18"}]),
            _illust(illust_id=2, x_restrict=0, tags=[{"name": "miku"}]),
        ]
    )
    service = PixivService(wrapper, _config(), FakeStorage())

    result = await service.random_by_tag(tags=["miku"], mode="safe", count=1)

    assert result.success is True
    assert [image.illust_id for image in result.images] == [2]


@pytest.mark.asyncio
async def test_random_by_tag_supports_multiple_images():
    wrapper = FakeClientWrapper()
    wrapper.search_result = SimpleNamespace(
        illusts=[
            _illust(illust_id=1, tags=[{"name": "miku"}]),
            _illust(illust_id=2, tags=[{"name": "miku"}]),
        ]
    )
    service = PixivService(wrapper, _config(), FakeStorage())

    result = await service.random_by_tag(tags=["miku"], mode="safe", count=2)

    assert result.success is True
    assert len(result.images) == 2
    assert {image.illust_id for image in result.images} == {1, 2}


@pytest.mark.asyncio
async def test_random_ranking_prefers_unseen_items():
    wrapper = FakeClientWrapper()
    wrapper.ranking_result = SimpleNamespace(
        illusts=[
            _illust(illust_id=1, tags=[{"name": "rank"}]),
            _illust(illust_id=2, tags=[{"name": "rank"}]),
        ]
    )
    service = PixivService(wrapper, _config(), FakeStorage(recent_ids={1}))

    result = await service.random_ranking(mode="safe", count=1)

    assert result.success is True
    assert [image.illust_id for image in result.images] == [2]


@pytest.mark.asyncio
async def test_random_by_tag_without_tags_uses_r18_pool_when_mode_is_r18():
    wrapper = FakeClientWrapper()
    wrapper.ranking_result = SimpleNamespace(
        illusts=[_illust(illust_id=10, x_restrict=1, tags=[{"name": "R-18"}])]
    )
    wrapper.recommended_result = SimpleNamespace(
        illusts=[_illust(illust_id=20, x_restrict=0, tags=[{"name": "safe"}])]
    )
    service = PixivService(wrapper, _config(), FakeStorage())

    result = await service.random_by_tag(mode="r18", count=1)

    assert result.success is True
    assert [image.illust_id for image in result.images] == [10]


@pytest.mark.asyncio
async def test_random_illust_supports_multiple_images():
    wrapper = FakeClientWrapper()
    wrapper.recommended_result = SimpleNamespace(
        illusts=[
            _illust(illust_id=20, x_restrict=0, tags=[{"name": "safe"}]),
            _illust(illust_id=21, x_restrict=0, tags=[{"name": "safe"}]),
            _illust(illust_id=22, x_restrict=0, tags=[{"name": "safe"}]),
        ]
    )
    service = PixivService(wrapper, _config(), FakeStorage())

    result = await service.random_by_tag(mode="safe", count=2)

    assert result.success is True
    assert len(result.images) == 2
    assert {image.illust_id for image in result.images}.issubset({20, 21, 22})


@pytest.mark.asyncio
async def test_service_reports_missing_refresh_token():
    wrapper = FakeClientWrapper(authenticated=False)
    service = PixivService(wrapper, _config(refresh_token=""), FakeStorage())

    result = await service.random_by_tag(tags=["miku"], count=1)

    assert result.success is False
    assert result.error == "missing_refresh_token"


@pytest.mark.asyncio
async def test_service_reports_auth_failure():
    wrapper = FakeClientWrapper(authenticated=False)
    service = PixivService(wrapper, _config(refresh_token="token"), FakeStorage())

    result = await service.random_by_tag(tags=["miku"], count=1)

    assert result.success is False
    assert result.error == "auth_failed"


@pytest.mark.asyncio
async def test_service_returns_no_results_when_filtered_pool_is_empty():
    wrapper = FakeClientWrapper()
    wrapper.search_result = SimpleNamespace(
        illusts=[_illust(illust_id=1, x_restrict=1, tags=[{"name": "R-18"}])]
    )
    service = PixivService(wrapper, _config(), FakeStorage())

    result = await service.random_by_tag(tags=["miku"], mode="safe", count=1)

    assert result.success is False
    assert result.error == "no_results"


@pytest.mark.asyncio
async def test_service_rejects_invalid_count():
    wrapper = FakeClientWrapper()
    service = PixivService(wrapper, _config(), FakeStorage())

    result = await service.random_by_tag(tags=["miku"], count=6)

    assert result.success is False
    assert result.error == "invalid_count"


@pytest.mark.asyncio
async def test_default_excluded_tags_are_applied_automatically():
    wrapper = FakeClientWrapper()
    wrapper.search_result = SimpleNamespace(
        illusts=[
            _illust(illust_id=1, tags=[{"name": "miku"}, {"name": "ntr"}]),
            _illust(illust_id=2, tags=[{"name": "miku"}]),
        ]
    )
    config = _config()
    config.default_excluded_tags = ["ntr"]
    service = PixivService(wrapper, config, FakeStorage())

    result = await service.random_by_tag(tags=["miku"], count=2)

    assert result.success is True
    assert [image.illust_id for image in result.images] == [2]
