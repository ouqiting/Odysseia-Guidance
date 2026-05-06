import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from src.chat.features.pixiv.tag_utils import filter_illusts, normalize_tag_inputs


def _illust(*, illust_id: int, x_restrict: int = 0, ai_type: int = 0, tags=None):
    return SimpleNamespace(
        id=illust_id,
        x_restrict=x_restrict,
        illust_ai_type=ai_type,
        tags=tags or [],
    )


def test_normalize_tag_inputs_supports_negative_tags():
    normalized = normalize_tag_inputs(["初音未来", "-guro"], ["blood"])

    assert normalized.include_tags == ["初音未来"]
    assert normalized.exclude_tags == ["guro", "blood"]
    assert normalized.search_text == "初音未来"


def test_normalize_tag_inputs_detects_conflicts():
    with pytest.raises(ValueError):
        normalize_tag_inputs(["初音未来", "-初音未来"], [])


def test_filter_illusts_applies_r18_ai_and_excluded_tags():
    illusts = [
        _illust(illust_id=1, x_restrict=0, ai_type=0, tags=[{"name": "vocaloid"}]),
        _illust(illust_id=2, x_restrict=1, ai_type=0, tags=[{"name": "R-18"}]),
        _illust(illust_id=3, x_restrict=0, ai_type=2, tags=[{"name": "AI生成"}]),
        _illust(illust_id=4, x_restrict=0, ai_type=0, tags=[{"name": "guro"}]),
    ]

    filtered = filter_illusts(
        illusts,
        mode="safe",
        allow_ai=False,
        excluded_tags=["guro"],
    )

    assert [item.id for item in filtered] == [1]

