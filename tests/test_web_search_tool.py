import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.chat.features.tools.functions import web_search_tool
from src.chat.features.tools.functions.web_search_tool import (
    _dedupe_sources,
    _format_kimi_search_answer,
    _split_moonshot_api_keys,
    search_web,
)


@pytest.mark.asyncio
async def test_search_web_returns_clear_error_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("GROK_URL", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    result = await search_web({"question": "上网搜一下 ds2api 是什么", "mode": "search"})

    assert result["search_executed"] is False
    assert "MOONSHOT_API_KEY" in result["error"]


@pytest.mark.asyncio
async def test_search_web_fetch_mode_requires_url():
    result = await search_web({"question": "看下这个网页", "mode": "fetch"})

    assert result["fetch_executed"] is False
    assert "url" in result["error"]


@pytest.mark.asyncio
async def test_fetch_mode_does_not_call_grok(monkeypatch: pytest.MonkeyPatch):
    called = {"fetch": False, "grok": False}

    async def fake_fetch(url: str):
        called["fetch"] = True
        return {
            "channel": "kimi_fetch",
            "enabled": True,
            "fetch_executed": True,
            "title": "T",
            "url": url,
            "answer": "网页正文",
            "sources": [{"title": "T", "url": url}],
        }

    async def fake_grok(question: str):
        called["grok"] = True
        return {"channel": "grok", "enabled": True, "answer": "grok", "sources": []}

    monkeypatch.setattr(web_search_tool, "_fetch_with_kimi", fake_fetch)
    monkeypatch.setattr(web_search_tool, "_search_with_grok", fake_grok)

    result = await search_web({"url": "https://example.com/a", "mode": "fetch"})

    assert called["fetch"] is True
    assert called["grok"] is False
    assert result["fetch_executed"] is True
    assert result["answer"] == "网页正文"


@pytest.mark.asyncio
async def test_search_mode_calls_kimi_and_grok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROK_URL", "https://grok.example/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    called = {"kimi": False, "grok": False}

    async def fake_kimi(question: str):
        called["kimi"] = True
        return {
            "channel": "kimi",
            "enabled": True,
            "search_executed": True,
            "answer": "kimi 片段",
            "sources": [{"title": "A", "url": "https://a.com"}],
            "results": [],
        }

    async def fake_grok(question: str):
        called["grok"] = True
        return {
            "channel": "grok",
            "enabled": True,
            "answer": "grok 答案",
            "sources": [],
        }

    monkeypatch.setattr(web_search_tool, "_search_with_kimi_pro", fake_kimi)
    monkeypatch.setattr(web_search_tool, "_search_with_grok", fake_grok)

    result = await search_web({"question": "问题", "mode": "search"})

    assert called["kimi"] is True
    assert called["grok"] is True
    assert "Kimi 联网搜索结果" in result["answer"]
    assert "Grok 辅助结果" in result["answer"]
    assert result["search_executed"] is True


def test_split_moonshot_api_keys_supports_multiple_formats():
    keys = _split_moonshot_api_keys("k1,k2，k3\nk1")

    assert keys == ["k1", "k2", "k3"]


def test_format_kimi_search_answer_includes_chunks_and_sources():
    results = [
        {
            "title": "示例标题",
            "url": "https://example.com/a",
            "date": "2026-01-01",
            "site_name": "示例站点",
            "snippet": "摘要内容",
            "chunks": [{"text": "正文片段", "score": 1.0}],
        }
    ]

    text = _format_kimi_search_answer(results)

    assert "示例标题" in text
    assert "https://example.com/a" in text
    assert "正文片段" in text


def test_dedupe_sources_removes_duplicates():
    sources = _dedupe_sources(
        [
            ("A", "https://example.com/a"),
            ("A", "https://example.com/a"),
            ("", ""),
            ("B", "https://example.com/b"),
        ]
    )

    assert sources == [
        {"title": "A", "url": "https://example.com/a"},
        {"title": "B", "url": "https://example.com/b"},
    ]
