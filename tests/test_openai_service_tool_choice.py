# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.chat.services.tool_intent_service import (
    extract_function_tool_names,
    resolve_proactive_tool_choice,
)


def test_resolve_proactive_tool_choice_matches_search_web():
    result = resolve_proactive_tool_choice(
        "你上网搜一下今天的 AI 新闻",
        ["search_web", "tarot_reading"],
    )

    assert result == {
        "type": "function",
        "function": {"name": "search_web"},
    }


def test_resolve_proactive_tool_choice_matches_tarot_reading():
    result = resolve_proactive_tool_choice(
        "帮我占卜一下最近的感情运势",
        ["search_web", "tarot_reading"],
    )

    assert result == {
        "type": "function",
        "function": {"name": "tarot_reading"},
    }


def test_resolve_proactive_tool_choice_returns_none_when_tool_unavailable():
    result = resolve_proactive_tool_choice(
        "请联网查一下这个新闻",
        ["tarot_reading"],
    )

    assert result is None


def test_extract_function_tool_names_skips_invalid_entries():
    tool_names = extract_function_tool_names(
        [
            {"type": "function", "function": {"name": "search_web"}},
            {"type": "web_search"},
            {"type": "function", "function": {"name": ""}},
            {"type": "function", "function": {"name": "tarot_reading"}},
        ]
    )

    assert tool_names == ["search_web", "tarot_reading"]
