# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.chat.services.tool_intent_service import (
    extract_function_tool_names,
    extract_forced_tool_name,
    resolve_proactive_tool_choice,
)


def test_extract_forced_tool_name_supports_alias_mapping():
    result = extract_forced_tool_name(
        "来给我唱个歌吧 tool:tts",
        ["tts_tool", "pixiv_tool"],
    )

    assert result == "tts_tool"


def test_resolve_proactive_tool_choice_matches_explicit_tool_name():
    result = resolve_proactive_tool_choice(
        "帮我找点图 tool:pixiv",
        ["tts_tool", "pixiv_tool"],
    )

    assert result == {
        "type": "function",
        "function": {"name": "pixiv_tool"},
    }


def test_resolve_proactive_tool_choice_supports_direct_tool_name():
    result = resolve_proactive_tool_choice(
        "请直接朗读这段文字 tool:tts_tool",
        ["tts_tool", "pixiv_tool"],
    )

    assert result == {
        "type": "function",
        "function": {"name": "tts_tool"},
    }


def test_resolve_proactive_tool_choice_returns_none_when_tool_unavailable():
    result = resolve_proactive_tool_choice(
        "请直接唱出来 tool:tts",
        ["pixiv_tool"],
    )

    assert result is None


def test_extract_forced_tool_name_returns_none_without_explicit_directive():
    result = extract_forced_tool_name(
        "来给我唱个歌吧",
        ["tts_tool", "pixiv_tool"],
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
