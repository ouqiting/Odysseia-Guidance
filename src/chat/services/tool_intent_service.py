# -*- coding: utf-8 -*-

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_PROACTIVE_TOOL_CHOICE_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(上网搜一下|联网|上网|本子|查一下|链接|网站|网址|下载地址)"),
        "search_web",
    ),
    (
        re.compile(r"占卜"),
        "tarot_reading",
    ),
)


def extract_function_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    tool_names: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            continue
        tool_name = str(function_payload.get("name") or "").strip()
        if tool_name:
            tool_names.append(tool_name)
    return tool_names


def resolve_proactive_tool_choice(
    user_message: str,
    available_tool_names: List[str],
) -> Optional[Dict[str, Any]]:
    normalized_message = str(user_message or "").strip()
    if not normalized_message or not available_tool_names:
        return None

    available_tool_name_set = set(available_tool_names)
    for pattern, tool_name in _PROACTIVE_TOOL_CHOICE_RULES:
        if not pattern.search(normalized_message):
            continue
        if tool_name not in available_tool_name_set:
            log.info(
                "检测到工具调用意图，但目标工具当前不可用 | tool=%s | message=%s",
                tool_name,
                normalized_message[:200],
            )
            return None
        return {
            "type": "function",
            "function": {"name": tool_name},
        }

    return None
