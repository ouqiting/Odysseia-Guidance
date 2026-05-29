# -*- coding: utf-8 -*-

import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_FORCED_TOOL_PATTERN = re.compile(r"(?:^|\s)tool:([^\s]+)", re.IGNORECASE)
_TOOL_NAME_ALIAS_MAP = {
    "tts": "xiaomi_tts_tool",
    "pixiv": "pixiv_tool",
    "塔罗": "tarot_reading",
    "otto": "otto_tool",
    "tts1": "new_tts_tool",
    "总结": "summarize_channel",
    "搜索": "search_web",
}


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


def extract_forced_tool_name(
    user_message: str,
    available_tool_names: List[str],
) -> Optional[str]:
    normalized_message = str(user_message or "").strip()
    if not normalized_message or not available_tool_names:
        return None

    match = _FORCED_TOOL_PATTERN.search(normalized_message)
    if not match:
        return None

    requested_tool_name = match.group(1).strip()
    if not requested_tool_name:
        return None

    lowered_requested_tool_name = requested_tool_name.lower()
    resolved_tool_name = _TOOL_NAME_ALIAS_MAP.get(
        lowered_requested_tool_name,
        requested_tool_name,
    )
    available_tool_name_set = set(available_tool_names)

    if resolved_tool_name in available_tool_name_set:
        return resolved_tool_name

    log.info(
        "检测到显式工具指定，但目标工具当前不可用 | requested=%s | resolved=%s | message=%s",
        requested_tool_name,
        resolved_tool_name,
        normalized_message[:200],
    )
    return None


def resolve_proactive_tool_choice(
    user_message: str,
    available_tool_names: List[str],
) -> Optional[Dict[str, Any]]:
    forced_tool_name = extract_forced_tool_name(user_message, available_tool_names)
    if not forced_tool_name:
        return None

    return {
        "type": "function",
        "function": {"name": forced_tool_name},
    }
