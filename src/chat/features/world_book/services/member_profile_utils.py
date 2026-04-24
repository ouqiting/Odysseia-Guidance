# -*- coding: utf-8 -*-

import ast
import json
import logging
from typing import Any, Dict, Mapping

log = logging.getLogger(__name__)


def coerce_source_metadata(raw_source_metadata: Any) -> Dict[str, Any]:
    if isinstance(raw_source_metadata, dict):
        return dict(raw_source_metadata)

    if isinstance(raw_source_metadata, str) and raw_source_metadata.strip():
        try:
            parsed = json.loads(raw_source_metadata)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            parsed = ast.literal_eval(raw_source_metadata)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError):
            pass

    return {}


def parse_member_profile(raw_data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    从原始数据库行中解析出成员档案核心字段。
    """
    full_text = raw_data.get("full_text", "")
    source_metadata = coerce_source_metadata(raw_data.get("source_metadata"))

    if full_text:
        if "名称:" in full_text and "Discord ID:" in full_text:
            lines = full_text.strip().split("\n")
            temp_data: Dict[str, Any] = {}
            for line in lines:
                if ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                field_name = {
                    "名称": "name",
                    "Discord ID": "discord_id",
                    "性格特点": "personality",
                    "背景信息": "background",
                    "喜好偏好": "preferences",
                }.get(key.strip())
                if field_name:
                    temp_data[field_name] = value.strip()
            if temp_data.get("name"):
                return temp_data

        try:
            cleaned_full_text = full_text.strip()
            if cleaned_full_text.startswith("{"):
                data = json.loads(cleaned_full_text)
                if isinstance(data, dict) and data.get("name"):
                    return data
        except (json.JSONDecodeError, TypeError):
            pass

    if source_metadata:
        try:
            content_json = source_metadata.get("content_json")
            if content_json:
                parsed_content = (
                    json.loads(content_json)
                    if isinstance(content_json, str)
                    else content_json
                )
                if isinstance(parsed_content, dict) and parsed_content.get("name"):
                    return parsed_content

            if source_metadata.get("name"):
                return source_metadata
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("无法从 source_metadata 解析档案数据: %s", exc)

    return {
        "name": raw_data.get("title", ""),
        "discord_id": raw_data.get("discord_id", ""),
        "personality": "",
        "background": "",
        "preferences": "",
    }


def build_member_profile_storage(
    *,
    name: str,
    discord_id: Any,
    personality: str = "",
    background: str = "",
    preferences: str = "",
    extra_source_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    将成员档案核心字段构建为统一的 full_text / source_metadata 存储格式。
    """
    clean_name = str(name or "").strip()
    clean_discord_id = str(discord_id or "").strip()
    clean_personality = str(personality or "").strip()
    clean_background = str(background or "").strip()
    clean_preferences = str(preferences or "").strip()

    full_text = (
        f"名称: {clean_name}\n"
        f"Discord ID: {clean_discord_id}\n"
        f"性格特点: {clean_personality}\n"
        f"背景信息: {clean_background}\n"
        f"喜好偏好: {clean_preferences}"
    ).strip()

    source_metadata = dict(extra_source_metadata or {})
    source_metadata.update(
        {
            "name": clean_name,
            "discord_id": clean_discord_id,
            "personality": clean_personality,
            "background": clean_background,
            "preferences": clean_preferences,
        }
    )

    return {
        "full_text": full_text,
        "source_metadata": source_metadata,
    }


def profile_details_are_empty(raw_data: Mapping[str, Any]) -> bool:
    """
    判断 personality/background/preferences 是否都为空。

    兼容旧的 auto_create 占位数据：
    - source=auto_create
    - personality 只是默认复制了用户名/title
    - background/preferences 为空
    这种情况仍视为“未填写”，允许后续自动补全。
    """
    parsed = parse_member_profile(raw_data)
    metadata = coerce_source_metadata(raw_data.get("source_metadata"))

    name = str(parsed.get("name", "") or "").strip()
    title = str(raw_data.get("title", "") or "").strip()
    personality = str(parsed.get("personality", "") or "").strip()
    background = str(parsed.get("background", "") or "").strip()
    preferences = str(parsed.get("preferences", "") or "").strip()

    if (
        metadata.get("source") == "auto_create"
        and personality
        and not background
        and not preferences
        and personality in {name, title}
    ):
        personality = ""

    return not any([personality, background, preferences])
