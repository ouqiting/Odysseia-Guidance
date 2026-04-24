# -*- coding: utf-8 -*-

from typing import Mapping, Any, Dict

from src.chat.features.world_book.services.member_profile_utils import (
    build_member_profile_storage,
    parse_member_profile,
)


def _parse_raw_profile_data(raw_data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    向后兼容的内部别名，避免改动所有调用点。
    """
    return parse_member_profile(raw_data)


def format_member_profile(raw_data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    接收一个原始的数据库行，返回一个包含格式化好的
    full_text 和 source_metadata 的字典。
    这个逻辑是从 EditCommunityMemberModal.on_submit 中提取的。
    """
    parsed_data = _parse_raw_profile_data(raw_data)
    return build_member_profile_storage(
        name=parsed_data.get("name", ""),
        discord_id=parsed_data.get("discord_id", ""),
        personality=parsed_data.get("personality", ""),
        background=parsed_data.get("background", ""),
        preferences=parsed_data.get("preferences", ""),
    )
