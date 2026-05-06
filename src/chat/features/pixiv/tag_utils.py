# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any


R18_BADWORDS = {s.lower() for s in ["R-18", "R18", "R-18G", "R18G", "R18+", "R18+G"]}
AI_BADWORDS = {s.lower() for s in ["AI", "AI生成", "AI-generated", "AI辅助"]}


@dataclass(slots=True)
class NormalizedTagQuery:
    include_tags: list[str]
    exclude_tags: list[str]
    search_text: str


def _split_tag_string(raw: str) -> list[str]:
    chunks = []
    normalized = str(raw or "").replace("\n", ",")
    for part in normalized.split(","):
        cleaned = part.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


def normalize_tag_inputs(
    tags: list[str] | str | None,
    exclude_tags: list[str] | str | None = None,
) -> NormalizedTagQuery:
    include_parts: list[str] = []
    exclude_parts: list[str] = []

    raw_parts: list[str] = []
    if isinstance(tags, str):
        raw_parts.extend(_split_tag_string(tags))
    elif isinstance(tags, list):
        for item in tags:
            if isinstance(item, str):
                raw_parts.extend(_split_tag_string(item))

    for item in raw_parts:
        if item.startswith("-"):
            excluded = item[1:].strip().lower()
            if excluded:
                exclude_parts.append(excluded)
        else:
            include_parts.append(item)

    if isinstance(exclude_tags, str):
        exclude_parts.extend(tag.lower() for tag in _split_tag_string(exclude_tags))
    elif isinstance(exclude_tags, list):
        for item in exclude_tags:
            if isinstance(item, str):
                exclude_parts.extend(tag.lower() for tag in _split_tag_string(item))

    deduped_include = list(
        dict.fromkeys(tag.strip() for tag in include_parts if str(tag).strip())
    )
    deduped_exclude = list(
        dict.fromkeys(tag.strip().lower() for tag in exclude_parts if str(tag).strip())
    )

    include_lower = {tag.lower() for tag in deduped_include}
    conflicts = [tag for tag in deduped_exclude if tag in include_lower]
    if conflicts:
        raise ValueError(
            "标签冲突：以下标签同时出现在包含和排除列表中：" + "、".join(conflicts)
        )
    if not deduped_include:
        raise ValueError("请至少提供一个包含标签。")

    return NormalizedTagQuery(
        include_tags=deduped_include,
        exclude_tags=deduped_exclude,
        search_text=" ".join(deduped_include),
    )


def normalize_excluded_tags(exclude_tags: list[str] | str | None) -> list[str]:
    normalized_parts: list[str] = []
    if isinstance(exclude_tags, str):
        candidates = _split_tag_string(exclude_tags)
    elif isinstance(exclude_tags, list):
        candidates = []
        for item in exclude_tags:
            if isinstance(item, str):
                candidates.extend(_split_tag_string(item))
    else:
        candidates = []

    for item in candidates:
        cleaned = item.strip()
        if cleaned.startswith("-"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            normalized_parts.append(cleaned.lower())

    return list(dict.fromkeys(normalized_parts))


def _get_value(source: Any, *keys: str) -> Any:
    if isinstance(source, dict):
        for key in keys:
            if key in source:
                return source.get(key)
        return None

    for key in keys:
        if hasattr(source, key):
            return getattr(source, key)
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def extract_tag_names(tags: Any) -> list[str]:
    if not tags:
        return []
    if not isinstance(tags, (list, tuple, set)):
        tags = [tags]

    names: list[str] = []
    for tag in tags:
        if isinstance(tag, str):
            cleaned = tag.strip()
            if cleaned:
                names.append(cleaned)
            continue
        name = _get_value(tag, "name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def is_r18(item: Any) -> bool:
    x_restrict = _to_int(_get_value(item, "x_restrict", "xRestrict"))
    if x_restrict is not None and x_restrict > 0:
        return True

    for tag_name in extract_tag_names(_get_value(item, "tags")):
        if tag_name.lower() in R18_BADWORDS:
            return True
    return False


def is_ai(item: Any) -> bool:
    ai_type = _to_int(_get_value(item, "illust_ai_type", "illustAiType"))
    if ai_type == 2:
        return True

    for tag_name in extract_tag_names(_get_value(item, "tags")):
        if tag_name.lower() in AI_BADWORDS:
            return True
    return False


def has_excluded_tags(item: Any, excluded_tags: list[str]) -> bool:
    if not excluded_tags:
        return False

    excluded_lower = [tag.lower() for tag in excluded_tags]
    for tag_name in extract_tag_names(_get_value(item, "tags")):
        lowered = tag_name.lower()
        if any(excluded in lowered for excluded in excluded_lower):
            return True
    return False


def filter_illusts(
    illusts: list[Any],
    *,
    mode: str,
    allow_ai: bool,
    excluded_tags: list[str] | None = None,
) -> list[Any]:
    filtered: list[Any] = []
    normalized_mode = str(mode or "safe").strip().lower()
    excluded = excluded_tags or []

    for item in illusts:
        if normalized_mode == "safe" and is_r18(item):
            continue
        if normalized_mode == "r18" and not is_r18(item):
            continue
        if not allow_ai and is_ai(item):
            continue
        if has_excluded_tags(item, excluded):
            continue
        filtered.append(item)

    return filtered


def format_tags(tags: Any) -> str:
    names = extract_tag_names(tags)
    return ", ".join(names[:12]) if names else "无"


def build_detail_caption(item: Any) -> str:
    title = str(_get_value(item, "title") or "未命名作品")
    illust_id = _get_value(item, "id") or "未知"
    user = _get_value(item, "user")
    author = "未知作者"
    if user is not None:
        author = str(_get_value(user, "name") or author)
    tag_text = format_tags(_get_value(item, "tags"))
    return (
        f"标题: {title}\n"
        f"作者: {author}\n"
        f"标签: {tag_text}\n"
        f"链接: https://www.pixiv.net/artworks/{illust_id}"
    )
