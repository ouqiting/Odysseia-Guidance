# -*- coding: utf-8 -*-

import random
from typing import Any

from .config import PixivConfig
from .image_sender import build_file_name, extract_best_image_url
from .models import PixivImageResult, PixivToolResult
from .storage import PixivStorage
from .tag_utils import (
    build_detail_caption,
    extract_tag_names,
    filter_illusts,
    normalize_excluded_tags,
    normalize_tag_inputs,
)

class PixivService:
    def __init__(self, client_wrapper: Any, config: PixivConfig, storage: PixivStorage):
        self.client_wrapper = client_wrapper
        self.client = client_wrapper.client_api
        self.config = config
        self.storage = storage

    async def random_by_tag(
        self,
        *,
        tags=None,
        exclude_tags=None,
        mode: str | None = None,
        allow_ai: bool | None = None,
        count: int = 1,
    ) -> PixivToolResult:
        normalized_count, validation = self._normalize_count(count)
        if validation is not None:
            return validation

        auth_error = await self._ensure_authenticated()
        if auth_error:
            return auth_error

        has_tags = bool(tags)
        if has_tags:
            normalized_or_error = self._normalize_query(tags, exclude_tags)
            if isinstance(normalized_or_error, PixivToolResult):
                return normalized_or_error

            results = await self._search_illusts(
                normalized_or_error.search_text,
                pages=3,
                sort="popular_desc",
            )
            filtered = filter_illusts(
                results,
                mode=self._resolve_mode(mode),
                allow_ai=self._resolve_allow_ai(allow_ai),
                excluded_tags=self._merge_excluded_tags(normalized_or_error.exclude_tags),
            )
            selected = await self._pick_random_many(filtered, normalized_count)
            if not selected:
                return PixivToolResult(False, "未找到符合条件的作品。", error="no_results")

            return self._success_result(
                selected,
                f"已按标签随机发送 {len(selected)} 张 Pixiv 插画。",
            )

        mode_value = self._resolve_mode(mode)
        if mode_value == "r18":
            ranking_result = await self.client_wrapper.call_pixiv_api(
                self.client.illust_ranking,
                mode="day_r18",
            )
            results = list(getattr(ranking_result, "illusts", []) or [])
        else:
            recommended_result = await self.client_wrapper.call_pixiv_api(
                self.client.illust_recommended
            )
            results = list(getattr(recommended_result, "illusts", []) or [])

        excluded = []
        if exclude_tags:
            excluded = normalize_excluded_tags(exclude_tags)

        filtered = filter_illusts(
            results,
            mode=mode_value,
            allow_ai=self._resolve_allow_ai(allow_ai),
            excluded_tags=self._merge_excluded_tags(excluded),
        )
        selected = await self._pick_random_many(filtered, normalized_count)
        if not selected:
            return PixivToolResult(False, "未找到符合条件的作品。", error="no_results")

        return self._success_result(
            selected,
            f"已随机发送 {len(selected)} 张 Pixiv 插画。",
        )

    async def random_ranking(
        self,
        *,
        mode: str | None = None,
        allow_ai: bool | None = None,
        exclude_tags=None,
        count: int = 1,
    ) -> PixivToolResult:
        normalized_count, validation = self._normalize_count(count)
        if validation is not None:
            return validation

        auth_error = await self._ensure_authenticated()
        if auth_error:
            return auth_error

        mode_value = self._resolve_mode(mode)
        ranking_result = await self.client_wrapper.call_pixiv_api(
            self.client.illust_ranking,
            mode="day_r18" if mode_value == "r18" else "day",
        )
        results = list(getattr(ranking_result, "illusts", []) or [])

        excluded = []
        if exclude_tags:
            excluded = normalize_excluded_tags(exclude_tags)

        filtered = filter_illusts(
            results,
            mode=mode_value,
            allow_ai=self._resolve_allow_ai(allow_ai),
            excluded_tags=self._merge_excluded_tags(excluded),
        )
        selected = await self._pick_random_many(filtered, normalized_count)
        if not selected:
            return PixivToolResult(False, "未找到符合条件的作品。", error="no_results")

        return self._success_result(
            selected,
            f"已从排行榜随机发送 {len(selected)} 张 Pixiv 插画。",
        )

    async def mark_sent(self, illust_id: int) -> None:
        await self.storage.mark_sent(illust_id)

    async def _ensure_authenticated(self) -> PixivToolResult | None:
        if not self.config.refresh_token:
            return PixivToolResult(
                False,
                self.config.get_missing_token_message(),
                error="missing_refresh_token",
            )

        if not await self.client_wrapper.authenticate():
            return PixivToolResult(
                False,
                self.config.get_auth_error_message(),
                error="auth_failed",
            )

        return None

    def _normalize_query(self, tags, exclude_tags):
        try:
            return normalize_tag_inputs(tags, exclude_tags)
        except ValueError as exc:
            return PixivToolResult(False, str(exc), error="invalid_tags")

    def _resolve_mode(self, mode: str | None) -> str:
        normalized = str(mode or self.config.default_mode or "safe").strip().lower()
        if normalized not in {"safe", "r18"}:
            normalized = self.config.default_mode
        return normalized

    def _resolve_allow_ai(self, allow_ai: bool | None) -> bool:
        return self.config.allow_ai_default if allow_ai is None else bool(allow_ai)

    def _merge_excluded_tags(self, extra_excluded_tags: list[str] | None) -> list[str]:
        merged = list(self.config.default_excluded_tags or [])
        if extra_excluded_tags:
            merged.extend(tag.lower() for tag in extra_excluded_tags if str(tag).strip())
        return list(dict.fromkeys(merged))

    def _normalize_count(self, count: int) -> tuple[int, PixivToolResult | None]:
        try:
            parsed = int(count)
        except Exception:
            parsed = 1
        if parsed < 1 or parsed > 5:
            return 1, PixivToolResult(
                False,
                "Pixiv 工具当前支持一次发送 1 到 5 张图。",
                error="invalid_count",
            )
        return parsed, None

    async def _search_illusts(self, query: str, *, pages: int, sort: str) -> list[Any]:
        all_illusts: list[Any] = []
        next_params: dict[str, Any] | None = {
            "word": query,
            "search_target": "partial_match_for_tags",
            "sort": sort,
            "filter": "for_ios",
        }

        for _ in range(max(1, pages)):
            if not next_params:
                break
            response = await self.client_wrapper.call_pixiv_api(
                self.client.search_illust,
                **next_params,
            )
            illusts = list(getattr(response, "illusts", []) or [])
            all_illusts.extend(illusts)
            next_url = getattr(response, "next_url", None)
            if next_url:
                next_params = self.client.parse_qs(next_url)
            else:
                next_params = None

        return all_illusts

    async def _pick_random_many(self, filtered: list[Any], count: int) -> list[Any]:
        if not filtered:
            return []

        await self.storage.prune_old_entries()
        recent_ids = await self.storage.get_recent_sent_ids(limit=100)
        unseen = [
            item
            for item in filtered
            if int(getattr(item, "id", 0) or 0) not in recent_ids
        ]
        pool = unseen or filtered
        if not pool:
            return []

        sample_size = min(len(pool), max(1, count))
        if sample_size >= len(pool):
            shuffled = list(pool)
            random.shuffle(shuffled)
            return shuffled[:sample_size]
        return random.sample(pool, sample_size)

    def _success_result(self, illusts: list[Any], message: str) -> PixivToolResult:
        images: list[PixivImageResult] = []
        for illust in illusts:
            illust_id = int(getattr(illust, "id"))
            title = str(getattr(illust, "title", "未命名作品"))
            user = getattr(illust, "user", None)
            author = (
                str(getattr(user, "name", "未知作者"))
                if user is not None
                else "未知作者"
            )
            image_url = extract_best_image_url(illust)
            images.append(
                PixivImageResult(
                    illust_id=illust_id,
                    title=title,
                    author=author,
                    caption=build_detail_caption(illust),
                    image_url=image_url,
                    file_name=build_file_name(title, illust_id, image_url),
                    tags=extract_tag_names(getattr(illust, "tags", [])),
                )
            )

        return PixivToolResult(True, message, images=images)
