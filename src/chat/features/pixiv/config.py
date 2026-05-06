# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(slots=True)
class PixivConfig:
    refresh_token: str
    proxy: str
    api_proxy_host: str
    image_proxy_host: str
    use_image_proxy: bool
    default_mode: str
    allow_ai_default: bool
    default_excluded_tags: list[str]
    refresh_interval_minutes: int
    random_dedupe_days: int

    @classmethod
    def from_env(cls) -> "PixivConfig":
        default_mode = str(
            os.getenv("PIXIV_R18_DEFAULT_MODE", "safe") or "safe"
        ).strip().lower()
        if default_mode not in {"safe", "r18"}:
            default_mode = "safe"

        refresh_interval_minutes = cls._safe_int(
            os.getenv("PIXIV_REFRESH_INTERVAL_MINUTES"),
            180,
            minimum=0,
        )
        random_dedupe_days = cls._safe_int(
            os.getenv("PIXIV_RANDOM_DEDUPE_DAYS"),
            7,
            minimum=1,
        )

        return cls(
            refresh_token=str(os.getenv("PIXIV_REFRESH_TOKEN", "") or "").strip(),
            proxy=str(os.getenv("PIXIV_PROXY", "") or "").strip(),
            api_proxy_host=str(os.getenv("PIXIV_API_PROXY_HOST", "") or "").strip(),
            image_proxy_host=str(
                os.getenv("PIXIV_IMAGE_PROXY_HOST", "i.pixiv.re") or "i.pixiv.re"
            ).strip(),
            use_image_proxy=_parse_bool(os.getenv("PIXIV_USE_IMAGE_PROXY"), True),
            default_mode=default_mode,
            allow_ai_default=_parse_bool(os.getenv("PIXIV_ALLOW_AI_DEFAULT"), False),
            default_excluded_tags=cls._parse_tag_list(
                os.getenv("PIXIV_DEFAULT_EXCLUDED_TAGS", "")
            ),
            refresh_interval_minutes=refresh_interval_minutes,
            random_dedupe_days=random_dedupe_days,
        )

    @staticmethod
    def _safe_int(raw: str | None, default: int, minimum: int = 0) -> int:
        try:
            parsed = int(str(raw).strip())
        except Exception:
            parsed = default
        return max(minimum, parsed)

    @staticmethod
    def _parse_tag_list(raw: str | None) -> list[str]:
        if not raw:
            return []
        normalized = str(raw).replace("\n", ",")
        tags: list[str] = []
        for part in normalized.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            if cleaned.startswith("-"):
                cleaned = cleaned[1:].strip()
            if cleaned:
                tags.append(cleaned.lower())
        return list(dict.fromkeys(tags))

    def get_requests_kwargs(self) -> dict:
        if not self.proxy:
            return {}
        return {"proxies": {"http": self.proxy, "https": self.proxy}}

    def get_missing_token_message(self) -> str:
        return "Pixiv 未配置 `PIXIV_REFRESH_TOKEN`，暂时无法使用搜图功能。"

    def get_auth_error_message(self) -> str:
        return (
            "Pixiv API 认证失败，请检查 `PIXIV_REFRESH_TOKEN` 是否有效，"
            "以及 `PIXIV_PROXY` / `PIXIV_API_PROXY_HOST` 是否配置正确。"
        )
