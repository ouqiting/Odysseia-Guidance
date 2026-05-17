# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.chat.utils.database import chat_db_manager, get_beijing_today_str

log = logging.getLogger(__name__)

OPENAI_FALLBACK_SECONDARY_MODEL_KEY = "openai_fallback_secondary_model"
OPENAI_FALLBACK_TERTIARY_MODEL_KEY = "openai_fallback_tertiary_model"
SUPPORTED_OPENAI_FALLBACK_MODELS = (
    "deepseek-chat",
    "deepseek-v4-pro",
    "kimi-k2.5",
    "custom",
)


@dataclass(frozen=True)
class OpenAIFallbackState:
    date: str
    order: List[str]
    failed_channels: List[str]

    @property
    def active_order(self) -> List[str]:
        failed_set = set(self.failed_channels)
        return [channel for channel in self.order if channel not in failed_set]


class OpenAIFallbackService:
    """管理 OpenAI 兼容模型的三渠道顺序与按天锁定状态。"""

    def __init__(self) -> None:
        self.db_manager = chat_db_manager
        self._state_by_primary_model: Dict[str, OpenAIFallbackState] = {}

    @staticmethod
    def is_supported_model(model_name: Optional[str]) -> bool:
        return str(model_name or "").strip() in SUPPORTED_OPENAI_FALLBACK_MODELS

    @staticmethod
    def build_channel_order(
        primary_model: Optional[str],
        secondary_model: Optional[str],
        tertiary_model: Optional[str],
    ) -> List[str]:
        if not OpenAIFallbackService.is_supported_model(primary_model):
            return []

        ordered: List[str] = []
        for raw_model in (primary_model, secondary_model, tertiary_model):
            model_name = str(raw_model or "").strip()
            if not model_name or model_name in ordered:
                continue
            if not OpenAIFallbackService.is_supported_model(model_name):
                continue
            ordered.append(model_name)
        return ordered

    @staticmethod
    def _build_default_state(order: Sequence[str], today_str: str) -> OpenAIFallbackState:
        return OpenAIFallbackState(
            date=today_str,
            order=list(order),
            failed_channels=[],
        )

    @staticmethod
    def _get_today_str() -> str:
        return get_beijing_today_str()

    @staticmethod
    def _normalize_failed_channels(
        failed_channels: Sequence[Any], order: Sequence[str]
    ) -> List[str]:
        order_set = set(order)
        normalized: List[str] = []
        for item in failed_channels:
            channel_name = str(item or "").strip()
            if not channel_name or channel_name not in order_set:
                continue
            if channel_name in normalized:
                continue
            normalized.append(channel_name)
        return normalized

    @staticmethod
    def _build_state_cache_key(primary_model: Optional[str]) -> str:
        return str(primary_model or "").strip()

    async def get_configured_channel_order(
        self, primary_model: Optional[str]
    ) -> List[str]:
        secondary_model = await self.db_manager.get_global_setting(
            OPENAI_FALLBACK_SECONDARY_MODEL_KEY
        )
        tertiary_model = await self.db_manager.get_global_setting(
            OPENAI_FALLBACK_TERTIARY_MODEL_KEY
        )
        return self.build_channel_order(primary_model, secondary_model, tertiary_model)

    async def get_daily_state(self, primary_model: Optional[str]) -> OpenAIFallbackState:
        order = await self.get_configured_channel_order(primary_model)
        today_str = self._get_today_str()
        if not order:
            return self._build_default_state([], today_str)

        cache_key = self._build_state_cache_key(primary_model)
        cached_state = self._state_by_primary_model.get(cache_key)
        if cached_state is None:
            default_state = self._build_default_state(order, today_str)
            self._state_by_primary_model[cache_key] = default_state
            return default_state

        if cached_state.date != today_str or cached_state.order != list(order):
            reset_state = self._build_default_state(order, today_str)
            self._state_by_primary_model[cache_key] = reset_state
            return reset_state

        normalized_state = OpenAIFallbackState(
            date=cached_state.date,
            order=list(order),
            failed_channels=self._normalize_failed_channels(
                cached_state.failed_channels,
                order,
            ),
        )
        self._state_by_primary_model[cache_key] = normalized_state
        return normalized_state

    async def save_state(
        self, state: OpenAIFallbackState, *, primary_model: Optional[str]
    ) -> None:
        cache_key = self._build_state_cache_key(primary_model)
        self._state_by_primary_model[cache_key] = OpenAIFallbackState(
            date=state.date,
            order=list(state.order),
            failed_channels=list(state.failed_channels),
        )

    async def mark_channel_failed(
        self, *, primary_model: Optional[str], channel_name: str
    ) -> OpenAIFallbackState:
        state = await self.get_daily_state(primary_model)
        normalized_channel_name = str(channel_name or "").strip()
        if not normalized_channel_name or normalized_channel_name not in state.order:
            return state

        if normalized_channel_name in state.failed_channels:
            return state

        updated_state = OpenAIFallbackState(
            date=state.date,
            order=list(state.order),
            failed_channels=[*state.failed_channels, normalized_channel_name],
        )
        await self.save_state(updated_state, primary_model=primary_model)
        return updated_state

    async def reset_state_for_current_order(
        self, primary_model: Optional[str]
    ) -> OpenAIFallbackState:
        state = await self.get_daily_state(primary_model)
        reset_state = self._build_default_state(state.order, state.date)
        await self.save_state(reset_state, primary_model=primary_model)
        return reset_state


openai_fallback_service = OpenAIFallbackService()
