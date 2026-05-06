from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.chat.services import context_service_test as context_module


class FakeGuild:
    def get_member(self, user_id: int):
        return None


class FakeBot:
    def __init__(self, channels=None, cached_messages=None):
        self._channels = {channel.id: channel for channel in channels or []}
        self.cached_messages = list(cached_messages or [])

    def get_channel(self, channel_id):
        return self._channels.get(channel_id)

    async def fetch_channel(self, channel_id):
        return self._channels.get(channel_id)


class FakeChannel:
    def __init__(self, channel_id=123, history_messages=None, fetched_messages=None):
        self.id = channel_id
        self.name = f"channel-{channel_id}"
        self._history_messages = list(history_messages or [])
        self._fetched_messages = dict(fetched_messages or {})
        self.fetch_message = AsyncMock(side_effect=self._fetch_message)

    async def _fetch_message(self, message_id):
        result = self._fetched_messages.get(message_id)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise RuntimeError(f"missing message {message_id}")
        return result

    async def history(self, limit=None, before=None):
        messages = list(self._history_messages)
        if before is not None:
            before_id = getattr(before, "id", before)
            messages = [message for message in messages if message.id < before_id]

        messages.sort(key=lambda message: message.id, reverse=True)
        if limit is not None:
            messages = messages[:limit]

        for message in messages:
            yield message


def build_message(
    message_id: int,
    channel: FakeChannel,
    *,
    author_name: str,
    content: str,
    created_at: datetime,
    is_bot: bool = False,
    reply_to_message_id: int | None = None,
    resolved_message=None,
):
    reference = None
    if reply_to_message_id is not None:
        reference = SimpleNamespace(
            message_id=reply_to_message_id,
            resolved=resolved_message,
            cached_message=resolved_message,
        )

    return SimpleNamespace(
        id=message_id,
        author=SimpleNamespace(display_name=author_name, bot=is_bot),
        content=content,
        created_at=created_at,
        reference=reference,
        attachments=[],
        channel=channel,
        guild=FakeGuild(),
        type=(
            context_module.discord.MessageType.reply
            if reply_to_message_id is not None
            else context_module.discord.MessageType.default
        ),
        flags=None,
        ephemeral=False,
    )


@pytest.fixture
def configured_context_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(context_module.config, "DATA_DIR", str(data_dir))
    monkeypatch.setitem(
        sys.modules,
        "src.chat.features.chat_settings.services.chat_settings_service",
        SimpleNamespace(
            chat_settings_service=SimpleNamespace(
                is_active_chat_cache_enabled=AsyncMock(return_value=True)
            )
        ),
    )
    return data_dir


def set_active_cache_enabled(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setitem(
        sys.modules,
        "src.chat.features.chat_settings.services.chat_settings_service",
        SimpleNamespace(
            chat_settings_service=SimpleNamespace(
                is_active_chat_cache_enabled=AsyncMock(return_value=enabled)
            )
        ),
    )


def extract_history_text(context: list[dict]) -> str:
    assert context
    assert context[0]["role"] == "user"
    return context[0]["parts"][0]


@pytest.mark.asyncio
async def test_flush_pending_messages_persists_small_active_cache_batch(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=101)
    bot = FakeBot(channels=[channel])
    service = context_module.ContextServiceTest(bot)
    now = datetime.now(timezone.utc)

    first = build_message(
        1001,
        channel,
        author_name="Alice",
        content="第一条消息",
        created_at=now,
    )
    second = build_message(
        1002,
        channel,
        author_name="Bot",
        content="第二条消息",
        created_at=now + timedelta(seconds=1),
        is_bot=True,
    )

    await service.record_message_for_active_cache(first)
    await service.record_message_for_active_cache(second)
    await service.flush_pending_active_messages()

    rebuilt_service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    context = await rebuilt_service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)

    assert "第一条消息" in history_text
    assert "第二条消息" in history_text
    assert (
        configured_context_module / "active_chat_cache" / f"active_cache_{channel.id}.json"
    ).exists()


@pytest.mark.asyncio
async def test_active_cache_writes_each_message_immediately(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=111)
    bot = FakeBot(channels=[channel])
    service = context_module.ContextServiceTest(bot)
    message = build_message(
        1101,
        channel,
        author_name="Alice",
        content="实时落盘消息",
        created_at=datetime.now(timezone.utc),
    )

    await service.record_message_for_active_cache(message)
    task = service.active_flush_tasks.get(channel.id)
    if task is not None:
        await task

    cache_file = (
        configured_context_module
        / "active_chat_cache"
        / f"active_cache_{channel.id}.json"
    )
    saved_entries = context_module.json.loads(cache_file.read_text(encoding="utf-8"))

    assert len(saved_entries) == 1
    assert saved_entries[0]["content"] == "实时落盘消息"


@pytest.mark.asyncio
async def test_clear_high_priority_triggers_active_cache_flush(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=202)
    bot = FakeBot(channels=[channel])
    service = context_module.ContextServiceTest(bot)
    now = datetime.now(timezone.utc)
    message = build_message(
        2001,
        channel,
        author_name="Alice",
        content="高优先级缓存消息",
        created_at=now,
    )

    service.mark_channel_high_priority(channel.id)
    await service.record_message_for_active_cache(message)

    cache_file = (
        configured_context_module
        / "active_chat_cache"
        / f"active_cache_{channel.id}.json"
    )
    assert not cache_file.exists()

    service.clear_channel_high_priority(channel.id)
    await asyncio.sleep(0)
    task = service.active_flush_tasks.get(channel.id)
    if task is not None:
        await task

    rebuilt_service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    context = await rebuilt_service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)

    assert cache_file.exists()
    assert "高优先级缓存消息" in history_text


@pytest.mark.asyncio
async def test_missing_reply_cache_only_drops_reply_hint_not_history_body(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=303, fetched_messages={9999: RuntimeError("missing")})
    bot = FakeBot(channels=[channel])
    service = context_module.ContextServiceTest(bot)
    now = datetime.now(timezone.utc)

    first = build_message(
        3001,
        channel,
        author_name="Alice",
        content="原始消息",
        created_at=now,
    )
    second = build_message(
        3002,
        channel,
        author_name="Bob",
        content="回复消息",
        created_at=now + timedelta(seconds=1),
        reply_to_message_id=9999,
    )

    await service.record_message_for_active_cache(first)
    await service.record_message_for_active_cache(second)
    await service.flush_pending_active_messages(channel.id)

    context = await service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)
    history_lines = history_text.replace("这是本频道最近的对话记录:\n\n", "").split("\n\n")

    assert "原始消息" in history_text
    assert "回复消息" in history_text
    assert len(history_lines) == 2
    channel.fetch_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_reply_cache_persists_full_history_snapshot_and_reply_author(
    configured_context_module: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    set_active_cache_enabled(monkeypatch, False)
    channel = FakeChannel(channel_id=313)
    now = datetime.now(timezone.utc)
    first = build_message(
        3101,
        channel,
        author_name="Alice",
        content="第一条",
        created_at=now,
    )
    second = build_message(
        3102,
        channel,
        author_name="Bob",
        content="第二条",
        created_at=now + timedelta(seconds=1),
        reply_to_message_id=3101,
        resolved_message=first,
    )
    channel._history_messages = [first, second]

    service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    await service.get_formatted_channel_history_new(channel.id, user_id=1, guild_id=1)

    cache_file = (
        configured_context_module
        / "reply_message_cache"
        / f"reply_cache_{channel.id}.json"
    )
    saved_entries = context_module.json.loads(cache_file.read_text(encoding="utf-8"))

    assert len(saved_entries) == 2
    assert saved_entries[0]["content"] == "第一条"
    assert saved_entries[1]["content"] == "第二条"
    assert saved_entries[1]["reply_to_author_display_name"] == "Alice"


@pytest.mark.asyncio
async def test_reply_cache_can_restore_history_without_active_cache(
    configured_context_module: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    set_active_cache_enabled(monkeypatch, False)
    channel = FakeChannel(channel_id=323)
    now = datetime.now(timezone.utc)
    first = context_module.CachedReplyMessage(
        message_id=3201,
        author_display_name="Alice",
        content="缓存第一条",
        created_at_ts=now.timestamp(),
    )
    second = context_module.CachedReplyMessage(
        message_id=3202,
        author_display_name="Bob",
        content="缓存第二条",
        created_at_ts=(now + timedelta(seconds=1)).timestamp(),
        reply_to_message_id=3201,
        reply_to_author_display_name="Alice",
    )
    reply_cache_dir = configured_context_module / "reply_message_cache"
    reply_cache_dir.mkdir(exist_ok=True)
    cache_file = reply_cache_dir / f"reply_cache_{channel.id}.json"
    cache_file.write_text(
        context_module.json.dumps(
            [first.to_dict(), second.to_dict()], ensure_ascii=False
        ),
        encoding="utf-8",
    )

    service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    context = await service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)

    assert "缓存第一条" in history_text
    assert "缓存第二条" in history_text
    assert "[Bob][回复 Alice]: 缓存第二条" in history_text


@pytest.mark.asyncio
async def test_legacy_reply_cache_is_ignored_and_rewritten(
    configured_context_module: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    set_active_cache_enabled(monkeypatch, False)
    channel = FakeChannel(channel_id=333)
    now = datetime.now(timezone.utc)
    first = build_message(
        3301,
        channel,
        author_name="Alice",
        content="新格式第一条",
        created_at=now,
    )
    second = build_message(
        3302,
        channel,
        author_name="Bob",
        content="新格式第二条",
        created_at=now + timedelta(seconds=1),
    )
    channel._history_messages = [first, second]

    reply_cache_dir = configured_context_module / "reply_message_cache"
    reply_cache_dir.mkdir(exist_ok=True)
    cache_file = reply_cache_dir / f"reply_cache_{channel.id}.json"
    cache_file.write_text(
        context_module.json.dumps(
            [
                {"message_id": 1, "author_display_name": "旧用户A"},
                {"message_id": 2, "author_display_name": "旧用户B"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    context = await service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)
    saved_entries = context_module.json.loads(cache_file.read_text(encoding="utf-8"))

    assert "新格式第一条" in history_text
    assert "新格式第二条" in history_text
    assert all("content" in entry for entry in saved_entries)
    assert {entry["message_id"] for entry in saved_entries} == {3301, 3302}


@pytest.mark.asyncio
async def test_delete_messages_from_reply_and_active_caches(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=414)
    now = datetime.now(timezone.utc)

    reply_cache_dir = configured_context_module / "reply_message_cache"
    reply_cache_dir.mkdir(exist_ok=True)
    reply_cache_file = reply_cache_dir / f"reply_cache_{channel.id}.json"
    reply_cache_file.write_text(
        context_module.json.dumps(
            [
                context_module.CachedReplyMessage(
                    message_id=4101,
                    author_display_name="Alice",
                    content="reply-1",
                    created_at_ts=now.timestamp(),
                ).to_dict(),
                context_module.CachedReplyMessage(
                    message_id=4102,
                    author_display_name="Bob",
                    content="reply-2",
                    created_at_ts=(now + timedelta(seconds=1)).timestamp(),
                ).to_dict(),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    active_cache_dir = configured_context_module / "active_chat_cache"
    active_cache_dir.mkdir(exist_ok=True)
    active_cache_file = active_cache_dir / f"active_cache_{channel.id}.json"
    active_cache_file.write_text(
        context_module.json.dumps(
            [
                context_module.ActiveCachedMessage(
                    message_id=4101,
                    author_display_name="Alice",
                    content="active-1",
                    created_at_ts=now.timestamp(),
                ).to_dict(),
                context_module.ActiveCachedMessage(
                    message_id=4103,
                    author_display_name="Carol",
                    content="active-3",
                    created_at_ts=(now + timedelta(seconds=2)).timestamp(),
                ).to_dict(),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    removed = await service.delete_messages_from_caches(channel.id, [4101])

    saved_reply_entries = context_module.json.loads(
        reply_cache_file.read_text(encoding="utf-8")
    )
    saved_active_entries = context_module.json.loads(
        active_cache_file.read_text(encoding="utf-8")
    )

    assert removed is True
    assert {entry["message_id"] for entry in saved_reply_entries} == {4102}
    assert {entry["message_id"] for entry in saved_active_entries} == {4103}


@pytest.mark.asyncio
async def test_delete_messages_from_caches_removes_pending_active_buffer(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=424)
    bot = FakeBot(channels=[channel])
    service = context_module.ContextServiceTest(bot)
    now = datetime.now(timezone.utc)

    first = build_message(
        4201,
        channel,
        author_name="Alice",
        content="待删除消息",
        created_at=now,
    )
    second = build_message(
        4202,
        channel,
        author_name="Bob",
        content="保留消息",
        created_at=now + timedelta(seconds=1),
    )

    service.mark_channel_high_priority(channel.id)
    await service.record_message_for_active_cache(first)
    await service.record_message_for_active_cache(second)

    removed = await service.delete_messages_from_caches(channel.id, [4201])
    pending_buffer = service._get_pending_active_buffer(channel.id)

    assert removed is True
    assert 4201 not in pending_buffer
    assert 4202 in pending_buffer

    service.clear_channel_high_priority(channel.id)
    await asyncio.sleep(0)
    task = service.active_flush_tasks.get(channel.id)
    if task is not None:
        await task

    cache_file = (
        configured_context_module
        / "active_chat_cache"
        / f"active_cache_{channel.id}.json"
    )
    saved_entries = context_module.json.loads(cache_file.read_text(encoding="utf-8"))

    assert {entry["message_id"] for entry in saved_entries} == {4202}


@pytest.mark.asyncio
async def test_persisted_active_cache_and_live_cache_are_deduplicated(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=404)
    persisted_entry = context_module.ActiveCachedMessage(
        message_id=4001,
        author_display_name="Alice",
        content="旧内容",
        created_at_ts=100.0,
    )
    active_cache_dir = configured_context_module / "active_chat_cache"
    active_cache_dir.mkdir(exist_ok=True)
    cache_file = active_cache_dir / f"active_cache_{channel.id}.json"
    cache_file.write_text(
        context_module.json.dumps([persisted_entry.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )

    live_message = build_message(
        4001,
        channel,
        author_name="Alice",
        content="新内容",
        created_at=datetime.now(timezone.utc),
    )
    service = context_module.ContextServiceTest(
        FakeBot(channels=[channel], cached_messages=[live_message])
    )

    context = await service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)

    assert "新内容" in history_text
    assert "旧内容" not in history_text
    assert history_text.count("新内容") == 1


@pytest.mark.asyncio
async def test_reply_cache_persists_reference_targets_to_avoid_repeat_fetch(
    configured_context_module: Path,
):
    channel = FakeChannel(channel_id=434)
    now = datetime.now(timezone.utc)
    referenced_message = build_message(
        4300,
        channel,
        author_name="Alice",
        content="被引用的旧消息",
        created_at=now - timedelta(minutes=10),
    )
    reply_message = build_message(
        4301,
        channel,
        author_name="Bob",
        content="引用旧消息的新回复",
        created_at=now,
        reply_to_message_id=4300,
    )
    channel._fetched_messages = {4300: referenced_message}

    service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    await service.record_message_for_active_cache(reply_message)
    await service.flush_pending_active_messages(channel.id)

    await service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    channel.fetch_message.assert_awaited_once_with(4300)

    reply_cache_file = (
        configured_context_module
        / "reply_message_cache"
        / f"reply_cache_{channel.id}.json"
    )
    saved_entries = context_module.json.loads(
        reply_cache_file.read_text(encoding="utf-8")
    )
    assert {entry["message_id"] for entry in saved_entries} == {4300, 4301}
    assert next(entry for entry in saved_entries if entry["message_id"] == 4300)[
        "is_history_message"
    ] is False

    channel.fetch_message.reset_mock()

    rebuilt_service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    await rebuilt_service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )

    channel.fetch_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_cache_restore_history_ignores_reference_only_entries(
    configured_context_module: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    set_active_cache_enabled(monkeypatch, False)
    channel = FakeChannel(channel_id=444)
    now = datetime.now(timezone.utc)
    history_entry = context_module.CachedReplyMessage(
        message_id=4401,
        author_display_name="Bob",
        content="最近消息",
        created_at_ts=now.timestamp(),
        reply_to_message_id=4400,
        reply_to_author_display_name="Alice",
    )
    reference_only_entry = context_module.CachedReplyMessage(
        message_id=4400,
        author_display_name="Alice",
        content="较早的被引用消息",
        created_at_ts=(now - timedelta(minutes=5)).timestamp(),
        is_history_message=False,
    )

    reply_cache_dir = configured_context_module / "reply_message_cache"
    reply_cache_dir.mkdir(exist_ok=True)
    cache_file = reply_cache_dir / f"reply_cache_{channel.id}.json"
    cache_file.write_text(
        context_module.json.dumps(
            [reference_only_entry.to_dict(), history_entry.to_dict()],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = context_module.ContextServiceTest(FakeBot(channels=[channel]))
    context = await service.get_formatted_channel_history_new(
        channel.id, user_id=1, guild_id=1
    )
    history_text = extract_history_text(context)

    assert "最近消息" in history_text
    assert "[Bob][回复 Alice]: 最近消息" in history_text
    assert "较早的被引用消息" not in history_text
