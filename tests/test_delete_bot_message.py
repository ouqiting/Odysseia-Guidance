# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import discord
from discord.ext import commands
from discord import app_commands

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src import config
from src.chat.features.admin_panel.cogs.admin_panel_cog import (
    AdminPanelCog,
    parse_discord_message_link,
    is_admin_or_dev,
    check_admin_or_dev,
)


def test_parse_discord_message_link():
    """测试不同格式的 Discord 消息链接解析"""
    # 标准链接
    link1 = "https://discord.com/channels/123456789/987654321/1122334455"
    assert parse_discord_message_link(link1) == (123456789, 987654321, 1122334455)

    # 包含末尾斜杠
    link2 = "https://discord.com/channels/123456789/987654321/1122334455/"
    assert parse_discord_message_link(link2) == (123456789, 987654321, 1122334455)

    # canary 域名
    link3 = "https://canary.discord.com/channels/123456789/987654321/1122334455"
    assert parse_discord_message_link(link3) == (123456789, 987654321, 1122334455)

    # ptb 域名
    link4 = "https://ptb.discord.com/channels/123456789/987654321/1122334455"
    assert parse_discord_message_link(link4) == (123456789, 987654321, 1122334455)

    # discordapp.com 历史域名
    link5 = "https://discordapp.com/channels/123456789/987654321/1122334455"
    assert parse_discord_message_link(link5) == (123456789, 987654321, 1122334455)

    # 私信 @me 链接
    link6 = "https://discord.com/channels/@me/987654321/1122334455"
    assert parse_discord_message_link(link6) == ("@me", 987654321, 1122334455)

    # 带前后空格和 <>
    link7 = "  <https://discord.com/channels/123456789/987654321/1122334455>  \n"
    assert parse_discord_message_link(link7) == (123456789, 987654321, 1122334455)

    # 带 query 参数
    link8 = "https://discord.com/channels/123456789/987654321/1122334455?feature=search"
    assert parse_discord_message_link(link8) == (123456789, 987654321, 1122334455)

    # 无效链接
    assert parse_discord_message_link("") is None
    assert parse_discord_message_link("https://google.com") is None
    assert parse_discord_message_link("https://discord.com/channels/123/456") is None
    assert parse_discord_message_link("https://discord.com/channels/abc/def/ghi") is None
    assert parse_discord_message_link("random text") is None


@pytest.mark.asyncio
async def test_is_admin_or_dev_permissions():
    """测试管理员与开发者权限校验逻辑"""
    # 验证命令上挂载了 check
    bot = MagicMock(spec=commands.Bot)
    cog = AdminPanelCog(bot)
    assert check_admin_or_dev in cog.delete_bot_message.checks

    with patch.object(config, "DEVELOPER_USER_IDS", {1001, 1002}), patch.object(
        config, "ADMIN_ROLE_IDS", {2001, 2002}
    ):
        # 1. 开发者用户（无论角色如何）
        dev_interaction = MagicMock(spec=discord.Interaction)
        dev_interaction.user = MagicMock()
        dev_interaction.user.id = 1001
        dev_interaction.user.roles = []
        assert await check_admin_or_dev(dev_interaction) is True

        # 2. 拥有管理员角色的普通用户
        admin_interaction = MagicMock(spec=discord.Interaction)
        admin_interaction.user = MagicMock()
        admin_interaction.user.id = 9999  # 不是开发者
        role = MagicMock()
        role.id = 2001  # 是管理员角色
        admin_interaction.user.roles = [role]
        assert await check_admin_or_dev(admin_interaction) is True

        # 3. 既不是开发者也没有管理员角色的普通用户
        normal_interaction = MagicMock(spec=discord.Interaction)
        normal_interaction.user = MagicMock()
        normal_interaction.user.id = 9999
        normal_role = MagicMock()
        normal_role.id = 3001
        normal_interaction.user.roles = [normal_role]
        assert await check_admin_or_dev(normal_interaction) is False

        # 4. 私信中（无 roles），非开发者
        dm_normal = MagicMock(spec=discord.Interaction)
        dm_normal.user = MagicMock()
        dm_normal.user.id = 9999
        dm_normal.user.roles = []
        assert await check_admin_or_dev(dm_normal) is False

        # 5. 私信中（无 roles），开发者
        dm_dev = MagicMock(spec=discord.Interaction)
        dm_dev.user = MagicMock()
        dm_dev.user.id = 1002
        dm_dev.user.roles = []
        assert await check_admin_or_dev(dm_dev) is True


@pytest.mark.asyncio
async def test_delete_bot_message_invalid_link():
    """测试传入无效消息链接时的提示"""
    bot = MagicMock(spec=commands.Bot)
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()
    interaction.user.id = 1001

    await cog.delete_bot_message.callback(cog, interaction, "https://invalid-link.com")

    interaction.response.defer.assert_called_once_with(ephemeral=True)
    interaction.followup.send.assert_called_once()
    assert "无效的消息链接格式" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_channel_not_found():
    """测试目标频道不存在的情况"""
    bot = MagicMock(spec=commands.Bot)
    bot.get_channel.return_value = None
    bot.fetch_channel = AsyncMock(
        side_effect=discord.NotFound(MagicMock(), "Channel not found")
    )
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "找不到指定的频道" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_message_not_found():
    """测试目标消息不存在的情况"""
    bot = MagicMock(spec=commands.Bot)
    channel = MagicMock()
    channel.fetch_message = AsyncMock(
        side_effect=discord.NotFound(MagicMock(), "Message not found")
    )
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "找不到指定的消息" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_not_authored_by_bot():
    """测试当目标消息不是 Bot 发送时的拒绝处理"""
    bot = MagicMock(spec=commands.Bot)
    bot.user = MagicMock()
    bot.user.id = 8888  # Bot ID

    channel = MagicMock()
    target_msg = MagicMock()
    target_msg.author = MagicMock()
    target_msg.author.id = 9999  # 消息发送者是其他用户
    target_msg.delete = AsyncMock()

    channel.fetch_message = AsyncMock(return_value=target_msg)
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    target_msg.delete.assert_not_called()
    interaction.followup.send.assert_called_once()
    assert "只能删除由本 Bot 发送的消息" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_success():
    """测试成功删除 Bot 发送的消息"""
    bot = MagicMock(spec=commands.Bot)
    bot.user = MagicMock()
    bot.user.id = 8888  # Bot ID

    channel = MagicMock()
    target_msg = MagicMock()
    target_msg.author = MagicMock()
    target_msg.author.id = 8888  # 确实是 Bot 发送的消息
    target_msg.delete = AsyncMock()

    channel.fetch_message = AsyncMock(return_value=target_msg)
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    target_msg.delete.assert_awaited_once()
    interaction.followup.send.assert_called_once()
    assert "消息删除成功" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_forbidden_on_delete():
    """测试删除时缺少权限（Forbidden）的异常处理"""
    bot = MagicMock(spec=commands.Bot)
    bot.user = MagicMock()
    bot.user.id = 8888

    channel = MagicMock()
    target_msg = MagicMock()
    target_msg.author = MagicMock()
    target_msg.author.id = 8888
    target_msg.delete = AsyncMock(
        side_effect=discord.Forbidden(MagicMock(), "Missing Permissions")
    )

    channel.fetch_message = AsyncMock(return_value=target_msg)
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "Bot 缺少删除该消息的权限" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_on_delete_bot_message_error_check_failure():
    """测试权限检查失败时的错误处理响应"""
    bot = MagicMock(spec=commands.Bot)
    cog = AdminPanelCog(bot)

    # 1. interaction 尚未响应过
    interaction1 = MagicMock(spec=discord.Interaction)
    interaction1.response.is_done.return_value = False
    interaction1.response.send_message = AsyncMock()

    await cog.on_delete_bot_message_error(
        interaction1, app_commands.CheckFailure("No permission")
    )
    interaction1.response.send_message.assert_called_once_with(
        "抱歉，您没有权限使用此命令。", ephemeral=True
    )

    # 2. interaction 已经 response 过（例如已 defer）
    interaction2 = MagicMock(spec=discord.Interaction)
    interaction2.response.is_done.return_value = True
    interaction2.followup.send = AsyncMock()

    await cog.on_delete_bot_message_error(
        interaction2, app_commands.CheckFailure("No permission")
    )
    interaction2.followup.send.assert_called_once_with(
        "抱歉，您没有权限使用此命令。", ephemeral=True
    )


@pytest.mark.asyncio
async def test_delete_bot_message_channel_forbidden():
    """测试获取频道时权限不足的情况"""
    bot = MagicMock(spec=commands.Bot)
    bot.get_channel.return_value = None
    bot.fetch_channel = AsyncMock(
        side_effect=discord.Forbidden(MagicMock(), "Cannot access channel")
    )
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "Bot 没有权限访问该频道" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_channel_unsupported():
    """测试频道类型不支持 fetch_message 的情况"""
    bot = MagicMock(spec=commands.Bot)
    channel = object()  # 没有 fetch_message 属性
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "该频道类型不支持获取消息" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_fetch_message_forbidden():
    """测试获取消息时缺少历史记录权限的情况"""
    bot = MagicMock(spec=commands.Bot)
    channel = MagicMock()
    channel.fetch_message = AsyncMock(
        side_effect=discord.Forbidden(MagicMock(), "Missing Read History")
    )
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "Bot 没有在该频道查看消息历史的权限" in interaction.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_bot_message_already_deleted():
    """测试删除时消息已被删除（NotFound）的情况"""
    bot = MagicMock(spec=commands.Bot)
    bot.user = MagicMock()
    bot.user.id = 8888

    channel = MagicMock()
    target_msg = MagicMock()
    target_msg.author = MagicMock()
    target_msg.author.id = 8888
    target_msg.delete = AsyncMock(
        side_effect=discord.NotFound(MagicMock(), "Already deleted")
    )

    channel.fetch_message = AsyncMock(return_value=target_msg)
    bot.get_channel.return_value = channel
    cog = AdminPanelCog(bot)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.user = MagicMock()

    link = "https://discord.com/channels/123/456/789"
    await cog.delete_bot_message.callback(cog, interaction, link)

    interaction.followup.send.assert_called_once()
    assert "该消息已经被删除" in interaction.followup.send.call_args[0][0]

