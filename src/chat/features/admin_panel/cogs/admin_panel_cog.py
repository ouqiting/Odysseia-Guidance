# -*- coding: utf-8 -*-
from __future__ import annotations

import discord
from discord.ext import commands
from discord import app_commands
import logging
import re
from typing import Optional, Tuple, Union

from src import config

log = logging.getLogger(__name__)

DISCORD_MESSAGE_LINK_PATTERN = re.compile(
    r"^https?://(?:(?:canary\.|ptb\.)?discord(?:app)?\.com)/channels/(?P<guild_id>[0-9]+|@me)/(?P<channel_id>[0-9]+)/(?P<message_id>[0-9]+)(?:[/?].*)?$"
)


def parse_discord_message_link(
    link: str,
) -> Optional[Tuple[Union[int, str], int, int]]:
    """从 Discord 消息链接中解析出 (guild_id, channel_id, message_id)。

    支持 discord.com, discordapp.com, canary, ptb 域名。
    支持带 <> 的链接、前后空白以及带有 query 参数的链接。
    若解析失败返回 None。
    """
    if not link:
        return None
    clean_link = link.strip().strip("<>")
    match = DISCORD_MESSAGE_LINK_PATTERN.match(clean_link)
    if not match:
        return None
    guild_part = match.group("guild_id")
    guild_id = "@me" if guild_part == "@me" else int(guild_part)
    channel_id = int(match.group("channel_id"))
    message_id = int(match.group("message_id"))
    return guild_id, channel_id, message_id


# --- 权限检查 ---
async def check_admin_or_dev(interaction: discord.Interaction) -> bool:
    """检查用户是否是管理员或开发者"""
    if not config.ADMIN_ROLE_IDS and not config.DEVELOPER_USER_IDS:
        log.warning("ADMIN_ROLE_IDS 和 DEVELOPER_USER_IDS 未在 .env 文件中配置。")
        return False

    user = interaction.user
    user_id = user.id
    is_dev = user_id in config.DEVELOPER_USER_IDS

    user_roles = {role.id for role in getattr(user, "roles", [])}
    is_admin = bool(
        config.ADMIN_ROLE_IDS and not user_roles.isdisjoint(config.ADMIN_ROLE_IDS)
    )

    result = is_admin or is_dev
    if not result:
        log.warning(
            f"用户 {interaction.user} (ID: {user_id}) 尝试执行管理员命令失败（权限不足）。"
        )
    return result


def is_admin_or_dev():
    """检查用户是否是管理员或开发者装饰器"""
    return app_commands.check(check_admin_or_dev)


class AdminPanelCog(commands.Cog):
    """包含仅限管理员和开发者使用的命令"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="数据库管理面板", description="以交互方式浏览和管理数据库内容"
    )
    @is_admin_or_dev()
    async def db_view(self, interaction: discord.Interaction):
        """启动数据库浏览器"""
        log.info(
            f"管理员 {interaction.user.display_name} (ID: {interaction.user.id}) 启动了数据库浏览器。"
        )
        try:
            from src.chat.features.admin_panel.ui.views.db_management_view import (
                DBManagementView,
            )

            await interaction.response.defer(ephemeral=True)
            view = DBManagementView(interaction.user.id)
            # 先发送一个临时的加载消息
            message = await interaction.followup.send(
                "正在加载数据库浏览器...", ephemeral=True
            )
            view.message = message
            # 手动触发第一次视图更新，显示完整的UI
            await view.update_view()
        except Exception as e:
            log.error(f"启动数据库浏览器时发生未知错误: {e}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"启动数据库浏览器时发生未知错误: {e}", ephemeral=True
                )

    @db_view.error
    async def on_db_view_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ):
        """db_view 命令的错误处理程序"""
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message(
                "抱歉，您没有权限使用此命令。", ephemeral=True
            )
        else:
            log.error(f"执行 db_view 命令时发生未处理的错误: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"执行命令时发生未知错误: {error}", ephemeral=True
                )

    @app_commands.command(name="删除信息", description="删除Bot自身发送的消息")
    @app_commands.describe(
        message_link="要删除的Bot消息链接（例如 https://discord.com/channels/服务器ID/频道ID/消息ID）"
    )
    @is_admin_or_dev()
    async def delete_bot_message(
        self, interaction: discord.Interaction, message_link: str
    ):
        """通过消息链接删除 Bot 自身发送的消息"""
        log.info(
            f"用户 {interaction.user} (ID: {interaction.user.id}) 请求删除消息: {message_link}"
        )
        await interaction.response.defer(ephemeral=True)

        parsed = parse_discord_message_link(message_link)
        if not parsed:
            await interaction.followup.send(
                "无效的消息链接格式。请输入有效的 Discord 消息链接，例如：\n"
                "`https://discord.com/channels/服务器ID/频道ID/消息ID`",
                ephemeral=True,
            )
            return

        guild_id, channel_id, message_id = parsed

        # 获取频道
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except discord.NotFound:
                await interaction.followup.send(
                    "找不到指定的频道，可能是频道已被删除或链接无效。",
                    ephemeral=True,
                )
                return
            except discord.Forbidden:
                await interaction.followup.send(
                    "Bot 没有权限访问该频道。", ephemeral=True
                )
                return
            except Exception as e:
                log.error(f"获取频道 {channel_id} 时发生错误: {e}", exc_info=True)
                await interaction.followup.send(
                    f"获取频道失败: {e}", ephemeral=True
                )
                return

        if not hasattr(channel, "fetch_message"):
            await interaction.followup.send(
                "该频道类型不支持获取消息。", ephemeral=True
            )
            return

        # 获取目标消息
        try:
            target_message = await channel.fetch_message(message_id)
        except discord.NotFound:
            await interaction.followup.send(
                "找不到指定的消息，该消息可能已被删除或链接无效。",
                ephemeral=True,
            )
            return
        except discord.Forbidden:
            await interaction.followup.send(
                "Bot 没有在该频道查看消息历史的权限。", ephemeral=True
            )
            return
        except Exception as e:
            log.error(
                f"获取消息 {message_id} (频道 {channel_id}) 时发生错误: {e}",
                exc_info=True,
            )
            await interaction.followup.send(
                f"获取消息失败: {e}", ephemeral=True
            )
            return

        # 校验消息作者是否为 Bot 自身
        bot_user_id = self.bot.user.id if self.bot.user else None
        if bot_user_id is None or target_message.author.id != bot_user_id:
            await interaction.followup.send(
                "只能删除由本 Bot 发送的消息，无法删除其他用户发送的消息。",
                ephemeral=True,
            )
            return

        # 执行删除
        try:
            await target_message.delete()
            log.info(
                f"管理员 {interaction.user} (ID: {interaction.user.id}) 成功删除了 Bot 消息 "
                f"(频道: {channel_id}, 消息ID: {message_id})"
            )
            await interaction.followup.send("消息删除成功！", ephemeral=True)
        except discord.NotFound:
            await interaction.followup.send("该消息已经被删除。", ephemeral=True)
        except discord.Forbidden:
            await interaction.followup.send(
                "Bot 缺少删除该消息的权限。", ephemeral=True
            )
        except Exception as e:
            log.error(f"删除消息 {message_id} 时发生错误: {e}", exc_info=True)
            await interaction.followup.send(
                f"删除消息失败: {e}", ephemeral=True
            )

    @delete_bot_message.error
    async def on_delete_bot_message_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ):
        """delete_bot_message 命令的错误处理程序"""
        if isinstance(error, app_commands.CheckFailure):
            if interaction.response.is_done():
                await interaction.followup.send(
                    "抱歉，您没有权限使用此命令。", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "抱歉，您没有权限使用此命令。", ephemeral=True
                )
        else:
            log.error(
                f"执行 delete_bot_message 命令时发生未处理的错误: {error}",
                exc_info=True,
            )
            msg = f"执行命令时发生未知错误: {error}"
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)


async def setup(bot: commands.Bot):
    """将这个Cog添加到机器人中"""
    await bot.add_cog(AdminPanelCog(bot))
