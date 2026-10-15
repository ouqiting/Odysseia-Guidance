# -*- coding: utf-8 -*-

"""`/投喂` 命令的单元测试。

文字阶段的模型链路已下沉到 feeding_text_service，因此这里只 mock 该服务，
专注验证奖励结算与 Embed 渲染。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from src.chat.features.affection.cogs.feeding_cog import FeedingCog
from src.chat.features.affection.service.feeding_text_service import (
    FeedingTextTimeoutError,
)


class TestFeedingCog(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 数据库 / 生图 / 提示词服务都不参与本测试，直接替换为替身
        self.db_patcher = patch(
            "src.chat.features.affection.cogs.feeding_cog.chat_db_manager"
        )
        self.mock_db = self.db_patcher.start()
        self.addCleanup(self.db_patcher.stop)
        self.mock_db.is_channel_muted = AsyncMock(return_value=False)
        self.mock_db.get_global_setting = AsyncMock(return_value=None)

        self.image_service_patcher = patch(
            "src.chat.features.affection.cogs.feeding_cog.gpt_image_service",
            new=MagicMock(is_available=False),
        )
        self.image_service_patcher.start()
        self.addCleanup(self.image_service_patcher.stop)

        self.prompt_patcher = patch(
            "src.chat.features.affection.cogs.feeding_cog.prompt_service"
        )
        self.mock_prompt_service = self.prompt_patcher.start()
        self.addCleanup(self.prompt_patcher.stop)
        self.mock_prompt_service.get_prompt.return_value = (
            "<character>神所娘人设</character>"
        )

        self.bot = MagicMock()
        self.cog = FeedingCog(self.bot)

        # Mock services
        self.cog.affection_service = AsyncMock()
        self.cog.coin_service = AsyncMock()
        self.cog.feeding_service = AsyncMock()
        self.cog.feeding_text_service = AsyncMock()

        # Mock interaction
        self.interaction = AsyncMock(spec=discord.Interaction)
        self.interaction.user.id = 123456789
        self.interaction.user.display_name = "tester"
        self.interaction.user.display_avatar.url = "https://cdn.example.com/avatar.png"
        self.interaction.channel = MagicMock(spec=discord.TextChannel)
        self.interaction.channel.id = 987654321
        self.interaction.response.send_message = AsyncMock()
        self.interaction.edit_original_response = AsyncMock()

        # Mock attachment
        self.image_attachment = MagicMock(spec=discord.Attachment)
        self.image_attachment.content_type = "image/png"
        self.image_attachment.read = AsyncMock(return_value=b"fake_image_bytes")
        self.image_attachment.filename = "test.png"

        # 默认放行投喂
        self.cog.feeding_service.can_feed.return_value = (True, "")

        # app_commands 装饰器会把方法包装成 Command 对象，测试需要取回原始回调
        self.feed = self.cog.feed.callback

    async def test_feed_with_positive_coins(self):
        """测试投喂获得正数类脑币奖励的情况"""
        self.cog.feeding_text_service.generate_text_with_image.return_value = (
            "评价<affection:+5;coins:+50>"
        )

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.affection_service.add_affection_points.assert_awaited_once_with(
            123456789, 5
        )
        self.cog.coin_service.add_coins.assert_awaited_once_with(
            123456789, 50, reason="投喂奖励"
        )
        self.cog.feeding_service.record_feeding.assert_awaited_once_with(123456789)

        embed = self.interaction.edit_original_response.call_args.kwargs["embed"]
        self.assertIn("评价", embed.description)
        self.assertIn("你获得了 50 枚类脑币！", embed.description)

    async def test_feed_with_negative_coins(self):
        """测试投喂获得负数类脑币奖励的情况"""
        self.cog.feeding_text_service.generate_text_with_image.return_value = (
            "评价<affection:-2;coins:-30>"
        )

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.affection_service.add_affection_points.assert_awaited_once_with(
            123456789, -2
        )
        # 核心验证：不应调用 add_coins 或 remove_coins
        self.cog.coin_service.add_coins.assert_not_called()
        self.cog.coin_service.remove_coins.assert_not_called()

        embed = self.interaction.edit_original_response.call_args.kwargs["embed"]
        self.assertNotIn("你获得了", embed.description)

    async def test_feed_with_zero_coins(self):
        """测试投喂获得零类脑币奖励的情况"""
        self.cog.feeding_text_service.generate_text_with_image.return_value = (
            "评价<affection:+1;coins:0>"
        )

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.affection_service.add_affection_points.assert_awaited_once_with(
            123456789, 1
        )
        self.cog.coin_service.add_coins.assert_not_called()
        self.cog.coin_service.remove_coins.assert_not_called()

        embed = self.interaction.edit_original_response.call_args.kwargs["embed"]
        self.assertNotIn("你获得了", embed.description)

    async def test_text_stage_receives_original_image_bytes(self):
        """文字链路自行压缩图片，因此拿到的是原图字节。"""
        self.cog.feeding_text_service.generate_text_with_image.return_value = (
            "评价<affection:+1;coins:+10>"
        )

        await self.feed(self.cog, self.interaction, self.image_attachment)

        call_kwargs = (
            self.cog.feeding_text_service.generate_text_with_image.call_args.kwargs
        )
        self.assertEqual(call_kwargs["image_bytes"], b"fake_image_bytes")
        self.assertEqual(call_kwargs["mime_type"], "image/png")
        self.assertIn("神所娘", call_kwargs["prompt"])

    async def test_feed_failure_message_when_text_stage_returns_none(self):
        """文字链路失败（None）时不发放奖励，只回复失败提示。"""
        self.cog.feeding_text_service.generate_text_with_image.return_value = None

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.affection_service.add_affection_points.assert_not_called()
        self.cog.coin_service.add_coins.assert_not_called()
        self.cog.feeding_service.record_feeding.assert_not_called()
        self.assertEqual(
            self.interaction.edit_original_response.call_args.kwargs["content"],
            "抱歉，我有点累了，暂时无法评价呢。",
        )

    async def test_timeout_message_when_text_stage_exceeds_limit(self):
        """文字阶段超时（120 秒）时提示生成投喂文本超时，且不发放奖励。"""
        self.cog.feeding_text_service.generate_text_with_image.side_effect = (
            FeedingTextTimeoutError(120.0)
        )

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.affection_service.add_affection_points.assert_not_called()
        self.cog.coin_service.add_coins.assert_not_called()
        self.cog.feeding_service.record_feeding.assert_not_called()
        self.assertIn(
            "生成投喂文本超时",
            self.interaction.edit_original_response.call_args.kwargs["content"],
        )

    async def test_cooldown_blocks_feeding(self):
        """冷却中直接回复提示，不进入文字链路。"""
        self.cog.feeding_service.can_feed.return_value = (False, "饱啦饱啦")

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.feeding_text_service.generate_text_with_image.assert_not_called()
        self.assertEqual(
            self.interaction.response.send_message.call_args.args[0], "饱啦饱啦"
        )

    async def test_unparsable_response_falls_back_to_default_rewards(self):
        """解析不到奖励标签时，用整段回复作为评价并给予默认奖励。"""
        self.cog.feeding_text_service.generate_text_with_image.return_value = "唔…好吃"

        await self.feed(self.cog, self.interaction, self.image_attachment)

        self.cog.affection_service.add_affection_points.assert_awaited_once_with(
            123456789, 1
        )
        self.cog.coin_service.add_coins.assert_awaited_once_with(
            123456789, 10, reason="投喂奖励"
        )
        embed = self.interaction.edit_original_response.call_args.kwargs["embed"]
        self.assertIn("唔…好吃", embed.description)


if __name__ == "__main__":
    unittest.main()
