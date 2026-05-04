import discord
from discord import ButtonStyle, Interaction
from discord.ui import Button, View

from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)


class LogSettingsView(View):
    """全局日志设置面板。"""

    def __init__(self, full_context_enabled: bool, final_reply_enabled: bool):
        super().__init__(timeout=180)
        self.service = chat_settings_service
        self.full_context_enabled = full_context_enabled
        self.final_reply_enabled = final_reply_enabled
        self._build_items()

    @classmethod
    async def create(cls):
        full_context_enabled = await chat_settings_service.get_full_context_logging_enabled()
        final_reply_enabled = await chat_settings_service.get_final_reply_logging_enabled()
        return cls(full_context_enabled, final_reply_enabled)

    def render_content(self) -> str:
        return (
            "这里控制全局日志输出。\n\n"
            f"完整上下文：**{'开' if self.full_context_enabled else '关'}**\n"
            "记录发送给模型的完整上下文。\n\n"
            f"最终回复：**{'开' if self.final_reply_enabled else '关'}**\n"
            "记录最终返回给用户的回复内容。"
        )

    def _build_items(self) -> None:
        self.clear_items()

        full_context_button = Button(
            label=f"完整上下文: {'开' if self.full_context_enabled else '关'}",
            style=ButtonStyle.green if self.full_context_enabled else ButtonStyle.red,
            custom_id="log_toggle_full_context",
            row=0,
        )
        full_context_button.callback = self.on_toggle_full_context
        self.add_item(full_context_button)

        final_reply_button = Button(
            label=f"最终回复: {'开' if self.final_reply_enabled else '关'}",
            style=ButtonStyle.green if self.final_reply_enabled else ButtonStyle.red,
            custom_id="log_toggle_final_reply",
            row=0,
        )
        final_reply_button.callback = self.on_toggle_final_reply
        self.add_item(final_reply_button)

    async def on_toggle_full_context(self, interaction: Interaction):
        self.full_context_enabled = not self.full_context_enabled
        await self.service.set_full_context_logging_enabled(self.full_context_enabled)
        self._build_items()
        await interaction.response.edit_message(content=self.render_content(), view=self)

    async def on_toggle_final_reply(self, interaction: Interaction):
        self.final_reply_enabled = not self.final_reply_enabled
        await self.service.set_final_reply_logging_enabled(self.final_reply_enabled)
        self._build_items()
        await interaction.response.edit_message(content=self.render_content(), view=self)
