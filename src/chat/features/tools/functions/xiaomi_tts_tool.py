# -*- coding: utf-8 -*-

import os
import logging
import io
import discord
import re
import asyncio
import base64
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI
from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)


def _get_mimo_api_key() -> str:
    key = os.getenv("XIAOMI_TTS_KEY")
    if not key:
        raise RuntimeError(
            "XIAOMI_TTS_KEY 未在环境变量中设置。请在项目根目录的 .env 文件中配置 "
            "XIAOMI_TTS_KEY=<你的小米MiMo API Key>。"
        )
    return key


class XiaomiTTSParams(BaseModel):
    text: str = Field(
        ...,
        description="""要转换成语音的文字内容。"""
    )
    filename: str = Field(
        ...,
        description="音频文件的名字。请根据内容生成一个10字以内的标题，如'傲娇抱怨'、'开心问候'等。无特殊要求中文名。"
    )


@tool_metadata(
    name="小米tts",
    description="使用小米MiMo-V2.5-TTS生成高质量语音，支持情绪标签和音频标签控制，冰糖音色。",
    emoji="🍬",
    category="工具",
)
async def xiaomi_tts_tool(
    params: XiaomiTTSParams,
    **kwargs,
) -> str:
    """
    [工具说明]
    这是小米MiMo-V2.5-TTS高质量语音合成工具，使用冰糖音色（年轻女声）。
    当用户提及"TTS"、"生成语音"或想生成语音时使用。

    [情绪标签用法]
    必须放在文本最前面，用半角括号包裹：
    (开心)、(惊讶)、(唱歌)、(悲伤)、(愤怒)、(温柔)、(慵懒)、(冷漠)、(傲娇)、(撒娇)

    [音频标签用法]
    可插入在文本任意位置，用全角括号包裹，实现细粒度控制：
    - 停顿/呼吸：（沉默片刻）、（深呼吸）、（叹气）、（喘息）
    - 情绪状态：（苦笑）、（紧张）、（兴奋）、（委屈）、（害羞）
    - 笑/哭：（轻笑）、（大笑）、（冷笑）、（啜泣）、（哽咽）

    [示例]
    (开心)主人～你终于来啦！（兴奋）我等你好久了呢～
    (惊讶)什么？！（沉默片刻）这……这不可能吧？
    如果我当时……（沉默片刻）哪怕再坚持一秒钟，结果是不是就不一样了？（苦笑）呵，没如果了。
    (唱歌)一闪一闪亮晶晶，满天都是小星星。

    [执行逻辑]
    - 工具会根据文本内容调用小米MiMo-V2.5-TTS生成音频。
    - 生成一个 .wav 音频流并直接发送至 Discord 频道。
    """
    channel = kwargs.get("channel")
    if not channel or not isinstance(channel, discord.abc.Messageable):
        return "错误：找不到有效的消息频道。"

    if not isinstance(params, XiaomiTTSParams):
        try:
            params = XiaomiTTSParams(**params)
        except Exception as e:
            return f"参数解析失败: {e}"

    raw_name = params.filename or "神所娘的语音"
    clean_name = re.sub(r'[\\/:*?"<>|]', '', raw_name)[:10]
    display_filename = f"{clean_name}.wav"

    try:
        def call_xiaomi_api():
            client = OpenAI(
                api_key=_get_mimo_api_key(),
                base_url="https://api.xiaomimimo.com/v1",
            )
            completion = client.chat.completions.create(
                model="mimo-v2.5-tts",
                messages=[
                    {
                        "role": "assistant",
                        "content": params.text,
                    }
                ],
                audio={
                    "format": "wav",
                    "voice": "冰糖",
                },
            )
            message = completion.choices[0].message
            return base64.b64decode(message.audio.data)

        audio_bytes = await asyncio.to_thread(call_xiaomi_api)

        file = discord.File(fp=io.BytesIO(audio_bytes), filename=display_filename)
        await channel.send(file=file)

        return "成功：语音文件已发送。"

    except Exception as e:
        log.error(f"小米TTS运行异常: {e}")
        return f"错误：生成语音时发生故障: {e}"
