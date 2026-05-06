# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field

from src.chat.features.pixiv import image_sender as pixiv_image_sender
from src.chat.features.pixiv import runtime as pixiv_runtime
from src.chat.features.tools.tool_metadata import tool_metadata


class PixivToolParams(BaseModel):
    tags: list[str] = Field(
        default_factory=list,
        description="搜索标签列表。传了就按 tag 搜完随机抽，不传就是全随机，例如 ['初音未来', 'vocaloid']。",
    )
    exclude_tags: list[str] = Field(
        default_factory=list,
        description="排除标签列表，例如 ['guro', 'futa']。",
    )
    count: int = Field(
        default=1,
        description="返回数量。当前支持 1 到 5。",
        ge=1,
        le=5,
    )
    mode: str = Field(
        default="safe",
        description="内容模式。可选 safe 或 r18。",
    )
    allow_ai: bool = Field(
        default=False,
        description="是否允许 AI 生成作品。",
    )


@tool_metadata(
    name="Pixiv搜图",
    description="按 tag 随机抽 Pixiv 插画，或从 Pixiv 排行榜随机抽图，并直接发送到当前频道。",
    emoji="🖼️",
    category="工具",
)
async def pixiv_tool(
    action: str,
    params: PixivToolParams,
    **kwargs,
) -> str:
    """
    [工具说明]
    这是 Pixiv 插画工具，只有一个统一入口。
    仅支持以下 action：
    - `random_by_tag`: 传 tags 就按标签搜索后随机发送n张插画；不传 tags 就全随机
    - `random_ranking`: 从 Pixiv 排行榜里随机发送n张插画

    [参数要求]
    - `action` 必填，只能填上面三种之一
    - `params.tags`：当 action 为 `random_by_tag` 时可选；不传就是全随机
    - `params.exclude_tags`：可选，排除标签
    - `params.count`：当前支持 1 到 5，建议除非特殊情况1就行
    - `params.mode`：`safe` 或 `r18`
    - `params.allow_ai`：是否允许 AI 生成作品

    [执行逻辑]
    - 工具会直接把图片发送到当前 Discord 频道
    - 工具本身返回简短状态文本，供后续回复继续组织语言
    """
    channel = kwargs.get("channel")
    if not channel or not hasattr(channel, "send"):
        return "错误：找不到有效的消息频道。"

    if not isinstance(params, PixivToolParams):
        try:
            params = PixivToolParams(**params)
        except Exception as exc:
            return f"参数解析失败: {exc}"

    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"random_by_tag", "random_ranking"}:
        return "错误：不支持的 Pixiv action。"

    try:
        runtime = await pixiv_runtime.get_runtime()
    except Exception as exc:
        return f"错误：Pixiv 运行时初始化失败: {exc}"

    try:
        if normalized_action == "random_by_tag":
            result = await runtime.random_by_tag(
                tags=params.tags,
                exclude_tags=params.exclude_tags,
                mode=params.mode,
                allow_ai=params.allow_ai,
                count=params.count,
            )
        else:
            result = await runtime.random_ranking(
                mode=params.mode,
                allow_ai=params.allow_ai,
                exclude_tags=params.exclude_tags,
                count=params.count,
            )
    except Exception as exc:
        return f"错误：Pixiv 查询失败: {exc}"

    if not result.success or not result.images:
        return result.message

    sent_count = 0
    for image in result.images:
        sent, error = await pixiv_image_sender.send_illust_to_channel(
            channel,
            image,
            runtime.config,
        )
        if not sent:
            return error or "Pixiv 图片发送失败。"
        await runtime.mark_sent(image.illust_id)
        sent_count += 1

    return result.message if sent_count == len(result.images) else f"已发送 {sent_count} 张 Pixiv 插画。"
