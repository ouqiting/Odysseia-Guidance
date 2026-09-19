# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

MOONSHOT_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_SEARCH_PRO_TIMEOUT_SECONDS = 30.0
KIMI_FETCH_TIMEOUT_SECONDS = 30.0
KIMI_SEARCH_LIMIT = 5
KIMI_FETCH_MAX_MARKDOWN_CHARS = 20000
GROK_MODEL_NAME = "grok-chat-fast"
GROK_MAX_RETRIES = 3
GROK_TOTAL_TIMEOUT_SECONDS = 30.0
GROK_SEARCH_INSTRUCTIONS = """
你是一个联网检索助手。

要求：
1. 回答语言使用简体中文。
2. 输出必须整理清楚，按以下结构组织：
   - 摘要
   - 关键信息
   - 参考来源
3. 如果搜索结果存在时效性或来源冲突，要明确说明。
4. 不要编造来源；如果没有明确来源，就直接说明来源不足。
""".strip()


def _build_chat_completions_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized.endswith("/chat/completions"):
        normalized += "/chat/completions"
    return normalized


def _resolve_moonshot_base_url() -> str:
    configured = str(os.getenv("MOONSHOT_URL") or "").strip().rstrip("/")
    return configured or MOONSHOT_DEFAULT_BASE_URL


def _build_kimi_tools_url(endpoint: str) -> str:
    base_url = _resolve_moonshot_base_url()
    return f"{base_url}/tools/{endpoint.strip('/')}"


def _split_moonshot_api_keys(raw_api_keys: Optional[str]) -> List[str]:
    raw = str(raw_api_keys or "").strip()
    if not raw:
        return []

    parts = re.split(r"[,\n\r，]+", raw)
    result: List[str] = []
    seen = set()
    for part in parts:
        key = part.strip().strip('"').strip("'").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _dedupe_sources(sources: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for title, url in sources:
        clean_title = str(title or "").strip()
        clean_url = str(url or "").strip()
        key = (clean_title, clean_url)
        if not clean_url or key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "title": clean_title or clean_url,
                "url": clean_url,
            }
        )
    return deduped


def _extract_chat_completion_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices", []) or []
    for choice in choices:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text

        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        chunks.append(text)
            joined = "\n".join(chunks).strip()
            if joined:
                return joined

    return ""


def _extract_chat_completion_text_from_sse_body(body: str) -> str:
    chunks: List[str] = []

    for raw_line in str(body or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = data.get("choices", []) or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            content = delta.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)
                continue

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = str(item.get("text") or "")
                        if text:
                            chunks.append(text)

    return "".join(chunks).strip()


def _is_grok_configured() -> bool:
    grok_url = str(os.getenv("GROK_URL") or "").strip()
    grok_api_key = str(os.getenv("GROK_API_KEY") or "").strip()
    return bool(grok_url and grok_api_key)


def _format_combined_answer(
    *,
    kimi_answer: str,
    grok_answer: str,
) -> str:
    sections: List[str] = []

    if kimi_answer:
        sections.append(f"【Kimi 联网搜索结果】\n{kimi_answer}")

    if grok_answer:
        sections.append(f"【Grok 辅助结果】\n{grok_answer}")

    return "\n\n".join(section for section in sections if section).strip()


def _format_kimi_search_answer(results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []

    for item in results or []:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        date = str(item.get("date") or "").strip()
        site_name = str(item.get("site_name") or "").strip()
        snippet = str(item.get("snippet") or "").strip()

        lines: List[str] = []
        if title:
            lines.append(f"标题：{title}")
        if url:
            lines.append(f"链接：{url}")
        if site_name:
            lines.append(f"站点：{site_name}")
        if date:
            lines.append(f"日期：{date}")
        if snippet:
            lines.append(f"摘要：{snippet}")

        chunk_texts: List[str] = []
        for chunk in item.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "").strip()
            if text:
                chunk_texts.append(text)
        if chunk_texts:
            lines.append("正文片段：")
            lines.extend(chunk_texts)

        if lines:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip()


async def _search_with_kimi_pro(clean_question: str) -> Dict[str, Any]:
    api_keys = _split_moonshot_api_keys(os.getenv("MOONSHOT_API_KEY"))
    if not api_keys:
        log.warning("[WebSearchTool][Kimi] MOONSHOT_API_KEY 未配置。")
        return {
            "channel": "kimi",
            "enabled": False,
            "search_executed": False,
            "error": "未配置 MOONSHOT_API_KEY，无法执行联网搜索。",
        }

    api_url = _build_kimi_tools_url("search_pro")
    payload: Dict[str, Any] = {
        "text_query": clean_question,
        "limit": KIMI_SEARCH_LIMIT,
        "timeout_seconds": int(KIMI_SEARCH_PRO_TIMEOUT_SECONDS),
    }

    last_error = ""
    for index, api_key in enumerate(api_keys, start=1):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=KIMI_SEARCH_PRO_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    api_url,
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:
                last_error = (
                    f"HTTP {response.status_code}: {(response.text or '')[:2000]}"
                ).strip()
                log.warning(
                    "[WebSearchTool][Kimi] 第 %s 个 key 请求失败 | %s",
                    index,
                    last_error,
                )
                continue

            try:
                data = response.json()
            except json.JSONDecodeError:
                last_error = (response.text or "<empty>")[:2000]
                log.warning(
                    "[WebSearchTool][Kimi] 第 %s 个 key 返回非 JSON 响应 | %s",
                    index,
                    last_error,
                )
                continue

            results = data.get("search_results") or []
            answer_text = _format_kimi_search_answer(results)
            sources = _dedupe_sources(
                [
                    (
                        str(item.get("title") or "").strip(),
                        str(item.get("url") or "").strip(),
                    )
                    for item in results
                    if isinstance(item, dict)
                ]
            )

            result: Dict[str, Any] = {
                "channel": "kimi",
                "enabled": True,
                "search_executed": True,
                "model": "kimi-search-pro",
                "answer": answer_text,
                "sources": sources,
                "results": results,
            }

            if not results:
                result["error"] = "联网搜索未返回结果。"
            elif not answer_text:
                result["error"] = "联网搜索已执行，但未解析出可用片段。"

            return result
        except httpx.RequestError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "[WebSearchTool][Kimi] 第 %s 个 key 网络异常 | %s",
                index,
                last_error,
            )
            continue
        except Exception as exc:
            last_error = str(exc)
            log.error(
                "[WebSearchTool][Kimi] 第 %s 个 key 发生异常 | %s",
                index,
                last_error,
                exc_info=True,
            )
            continue

    return {
        "channel": "kimi",
        "enabled": True,
        "search_executed": False,
        "model": "kimi-search-pro",
        "error": "Kimi 联网搜索请求失败，已自动放弃。",
        "detail": last_error,
    }


async def _fetch_with_kimi(target_url: str) -> Dict[str, Any]:
    api_keys = _split_moonshot_api_keys(os.getenv("MOONSHOT_API_KEY"))
    if not api_keys:
        log.warning("[WebSearchTool][Kimi][Fetch] MOONSHOT_API_KEY 未配置。")
        return {
            "channel": "kimi_fetch",
            "enabled": False,
            "fetch_executed": False,
            "error": "未配置 MOONSHOT_API_KEY，无法抓取网页。",
        }

    api_url = _build_kimi_tools_url("fetch")
    payload: Dict[str, Any] = {"url": target_url}

    last_error = ""
    for index, api_key in enumerate(api_keys, start=1):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=KIMI_FETCH_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    api_url,
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:
                last_error = (
                    f"HTTP {response.status_code}: {(response.text or '')[:2000]}"
                ).strip()
                log.warning(
                    "[WebSearchTool][Kimi][Fetch] 第 %s 个 key 请求失败 | %s",
                    index,
                    last_error,
                )
                continue

            try:
                data = response.json()
            except json.JSONDecodeError:
                last_error = (response.text or "<empty>")[:2000]
                log.warning(
                    "[WebSearchTool][Kimi][Fetch] 第 %s 个 key 返回非 JSON 响应 | %s",
                    index,
                    last_error,
                )
                continue

            title = str(data.get("title") or "").strip()
            markdown = str(data.get("markdown") or "").strip()
            final_url = str(data.get("url") or target_url).strip() or target_url

            if len(markdown) > KIMI_FETCH_MAX_MARKDOWN_CHARS:
                markdown = (
                    markdown[:KIMI_FETCH_MAX_MARKDOWN_CHARS]
                    + "\n\n...(正文过长，已截断)"
                )

            result: Dict[str, Any] = {
                "channel": "kimi_fetch",
                "enabled": True,
                "fetch_executed": True,
                "title": title,
                "url": final_url,
                "answer": markdown,
                "sources": [
                    {
                        "title": title or final_url,
                        "url": final_url,
                    }
                ],
            }

            if not markdown:
                result["error"] = "网页抓取已执行，但未获取到正文内容。"

            return result
        except httpx.RequestError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "[WebSearchTool][Kimi][Fetch] 第 %s 个 key 网络异常 | %s",
                index,
                last_error,
            )
            continue
        except Exception as exc:
            last_error = str(exc)
            log.error(
                "[WebSearchTool][Kimi][Fetch] 第 %s 个 key 发生异常 | %s",
                index,
                last_error,
                exc_info=True,
            )
            continue

    return {
        "channel": "kimi_fetch",
        "enabled": True,
        "fetch_executed": False,
        "error": "Kimi 网页抓取请求失败，已自动放弃。",
        "detail": last_error,
    }


async def _search_with_grok(clean_question: str) -> Dict[str, Any]:
    grok_base_url = str(os.getenv("GROK_URL") or "").strip()
    grok_api_key = str(os.getenv("GROK_API_KEY") or "").strip()

    if not grok_base_url or not grok_api_key:
        return {
            "channel": "grok",
            "enabled": False,
            "skipped": True,
            "model": GROK_MODEL_NAME,
        }

    grok_url = _build_chat_completions_url(grok_base_url)
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": GROK_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": GROK_SEARCH_INSTRUCTIONS},
            {"role": "user", "content": clean_question},
        ],
    }

    last_error = ""
    started_at = time.monotonic()
    for attempt in range(1, GROK_MAX_RETRIES + 2):
        elapsed_seconds = time.monotonic() - started_at
        remaining_seconds = GROK_TOTAL_TIMEOUT_SECONDS - elapsed_seconds
        if remaining_seconds <= 0:
            last_error = (
                f"Grok 通道总超时，已超过 {GROK_TOTAL_TIMEOUT_SECONDS:.0f} 秒。"
            )
            log.error("[WebSearchTool][Grok] %s", last_error)
            return {
                "channel": "grok",
                "enabled": True,
                "search_executed": False,
                "model": GROK_MODEL_NAME,
                "error": "Grok 通道请求超时，已自动放弃。",
                "detail": last_error,
                "attempts": max(attempt - 1, 0),
            }

        try:
            timeout_seconds = min(KIMI_SEARCH_PRO_TIMEOUT_SECONDS, remaining_seconds)
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    grok_url,
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:
                last_error = (
                    f"HTTP {response.status_code}: {(response.text or '')[:2000]}"
                ).strip()
                raise httpx.HTTPStatusError(
                    "Grok 请求返回非 200 状态码。",
                    request=response.request,
                    response=response,
                )

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                sse_answer_text = _extract_chat_completion_text_from_sse_body(
                    response.text or ""
                )
                if sse_answer_text:
                    return {
                        "channel": "grok",
                        "enabled": True,
                        "search_executed": False,
                        "model": GROK_MODEL_NAME,
                        "answer": sse_answer_text,
                        "sources": [],
                        "attempts": attempt,
                        "response_format": "sse",
                    }

                last_error = (response.text or "<empty>")[:2000]
                raise ValueError("Grok 返回了非 JSON 响应。") from exc

            answer_text = _extract_chat_completion_text(data)
            if not answer_text:
                last_error = json.dumps(data, ensure_ascii=False)[:2000]
                raise ValueError("Grok 响应中未解析出正文。")

            return {
                "channel": "grok",
                "enabled": True,
                "search_executed": False,
                "model": GROK_MODEL_NAME,
                "answer": answer_text,
                "sources": [],
                "attempts": attempt,
                "response_format": "json",
            }
        except Exception as exc:
            if not last_error:
                last_error = str(exc)

            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= GROK_TOTAL_TIMEOUT_SECONDS:
                last_error = (
                    f"{last_error} | Grok 通道总超时，已超过 "
                    f"{GROK_TOTAL_TIMEOUT_SECONDS:.0f} 秒。"
                )
                log.error("[WebSearchTool][Grok] %s", last_error)
                return {
                    "channel": "grok",
                    "enabled": True,
                    "search_executed": False,
                    "model": GROK_MODEL_NAME,
                    "error": "Grok 通道请求超时，已自动放弃。",
                    "detail": last_error,
                    "attempts": attempt,
                }

            if attempt <= GROK_MAX_RETRIES:
                log.warning(
                    "[WebSearchTool][Grok] 第 %s 次请求失败，准备重试 | error=%s",
                    attempt,
                    last_error,
                )
                continue

            log.error(
                "[WebSearchTool][Grok] 请求失败，已达到最大重试次数 | error=%s",
                last_error,
            )
            return {
                "channel": "grok",
                "enabled": True,
                "search_executed": False,
                "model": GROK_MODEL_NAME,
                "error": "Grok 通道请求失败，已自动放弃。",
                "detail": last_error,
                "attempts": attempt,
            }


class WebSearchParams(BaseModel):
    question: str = Field(
        "",
        description="需要联网搜索的问题，例如“上网搜一下A项目是做什么的”里真正要搜索的那部分问题。抓取模式下可为空。",
    )
    mode: str = Field(
        "search",
        description="检索模式，`search`（联网搜索）或 `fetch`（网页抓取），默认 `search`。",
    )
    url: str = Field(
        "",
        description="网页抓取模式下的目标网页 URL，仅支持 http/https。搜索模式下留空。",
    )


@tool_metadata(
    name="联网搜索",
    description=(
        "联网检索网页信息。支持两种模式：mode=search 联网搜索，"
        "mode=fetch 网页抓取（只抓取指定 URL 的正文）。"
    ),
    emoji="🌐",
    category="工具",
)
async def search_web(params: WebSearchParams, **kwargs) -> Dict[str, Any]:
    """
    [工具说明]
    这是一个用于联网检索网页信息的工具，支持两种模式。

    [模式选择 - 必须显式传入 mode]
    - `mode="search"`：联网搜索。适用于用户想“上网搜”、“联网搜”、“网上查”、“帮我搜最新信息”等。
    - `mode="fetch"`：网页抓取。适用于用户给出一个明确的网址/链接，或要求“看一下这个网页/链接的内容”。
      只抓取该 URL 的正文

    [调用规则 - 高优先级]
    - 当用户明确提到“上网搜”、“联网搜”、“网上查”、“去网上搜一下”、“帮我搜最新信息”时，必须调用此工具，并传 `mode="search"`。
    - 当用户给出明确的 URL（http/https）或要求读取某个链接内容时，必须调用此工具，并传 `mode="fetch"`，同时把链接放进 `url`。
    - 当问题明显依赖最新网页信息时，应优先考虑此工具，而不是直接凭记忆回答。
    - 如果没有调用此工具，就不要假装自己已经上网搜索过。
    - `mode="search"` 时只需把“要搜索的问题”传给 `question`，不要夹带多余解释；`mode="fetch"` 时把目标链接传给 `url`。

    [结果使用规则]
    - 工具返回的 `answer` 是检索到的内容（搜索模式为搜索片段，抓取模式为网页正文），可直接基于它进行回复。
    - 工具返回的 `sources` 是实际检索到的网址列表；如果需要引用来源，应优先使用这些链接。
    - 如果工具返回 `search_executed=false` / `fetch_executed=false` 或 `error`，要如实告诉用户本次联网检索失败，不要编造结果。

    Args:
        params (WebSearchParams): 检索参数。
            - question: 需要联网搜索的问题，例如“上网搜一下A项目是做什么的”里真正要搜索的那部分问题。抓取模式下可为空。
            - mode: 检索模式，`"search"`（联网搜索）或 `"fetch"`（网页抓取），默认 `"search"`。
            - url: 网页抓取模式下的目标网页 URL，仅支持 http/https。搜索模式下留空。

    Returns:
        一个包含检索是否执行、整理后的答案、来源列表和错误信息的字典。
    """
    del kwargs

    if not isinstance(params, WebSearchParams):
        try:
            clean_dict = {
                str(key).strip().strip('"'): value
                for key, value in (params or {}).items()
            }
            params = WebSearchParams(**clean_dict)
        except Exception as e:
            log.error("创建 WebSearchParams 失败: %s", e)
            return {
                "search_executed": False,
                "fetch_executed": False,
                "error": f"参数格式不正确: {e}",
            }

    normalized_mode = str(params.mode or "search").strip().lower()
    if normalized_mode not in {"search", "fetch"}:
        normalized_mode = "search"

    clean_question = str(params.question or "").strip()
    clean_url = str(params.url or "").strip()

    if normalized_mode == "fetch":
        if not clean_url:
            return {
                "search_executed": False,
                "fetch_executed": False,
                "mode": "fetch",
                "error": "网页抓取模式需要提供 url。",
            }

        fetch_result = await _fetch_with_kimi(clean_url)
        fetch_answer = str(fetch_result.get("answer") or "").strip()

        result: Dict[str, Any] = {
            "search_executed": False,
            "fetch_executed": bool(fetch_result.get("fetch_executed")),
            "mode": "fetch",
            "query": clean_question or clean_url,
            "url": clean_url,
            "answer": fetch_answer,
            "sources": fetch_result.get("sources", []) or [],
            "channels": {
                "kimi_fetch": fetch_result,
            },
        }

        if fetch_result.get("error") and not fetch_answer:
            result["error"] = str(fetch_result.get("error"))
            if fetch_result.get("detail"):
                result["detail"] = fetch_result.get("detail")

        return result

    if not clean_question:
        return {
            "search_executed": False,
            "mode": "search",
            "error": "搜索问题不能为空。",
        }

    kimi_task = _search_with_kimi_pro(clean_question)
    grok_enabled = _is_grok_configured()

    if grok_enabled:
        kimi_result, grok_result = await asyncio.gather(
            kimi_task,
            _search_with_grok(clean_question),
        )
    else:
        kimi_result = await kimi_task
        grok_result = {
            "channel": "grok",
            "enabled": False,
            "skipped": True,
            "model": GROK_MODEL_NAME,
        }

    kimi_answer = str(kimi_result.get("answer") or "").strip()
    grok_answer = str(grok_result.get("answer") or "").strip()
    combined_answer = _format_combined_answer(
        kimi_answer=kimi_answer,
        grok_answer=grok_answer,
    )

    merged_sources_input: List[Tuple[str, str]] = []
    for channel_result in [kimi_result, grok_result]:
        for source in channel_result.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            merged_sources_input.append(
                (
                    str(source.get("title") or "").strip(),
                    str(source.get("url") or "").strip(),
                )
            )

    warnings: List[str] = []
    if grok_enabled and grok_result.get("error"):
        warnings.append("Grok 通道失败，已自动忽略。")

    result = {
        "search_executed": bool(kimi_result.get("search_executed")),
        "mode": "search",
        "query": clean_question,
        "model": "kimi-search-pro",
        "models": [
            "kimi-search-pro",
            *([GROK_MODEL_NAME] if grok_enabled else []),
        ],
        "answer": combined_answer or kimi_answer or grok_answer,
        "sources": _dedupe_sources(merged_sources_input),
        "channels": {
            "kimi": kimi_result,
            "grok": grok_result,
        },
    }

    if warnings:
        result["warnings"] = warnings

    if kimi_result.get("error") and not grok_answer:
        result["error"] = str(kimi_result.get("error"))
        if kimi_result.get("detail"):
            result["detail"] = kimi_result.get("detail")
    elif not combined_answer:
        result["error"] = "联网搜索未返回可用内容。"
    elif not kimi_result.get("search_executed"):
        result["error"] = str(
            kimi_result.get("error") or "Kimi 通道本次未实际执行联网搜索。"
        )

    return result
