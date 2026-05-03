# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import httpx

from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

SEARCH_API_BASE_URL = "https://ai-gateway.vercel.sh/v1"
SEARCH_API_URL = f"{SEARCH_API_BASE_URL}/responses"
SEARCH_MODEL_NAME = "openai/gpt-5.4-mini"
SEARCH_TIMEOUT_SECONDS = 25.0
SEARCH_INCLUDE_FIELDS = ["web_search_call.action.sources"]
GROK_MODEL_NAME = "grok-4.20-fast"
GROK_MAX_RETRIES = 3
GROK_TOTAL_TIMEOUT_SECONDS = 25.0
SEARCH_TOOL_INSTRUCTIONS = """
你是一个联网检索助手。

要求：
1. 必须先使用 web_search 工具，再回答用户问题。
2. 回答语言使用简体中文。
3. 输出必须整理清楚，按以下结构组织：
   - 摘要
   - 关键信息
   - 参考来源
4. 如果搜索结果存在时效性或来源冲突，要明确说明。
5. 不要编造来源；只引用实际搜索到的网址。
""".strip()
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


def _extract_text_from_output_item(item: Dict[str, Any]) -> str:
    if item.get("type") != "message":
        return ""

    chunks: List[str] = []
    for content_item in item.get("content", []) or []:
        if not isinstance(content_item, dict):
            continue
        if content_item.get("type") == "output_text":
            text = str(content_item.get("text") or "").strip()
            if text:
                chunks.append(text)

    return "\n".join(chunks).strip()


def _extract_sources_from_annotations(
    annotations: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    extracted: List[Tuple[str, str]] = []
    for annotation in annotations or []:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("type") != "url_citation":
            continue
        url = str(annotation.get("url") or "").strip()
        title = str(annotation.get("title") or url or "Untitled").strip()
        if url:
            extracted.append((title, url))
    return extracted


def _extract_sources_from_response(data: Dict[str, Any]) -> List[Dict[str, str]]:
    sources: List[Tuple[str, str]] = []

    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "web_search_call":
            action = item.get("action")
            if isinstance(action, dict):
                for source in action.get("sources", []) or []:
                    if not isinstance(source, dict):
                        continue
                    url = str(source.get("url") or "").strip()
                    title = str(source.get("title") or url or "Untitled").strip()
                    if url:
                        sources.append((title, url))

        for content_item in item.get("content", []) or []:
            if not isinstance(content_item, dict):
                continue
            sources.extend(
                _extract_sources_from_annotations(content_item.get("annotations", []) or [])
            )

    return _dedupe_sources(sources)


def _extract_answer_text(data: Dict[str, Any]) -> str:
    output_text = str(data.get("output_text") or "").strip()
    if output_text:
        return output_text

    chunks: List[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        text = _extract_text_from_output_item(item)
        if text:
            chunks.append(text)

    return "\n\n".join(chunks).strip()


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


def _has_web_search_call(data: Dict[str, Any]) -> bool:
    for item in data.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            return True
    return False


def _is_grok_configured() -> bool:
    grok_url = str(os.getenv("GROK_URL") or "").strip()
    grok_api_key = str(os.getenv("GROK_API_KEY") or "").strip()
    return bool(grok_url and grok_api_key)


def _format_combined_answer(
    *,
    gpt_answer: str,
    grok_answer: str,
) -> str:
    sections: List[str] = []

    if gpt_answer:
        sections.append(f"【GPT 联网搜索结果】\n{gpt_answer}")

    if grok_answer:
        sections.append(f"【Grok 辅助结果】\n{grok_answer}")

    return "\n\n".join(section for section in sections if section).strip()


async def _search_with_gpt(clean_question: str) -> Dict[str, Any]:
    api_key = str(os.getenv("SEARCH_API_KEY") or "").strip()
    if not api_key:
        log.warning("[WebSearchTool] SEARCH_API_KEY 未配置。")
        return {
            "channel": "gpt",
            "search_executed": False,
            "error": "未配置 SEARCH_API_KEY，无法执行联网搜索。",
        }

    payload: Dict[str, Any] = {
        "model": SEARCH_MODEL_NAME,
        "instructions": SEARCH_TOOL_INSTRUCTIONS,
        "input": clean_question,
        "tools": [
            {
                "type": "web_search",
                "external_web_access": True,
            }
        ],
        "tool_choice": "required",
        "include": list(SEARCH_INCLUDE_FIELDS),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_SECONDS) as client:
            response = await client.post(
                SEARCH_API_URL,
                headers=headers,
                json=payload,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body_preview = exc.response.text[:2000] if exc.response is not None else ""
        log.error(
            "[WebSearchTool][GPT] HTTP 请求失败 | status=%s | body=%s",
            exc.response.status_code if exc.response is not None else "N/A",
            body_preview,
        )
        return {
            "channel": "gpt",
            "search_executed": False,
            "error": f"联网搜索请求失败：HTTP {exc.response.status_code if exc.response is not None else 'N/A'}。",
            "detail": body_preview or str(exc),
        }
    except httpx.RequestError as exc:
        log.error("[WebSearchTool][GPT] 网络请求异常: %s", exc, exc_info=True)
        return {
            "channel": "gpt",
            "search_executed": False,
            "error": f"联网搜索网络异常：{type(exc).__name__}。",
            "detail": str(exc),
        }

    try:
        data = response.json()
    except json.JSONDecodeError:
        body_preview = response.text[:2000] if response.text else "<empty>"
        log.error("[WebSearchTool][GPT] 响应不是 JSON: %s", body_preview)
        return {
            "channel": "gpt",
            "search_executed": False,
            "error": "联网搜索返回了非 JSON 响应。",
            "detail": body_preview,
        }

    answer_text = _extract_answer_text(data)
    sources = _extract_sources_from_response(data)
    search_executed = _has_web_search_call(data)

    result: Dict[str, Any] = {
        "channel": "gpt",
        "search_executed": search_executed,
        "model": SEARCH_MODEL_NAME,
        "answer": answer_text,
        "sources": sources,
    }

    if not search_executed:
        result["error"] = "模型本次响应中未实际触发 web_search 工具。"
    elif not answer_text:
        result["error"] = "联网搜索已执行，但未解析出整理后的正文。"

    return result


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
            timeout_seconds = min(SEARCH_TIMEOUT_SECONDS, remaining_seconds)
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


@tool_metadata(
    name="联网搜索",
    description="当用户明确要求上网搜、联网搜、搜索最新网页信息时，使用此工具联网检索并整理结果。",
    emoji="🌐",
    category="工具",
)
async def search_web(question: str, **kwargs) -> Dict[str, Any]:
    """
    [工具说明]
    这是一个专门用于联网搜索网页信息的工具。

    [调用规则 - 高优先级]
    - 当用户明确提到“上网搜”、“联网搜”、“网上查”、“去网上搜一下”、“帮我搜最新信息”、“看一下网页/链接内容”时，你必须调用此工具。
    - 当问题明显依赖最新网页信息时，应优先考虑此工具，而不是直接凭记忆回答。
    - 如果没有调用此工具，就不要假装自己已经上网搜索过。
    - 传入参数时，只需要把“要搜索的问题”本身传给 `question`，不要夹带多余解释。

    [结果使用规则]
    - 工具返回的 `answer` 是已经整理好的搜索结果，可以直接基于它进行回复。
    - 工具返回的 `sources` 是实际搜索到的网址列表；如果需要引用来源，应优先使用这些链接。
    - 如果工具返回 `search_executed=false` 或 `error`，要如实告诉用户本次联网搜索失败，不要编造结果。

    Args:
        question (str): 需要联网搜索的问题，例如“上网搜一下 ds2api 项目是做什么的”里真正要搜索的那部分问题。

    Returns:
        一个包含搜索是否执行、整理后的答案、来源列表和错误信息的字典。
    """
    del kwargs

    clean_question = str(question or "").strip()
    if not clean_question:
        return {
            "search_executed": False,
            "error": "搜索问题不能为空。",
        }

    gpt_task = _search_with_gpt(clean_question)
    grok_enabled = _is_grok_configured()

    if grok_enabled:
        gpt_result, grok_result = await asyncio.gather(
            gpt_task,
            _search_with_grok(clean_question),
        )
    else:
        gpt_result = await gpt_task
        grok_result = {
            "channel": "grok",
            "enabled": False,
            "skipped": True,
            "model": GROK_MODEL_NAME,
        }

    gpt_answer = str(gpt_result.get("answer") or "").strip()
    grok_answer = str(grok_result.get("answer") or "").strip()
    combined_answer = _format_combined_answer(
        gpt_answer=gpt_answer,
        grok_answer=grok_answer,
    )

    merged_sources_input: List[Tuple[str, str]] = []
    for channel_result in [gpt_result, grok_result]:
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

    result: Dict[str, Any] = {
        "search_executed": bool(gpt_result.get("search_executed")),
        "query": clean_question,
        "model": SEARCH_MODEL_NAME,
        "models": [
            SEARCH_MODEL_NAME,
            *([GROK_MODEL_NAME] if grok_enabled else []),
        ],
        "answer": combined_answer or gpt_answer or grok_answer,
        "sources": _dedupe_sources(merged_sources_input),
        "channels": {
            "gpt": gpt_result,
            "grok": grok_result,
        },
    }

    if warnings:
        result["warnings"] = warnings

    if gpt_result.get("error") and not grok_answer:
        result["error"] = str(gpt_result.get("error"))
        if gpt_result.get("detail"):
            result["detail"] = gpt_result.get("detail")
    elif not combined_answer:
        result["error"] = "联网搜索未返回可用内容。"
    elif not gpt_result.get("search_executed"):
        result["error"] = str(
            gpt_result.get("error") or "GPT 通道本次未实际触发 web_search 工具。"
        )

    return result
