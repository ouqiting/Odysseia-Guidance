# -*- coding: utf-8 -*-

"""Google Vertex AI (快速模式 / Express Mode) 服务实现。

Vertex AI Express 模式实现：
- 使用 API Key 鉴权（快速模式，通过 URL query 参数 `?key={api_key}` 或 header 鉴权）；
- 端点 Base URL 计算规则：
  - 未配置 project_id 时，使用全局端点 https://aiplatform.googleapis.com/v1
  - 配置了 project_id 时，按 region 路由：
    - global: https://aiplatform.googleapis.com
    - us / eu: https://aiplatform.{region}.rep.googleapis.com
    - 其它（如 us-central1）: https://{region}-aiplatform.googleapis.com
    拼接为：{host}/v1/projects/{project_id}/locations/{region}
- 端点 Path：/publishers/google/models/{model}:{method}
- 完美适配 GeminiService 的调用流程（工具调用、多模态、思考链、上下文历史）。
"""

import asyncio
import base64
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

from src.chat.config import chat_config as app_config

log = logging.getLogger(__name__)

# 上游可重试的状态码
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class VertexFunctionCall:
    """与 google.genai.types.FunctionCall 兼容的函数调用对象。"""

    def __init__(self, name: str, args: Optional[Dict[str, Any]] = None):
        self.name = name
        self.args = args or {}

    def __repr__(self) -> str:
        return f"VertexFunctionCall(name={self.name!r}, args={self.args!r})"


class VertexPart:
    """与 google.genai.types.Part 兼容的 Part 对象。"""

    def __init__(
        self,
        text: Optional[str] = None,
        thought: bool = False,
        inline_data: Optional[Any] = None,
        function_call: Optional[VertexFunctionCall] = None,
        raw_part: Optional[Dict[str, Any]] = None,
    ):
        self.text = text or ""
        self.thought = thought
        self.inline_data = inline_data
        self.function_call = function_call
        self.raw_part = raw_part or {}

    def __repr__(self) -> str:
        if self.thought:
            return f"<VertexPart: thought={self.text[:30]!r}>"
        if self.function_call:
            return f"<VertexPart: function_call={self.function_call.name}>"
        return f"<VertexPart: text={self.text[:30]!r}>"


class VertexContent:
    """与 google.genai.types.Content 兼容的 Content 对象。"""

    def __init__(self, role: str = "model", parts: Optional[List[VertexPart]] = None):
        self.role = role
        self.parts = parts or []

    def __repr__(self) -> str:
        return f"VertexContent(role={self.role!r}, parts_count={len(self.parts)})"


class VertexCandidate:
    """与 google.genai.types.Candidate 兼容的 Candidate 对象。"""

    def __init__(
        self,
        content: Optional[VertexContent] = None,
        finish_reason: Optional[str] = None,
        safety_ratings: Optional[List[Any]] = None,
    ):
        self.content = content or VertexContent()
        self.finish_reason = finish_reason or "STOP"
        self.safety_ratings = safety_ratings or []

    def __repr__(self) -> str:
        return f"VertexCandidate(finish_reason={self.finish_reason!r})"


class VertexPromptFeedback:
    """与 google.genai.types.PromptFeedback 兼容的 PromptFeedback 对象。"""

    def __init__(self, block_reason: Optional[str] = None):
        self.block_reason = block_reason

    def __repr__(self) -> str:
        return f"VertexPromptFeedback(block_reason={self.block_reason!r})"


class VertexUsageMetadata:
    """与 google.genai.types.UsageMetadata 兼容的 Token 用量对象。"""

    def __init__(
        self,
        prompt_token_count: int = 0,
        candidates_token_count: int = 0,
        total_token_count: int = 0,
    ):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


class VertexGenerateContentResponse:
    """与 google.genai.types.GenerateContentResponse 兼容的统一响应对象。"""

    def __init__(
        self,
        candidates: Optional[List[VertexCandidate]] = None,
        prompt_feedback: Optional[VertexPromptFeedback] = None,
        usage_metadata: Optional[VertexUsageMetadata] = None,
    ):
        self.candidates = candidates or []
        self.prompt_feedback = prompt_feedback
        self.usage_metadata = usage_metadata or VertexUsageMetadata()

    @property
    def parts(self) -> List[VertexPart]:
        """返回首个 candidate 的所有 parts。"""
        if self.candidates and self.candidates[0].content:
            return self.candidates[0].content.parts
        return []

    @property
    def text(self) -> str:
        """返回非思考过程的正文回复拼接字符串。"""
        text_parts = [
            part.text
            for part in self.parts
            if not getattr(part, "thought", False) and part.text
        ]
        return "".join(text_parts).strip()

    @property
    def function_calls(self) -> List[VertexFunctionCall]:
        """提取首个 candidate 中的所有函数调用对象。"""
        calls = []
        for part in self.parts:
            if getattr(part, "function_call", None):
                calls.append(part.function_call)
        return calls

    def __repr__(self) -> str:
        return f"<VertexGenerateContentResponse: text={self.text[:40]!r}, candidates={len(self.candidates)}>"


class VertexAIOModels:
    """异步接口命名空间，提供与 client.aio.models.generate_content 一致的调用入口。"""

    def __init__(self, client: "VertexGeminiClient"):
        self._client = client

    async def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[Any] = None,
    ) -> VertexGenerateContentResponse:
        return await self._client.generate_content_async(
            model=model, contents=contents, config=config
        )


class VertexAIO:
    """提供 client.aio 异步命名空间。"""

    def __init__(self, client: "VertexGeminiClient"):
        self.models = VertexAIOModels(client)
        self._client = client

    async def aclose(self) -> None:
        await self._client.close_async()


class VertexSyncModels:
    """同步接口命名空间，提供与 client.models.generate_content 一致的调用入口。"""

    def __init__(self, client: "VertexGeminiClient"):
        self._client = client

    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[Any] = None,
    ) -> VertexGenerateContentResponse:
        return asyncio.run(
            self._client.generate_content_async(
                model=model, contents=contents, config=config
            )
        )


class VertexGeminiClient:
    """Vertex AI 快速模式 (Express Mode) 客户端。

    完全匹配 GeminiService 的预期接口（client.aio.models.generate_content）。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        override_base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = (
            api_key
            if api_key is not None
            else app_config.VERTEX_CONFIG.get("API_KEY", "")
        )
        self.project_id = (
            project_id
            if project_id is not None
            else app_config.VERTEX_CONFIG.get("PROJECT_ID", "")
        )
        self.region = (
            region
            if region is not None
            else app_config.VERTEX_CONFIG.get("REGION", "global")
        )
        self.override_base_url = override_base_url
        self.timeout = timeout

        self.aio = VertexAIO(self)
        self.models = VertexSyncModels(self)
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_base_url(self) -> str:
        """获取或计算 Vertex AI Base URL。"""
        if self.override_base_url:
            return self.override_base_url.rstrip("/")
        return app_config.get_vertexai_base_url(
            region=self.region, project_id=self.project_id
        )

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close_async(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._http_client.aclose())
                else:
                    loop.run_until_complete(self._http_client.aclose())
            except Exception:
                pass

    @classmethod
    def _serialize_part(cls, part: Any) -> Dict[str, Any]:
        """将单个 part 对象转换为 Vertex AI REST 请求体字典。"""
        if isinstance(part, str):
            return {"text": part}
        if isinstance(part, dict):
            return part

        # 检查是否是 Google GenAI / 自定义 Part
        result: Dict[str, Any] = {}

        # 1. 文本
        text = getattr(part, "text", None)
        if text:
            result["text"] = str(text)

        # 2. 内联数据 (Blob / 图片)
        inline_data = getattr(part, "inline_data", None)
        if inline_data:
            mime_type = getattr(inline_data, "mime_type", "image/png")
            data = getattr(inline_data, "data", b"")
            if isinstance(data, (bytes, bytearray)):
                b64_str = base64.b64encode(bytes(data)).decode("utf-8")
            else:
                b64_str = str(data)
            result["inlineData"] = {"mimeType": mime_type, "data": b64_str}

        # 3. 函数调用 (FunctionCall)
        function_call = getattr(part, "function_call", None)
        if function_call:
            name = getattr(function_call, "name", "")
            args = getattr(function_call, "args", {})
            if hasattr(args, "items"):
                args_dict = dict(args)
            elif isinstance(args, dict):
                args_dict = args
            else:
                args_dict = {}
            result["functionCall"] = {"name": name, "args": args_dict}

        # 4. 函数响应 (FunctionResponse)
        function_response = getattr(part, "function_response", None)
        if function_response:
            name = getattr(function_response, "name", "")
            resp = getattr(function_response, "response", {})
            if isinstance(resp, dict):
                resp_dict = resp
            elif hasattr(resp, "items"):
                resp_dict = dict(resp)
            else:
                resp_dict = {"result": resp}
            result["functionResponse"] = {"name": name, "response": resp_dict}

        # 5. 文件引用 (FileData)
        file_data = getattr(part, "file_data", None)
        if file_data:
            file_uri = getattr(file_data, "file_uri", "")
            mime_type = getattr(file_data, "mime_type", "")
            result["fileData"] = {"fileUri": file_uri, "mimeType": mime_type}

        return result if result else {"text": str(part)}

    @classmethod
    def _serialize_contents(cls, contents: Any) -> List[Dict[str, Any]]:
        """将 contents（可能为 Content 列表或单项）序列化为 REST API 结构。"""
        if not contents:
            return []
        if not isinstance(contents, list):
            contents = [contents]

        serialized_contents: List[Dict[str, Any]] = []
        for item in contents:
            if isinstance(item, str):
                serialized_contents.append({"role": "user", "parts": [{"text": item}]})
                continue
            if isinstance(item, dict):
                serialized_contents.append(item)
                continue

            role = getattr(item, "role", "user") or "user"
            raw_parts = getattr(item, "parts", []) or []
            serialized_parts = [cls._serialize_part(p) for p in raw_parts]
            serialized_contents.append({"role": role, "parts": serialized_parts})

        return serialized_contents

    @classmethod
    def _convert_callable_to_function_declaration(
        cls, func: Callable
    ) -> Dict[str, Any]:
        """将 Python 函数转为 Gemini FunctionDeclaration 结构。"""
        name = getattr(func, "__name__", "custom_tool")
        doc = inspect.getdoc(func) or ""

        # 尝试分析签名构建参数
        properties = {}
        required = []
        try:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation in (dict, Dict):
                    param_type = "object"
                elif param.annotation in (list, List):
                    param_type = "array"

                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}",
                }
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
        except Exception:
            pass

        return {
            "name": name,
            "description": doc[:1024] if doc else f"Tool {name}",
            "parameters": {
                "type": "OBJECT",
                "properties": properties,
                "required": required,
            },
        }

    @classmethod
    def _serialize_config(
        cls, config: Any
    ) -> Tuple[
        Dict[str, Any],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """将 GenerateContentConfig 序列化为 generationConfig, safetySettings, tools, systemInstruction。"""
        gen_config: Dict[str, Any] = {}
        safety_settings: List[Dict[str, Any]] = []
        tools: List[Dict[str, Any]] = []
        system_instruction: Optional[Dict[str, Any]] = None

        if not config:
            return gen_config, safety_settings, tools, system_instruction

        if isinstance(config, dict):
            # 字典形式提取
            for key in ("temperature", "top_p", "top_k", "max_output_tokens"):
                val = config.get(key)
                if val is not None:
                    # 驼峰命名转换
                    camel_map = {
                        "temperature": "temperature",
                        "top_p": "topP",
                        "top_k": "topK",
                        "max_output_tokens": "maxOutputTokens",
                    }
                    gen_config[camel_map[key]] = val

            thinking_cfg = config.get("thinking_config") or config.get("thinkingConfig")
            if isinstance(thinking_cfg, dict):
                t_obj: Dict[str, Any] = {}
                if "thinking_budget" in thinking_cfg:
                    t_obj["thinkingBudget"] = thinking_cfg["thinking_budget"]
                elif "thinkingBudget" in thinking_cfg:
                    t_obj["thinkingBudget"] = thinking_cfg["thinkingBudget"]

                if "thinking_level" in thinking_cfg:
                    t_obj["thinkingLevel"] = thinking_cfg["thinking_level"]
                elif "thinkingLevel" in thinking_cfg:
                    t_obj["thinkingLevel"] = thinking_cfg["thinkingLevel"]

                if "include_thoughts" in thinking_cfg:
                    t_obj["includeThoughts"] = thinking_cfg["include_thoughts"]
                elif "includeThoughts" in thinking_cfg:
                    t_obj["includeThoughts"] = thinking_cfg["includeThoughts"]

                if t_obj:
                    gen_config["thinkingConfig"] = t_obj

            return gen_config, safety_settings, tools, system_instruction

        # 对象形式提取 (google.genai.types.GenerateContentConfig)
        for attr, camel in [
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("top_k", "topK"),
            ("max_output_tokens", "maxOutputTokens"),
        ]:
            val = getattr(config, attr, None)
            if val is not None:
                gen_config[camel] = val

        # thinking_config
        thinking_cfg = getattr(config, "thinking_config", None)
        if thinking_cfg is not None:
            t_obj: Dict[str, Any] = {}
            budget = getattr(thinking_cfg, "thinking_budget", None)
            if budget is not None:
                t_obj["thinkingBudget"] = budget
            level = getattr(thinking_cfg, "thinking_level", None)
            if level is not None:
                t_obj["thinkingLevel"] = str(level)
            include_thoughts = getattr(thinking_cfg, "include_thoughts", None)
            if include_thoughts is not None:
                t_obj["includeThoughts"] = bool(include_thoughts)
            if t_obj:
                gen_config["thinkingConfig"] = t_obj

        # safety_settings
        raw_safety = getattr(config, "safety_settings", None)
        if raw_safety and isinstance(raw_safety, list):
            for s in raw_safety:
                cat = getattr(s, "category", None)
                cat_str = cat.name if hasattr(cat, "name") else str(cat or "")
                thresh = getattr(s, "threshold", None)
                thresh_str = (
                    thresh.name if hasattr(thresh, "name") else str(thresh or "")
                )
                if cat_str:
                    safety_settings.append(
                        {"category": cat_str, "threshold": thresh_str or "BLOCK_NONE"}
                    )

        # tools
        raw_tools = getattr(config, "tools", None)
        if raw_tools and isinstance(raw_tools, list):
            func_decls = []
            for t in raw_tools:
                if callable(t):
                    func_decls.append(cls._convert_callable_to_function_declaration(t))
                elif hasattr(t, "function_declarations") and t.function_declarations:
                    for fd in t.function_declarations:
                        func_decls.append(cls._serialize_part(fd))
                elif isinstance(t, dict):
                    tools.append(t)
            if func_decls:
                tools.append({"functionDeclarations": func_decls})

        # system_instruction
        raw_sys = getattr(config, "system_instruction", None)
        if raw_sys:
            if isinstance(raw_sys, str):
                system_instruction = {"parts": [{"text": raw_sys}]}
            else:
                system_instruction = {"parts": [cls._serialize_part(raw_sys)]}

        return gen_config, safety_settings, tools, system_instruction

    @classmethod
    def _parse_response_json(
        cls, response_json: Dict[str, Any]
    ) -> VertexGenerateContentResponse:
        """解析 Vertex AI 返回的原始 JSON 并包装为 VertexGenerateContentResponse。"""
        # 优先尝试使用官方 types 反序列化（若安装且非 mock）
        try:
            from google.genai import types as genai_types

            resp_cls = getattr(genai_types, "GenerateContentResponse", None)
            if isinstance(resp_cls, type) and hasattr(resp_cls, "model_validate"):
                return resp_cls.model_validate(response_json)
        except Exception:
            pass

        # 使用自研兼容对象反序列化
        raw_candidates = response_json.get("candidates") or []
        candidates: List[VertexCandidate] = []
        for rc in raw_candidates:
            raw_content = rc.get("content") or {}
            role = raw_content.get("role", "model")
            raw_parts = raw_content.get("parts") or []

            parts: List[VertexPart] = []
            for rp in raw_parts:
                text = rp.get("text", "")
                thought = bool(rp.get("thought", False))
                inline_data = rp.get("inlineData")
                function_call = None
                if "functionCall" in rp:
                    fc = rp["functionCall"]
                    function_call = VertexFunctionCall(
                        name=fc.get("name", ""),
                        args=fc.get("args") or {},
                    )
                parts.append(
                    VertexPart(
                        text=text,
                        thought=thought,
                        inline_data=inline_data,
                        function_call=function_call,
                        raw_part=rp,
                    )
                )

            finish_reason = rc.get("finishReason", "STOP")
            safety_ratings = rc.get("safetyRatings", [])
            candidates.append(
                VertexCandidate(
                    content=VertexContent(role=role, parts=parts),
                    finish_reason=finish_reason,
                    safety_ratings=safety_ratings,
                )
            )

        prompt_feedback = None
        if "promptFeedback" in response_json:
            pf = response_json["promptFeedback"]
            prompt_feedback = VertexPromptFeedback(
                block_reason=pf.get("blockReason"),
            )

        usage_metadata = VertexUsageMetadata()
        if "usageMetadata" in response_json:
            um = response_json["usageMetadata"]
            usage_metadata = VertexUsageMetadata(
                prompt_token_count=um.get("promptTokenCount", 0),
                candidates_token_count=um.get("candidatesTokenCount", 0),
                total_token_count=um.get("totalTokenCount", 0),
            )

        return VertexGenerateContentResponse(
            candidates=candidates,
            prompt_feedback=prompt_feedback,
            usage_metadata=usage_metadata,
        )

    async def generate_content_async(
        self,
        model: str,
        contents: Any,
        config: Optional[Any] = None,
    ) -> VertexGenerateContentResponse:
        """发送 Vertex AI 生成请求。

        - 自动进行快速模式的 URL 拼装与 API Key 参数注入；
        - 支持 429/503 可重试状态码重试。
        """
        if not self.api_key:
            raise ValueError(
                "未配置 Vertex AI API Key。请在 .env 中设置 VERTEX_API_KEY。"
            )

        base_url = self._get_base_url()
        endpoint_url = f"{base_url}/publishers/google/models/{model}:generateContent"

        # 序列化 payload
        gen_cfg, safety_cfg, tools_cfg, sys_inst = self._serialize_config(config)
        payload: Dict[str, Any] = {
            "contents": self._serialize_contents(contents),
        }
        if gen_cfg:
            payload["generationConfig"] = gen_cfg
        if safety_cfg:
            payload["safetySettings"] = safety_cfg
        if tools_cfg:
            payload["tools"] = tools_cfg
        if sys_inst:
            payload["systemInstruction"] = sys_inst

        # 鉴权：快速模式使用 Query 参数 ?key={api_key}
        params = {"key": self.api_key}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        client = self._get_http_client()

        max_attempts = 2
        last_error = None
        for attempt in range(max_attempts):
            try:
                log.info(
                    f"正在调用 Vertex AI 快速模式端点: {endpoint_url} (模型: {model}, 尝试 {attempt + 1}/{max_attempts})"
                )
                resp = await client.post(
                    endpoint_url,
                    params=params,
                    headers=headers,
                    json=payload,
                )

                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    error_msg = f"Vertex AI 返回可重试状态码 {resp.status_code}: {resp.text}"
                    log.warning(error_msg)
                    last_error = httpx.HTTPStatusError(
                        error_msg, request=resp.request, response=resp
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1.0)
                        continue
                    raise last_error

                resp.raise_for_status()
                response_json = resp.json()
                return self._parse_response_json(response_json)

            except httpx.HTTPStatusError as e:
                last_error = e
                log.error(
                    f"Vertex AI API 状态错误 ({e.response.status_code}): {e.response.text}",
                    exc_info=True,
                )
                raise
            except httpx.RequestError as e:
                last_error = e
                log.warning(f"Vertex AI 网络连接异常: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0)
                    continue
                raise

        raise last_error or RuntimeError("Vertex AI 请求未知失败")
