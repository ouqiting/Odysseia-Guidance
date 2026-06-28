import json
import pathlib
import sys

import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.chat.services.openai_models import custom_model as custom_model_module
from src.chat.services.openai_models.custom_model import (
    CustomModelChannelError,
    CustomModelClient,
)


def _build_runtime_config(
    *,
    api_keys: list[str] | None = None,
    source_type: str = "file",
) -> dict:
    normalized_api_keys = api_keys or ["key-alpha", "key-beta"]
    serialized_api_keys = ",".join(normalized_api_keys)
    source_value = (
        "/app/data/CUSTOM_MODEL_API_KEY.json"
        if source_type == "file"
        else serialized_api_keys
    )
    return {
        "base_url": "https://example.com/v1",
        "api_key": serialized_api_keys,
        "api_keys": list(normalized_api_keys),
        "api_key_source_type": source_type,
        "api_key_source_value": source_value,
        "api_key_file_path": (
            "/app/data/CUSTOM_MODEL_API_KEY.json" if source_type == "file" else None
        ),
        "api_key_error": None,
        "model_name": "custom-test-model",
        "enable_vision": False,
        "enable_video_input": False,
        "gateway_provider_timeout_ms": 4000,
        "gateway_provider_name": "",
        "stream_idle_timeout_seconds": 5.0,
        "accept_encoding": "identity",
        "timeout_detection_enabled": False,
    }


def test_build_runtime_config_from_preset_settings_uses_preset_values(
    monkeypatch: pytest.MonkeyPatch,
):
    # 锁定当前 env 配置，确保预设构建不依赖 / 不修改 env
    monkeypatch.setenv("CUSTOM_MODEL_URL", "https://env.example.com/v1")
    monkeypatch.setenv("CUSTOM_MODEL_API_KEY", "env-key-1,env-key-2")
    monkeypatch.setenv("CUSTOM_MODEL_NAME", "env-model-name")
    monkeypatch.setenv("CUSTOM_MODEL_ENABLE_VISION", "false")
    monkeypatch.setenv("CUSTOM_MODEL_ENABLE_VIDEO_INPUT", "false")

    client = CustomModelClient()
    rc = client.build_runtime_config_from_preset_settings(
        custom_model_url="https://preset.example.com/v1",
        custom_model_api_key="preset-key-1,preset-key-2",
        custom_model_name="preset-model-name",
        custom_model_enable_vision="true",
        custom_model_enable_video_input="false",
    )

    assert rc["base_url"] == "https://preset.example.com/v1"
    assert rc["model_name"] == "preset-model-name"
    assert rc["api_keys"] == ["preset-key-1", "preset-key-2"]
    assert rc["api_key"] == "preset-key-1,preset-key-2"
    assert rc["enable_vision"] is True
    assert rc["enable_video_input"] is False
    assert rc["api_key_source_type"] == "inline"
    assert rc["api_key_file_path"] is None
    assert rc["api_key_error"] is None
    # 非预设字段回退使用 env 配置
    assert isinstance(rc["gateway_provider_timeout_ms"], int)
    assert isinstance(rc["stream_idle_timeout_seconds"], float)

    # 当前 env 配置未被修改
    env_rc = client.refresh_from_env()
    assert env_rc["base_url"] == "https://env.example.com/v1"
    assert env_rc["model_name"] == "env-model-name"
    assert env_rc["api_keys"] == ["env-key-1", "env-key-2"]


def test_build_runtime_config_from_preset_settings_supports_file_key_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("CUSTOM_MODEL_URL", "https://env.example.com/v1")
    monkeypatch.setenv("CUSTOM_MODEL_API_KEY", "env-key-1")
    monkeypatch.setenv("CUSTOM_MODEL_NAME", "env-model-name")

    key_file = tmp_path / "preset-keys.json"
    key_file.write_text(
        json.dumps({"api_keys": ["vck_alpha", "vck_beta"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.chat.utils.custom_model_api_keys._validate_custom_model_api_key_file_path",
        lambda raw_value: str(key_file) if raw_value == "/data/preset-keys.json" else raw_value,
    )

    client = CustomModelClient()
    rc = client.build_runtime_config_from_preset_settings(
        custom_model_url="https://preset.example.com/v1",
        custom_model_api_key="/data/preset-keys.json",
        custom_model_name="preset-model-name",
        custom_model_enable_vision="false",
        custom_model_enable_video_input="false",
    )

    assert rc["api_key_source_type"] == "file"
    assert rc["api_keys"] == ["vck_alpha", "vck_beta"]
    assert rc["api_key"] == "vck_alpha,vck_beta"
    assert rc["base_url"] == "https://preset.example.com/v1"


def _build_success_response(
    request: httpx.Request, *, content: str = "ok"
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        request=request,
    )


@pytest.mark.asyncio
async def test_custom_file_keys_delete_failed_key_on_403(tmp_path):
    client = CustomModelClient()
    runtime_config = _build_runtime_config()
    file_path = tmp_path / "CUSTOM_MODEL_API_KEY.json"
    runtime_config["api_key_file_path"] = str(file_path)

    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"message": "forbidden"}},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(forbidden_handler)
    ) as http_client:
        with pytest.raises(CustomModelChannelError) as exc_info:
            await client.send(
                http_client=http_client,
                payload={"model": "custom-test-model", "messages": []},
                runtime_config=runtime_config,
            )

    exc = exc_info.value
    assert exc.failure_kind == "custom_file_all_keys_exhausted"
    assert exc.api_key_rotation_count == 2
    assert exc.total_api_keys == 2
    persisted_payload = json.loads(file_path.read_text(encoding="utf-8"))
    assert persisted_payload == {"api_keys": []}


@pytest.mark.asyncio
async def test_custom_file_keys_move_failed_key_to_end_on_402(
    monkeypatch: pytest.MonkeyPatch,
):
    client = CustomModelClient()
    runtime_config = _build_runtime_config()
    persisted_orders = []

    monkeypatch.setattr(
        custom_model_module,
        "persist_custom_model_api_keys_to_file",
        lambda file_path, api_keys: persisted_orders.append(list(api_keys)),
    )

    def payment_required_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={"error": {"message": "payment required"}},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(payment_required_handler)
    ) as http_client:
        with pytest.raises(CustomModelChannelError) as exc_info:
            await client.send(
                http_client=http_client,
                payload={"model": "custom-test-model", "messages": []},
                runtime_config=runtime_config,
            )

    exc = exc_info.value
    assert exc.failure_kind == "custom_file_all_keys_exhausted"
    assert exc.api_key_rotation_count == 2
    assert exc.total_api_keys == 2
    assert persisted_orders == [
        ["key-beta", "key-alpha"],
        ["key-alpha", "key-beta"],
    ]


@pytest.mark.asyncio
async def test_custom_file_keys_move_failed_key_to_end_on_insufficient_funds_text(
    monkeypatch: pytest.MonkeyPatch,
):
    client = CustomModelClient()
    runtime_config = _build_runtime_config()
    persisted_orders = []

    monkeypatch.setattr(
        custom_model_module,
        "persist_custom_model_api_keys_to_file",
        lambda file_path, api_keys: persisted_orders.append(list(api_keys)),
    )

    def insufficient_funds_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Insufficient funds. Please add credits to your account to continue using AI services.",
                    "type": "insufficient_funds",
                }
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(insufficient_funds_handler)
    ) as http_client:
        with pytest.raises(CustomModelChannelError) as exc_info:
            await client.send(
                http_client=http_client,
                payload={"model": "custom-test-model", "messages": []},
                runtime_config=runtime_config,
            )

    exc = exc_info.value
    assert exc.failure_kind == "custom_file_all_keys_exhausted"
    assert exc.api_key_rotation_count == 2
    assert exc.total_api_keys == 2
    assert persisted_orders == [
        ["key-beta", "key-alpha"],
        ["key-alpha", "key-beta"],
    ]


@pytest.mark.asyncio
async def test_custom_file_keys_temporarily_skip_429_without_persisting_order(
    monkeypatch: pytest.MonkeyPatch,
):
    client = CustomModelClient()
    runtime_config = _build_runtime_config(
        api_keys=["key-alpha", "key-beta", "key-gamma"]
    )
    request_key_order = []
    persisted_orders = []
    alpha_attempts = 0

    monkeypatch.setattr(
        custom_model_module,
        "persist_custom_model_api_keys_to_file",
        lambda file_path, api_keys: persisted_orders.append(list(api_keys)),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal alpha_attempts
        api_key = request.headers["Authorization"].removeprefix("Bearer ").strip()
        request_key_order.append(api_key)

        if api_key == "key-alpha":
            alpha_attempts += 1
            return httpx.Response(
                429,
                json={"error": {"message": "Too Many Requests"}},
                request=request,
            )

        return _build_success_response(request, content=f"reply from {api_key}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        first_result = await client.send(
            http_client=http_client,
            payload={"model": "custom-test-model", "messages": []},
            runtime_config=runtime_config,
        )
        second_result = await client.send(
            http_client=http_client,
            payload={"model": "custom-test-model", "messages": []},
            runtime_config=runtime_config,
        )

    assert (
        first_result["response"].json()["choices"][0]["message"]["content"]
        == "reply from key-beta"
    )
    assert (
        second_result["response"].json()["choices"][0]["message"]["content"]
        == "reply from key-beta"
    )
    assert request_key_order == [
        "key-alpha",
        "key-beta",
        "key-alpha",
        "key-beta",
    ]
    assert alpha_attempts == 2
    assert persisted_orders == []


@pytest.mark.asyncio
async def test_custom_file_keys_keep_first_key_after_429_then_move_402_key_to_end(
    monkeypatch: pytest.MonkeyPatch,
):
    client = CustomModelClient()
    runtime_config = _build_runtime_config(
        api_keys=["key-alpha", "key-beta", "key-gamma"]
    )
    request_key_order = []
    persisted_orders = []
    alpha_attempts = 0
    beta_attempts = 0

    monkeypatch.setattr(
        custom_model_module,
        "persist_custom_model_api_keys_to_file",
        lambda file_path, api_keys: persisted_orders.append(list(api_keys)),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal alpha_attempts, beta_attempts
        api_key = request.headers["Authorization"].removeprefix("Bearer ").strip()
        request_key_order.append(api_key)

        if api_key == "key-alpha":
            alpha_attempts += 1
            if alpha_attempts == 1:
                return httpx.Response(
                    429,
                    json={"error": {"message": "Too Many Requests"}},
                    request=request,
                )
            return _build_success_response(request, content="reply from key-alpha")

        if api_key == "key-beta":
            beta_attempts += 1
            return httpx.Response(
                402,
                json={"error": {"message": "payment required"}},
                request=request,
            )

        return _build_success_response(request, content="reply from key-gamma")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        first_result = await client.send(
            http_client=http_client,
            payload={"model": "custom-test-model", "messages": []},
            runtime_config=runtime_config,
        )
        second_result = await client.send(
            http_client=http_client,
            payload={"model": "custom-test-model", "messages": []},
            runtime_config=runtime_config,
        )

    assert (
        first_result["response"].json()["choices"][0]["message"]["content"]
        == "reply from key-gamma"
    )
    assert (
        second_result["response"].json()["choices"][0]["message"]["content"]
        == "reply from key-alpha"
    )
    assert request_key_order == [
        "key-alpha",
        "key-beta",
        "key-gamma",
        "key-alpha",
    ]
    assert persisted_orders == [["key-alpha", "key-gamma", "key-beta"]]


@pytest.mark.asyncio
async def test_custom_inline_keys_raise_only_after_all_keys_hit_429():
    client = CustomModelClient()
    runtime_config = _build_runtime_config(
        api_keys=["key-alpha", "key-beta"],
        source_type="inline",
    )
    request_key_order = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.headers["Authorization"].removeprefix("Bearer ").strip()
        request_key_order.append(api_key)
        return httpx.Response(
            429,
            json={"error": {"message": "Too Many Requests"}},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        with pytest.raises(CustomModelChannelError) as exc_info:
            await client.send(
                http_client=http_client,
                payload={"model": "custom-test-model", "messages": []},
                runtime_config=runtime_config,
            )

    exc = exc_info.value
    assert exc.failure_kind == "custom_all_keys_rate_limited"
    assert exc.api_key_rotation_count == 2
    assert exc.total_api_keys == 2
    assert request_key_order == ["key-alpha", "key-beta"]


@pytest.mark.asyncio
async def test_custom_keys_do_not_rotate_or_delete_on_free_tier_403(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """命中 'Free tier users do not have access to this model' 时，
    不删除/不轮换任何 API Key，让原始 HTTP 403 异常向上抛出，便于
    上层展示 'Custom 连接失败: ... 详情: ...'。"""
    client = CustomModelClient()
    runtime_config = _build_runtime_config()
    file_path = tmp_path / "CUSTOM_MODEL_API_KEY.json"
    file_path.write_text(
        json.dumps({"api_keys": ["key-alpha", "key-beta"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    request_key_order = []
    persisted_orders = []

    monkeypatch.setattr(
        custom_model_module,
        "persist_custom_model_api_keys_to_file",
        lambda file_path, api_keys: persisted_orders.append(list(api_keys)),
    )

    def free_tier_handler(request: httpx.Request) -> httpx.Response:
        api_key = request.headers["Authorization"].removeprefix("Bearer ").strip()
        request_key_order.append(api_key)
        return httpx.Response(
            403,
            json={
                "error": {
                    "message": "Free tier users do not have access to this model. Upgrade to paid credits for unrestricted access.",
                    "type": "invalid_request_error",
                }
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(free_tier_handler)
    ) as http_client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.send(
                http_client=http_client,
                payload={"model": "custom-test-model", "messages": []},
                runtime_config=runtime_config,
            )

    exc = exc_info.value
    assert exc.response.status_code == 403
    assert "403" in str(exc)
    # 原始响应体保留完整上游报错，上层可据此展示 "Custom 连接失败: ... 详情: ..."
    assert "Free tier users do not have access to this model" in exc.response.text
    # 只请求了一次，没有轮换到其他 Key
    assert request_key_order == ["key-alpha"]
    # 没有删除或持久化任何 Key 顺序变更
    assert persisted_orders == []
    assert json.loads(file_path.read_text(encoding="utf-8")) == {
        "api_keys": ["key-alpha", "key-beta"]
    }
