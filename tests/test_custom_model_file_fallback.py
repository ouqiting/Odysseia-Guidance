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
