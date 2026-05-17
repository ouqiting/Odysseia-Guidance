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


def _build_runtime_config() -> dict:
    return {
        "base_url": "https://example.com/v1",
        "api_key": "key-alpha,key-beta",
        "api_keys": ["key-alpha", "key-beta"],
        "api_key_source_type": "file",
        "api_key_source_value": "/app/data/CUSTOM_MODEL_API_KEY.json",
        "api_key_file_path": "/app/data/CUSTOM_MODEL_API_KEY.json",
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


@pytest.mark.asyncio
async def test_custom_file_keys_fail_after_all_rotations(monkeypatch: pytest.MonkeyPatch):
    client = CustomModelClient()
    runtime_config = _build_runtime_config()
    persisted_orders = []

    monkeypatch.setattr(
        custom_model_module,
        "persist_custom_model_api_keys_to_file",
        lambda file_path, api_keys: persisted_orders.append(list(api_keys)),
    )

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
    assert persisted_orders == [
        ["key-beta", "key-alpha"],
        ["key-alpha", "key-beta"],
    ]
