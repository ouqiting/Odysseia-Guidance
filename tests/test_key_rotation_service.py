import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.chat.services.key_rotation_service import (
    KeyRotationService,
    KeyStatus,
    NoAvailableKeyError,
)


@pytest.mark.asyncio
async def test_acquire_key_with_timeout_raises_when_all_keys_are_cooling():
    service = KeyRotationService(["test-key-1"])
    key_obj = next(iter(service.keys.values()))
    key_obj.status = KeyStatus.COOLING_DOWN
    key_obj.cooldown_until = time.time() + 60

    with pytest.raises(NoAvailableKeyError, match="当前无可用 Gemini API Key"):
        await service.acquire_key_with_timeout(
            wait_timeout_seconds=0.2,
            poll_interval_seconds=0.05,
        )
