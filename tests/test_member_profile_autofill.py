from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

try:
    import src.chat.features.personal_memory.services.personal_memory_service as personal_memory_service_module
except ModuleNotFoundError as exc:
    if exc.name == "sqlalchemy":
        personal_memory_service_module = None
    else:
        raise


class _FakeBegin:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeWriteSession:
    def __init__(self):
        self.executed = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def begin(self):
        return _FakeBegin()

    async def execute(self, stmt):
        self.executed.append(stmt)
        return None


@pytest.mark.asyncio
async def test_maybe_autofill_member_profile_from_memory_accepts_auto_create_placeholder(
    monkeypatch: pytest.MonkeyPatch,
):
    if personal_memory_service_module is None:
        pytest.skip("sqlalchemy is not installed in the current environment.")

    service = personal_memory_service_module.PersonalMemoryService()
    infer_mock = AsyncMock(
        return_value={
            "personality": "做事细致，会主动确认关键信息。",
            "background": "与神社娘保持长期互动，并重视记忆同步。",
            "preferences": "偏好把重要信息及时写入长期记录。",
        }
    )
    monkeypatch.setattr(service, "_infer_member_profile_fields_from_memory", infer_mock)

    profile = SimpleNamespace(
        id=10,
        discord_id="123",
        title="Auto User",
        full_text=(
            "名称: Auto User\n"
            "Discord ID: 123\n"
            "性格特点: Auto User\n"
            "背景信息: \n"
            "喜好偏好: "
        ),
        source_metadata={
            "name": "Auto User",
            "discord_id": "123",
            "personality": "Auto User",
            "background": "",
            "preferences": "",
            "source": "auto_create",
        },
    )

    result = await service.maybe_autofill_member_profile_from_memory(
        user_id=123,
        profile=profile,
        dialogue_text="用户: 请记住重要信息\nAI: 好，我会整理",
        existing_summary="### 长期记忆\n- 用户重视记忆同步\n\n### 近期动态\n- 刚要求更新记忆",
        persist=False,
    )

    assert result["status"] == "preview"
    assert result["fields"]["personality"] == "做事细致，会主动确认关键信息。"
    assert "性格特点: 做事细致，会主动确认关键信息。" in result["full_text"]
    assert (
        result["source_metadata"]["preferences"]
        == "偏好把重要信息及时写入长期记录。"
    )
    infer_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_autofill_member_profile_from_memory_writes_profile_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    if personal_memory_service_module is None:
        pytest.skip("sqlalchemy is not installed in the current environment.")

    service = personal_memory_service_module.PersonalMemoryService()
    write_session = _FakeWriteSession()
    infer_mock = AsyncMock(
        return_value={
            "personality": "慢热，但会认真维护长期关系。",
            "background": "和神社娘已有稳定互动历史。",
            "preferences": "偏好明确约定与持续性记录。",
        }
    )

    monkeypatch.setattr(service, "_infer_member_profile_fields_from_memory", infer_mock)
    monkeypatch.setattr(
        personal_memory_service_module,
        "AsyncSessionLocal",
        lambda: write_session,
    )

    profile = SimpleNamespace(
        id=88,
        discord_id="456",
        title="Write User",
        full_text=(
            "名称: Write User\n"
            "Discord ID: 456\n"
            "性格特点: \n"
            "背景信息: \n"
            "喜好偏好: "
        ),
        source_metadata={
            "name": "Write User",
            "discord_id": "456",
            "personality": "",
            "background": "",
            "preferences": "",
            "source": "auto_create",
        },
    )

    result = await service.maybe_autofill_member_profile_from_memory(
        user_id=456,
        profile=profile,
        dialogue_text="用户: 以后重要内容要记住\nAI: 我会持续整理",
        existing_summary="",
        persist=True,
        reindex_in_background=False,
    )

    assert result["status"] == "updated"
    assert len(write_session.executed) == 1

    compiled_params = write_session.executed[0].compile().params
    assert compiled_params["title"] == "Write User"
    assert "性格特点: 慢热，但会认真维护长期关系。" in compiled_params["full_text"]
    assert compiled_params["source_metadata"]["background"] == "和神社娘已有稳定互动历史。"
