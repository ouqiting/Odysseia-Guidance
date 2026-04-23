# -*- coding: utf-8 -*-

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import src.chat.services.message_processor as message_processor_module
from src.chat.services.message_processor import MessageProcessor


class _FakeAttachment:
    def __init__(self, filename: str, content_type: str, payload: bytes):
        self.filename = filename
        self.content_type = content_type
        self._payload = payload
        self.size = len(payload)

    async def read(self) -> bytes:
        return self._payload


@pytest.mark.asyncio
async def test_extract_text_from_attachments_formats_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        message_processor_module.chat_config,
        "TEXT_ATTACHMENT_PROCESSING_CONFIG",
        {
            "MAX_TEXT_ATTACHMENTS_PER_MESSAGE": 5,
            "MAX_TEXT_ATTACHMENT_SIZE_MB": 1,
            "MAX_TEXT_ATTACHMENT_CHARS": 12000,
            "SUPPORTED_TEXT_MIME_TYPES": {"application/json"},
        },
    )

    processor = MessageProcessor()
    attachments = [
        _FakeAttachment(
            "payload.json",
            "application/json",
            b'{"name":"Odysseia","meta":{"enabled":true}}',
        )
    ]

    result = await processor._extract_text_from_attachments(attachments)

    assert len(result) == 1
    assert result[0]["filename"] == "payload.json"
    assert result[0]["mime_type"] == "application/json"
    assert '"name": "Odysseia"' in result[0]["content"]
    assert '"enabled": true' in result[0]["content"]
    assert result[0]["truncated"] is False
