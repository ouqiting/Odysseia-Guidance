import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.chat.services.openai_models.deepseek_model import DeepSeekModelClient
from src.chat.services.openai_models.kimi_model import KimiModelClient
from src.chat.services.prompt_service import PromptService


def test_build_raw_image_part_preserves_remote_urls():
    image_data = {
        "mime_type": "image/png",
        "data": b"png-bytes",
        "source": "attachment",
        "image_url": "https://cdn.discordapp.com/attachments/1/2/example.png",
        "proxy_url": "https://media.discordapp.net/attachments/1/2/example.png",
    }

    raw_part = PromptService._build_raw_image_part(image_data)

    assert raw_part is not None
    assert raw_part["image_url"] == image_data["image_url"]
    assert raw_part["proxy_url"] == image_data["proxy_url"]


def test_kimi_prefers_remote_image_url_over_base64():
    client = KimiModelClient()
    image_url = "https://cdn.discordapp.com/attachments/1/2/example.png"

    content_blocks = client.build_turn_content(
        [
            {
                "type": "image",
                "mime_type": "image/png",
                "data": b"png-bytes",
                "image_url": image_url,
            }
        ]
    )

    assert content_blocks == [{"type": "image_url", "image_url": {"url": image_url}}]


def test_kimi_falls_back_to_data_uri_when_remote_url_invalid():
    client = KimiModelClient()

    content_blocks = client.build_turn_content(
        [
            {
                "type": "image",
                "mime_type": "image/png",
                "data": b"png-bytes",
                "image_url": "not-a-url",
            }
        ]
    )

    assert len(content_blocks) == 1
    assert content_blocks[0]["type"] == "image_url"
    assert content_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_deepseek_prefers_remote_image_url_over_base64():
    client = DeepSeekModelClient()
    image_url = "https://cdn.discordapp.com/attachments/1/2/example.jpg"

    content_blocks = client.build_turn_content(
        [
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": b"jpeg-bytes",
                "image_url": image_url,
            }
        ]
    )

    assert content_blocks == [{"type": "image_url", "image_url": {"url": image_url}}]
