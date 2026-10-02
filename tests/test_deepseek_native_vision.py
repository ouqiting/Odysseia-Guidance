import base64
import io
import unittest

from PIL import Image
from google.genai import types

from src.chat.services.openai_models.deepseek_model import DeepSeekModelClient


class TestDeepSeekNativeVision(unittest.TestCase):
    """deepseek-flash 图片直传（原生多模态）；deepseek-v4-pro 仍走 Moonshot 转述。"""

    def setUp(self):
        self.client = DeepSeekModelClient()
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
        self.png_bytes = buffer.getvalue()

    def test_supports_native_image_input_only_for_flash(self):
        self.assertTrue(self.client.supports_native_image_input("deepseek-flash"))
        self.assertFalse(self.client.supports_native_image_input("deepseek-v4-pro"))
        self.assertFalse(self.client.supports_native_image_input(None))

    def test_build_native_blocks_for_every_supported_part_type(self):
        parts = [
            "前缀文本",
            {"type": "image", "mime_type": "image/png", "data": self.png_bytes},
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "image_base64": base64.b64encode(b"jpeg-bytes").decode(),
            },
            {
                "type": "image",
                "mime_type": "image/png",
                "data_preview": self.png_bytes.hex(),
            },
            Image.open(io.BytesIO(self.png_bytes)),
            types.Part(inline_data=types.Blob(mime_type="image/png", data=self.png_bytes)),
        ]

        blocks = self.client.build_native_image_turn_content(parts)

        self.assertEqual(len(blocks), 6)
        self.assertEqual(blocks[0], {"type": "text", "text": "前缀文本"})
        for block in blocks[1:]:
            self.assertEqual(block["type"], "image_url")
            url = block["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/"))
            self.assertIn(";base64,", url)

    def test_unparseable_image_falls_back_to_text_note(self):
        blocks = self.client.build_native_image_turn_content(
            [{"type": "image", "mime_type": "image/png"}]
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")

    def test_extract_text_keeps_only_text_blocks(self):
        content = [
            {"type": "text", "text": "第一段"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}},
            {"type": "text", "text": "第二段"},
        ]
        self.assertEqual(
            self.client.extract_text_from_openai_content(content), "第一段\n第二段"
        )
        self.assertEqual(self.client.extract_text_from_openai_content(" 原文 "), "原文")

    def test_empty_parts_return_empty_blocks(self):
        self.assertEqual(self.client.build_native_image_turn_content([]), [])

    def test_gif_within_limit_is_passed_through_raw(self):
        gif_bytes = b"GIF89a" + b"\x00" * 1024
        blocks = self.client.build_native_image_turn_content(
            [
                {
                    "type": "image",
                    "mime_type": "image/gif",
                    "data": gif_bytes,
                    "source": "attachment",
                }
            ]
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "image_url")
        self.assertTrue(
            blocks[0]["image_url"]["url"].startswith("data:image/gif;base64,")
        )

    def test_oversized_gif_attachment_is_skipped(self):
        oversized = b"GIF89a" + b"\x00" * (8 * 1024 * 1024)
        blocks = self.client.build_native_image_turn_content(
            [
                {
                    "type": "image",
                    "mime_type": "image/gif",
                    "data": oversized,
                    "source": "attachment",
                }
            ]
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("体积超过限制", blocks[0]["text"])

    def test_oversized_emoji_gif_uses_stricter_limit(self):
        # 动态表情限制为 2MB：3MB 的 GIF 在 emoji 来源下应被跳过，作为附件则放行
        gif_bytes = b"GIF89a" + b"\x00" * (3 * 1024 * 1024)
        emoji_blocks = self.client.build_native_image_turn_content(
            [
                {
                    "type": "image",
                    "mime_type": "image/gif",
                    "data": gif_bytes,
                    "source": "emoji",
                }
            ]
        )
        attachment_blocks = self.client.build_native_image_turn_content(
            [
                {
                    "type": "image",
                    "mime_type": "image/gif",
                    "data": gif_bytes,
                    "source": "attachment",
                }
            ]
        )
        self.assertEqual(emoji_blocks[0]["type"], "text")
        self.assertEqual(attachment_blocks[0]["type"], "image_url")

    def test_oversized_gif_inline_data_is_skipped(self):
        oversized = b"GIF89a" + b"\x00" * (8 * 1024 * 1024)
        blocks = self.client.build_native_image_turn_content(
            [
                types.Part(
                    inline_data=types.Blob(mime_type="image/gif", data=oversized)
                )
            ]
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")


if __name__ == "__main__":
    unittest.main()
