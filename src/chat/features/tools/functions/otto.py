# -*- coding: utf-8 -*-

import asyncio
import io
import logging
import math
import re
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import discord
import yaml
from pydantic import BaseModel, Field
from pypinyin import Style, lazy_pinyin

from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VOICE_DIR = PROJECT_ROOT / "data" / "otto"
TOKENS_DIR = VOICE_DIR / "tokens"
PRESETS_DIR = VOICE_DIR / "ysddTokens"
PRESETS_FILE = VOICE_DIR / "presets.yml"

CHAR_PRESETS = {
    ".": "dian",
    "0": "ling",
    "1": "yi",
    "2": "er",
    "3": "san",
    "4": "si",
    "5": "wu",
    "6": "liu",
    "7": "qi",
    "8": "ba",
    "9": "jiu",
}

ASCII_PRESETS = {
    "a": ["ei"],
    "b": ["bi"],
    "c": ["xi"],
    "d": ["di"],
    "e": ["yi"],
    "f": ["ai", "fu"],
    "g": ["ji"],
    "h": ["ai", "chi"],
    "i": ["ai"],
    "j": ["zhei"],
    "k": ["ke", "ei"],
    "l": ["ai", "lu"],
    "m": ["ai", "mu"],
    "n": ["en"],
    "o": ["ou"],
    "p": ["pi"],
    "q": ["kiu"],
    "r": ["a"],
    "s": ["ai", "si"],
    "t": ["ti"],
    "u": ["you"],
    "v": ["wei"],
    "w": ["da", "bu", "liu"],
    "x": ["ai", "ke", "si"],
    "y": ["wai"],
    "z": ["zei"],
}


class OttoParams(BaseModel):
    text: str = Field(
        ...,
        description="要做成活字印刷音频的文本内容。支持中文、数字、英文字母。",
    )
    filename: str = Field(
        ...,
        description="音频文件名。请根据内容生成一个 10 字以内的标题，如“活字问候”“倒放整活”等。",
    )
    reverse: bool = Field(
        False,
        description="是否将整段生成后的音频整体倒放。true 为倒放，false 为正常播放。",
    )


@dataclass(frozen=True)
class Segment:
    kind: str
    value: str


class OttoEngine:
    def __init__(self, voice_dir: Path = VOICE_DIR) -> None:
        self.voice_dir = voice_dir
        self.tokens = self._scan_audio(TOKENS_DIR)
        self.presets = self._scan_audio(PRESETS_DIR)
        config = self._load_presets(PRESETS_FILE)
        self.original_presets = self._build_original_presets(config["presets"])
        self.pinyin_presets = self._build_pinyin_presets(config["presets"])

    @staticmethod
    def _scan_audio(directory: Path) -> dict[str, Path]:
        if not directory.exists():
            raise FileNotFoundError(f"音频目录不存在: {directory}")
        return {file.stem: file for file in directory.glob("*.wav")}

    @staticmethod
    def _load_presets(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"预设文件不存在: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        presets = data.get("presets") or {}
        return {"presets": presets}

    def _build_original_presets(self, presets: dict[str, list[str]]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for preset_id, aliases in presets.items():
            if preset_id not in self.presets:
                continue
            for alias in aliases:
                pairs.append((alias, preset_id))
        return sorted(pairs, key=lambda item: len(item[0]), reverse=True)

    def _build_pinyin_presets(self, presets: dict[str, list[str]]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for preset_id, aliases in presets.items():
            if preset_id not in self.presets:
                continue
            for alias in aliases:
                pairs.append((self._to_bracketed_pinyin(alias), preset_id))
        return sorted(pairs, key=lambda item: len(item[0]), reverse=True)

    def synthesize_bytes(self, text: str, reverse: bool = False) -> tuple[bytes, list[str]]:
        token_ids = self.parse(text)
        if not token_ids:
            raise ValueError("没有匹配到任何可拼接的音频片段。")
        files = [self._resolve_audio_file(token_id) for token_id in token_ids]
        return merge_wav_files(files, reverse=reverse), token_ids

    def parse(self, text: str) -> list[str]:
        cleaned = text.replace("\r", "").replace("\n", "").strip()
        prepared = self._prepare_original(cleaned)
        resolved: list[str] = []

        for segment in prepared:
            if segment.kind == "preset":
                resolved.append(segment.value)
                continue

            for item in self._tokenize(segment.value):
                if item.kind == "token":
                    resolved.append(item.value)
                elif item.kind == "ascii":
                    resolved.extend(self._resolve_ascii(item.value))
                else:
                    resolved.extend(self._resolve_chinese(item.value))

        return resolved[:11120]

    def _prepare_original(self, text: str) -> list[Segment]:
        segments: list[Segment] = [Segment("raw", text)]
        while True:
            replaced = False
            next_segments: list[Segment] = []
            for segment in segments:
                if segment.kind != "raw":
                    next_segments.append(segment)
                    continue

                matched = None
                for alias, preset_id in self.original_presets:
                    if alias in segment.value:
                        matched = (alias, preset_id)
                        break

                if not matched:
                    next_segments.append(segment)
                    continue

                alias, preset_id = matched
                before, _, after = segment.value.partition(alias)
                if before:
                    next_segments.append(Segment("raw", before))
                next_segments.append(Segment("preset", preset_id))
                if after:
                    next_segments.append(Segment("raw", after))
                replaced = True

            segments = next_segments
            if not replaced:
                return segments

    def _tokenize(self, text: str) -> list[Segment]:
        result: list[Segment] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                result.append(Segment("raw_chinese", "".join(current)))
                current.clear()

        for char in text:
            if char in CHAR_PRESETS and CHAR_PRESETS[char] in self.tokens:
                flush()
                result.append(Segment("token", CHAR_PRESETS[char]))
            elif char.lower() in ASCII_PRESETS:
                flush()
                result.append(Segment("ascii", char.lower()))
            elif is_chinese(char):
                current.append(char)
            else:
                flush()

        flush()
        return result

    def _resolve_ascii(self, char: str) -> list[str]:
        token_ids: list[str] = []
        for token_id in ASCII_PRESETS.get(char, []):
            if token_id in self.tokens:
                token_ids.append(token_id)
        return token_ids

    def _resolve_chinese(self, text: str) -> list[str]:
        pinyin_text = text if text.startswith("[") else self._to_bracketed_pinyin(text)
        for pinyin_alias, preset_id in self.pinyin_presets:
            index = pinyin_text.find(pinyin_alias)
            if index < 0:
                continue

            left = pinyin_text[:index].strip()
            right = pinyin_text[index + len(pinyin_alias):].strip()
            result: list[str] = []
            if left:
                result.extend(self._resolve_chinese(left))
            result.append(preset_id)
            if right:
                result.extend(self._resolve_chinese(right))
            return result

        token_ids: list[str] = []
        for piece in pinyin_text.split():
            token_id = piece[1:-1]
            if token_id in self.tokens:
                token_ids.append(token_id)
        return token_ids

    def _resolve_audio_file(self, token_id: str) -> Path:
        if token_id in self.tokens:
            return self.tokens[token_id]
        if token_id in self.presets:
            return self.presets[token_id]
        raise KeyError(f"找不到音频片段: {token_id}")

    @staticmethod
    def _to_bracketed_pinyin(text: str) -> str:
        syllables = lazy_pinyin(text, style=Style.NORMAL, strict=False, v_to_u=False)
        return " ".join(f"[{syllable.lower()}]" for syllable in syllables if syllable.strip())


def merge_wav_files(files: list[Path], reverse: bool = False) -> bytes:
    if not files:
        raise ValueError("没有可合并的 wav 文件。")

    sample_rate = 44100
    combined_samples: list[float] = []
    for file in files:
        samples, source_rate = read_wav_as_mono(file)
        if source_rate != sample_rate:
            samples = resample_linear(samples, source_rate, sample_rate)
        combined_samples.extend(samples)

    if reverse:
        combined_samples.reverse()

    pcm_data = float_to_pcm16(combined_samples)
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.setcomptype("NONE", "not compressed")
        wav_file.writeframes(pcm_data)
    return audio.getvalue()


def read_wav_as_mono(path: Path) -> tuple[list[float], int]:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"不是合法的 wav 文件: {path}")

    offset = 12
    fmt: tuple[int, int, int, int] | None = None
    raw_audio = b""

    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_data = data[offset + 8:offset + 8 + chunk_size]
        offset += 8 + chunk_size + (chunk_size % 2)

        if chunk_id == b"fmt ":
            audio_format, channels, sample_rate = struct.unpack_from("<HHI", chunk_data, 0)
            bits_per_sample = struct.unpack_from("<H", chunk_data, 14)[0]
            fmt = (audio_format, channels, sample_rate, bits_per_sample)
        elif chunk_id == b"data":
            raw_audio = chunk_data

    if fmt is None or not raw_audio:
        raise ValueError(f"wav 缺少 fmt/data chunk: {path}")

    audio_format, channels, sample_rate, bits_per_sample = fmt
    if audio_format == 1 and bits_per_sample == 16:
        sample_count = len(raw_audio) // 2
        samples = struct.unpack("<" + "h" * sample_count, raw_audio)
        floats = [sample / 32768.0 for sample in samples]
    elif audio_format == 3 and bits_per_sample == 32:
        sample_count = len(raw_audio) // 4
        floats = list(struct.unpack("<" + "f" * sample_count, raw_audio))
    else:
        raise ValueError(
            f"不支持的 wav 格式: format={audio_format}, bits={bits_per_sample}, file={path}"
        )

    if channels == 1:
        mono = floats
    elif channels == 2:
        mono = [(floats[i] + floats[i + 1]) * 0.5 for i in range(0, len(floats), 2)]
    else:
        raise ValueError(f"不支持的声道数: {channels}, file={path}")

    return mono, sample_rate


def resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if not samples or source_rate == target_rate:
        return samples

    target_length = max(1, round(len(samples) * target_rate / source_rate))
    if target_length == 1:
        return [samples[0]]

    scale = (len(samples) - 1) / (target_length - 1)
    output: list[float] = []
    for index in range(target_length):
        position = index * scale
        left = int(math.floor(position))
        right = min(left + 1, len(samples) - 1)
        ratio = position - left
        value = samples[left] * (1.0 - ratio) + samples[right] * ratio
        output.append(value)
    return output


def float_to_pcm16(samples: list[float]) -> bytes:
    pcm = bytearray()
    for sample in samples:
        clamped = max(-1.0, min(1.0, sample))
        value = int(round(clamped * 32767.0))
        pcm.extend(struct.pack("<h", value))
    return bytes(pcm)


def is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _clean_filename(raw_name: str) -> str:
    base = raw_name or "活字印刷"
    clean_name = re.sub(r'[\\/:*?"<>|]', "", base)[:10]
    return clean_name or "活字印刷"


_ENGINE: OttoEngine | None = None


def _get_engine() -> OttoEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = OttoEngine()
    return _ENGINE


@tool_metadata(
    name="活字印刷",
    description="将文本拼接成搞怪的活字印刷语音并发送到当前频道，可选整段倒放。",
    emoji="🗣️",
    category="工具",
)
async def otto_tool(
    params: OttoParams,
    **kwargs,
) -> str:
    """
    [工具说明]
    这是一个“活字印刷”语音拼接工具。
    当用户明确提及“活字印刷”时，你必须调用此工具，不要改用普通 TTS。
    你需要填写 `filename` 与 `text`，并根据用户要求决定 `reverse` 是否为 true。

    [效果说明]
    - 中文会优先命中整词预设，命不中时按拼音音节拼接。
    - 英文字母会故意按搞怪的逐字母中文近似发音来拼，比如 hello 会被拼成类似 ai chi yi ai lu ai lu ou。
    - `reverse=true` 时，不是倒序拼接词片，而是将整段成品音频整体倒放。

    [参数规则]
    - `filename`: 生成一个 10 字以内的中文标题，用作发到频道里的音频文件名。
    - `text`: 要做成活字印刷的原文。
    - `reverse`: 用户说“倒放”“反着放”“整段倒放”时设为 true，否则一般为 false。

    [执行逻辑]
    - 工具会直接生成一个 .wav 音频并发送到当前 Discord 频道。
    """
    channel = kwargs.get("channel")
    if not channel or not isinstance(channel, discord.abc.Messageable):
        return "错误：找不到有效的消息频道。"

    if not isinstance(params, OttoParams):
        try:
            params = OttoParams(**params)
        except Exception as e:
            return f"参数解析失败: {e}"

    text = str(params.text or "").strip()
    if not text:
        return "错误：活字印刷文本不能为空。"

    display_filename = f"{_clean_filename(params.filename)}.wav"

    try:
        audio_bytes, token_ids = await asyncio.to_thread(
            _get_engine().synthesize_bytes,
            text,
            params.reverse,
        )
        file = discord.File(fp=io.BytesIO(audio_bytes), filename=display_filename)
        await channel.send(file=file)
        reverse_note = "，已整段倒放" if params.reverse else ""
        log.info("活字印刷发送成功 | reverse=%s | tokens=%s", params.reverse, ",".join(token_ids[:40]))
        return f"成功：活字印刷音频已发送{reverse_note}。"
    except Exception as e:
        log.error(f"活字印刷生成异常: {e}", exc_info=True)
        return f"错误：生成活字印刷音频时发生故障: {e}"
