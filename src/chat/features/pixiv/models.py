# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class PixivImageResult:
    illust_id: int
    title: str
    author: str
    caption: str
    image_url: str
    file_name: str
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PixivToolResult:
    success: bool
    message: str
    images: list[PixivImageResult] = field(default_factory=list)
    error: Optional[str] = None
