from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExifData:
    timestamp: datetime | None
    aperture: float | None
    shutter: float | None
    iso: float | None
    orientation: int | None
    camera_make: str | None = None
    camera_model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ev(self) -> float | None:
        if self.aperture is None or self.shutter is None or self.iso is None:
            return None
        if self.aperture <= 0 or self.shutter <= 0 or self.iso <= 0:
            return None
        return float(np.log2((self.aperture ** 2) / self.shutter * (100.0 / self.iso)))


@dataclass
class Frame:
    path: Path
    exif: ExifData
    is_raw: bool


@dataclass
class Scene:
    frames: list[Frame]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("Scene requires at least one frame")

    @property
    def primary(self) -> Frame:
        return self.frames[0]

    @property
    def is_bracket(self) -> bool:
        return len(self.frames) >= 2


@dataclass
class PipelineResult:
    scene: Scene
    output_path: Path
    success: bool
    error: str | None = None
