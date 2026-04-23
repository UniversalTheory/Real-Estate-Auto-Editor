from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from reautoedit.core.colorspace import to_uint8
from reautoedit.fusion.align import align_mtb


@dataclass
class MertensConfig:
    contrast_weight: float = 1.0
    saturation_weight: float = 1.0
    exposure_weight: float = 1.0
    align: bool = True


def _to_bgr_u8(frame_rgb_f32: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb_f32, cv2.COLOR_RGB2BGR)
    return to_uint8(bgr)


def _normalize_shapes(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Crop all frames to the common min height/width so alignment/merge works."""
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    return [f[:h, :w] for f in frames]


def fuse_mertens(
    frames_rgb_f32: list[np.ndarray],
    config: MertensConfig | None = None,
) -> np.ndarray:
    """Mertens exposure-fuse a list of sRGB float32 [0,1] RGB frames.

    Returns an sRGB float32 [0,1] RGB image ready for the grading pipeline.
    """
    if not frames_rgb_f32:
        raise ValueError("fuse_mertens requires at least one frame")

    cfg = config or MertensConfig()
    frames_bgr_u8 = [_to_bgr_u8(f) for f in _normalize_shapes(frames_rgb_f32)]
    if cfg.align and len(frames_bgr_u8) > 1:
        frames_bgr_u8 = align_mtb(frames_bgr_u8)

    merger = cv2.createMergeMertens(
        cfg.contrast_weight, cfg.saturation_weight, cfg.exposure_weight
    )
    fused_bgr = merger.process(frames_bgr_u8)  # float32, may exceed [0,1]
    fused_bgr = np.clip(fused_bgr, 0.0, 1.0)
    fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)
    return fused_rgb.astype(np.float32)
