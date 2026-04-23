from __future__ import annotations

import numpy as np


def passthrough(frame_rgb_f32: np.ndarray) -> np.ndarray:
    """Single-frame "fusion": just normalize shape/dtype to pipeline contract."""
    return np.clip(frame_rgb_f32.astype(np.float32), 0.0, 1.0)
