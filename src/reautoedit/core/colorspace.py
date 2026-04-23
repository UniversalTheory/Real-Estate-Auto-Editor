from __future__ import annotations

import numpy as np


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB OETF to a linear float image in [0, 1]."""
    linear = np.clip(linear, 0.0, 1.0)
    threshold = 0.0031308
    low = 12.92 * linear
    high = 1.055 * np.power(np.clip(linear, threshold, 1.0), 1.0 / 2.4) - 0.055
    return np.where(linear <= threshold, low, high).astype(np.float32)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """Apply sRGB EOTF, converting sRGB-encoded values in [0, 1] to linear."""
    encoded = np.clip(encoded, 0.0, 1.0)
    threshold = 0.04045
    low = encoded / 12.92
    high = np.power((np.clip(encoded, threshold, 1.0) + 0.055) / 1.055, 2.4)
    return np.where(encoded <= threshold, low, high).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 [0, 255] with rounding."""
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def from_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a uint8 image to float32 in [0, 1]."""
    return (img.astype(np.float32)) / 255.0
