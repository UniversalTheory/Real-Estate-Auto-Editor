from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def decode_jpeg(path: Path) -> np.ndarray:
    """Decode a JPEG to an sRGB float32 image in [0, 1], honoring EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr
