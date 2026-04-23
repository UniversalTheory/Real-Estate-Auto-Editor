from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from reautoedit.core.colorspace import to_uint8
from reautoedit.export.icc import srgb_profile_bytes


class OutputFormat(str, Enum):
    JPG = "jpg"
    PNG = "png"

    @property
    def extension(self) -> str:
        return ".jpg" if self is OutputFormat.JPG else ".png"


def write_image(
    img_rgb_f32: np.ndarray,
    out_path: Path,
    fmt: OutputFormat,
    jpeg_quality: int = 92,
) -> Path:
    """Write a float32 sRGB RGB image in [0,1] to disk. Returns the actual path used."""
    out_path = out_path.with_suffix(fmt.extension)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = to_uint8(img_rgb_f32)
    image = Image.fromarray(arr, mode="RGB")

    save_kwargs: dict[str, object] = {"icc_profile": srgb_profile_bytes()}
    if fmt is OutputFormat.JPG:
        save_kwargs.update(
            format="JPEG",
            quality=int(jpeg_quality),
            subsampling=1,
            optimize=True,
        )
    else:
        save_kwargs.update(format="PNG", compress_level=6)

    image.save(out_path, **save_kwargs)
    return out_path
