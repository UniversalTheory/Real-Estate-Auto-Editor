from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy


def decode_raw(path: Path, half_size: bool = False) -> np.ndarray:
    """Decode a RAW file to an sRGB float32 image in [0, 1].

    Uses camera white balance and a standard sRGB output. ``half_size`` returns
    a 2x-downsampled image for preview use.
    """
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
            half_size=half_size,
        )
    return (rgb16.astype(np.float32)) / 65535.0
