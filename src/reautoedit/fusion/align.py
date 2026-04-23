from __future__ import annotations

import cv2
import numpy as np


def align_mtb(frames_bgr_u8: list[np.ndarray]) -> list[np.ndarray]:
    """Median-threshold-bitmap alignment for handheld brackets.

    Input frames must be uint8 BGR of identical shape. Output is a new list
    with the same shape, translation-aligned to the first frame.
    """
    if not frames_bgr_u8:
        return []
    aligner = cv2.createAlignMTB()
    aligned: list[np.ndarray] = [f.copy() for f in frames_bgr_u8]
    aligner.process(frames_bgr_u8, aligned)
    return aligned
