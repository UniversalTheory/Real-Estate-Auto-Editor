from __future__ import annotations

from datetime import datetime
from fractions import Fraction
from pathlib import Path

import exifread

from reautoedit.core.types import ExifData

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | JPEG_EXTENSIONS


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def _ratio_to_float(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Fraction):
            return float(value)
        if hasattr(value, "values"):
            vals = list(value.values)
            if not vals:
                return None
            v0 = vals[0]
            if hasattr(v0, "num") and hasattr(v0, "den"):
                return _ratio_to_float(v0.num, v0.den)
            if isinstance(v0, (tuple, list)):
                if len(v0) == 1:
                    return float(v0[0])
                if len(v0) >= 2:
                    return _ratio_to_float(v0[0], v0[1])
                return None
            if isinstance(v0, (int, float)):
                if len(vals) == 2 and isinstance(vals[1], (int, float)):
                    return _ratio_to_float(v0, vals[1])
                return float(v0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _to_int(value: object) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None


def _parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def read_exif(path: Path) -> ExifData:
    with path.open("rb") as fh:
        tags = exifread.process_file(fh, details=False)

    dt_raw = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    timestamp = _parse_datetime(str(dt_raw) if dt_raw is not None else None)

    aperture = _to_float(tags.get("EXIF FNumber"))
    shutter = _to_float(tags.get("EXIF ExposureTime"))
    iso = _to_float(tags.get("EXIF ISOSpeedRatings"))
    orientation = _to_int(tags.get("Image Orientation"))

    camera_make = str(tags["Image Make"]).strip() if "Image Make" in tags else None
    camera_model = str(tags["Image Model"]).strip() if "Image Model" in tags else None

    if timestamp is None:
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            timestamp = None

    return ExifData(
        timestamp=timestamp,
        aperture=aperture,
        shutter=shutter,
        iso=iso,
        orientation=orientation,
        camera_make=camera_make,
        camera_model=camera_model,
        raw={str(k): str(v) for k, v in tags.items()},
    )
