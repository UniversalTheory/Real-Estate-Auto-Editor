from __future__ import annotations

from PIL import ImageCms

_SRGB_PROFILE: bytes | None = None


def srgb_profile_bytes() -> bytes:
    """Return the bytes of a standard sRGB ICC profile, cached across calls."""
    global _SRGB_PROFILE
    if _SRGB_PROFILE is None:
        profile = ImageCms.createProfile("sRGB")
        _SRGB_PROFILE = ImageCms.ImageCmsProfile(profile).tobytes()
    return _SRGB_PROFILE
