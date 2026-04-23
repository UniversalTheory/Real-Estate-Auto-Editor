from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from reautoedit.core.types import ExifData, Frame, Scene
from reautoedit.ingest.exif import is_raw, is_supported, read_exif

DEFAULT_GAP_SECONDS = 3.0
MAX_BRACKET_SIZE = 7
MIN_BRACKET_SIZE = 2
MIN_BRACKET_EV_RANGE = 1.0
EV_EQUAL_TOLERANCE = 0.1


@dataclass
class GroupingConfig:
    gap_seconds: float = DEFAULT_GAP_SECONDS
    max_bracket_size: int = MAX_BRACKET_SIZE
    min_bracket_size: int = MIN_BRACKET_SIZE
    min_bracket_ev_range: float = MIN_BRACKET_EV_RANGE


def scan_folder(folder: Path) -> list[Frame]:
    """Read every supported image in ``folder`` (non-recursive) as a ``Frame``."""
    frames: list[Frame] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not is_supported(path):
            continue
        exif = read_exif(path)
        frames.append(Frame(path=path, exif=exif, is_raw=is_raw(path)))
    return frames


def _sort_key(frame: Frame) -> tuple[float, str]:
    ts = frame.exif.timestamp
    epoch = ts.timestamp() if isinstance(ts, datetime) else 0.0
    return (epoch, frame.path.name)


def _evs_effectively_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= EV_EQUAL_TOLERANCE


def _timestamp_gap(a: ExifData, b: ExifData) -> float | None:
    if a.timestamp is None or b.timestamp is None:
        return None
    delta: timedelta = b.timestamp - a.timestamp
    return abs(delta.total_seconds())


def group_frames(
    frames: Iterable[Frame],
    config: GroupingConfig | None = None,
) -> list[Scene]:
    """Partition frames into scenes.

    A new scene starts when any of the following is true relative to the
    current run:

    * the timestamp gap exceeds ``config.gap_seconds``
    * orientation changes
    * the incoming frame's EV matches an EV already in the current run
      (brackets should never repeat an EV)
    * the current run already has ``config.max_bracket_size`` frames

    After partitioning, any "scene" that doesn't meet bracket criteria
    (size 2..max, EV range >= min_bracket_ev_range, all EVs distinct) is
    split back into one single-frame scene per frame.
    """
    cfg = config or GroupingConfig()
    ordered = sorted(frames, key=_sort_key)
    if not ordered:
        return []

    runs: list[list[Frame]] = [[ordered[0]]]
    for frame in ordered[1:]:
        run = runs[-1]
        prev = run[-1]
        gap = _timestamp_gap(prev.exif, frame.exif)

        start_new = False
        if len(run) >= cfg.max_bracket_size:
            start_new = True
        elif gap is not None and gap > cfg.gap_seconds:
            start_new = True
        elif (
            prev.exif.orientation is not None
            and frame.exif.orientation is not None
            and prev.exif.orientation != frame.exif.orientation
        ):
            start_new = True
        elif any(_evs_effectively_equal(f.exif.ev, frame.exif.ev) for f in run):
            start_new = True

        if start_new:
            runs.append([frame])
        else:
            run.append(frame)

    scenes: list[Scene] = []
    for run in runs:
        if _is_valid_bracket(run, cfg):
            scenes.append(Scene(frames=run))
        else:
            scenes.extend(Scene(frames=[f]) for f in run)
    return scenes


def _is_valid_bracket(run: list[Frame], cfg: GroupingConfig) -> bool:
    if not (cfg.min_bracket_size <= len(run) <= cfg.max_bracket_size):
        return False
    evs = [f.exif.ev for f in run]
    if any(ev is None for ev in evs):
        return False
    ev_vals = [float(ev) for ev in evs]
    if max(ev_vals) - min(ev_vals) < cfg.min_bracket_ev_range:
        return False
    # all distinct within tolerance
    for i, a in enumerate(ev_vals):
        for b in ev_vals[i + 1 :]:
            if abs(a - b) <= EV_EQUAL_TOLERANCE:
                return False
    return True
