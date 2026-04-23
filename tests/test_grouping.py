from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from reautoedit.core.types import ExifData, Frame
from reautoedit.ingest.scene_grouping import GroupingConfig, group_frames


def _exif(
    ts: datetime,
    *,
    aperture: float = 8.0,
    shutter: float = 1 / 125,
    iso: float = 100.0,
    orientation: int = 1,
) -> ExifData:
    return ExifData(
        timestamp=ts,
        aperture=aperture,
        shutter=shutter,
        iso=iso,
        orientation=orientation,
    )


def _frame(name: str, exif: ExifData) -> Frame:
    return Frame(path=Path(name), exif=exif, is_raw=False)


def _shutter_for_ev(target_ev: float, aperture: float = 8.0, iso: float = 100.0) -> float:
    # ev = log2((a^2 / t) * (100/iso))  =>  t = (a^2 * 100/iso) / 2^ev
    return (aperture ** 2) * (100.0 / iso) / (2 ** target_ev)


def test_three_shot_bracket_is_one_scene():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    frames = [
        _frame("A.nef", _exif(t0, shutter=_shutter_for_ev(10))),
        _frame("B.nef", _exif(t0 + timedelta(seconds=1), shutter=_shutter_for_ev(12))),
        _frame("C.nef", _exif(t0 + timedelta(seconds=2), shutter=_shutter_for_ev(14))),
    ]
    scenes = group_frames(frames)
    assert len(scenes) == 1
    assert len(scenes[0].frames) == 3
    assert scenes[0].is_bracket


def test_gap_splits_scenes():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    frames = [
        _frame("A.nef", _exif(t0, shutter=_shutter_for_ev(10))),
        _frame("B.nef", _exif(t0 + timedelta(seconds=1), shutter=_shutter_for_ev(12))),
        _frame("C.nef", _exif(t0 + timedelta(seconds=2), shutter=_shutter_for_ev(14))),
        # 30s gap
        _frame("D.nef", _exif(t0 + timedelta(seconds=32), shutter=_shutter_for_ev(10))),
        _frame("E.nef", _exif(t0 + timedelta(seconds=33), shutter=_shutter_for_ev(12))),
        _frame("F.nef", _exif(t0 + timedelta(seconds=34), shutter=_shutter_for_ev(14))),
    ]
    scenes = group_frames(frames)
    assert len(scenes) == 2
    assert all(s.is_bracket for s in scenes)


def test_orientation_change_splits():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    frames = [
        _frame("A.nef", _exif(t0, shutter=_shutter_for_ev(10), orientation=1)),
        _frame("B.nef", _exif(t0 + timedelta(seconds=1), shutter=_shutter_for_ev(12), orientation=1)),
        _frame("C.nef", _exif(t0 + timedelta(seconds=2), shutter=_shutter_for_ev(14), orientation=6)),
    ]
    scenes = group_frames(frames)
    # A,B form a bracket; C is alone -> single-frame scene
    assert len(scenes) == 2
    assert scenes[0].is_bracket and len(scenes[0].frames) == 2
    assert not scenes[1].is_bracket


def test_repeated_ev_does_not_group_as_bracket():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    # Three shots at the same EV, close in time: should split into singles
    frames = [
        _frame("A.jpg", _exif(t0, shutter=_shutter_for_ev(10))),
        _frame("B.jpg", _exif(t0 + timedelta(seconds=1), shutter=_shutter_for_ev(10))),
        _frame("C.jpg", _exif(t0 + timedelta(seconds=2), shutter=_shutter_for_ev(10))),
    ]
    scenes = group_frames(frames)
    assert len(scenes) == 3
    assert all(not s.is_bracket for s in scenes)


def test_mixed_brackets_and_singles():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    frames = [
        # scene 1: 3-shot bracket
        _frame("A.nef", _exif(t0, shutter=_shutter_for_ev(10))),
        _frame("B.nef", _exif(t0 + timedelta(seconds=1), shutter=_shutter_for_ev(12))),
        _frame("C.nef", _exif(t0 + timedelta(seconds=2), shutter=_shutter_for_ev(14))),
        # single shot much later
        _frame("D.jpg", _exif(t0 + timedelta(minutes=5), shutter=_shutter_for_ev(11))),
        # scene 3: 5-shot bracket even later
        _frame("E.nef", _exif(t0 + timedelta(minutes=10), shutter=_shutter_for_ev(9))),
        _frame("F.nef", _exif(t0 + timedelta(minutes=10, seconds=1), shutter=_shutter_for_ev(11))),
        _frame("G.nef", _exif(t0 + timedelta(minutes=10, seconds=2), shutter=_shutter_for_ev(13))),
        _frame("H.nef", _exif(t0 + timedelta(minutes=10, seconds=3), shutter=_shutter_for_ev(15))),
        _frame("I.nef", _exif(t0 + timedelta(minutes=10, seconds=4), shutter=_shutter_for_ev(17))),
    ]
    scenes = group_frames(frames)
    assert [len(s.frames) for s in scenes] == [3, 1, 5]
    assert scenes[0].is_bracket
    assert not scenes[1].is_bracket
    assert scenes[2].is_bracket


def test_max_bracket_size_caps_run():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    # 8 distinct EVs, close timestamps: cap at 7
    frames = [
        _frame(f"F{i}.nef", _exif(t0 + timedelta(seconds=i), shutter=_shutter_for_ev(8 + i)))
        for i in range(8)
    ]
    scenes = group_frames(frames, GroupingConfig(max_bracket_size=7))
    # First 7 form a bracket; 8th is split off as single
    assert len(scenes) == 2
    assert len(scenes[0].frames) == 7 and scenes[0].is_bracket
    assert len(scenes[1].frames) == 1


def test_insufficient_ev_range_not_a_bracket():
    t0 = datetime(2026, 4, 23, 12, 0, 0)
    # Three distinct EVs but only 0.6 stops of range
    frames = [
        _frame("A.nef", _exif(t0, shutter=_shutter_for_ev(10.0))),
        _frame("B.nef", _exif(t0 + timedelta(seconds=1), shutter=_shutter_for_ev(10.3))),
        _frame("C.nef", _exif(t0 + timedelta(seconds=2), shutter=_shutter_for_ev(10.6))),
    ]
    scenes = group_frames(frames)
    # Split into singles since ev_range < 1.0
    assert len(scenes) == 3
    assert all(not s.is_bracket for s in scenes)


def test_empty_input_returns_empty():
    assert group_frames([]) == []


def test_missing_timestamp_does_not_crash():
    frames = [
        _frame("A.jpg", ExifData(None, 8.0, 1 / 125, 100.0, 1)),
        _frame("B.jpg", ExifData(None, 8.0, 1 / 60, 100.0, 1)),
    ]
    scenes = group_frames(frames)
    assert len(scenes) >= 1


def test_ev_formula_matches_known_value():
    # f/8, 1/125, ISO 100 -> EV ~ 13.0
    exif = _exif(datetime(2026, 4, 23), aperture=8.0, shutter=1 / 125, iso=100.0)
    assert exif.ev is not None
    assert pytest.approx(exif.ev, abs=0.02) == float(np.log2((8.0 ** 2) / (1 / 125) * (100.0 / 100.0)))
