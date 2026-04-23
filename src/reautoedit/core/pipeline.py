from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reautoedit.core.types import Frame, PipelineResult, Scene
from reautoedit.export.writer import OutputFormat, write_image
from reautoedit.fusion.mertens import MertensConfig, fuse_mertens
from reautoedit.fusion.single_frame import passthrough
from reautoedit.ingest.jpeg_decode import decode_jpeg
from reautoedit.ingest.raw_decode import decode_raw
from reautoedit.ingest.scene_grouping import GroupingConfig, group_frames, scan_folder

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_format: OutputFormat = OutputFormat.JPG
    jpeg_quality: int = 92
    grouping: GroupingConfig | None = None
    mertens: MertensConfig | None = None


def _decode_frame(frame: Frame) -> np.ndarray:
    if frame.is_raw:
        return decode_raw(frame.path)
    return decode_jpeg(frame.path)


def _fuse_scene(scene: Scene, config: PipelineConfig) -> np.ndarray:
    decoded = [_decode_frame(f) for f in scene.frames]
    if len(decoded) == 1:
        return passthrough(decoded[0])
    return fuse_mertens(decoded, config.mertens)


def _output_path(scene: Scene, out_dir: Path, fmt: OutputFormat) -> Path:
    stem = scene.primary.path.stem
    return out_dir / f"{stem}{fmt.extension}"


def process_folder(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
    progress: Callable[[int, int, Scene], None] | None = None,
) -> list[PipelineResult]:
    """Run the full pipeline on ``input_dir`` -> ``output_dir``."""
    cfg = config or PipelineConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = scan_folder(input_dir)
    scenes = group_frames(frames, cfg.grouping)

    results: list[PipelineResult] = []
    for i, scene in enumerate(scenes):
        if progress is not None:
            progress(i, len(scenes), scene)
        try:
            fused = _fuse_scene(scene, cfg)
            out = _output_path(scene, output_dir, cfg.output_format)
            written = write_image(fused, out, cfg.output_format, cfg.jpeg_quality)
            results.append(PipelineResult(scene=scene, output_path=written, success=True))
        except Exception as exc:
            log.exception("Pipeline failed on scene with primary %s", scene.primary.path)
            results.append(
                PipelineResult(
                    scene=scene,
                    output_path=_output_path(scene, output_dir, cfg.output_format),
                    success=False,
                    error=str(exc),
                )
            )
    return results
