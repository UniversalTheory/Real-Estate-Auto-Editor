from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from reautoedit.core.pipeline import PipelineConfig, process_folder
from reautoedit.core.types import Scene
from reautoedit.export.writer import OutputFormat

app = typer.Typer(add_completion=False, help="Real estate photo auto-editor")


@app.command()
def process(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")],
    format: Annotated[OutputFormat, typer.Option("--format", "-f", case_sensitive=False)] = OutputFormat.JPG,
    jpeg_quality: Annotated[int, typer.Option("--jpeg-quality", min=1, max=100)] = 92,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """HDR-fuse brackets in INPUT_DIR and write results to --out."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(output_format=format, jpeg_quality=jpeg_quality)

    def progress(idx: int, total: int, scene: Scene) -> None:
        kind = "bracket" if scene.is_bracket else "single"
        typer.echo(
            f"[{idx + 1}/{total}] {kind} ({len(scene.frames)}): {scene.primary.path.name}"
        )

    results = process_folder(input_dir, out, config, progress=progress)

    ok = sum(1 for r in results if r.success)
    failed = len(results) - ok
    typer.echo(f"Done. {ok} succeeded, {failed} failed.")
    for r in results:
        if not r.success:
            typer.echo(f"  FAIL {r.scene.primary.path.name}: {r.error}", err=True)


if __name__ == "__main__":
    app()
