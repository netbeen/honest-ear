"""CLI entrypoints for the HonestEar Phase 1 prototype."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from honest_ear.config import get_settings
from honest_ear.pipeline import run_pipeline
from honest_ear.samples import load_sample_records


app = typer.Typer(help="HonestEar Phase 1 local pipeline tools.")


@app.command()
def process(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    mode: str = typer.Option("accuracy", help="Correction mode: fluency or accuracy."),
    speak_reply: bool = typer.Option(True, help="Whether to synthesize the reply with local TTS."),
) -> None:
    """Runs the end-to-end pipeline for one local audio file."""

    result = run_pipeline(audio_path=audio_path, mode=mode, speak_reply=speak_reply, settings=get_settings())
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("list-samples")
def list_samples() -> None:
    """Prints the built-in Phase 1 evaluation samples as JSON."""

    settings = get_settings()
    records = load_sample_records(settings.sample_dataset_path)
    typer.echo(json.dumps([record.model_dump(mode="json") for record in records], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()

