"""Utilities for downloading ASR models into the project-local models directory."""

from __future__ import annotations

from pathlib import Path

import typer


app = typer.Typer(help="下载 HonestEar 使用的本地 ASR 模型。")


def _sanitize_model_name(model_name: str) -> str:
    """Converts a model identifier into a safe relative directory name."""

    return model_name.replace("/", "--")


def _download_whisper_model(model_name: str, output_dir: Path) -> Path:
    """Downloads one faster-whisper model into the target directory."""

    from faster_whisper import download_model

    target_dir = output_dir / "whisper" / _sanitize_model_name(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = Path(download_model(model_name, output_dir=str(target_dir)))
    return resolved_path


def _download_wav2vec2_model(model_name: str, output_dir: Path) -> Path:
    """Downloads one wav2vec2 processor and model into the target directory."""

    from transformers import AutoModelForCTC, AutoProcessor

    target_dir = output_dir / "wav2vec2" / _sanitize_model_name(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)

    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    return target_dir


@app.command("asr")
def download_asr_models(
    whisper_model: str = typer.Option("small.en", help="Whisper 模型名，例如 small.en 或 Systran/faster-whisper-small.en。"),
    wav2vec2_model: str = typer.Option(
        "facebook/wav2vec2-large-960h-lv60-self",
        help="wav2vec2 模型名，例如 facebook/wav2vec2-large-960h-lv60-self。",
    ),
    output_dir: Path = typer.Option(
        Path("models"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="模型下载根目录。",
    ),
    whisper_only: bool = typer.Option(False, help="只下载 Whisper 模型。"),
    wav2vec2_only: bool = typer.Option(False, help="只下载 wav2vec2 模型。"),
) -> None:
    """Downloads ASR models into the repository-local models directory."""

    if whisper_only and wav2vec2_only:
        raise typer.BadParameter("不能同时设置 --whisper-only 和 --wav2vec2-only。")

    if not wav2vec2_only:
        whisper_path = _download_whisper_model(whisper_model, output_dir)
        typer.echo(f"whisper 下载完成: {whisper_path}")

    if not whisper_only:
        wav2vec2_path = _download_wav2vec2_model(wav2vec2_model, output_dir)
        typer.echo(f"wav2vec2 下载完成: {wav2vec2_path}")

    typer.echo("建议将 .env 中的 WHISPER_MODEL_SIZE 和 WAV2VEC2_MODEL_NAME 改为上述本地目录路径。")


if __name__ == "__main__":
    app()

