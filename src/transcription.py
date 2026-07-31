from __future__ import annotations

from pathlib import Path


def ensure_mlx_whisper_available() -> None:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc


def transcribe_mp3(model_name: str, audio_path: Path, language: str | None = None) -> str:
    ensure_mlx_whisper_available()
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_name,
        language=language,
        verbose=False,
    )
    text = result.get("text", "").strip()
    return text + ("\n" if text else "")
