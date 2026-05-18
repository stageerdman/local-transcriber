from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_faster_whisper_available() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc


def create_model(model_name: str = "medium") -> Any:
    ensure_faster_whisper_available()
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_mp3(model: Any, audio_path: Path, language: str = "en") -> str:
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
    )
    lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    return "\n".join(lines).strip() + ("\n" if lines else "")
