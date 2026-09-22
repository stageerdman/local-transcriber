from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscribedSegment:
    start: float
    end: float
    text: str


def _is_parakeet_model(model_name: str) -> bool:
    return "parakeet" in model_name.lower()


def ensure_mlx_whisper_available() -> None:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc


def ensure_parakeet_mlx_available() -> None:
    try:
        import parakeet_mlx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "parakeet-mlx is not installed. Run: pip install -r requirements.txt"
        ) from exc


# Loaded Parakeet models, keyed by repo id - kept warm in-process for the
# lifetime of the engine child process, same idea as mlx_whisper's own
# ModelHolder cache below.
_parakeet_models: dict[str, Any] = {}


def _get_parakeet_model(model_name: str) -> Any:
    if model_name not in _parakeet_models:
        ensure_parakeet_mlx_available()
        from parakeet_mlx import from_pretrained

        _parakeet_models[model_name] = from_pretrained(model_name)
    return _parakeet_models[model_name]


def ensure_model_loaded(model_name: str) -> None:
    """Load (downloading if needed) and cache the model ahead of transcription.

    Both backends lazily load/download on first use and cache in-process for
    later calls with the same model. Pre-warming it here lets the UI show a
    distinct "downloading/loading model" phase instead of that time being
    silently folded into "transcribing" (which throws off the
    audio-duration-based progress estimate on a model's first use).
    """
    if _is_parakeet_model(model_name):
        try:
            _get_parakeet_model(model_name)
        except Exception:
            # Best-effort only: fall back to letting transcribe() load it.
            pass
        return

    try:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.get_model(model_name, mx.float16)
    except Exception:
        # Best-effort only: if mlx_whisper's internals change, fall back to
        # letting transcribe() load the model itself.
        pass


def transcribe_mp3_segments(
    model_name: str, audio_path: Path, language: str | None = None
) -> list[TranscribedSegment]:
    """Transcribe, keeping per-segment/sentence timestamps.

    Timestamps are what let the pause-based dialogue blocking (see
    `src.dialogue`) and multi-track/interruption merging work. The worker
    joins segment text back into plain paragraphs for single-track jobs, and
    into speaker-labeled blocks for multi-track jobs.
    """
    if _is_parakeet_model(model_name):
        return _transcribe_with_parakeet(model_name, audio_path)

    ensure_mlx_whisper_available()
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_name,
        language=language,
        verbose=False,
    )
    return [
        TranscribedSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg.get("text", "").strip())
        for seg in result.get("segments", [])
        if seg.get("text", "").strip()
    ]


# parakeet_mlx.transcribe()'s own `chunk_duration` default is None - "process
# the whole file in one pass" - which is fine for short clips but allocates
# an attention buffer that scales with the *entire* file length. On a
# ~40-minute recording that blew past Metal's max buffer size and crashed
# the job (`[metal::malloc] Attempting to allocate 62011956480 bytes...`).
# 120s/15s matches parakeet-mlx's own CLI defaults, which chunk specifically
# to avoid this - timestamps come back stitched to the full file regardless.
_PARAKEET_CHUNK_DURATION_SECONDS = 120.0
_PARAKEET_CHUNK_OVERLAP_SECONDS = 15.0


def _transcribe_with_parakeet(model_name: str, audio_path: Path) -> list[TranscribedSegment]:
    # Parakeet is multilingual and auto-detects the spoken language itself -
    # unlike mlx_whisper.transcribe(), parakeet_mlx's transcribe() takes no
    # language parameter to force one, so `language` isn't threaded through
    # here (the app-level language picker is simply a no-op for this model).
    model = _get_parakeet_model(model_name)
    result = model.transcribe(
        audio_path,
        chunk_duration=_PARAKEET_CHUNK_DURATION_SECONDS,
        overlap_duration=_PARAKEET_CHUNK_OVERLAP_SECONDS,
    )
    return [
        TranscribedSegment(start=float(sentence.start), end=float(sentence.end), text=sentence.text.strip())
        for sentence in result.sentences
        if sentence.text.strip()
    ]
