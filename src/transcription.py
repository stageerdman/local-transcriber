from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Called during transcription with a fraction in [0, 1] of the audio processed
# so far. Not every backend can report it (see `transcribe_mp3_segments`).
ProgressCallback = Callable[[float], None]


@dataclass
class TranscribedSegment:
    start: float
    end: float
    text: str


# Logical id for the locally-converted Czech-finetuned Whisper large-v3 (see
# `scripts/convert_czech_model.py`). Kept as a short token in the model catalog
# so the catalog stays portable; `_resolve_model` maps it to the on-disk path.
CZECH_FT_ID = "czech-large-v3-mlx"
CZECH_FT_PATH = (
    Path.home() / ".cache" / "local-transcriber" / "models"
    / "whisper-large-v3-czech-cv13-mlx"
)


def _resolve_model(model_name: str) -> str:
    """Map a catalog model id to what the backend actually loads.

    Only the Czech-finetune token needs resolving (to its converted local MLX
    dir); every other id is an mlx-community repo passed through unchanged.
    """
    if model_name == CZECH_FT_ID:
        if not CZECH_FT_PATH.exists():
            raise RuntimeError(
                "The Czech-finetuned model isn't installed yet. Run "
                "`python scripts/convert_czech_model.py` once to convert it."
            )
        return str(CZECH_FT_PATH)
    return model_name


def _is_parakeet_model(model_name: str) -> bool:
    return "parakeet" in model_name.lower()


# Tuned Whisper decode options (validated in the transcription-quality-research
# update). `condition_on_previous_text=False` is the single most important flag:
# it stops one hallucinated/looped segment from poisoning every later window -
# the dominant failure mode on conversational audio with pauses, and on the
# subtraction-residual tracks produced by mic-bleed removal. The shortened
# temperature ladder drops the high-temp retries that invent text. No
# `initial_prompt` on purpose - a content-ful Czech prompt leaked into the
# output as fabricated text in testing.
WHISPER_DECODE_OPTS: dict[str, Any] = {
    "condition_on_previous_text": False,
    "temperature": (0.0, 0.2, 0.4),
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
}


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

        ModelHolder.get_model(_resolve_model(model_name), mx.float16)
    except Exception:
        # Best-effort only: if mlx_whisper's internals change, fall back to
        # letting transcribe() load the model itself.
        pass


def transcribe_mp3_segments(
    model_name: str,
    audio_path: Path,
    language: str | None = None,
    progress_callback: ProgressCallback | None = None,
    use_vad: bool = True,
) -> list[TranscribedSegment]:
    """Transcribe, keeping per-segment/sentence timestamps.

    Timestamps are what let the pause-based dialogue blocking (see
    `src.dialogue`) and multi-track/interruption merging work. The worker
    joins segment text back into plain paragraphs for single-track jobs, and
    into speaker-labeled blocks for multi-track jobs.

    `progress_callback`, if given, is invoked with a fraction in [0, 1] of the
    audio processed so far. Both backends report it: Parakeet per chunk,
    Whisper per VAD speech-region.

    `use_vad` (Whisper path only): run Silero VAD first and transcribe only the
    speech regions. This is the fix for Whisper's silence-hallucination - on
    quiet stretches and the subtraction-residual tracks from mic-bleed removal,
    Whisper otherwise invents boilerplate ("Titulky vytvořil...") or loops.
    Parakeet has no such failure mode and ignores this flag.
    """
    if _is_parakeet_model(model_name):
        return _transcribe_with_parakeet(model_name, audio_path, progress_callback)

    repo = _resolve_model(model_name)
    if use_vad:
        return _transcribe_whisper_vad(repo, audio_path, language, progress_callback)
    return _transcribe_whisper_plain(repo, audio_path, language)


def _transcribe_whisper_plain(
    repo: str, audio_path: Path, language: str | None
) -> list[TranscribedSegment]:
    ensure_mlx_whisper_available()
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo,
        language=language,
        verbose=False,
        **WHISPER_DECODE_OPTS,
    )
    return [
        TranscribedSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg.get("text", "").strip())
        for seg in result.get("segments", [])
        if seg.get("text", "").strip()
    ]


# Silero VAD chunking: keep each decode under Whisper's 30 s window, pad speech a
# little so leading/trailing phonemes aren't clipped, and bridge short gaps so we
# don't fragment one sentence into many tiny decodes.
_VAD_SAMPLE_RATE = 16000
_VAD_MAX_CHUNK_SECONDS = 28.0
_VAD_MERGE_GAP_SECONDS = 0.4


def _speech_chunks(audio) -> list[tuple[int, int]]:
    """Silero VAD -> list of (start_sample, end_sample) speech chunks.

    Regions from `get_speech_timestamps` are grouped greedily into chunks up to
    `_VAD_MAX_CHUNK_SECONDS` so each is a single Whisper decode window.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        min_silence_duration_ms=400,
        speech_pad_ms=200,
        max_speech_duration_s=_VAD_MAX_CHUNK_SECONDS,
    )
    regions = get_speech_timestamps(audio, options, sampling_rate=_VAD_SAMPLE_RATE)
    return _group_regions(
        [(int(r["start"]), int(r["end"])) for r in regions],
        max_len=int(_VAD_MAX_CHUNK_SECONDS * _VAD_SAMPLE_RATE),
        merge_gap=int(_VAD_MERGE_GAP_SECONDS * _VAD_SAMPLE_RATE),
    )


def _group_regions(
    regions: list[tuple[int, int]], max_len: int, merge_gap: int
) -> list[tuple[int, int]]:
    """Greedily merge adjacent speech regions into chunks.

    A region is appended to the previous chunk when the gap between them is
    small (<= merge_gap) and the combined span stays under max_len; otherwise it
    starts a new chunk. Pure function of its inputs - unit-tested directly.
    """
    chunks: list[tuple[int, int]] = []
    for s, e in regions:
        if chunks and s - chunks[-1][1] <= merge_gap and e - chunks[-1][0] <= max_len:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return chunks


def _transcribe_whisper_vad(
    repo: str,
    audio_path: Path,
    language: str | None,
    progress_callback: ProgressCallback | None,
) -> list[TranscribedSegment]:
    ensure_mlx_whisper_available()
    import mlx_whisper
    from mlx_whisper.audio import load_audio

    audio = load_audio(str(audio_path))  # float32 mono @ 16 kHz
    chunks = _speech_chunks(audio)
    if not chunks:
        return []

    total_speech = sum(e - s for s, e in chunks) or 1
    done_speech = 0
    segments: list[TranscribedSegment] = []
    for start, end in chunks:
        offset = start / _VAD_SAMPLE_RATE
        result = mlx_whisper.transcribe(
            audio[start:end],
            path_or_hf_repo=repo,
            language=language,
            verbose=False,
            **WHISPER_DECODE_OPTS,
        )
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(TranscribedSegment(
                    start=offset + float(seg["start"]),
                    end=offset + float(seg["end"]),
                    text=text,
                ))
        done_speech += end - start
        if progress_callback is not None:
            progress_callback(min(1.0, done_speech / total_speech))
    return segments


# parakeet_mlx.transcribe()'s own `chunk_duration` default is None - "process
# the whole file in one pass" - which is fine for short clips but allocates
# an attention buffer that scales with the *entire* file length. On a
# ~40-minute recording that blew past Metal's max buffer size and crashed
# the job (`[metal::malloc] Attempting to allocate 62011956480 bytes...`).
# 120s/15s matches parakeet-mlx's own CLI defaults, which chunk specifically
# to avoid this - timestamps come back stitched to the full file regardless.
_PARAKEET_CHUNK_DURATION_SECONDS = 120.0
_PARAKEET_CHUNK_OVERLAP_SECONDS = 15.0


def _transcribe_with_parakeet(
    model_name: str, audio_path: Path, progress_callback: ProgressCallback | None = None
) -> list[TranscribedSegment]:
    # Parakeet is multilingual and auto-detects the spoken language itself -
    # unlike mlx_whisper.transcribe(), parakeet_mlx's transcribe() takes no
    # language parameter to force one, so `language` isn't threaded through
    # here (the app-level language picker is simply a no-op for this model).
    model = _get_parakeet_model(model_name)

    # parakeet-mlx calls this with (current_sample, total_samples) before each
    # chunk. Only fires when the file is long enough to be chunked; short
    # files return in one pass and the caller keeps its time-estimate bar.
    chunk_callback = None
    if progress_callback is not None:
        def chunk_callback(current: int, total: int) -> None:
            if total > 0:
                progress_callback(max(0.0, min(1.0, current / total)))

    result = model.transcribe(
        audio_path,
        chunk_duration=_PARAKEET_CHUNK_DURATION_SECONDS,
        overlap_duration=_PARAKEET_CHUNK_OVERLAP_SECONDS,
        chunk_callback=chunk_callback,
    )
    return [
        TranscribedSegment(start=float(sentence.start), end=float(sentence.end), text=sentence.text.strip())
        for sentence in result.sentences
        if sentence.text.strip()
    ]
