from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Called during transcription with a fraction in [0, 1] of the audio processed
# so far. Not every backend can report it (see `transcribe_mp3_segments`).
ProgressCallback = Callable[[float], None]

# Called once per completed chunk with (chunk_index, that chunk's segments). It
# is how resumable transcription persists finished work *as it happens*, so a
# stop/crash mid-file only loses the in-flight chunk. Whisper fires it per VAD
# speech region (frame-accurate, resumable mid-file); Parakeet fires it once for
# the whole file (chunk 0), since its chunking is internal and opaque.
ChunkSink = Callable[[int, "list[TranscribedSegment]"], None]


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


# Cloud backend: Groq hosts stock Whisper weights on their own LPU hardware,
# not a different model - just a much faster place to run the same weights
# (confirmed ~7-8x faster than local MLX in the transcription-quality-research
# update's benchmarking). Prefixing the catalog id keeps it visually distinct
# from on-device repo ids and makes dispatch a cheap string check.
GROQ_MODEL_PREFIX = "groq:"


def _is_groq_model(model_name: str) -> bool:
    return model_name.startswith(GROQ_MODEL_PREFIX)


# The Cloud quality tier's one model choice. large-v3-turbo over plain
# large-v3: in blind-ish comparison on real Czech call audio (the
# transcription-quality-research update), turbo didn't show large-v3's
# hallucinated-repeat-loop failure on a quiet opening, and it's both faster
# and cheaper on Groq's pricing.
GROQ_ID = f"{GROQ_MODEL_PREFIX}whisper-large-v3-turbo"


# Tuned Whisper decode options (validated in the transcription-quality-research
# update). `condition_on_previous_text=False` is the single most important flag:
# it stops one hallucinated/looped segment from poisoning every later window -
# the dominant failure mode on conversational audio with pauses, and on the
# subtraction-residual tracks produced by mic-bleed removal. The shortened
# temperature ladder drops the high-temp retries that invent text. No
# `initial_prompt` on purpose - a content-ful Czech prompt leaked into the
# output as fabricated text in testing.
#
# `word_timestamps=True` turns on DTW alignment so each segment carries per-word
# start/end. We anchor a turn to its first word's start (see `_seg_times`):
# Whisper's segment-level starts are only accurate to a second or two and drift
# early in fast cross-talk, which showed up as read-along turns highlighting
# ~2s ahead of the audio. Word starts are frame-accurate.
WHISPER_DECODE_OPTS: dict[str, Any] = {
    "condition_on_previous_text": False,
    "temperature": (0.0, 0.2, 0.4),
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "word_timestamps": True,
}


def _seg_times(seg: dict) -> tuple[float, float]:
    """(start, end) for a Whisper segment, preferring its word-level times.

    Falls back to the segment's own start/end when word timestamps are absent
    (e.g. a segment that produced no aligned words), so this is safe whether or
    not `word_timestamps` was on."""
    words = [w for w in seg.get("words", []) if w.get("start") is not None]
    if words:
        return float(words[0]["start"]), float(words[-1]["end"])
    return float(seg["start"]), float(seg["end"])


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

    if _is_groq_model(model_name):
        return  # nothing to preload - each call is a fresh API request

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
    chunk_sink: ChunkSink | None = None,
    start_chunk: int = 0,
) -> list[TranscribedSegment]:
    """Transcribe, keeping per-segment/sentence timestamps.

    Timestamps are what let the pause-based dialogue blocking (see
    `src.dialogue`) and multi-track/interruption merging work. The worker
    joins segment text back into plain paragraphs for single-track jobs, and
    into speaker-labeled blocks for multi-track jobs.

    `progress_callback`, if given, is invoked with a fraction in [0, 1] of the
    audio processed so far: Parakeet per chunk, Whisper per VAD speech-region,
    Groq once (at completion - it's a single blocking API call).

    `use_vad` (Whisper path only): run Silero VAD first and transcribe only the
    speech regions. This is the fix for Whisper's silence-hallucination - on
    quiet stretches and the subtraction-residual tracks from mic-bleed removal,
    Whisper otherwise invents boilerplate ("Titulky vytvořil...") or loops.
    Parakeet has no such failure mode and ignores this flag.

    `chunk_sink`/`start_chunk` power resumable transcription. `chunk_sink` is
    called with each completed chunk's segments as it finishes (so the caller
    can persist them immediately). `start_chunk` skips that many leading chunks -
    on resume the caller already has them persisted. The VAD chunk list is a
    deterministic function of the audio, so skipping the first N and keeping the
    previously-saved N reproduces the full transcript exactly. Only the Whisper
    VAD path can skip mid-file; Parakeet always runs the whole file (its single
    chunk 0), so `start_chunk` is a no-op there and the caller re-runs the unit.

    Returns only the segments produced by chunks actually run in *this* call
    (i.e. from `start_chunk` onward); resumed-over chunks come from the caller's
    persisted log, not from here.
    """
    if _is_parakeet_model(model_name):
        return _transcribe_with_parakeet(model_name, audio_path, progress_callback, chunk_sink)

    if _is_groq_model(model_name):
        return _transcribe_with_groq(model_name, audio_path, language, progress_callback, chunk_sink)

    repo = _resolve_model(model_name)
    if use_vad:
        return _transcribe_whisper_vad(
            repo, audio_path, language, progress_callback, chunk_sink, start_chunk
        )
    return _transcribe_whisper_plain(repo, audio_path, language, chunk_sink)


def _transcribe_whisper_plain(
    repo: str, audio_path: Path, language: str | None, chunk_sink: ChunkSink | None = None
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
    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            start, end = _seg_times(seg)
            segments.append(TranscribedSegment(start=start, end=end, text=text))
    # The plain path is one un-chunked decode: the whole file is chunk 0.
    if chunk_sink is not None:
        chunk_sink(0, segments)
    return segments


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
    chunk_sink: ChunkSink | None = None,
    start_chunk: int = 0,
) -> list[TranscribedSegment]:
    ensure_mlx_whisper_available()
    import mlx_whisper
    from mlx_whisper.audio import load_audio

    audio = load_audio(str(audio_path))  # float32 mono @ 16 kHz
    chunks = _speech_chunks(audio)
    if not chunks:
        return []

    total_speech = sum(e - s for s, e in chunks) or 1
    # On resume, count the skipped chunks as already-done speech so the bar picks
    # up where it left off instead of jumping back to 0.
    done_speech = sum(e - s for s, e in chunks[:start_chunk])
    if start_chunk and progress_callback is not None:
        progress_callback(min(1.0, done_speech / total_speech))
    segments: list[TranscribedSegment] = []
    for index in range(start_chunk, len(chunks)):
        start, end = chunks[index]
        offset = start / _VAD_SAMPLE_RATE
        result = mlx_whisper.transcribe(
            audio[start:end],
            path_or_hf_repo=repo,
            language=language,
            verbose=False,
            **WHISPER_DECODE_OPTS,
        )
        chunk_segments: list[TranscribedSegment] = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                seg_start, seg_end = _seg_times(seg)
                chunk_segments.append(TranscribedSegment(
                    start=offset + seg_start,
                    end=offset + seg_end,
                    text=text,
                ))
        # Hand off this chunk's segments for durable persistence BEFORE moving on,
        # so a stop/crash after this point keeps the chunk's work.
        if chunk_sink is not None:
            chunk_sink(index, chunk_segments)
        segments.extend(chunk_segments)
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
    model_name: str,
    audio_path: Path,
    progress_callback: ProgressCallback | None = None,
    chunk_sink: ChunkSink | None = None,
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
    segments = [
        TranscribedSegment(start=float(sentence.start), end=float(sentence.end), text=sentence.text.strip())
        for sentence in result.sentences
        if sentence.text.strip()
    ]
    # Parakeet's chunking is internal and opaque, so the whole file resumes as a
    # single unit: emit it all as chunk 0 once the one call returns.
    if chunk_sink is not None:
        chunk_sink(0, segments)
    return segments


def _transcribe_with_groq(
    model_name: str,
    audio_path: Path,
    language: str | None,
    progress_callback: ProgressCallback | None = None,
    chunk_sink: ChunkSink | None = None,
) -> list[TranscribedSegment]:
    """Transcribe via Groq's hosted Whisper API (one HTTP call, whole file).

    Needs `GROQ_API_KEY` in the environment (loaded from `.env` at app
    startup, inherited by this engine subprocess) and the `groq` package.
    `verbose_json` is required, not the default `json`/`text`, since it's the
    only response format that returns per-segment start/end timestamps - the
    `dialogue`/multi-track pipeline is built entirely on those.

    No mid-call progress or chunking is possible (it's a single blocking
    upload+inference request), so this matches Parakeet's shape: the whole
    result lands as chunk 0, and `progress_callback` (if given) only ever
    fires once, at completion.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Cloud transcription isn't set up - missing API key. Switch to "
            "Faster or Best to transcribe this one on-device. (To enable Cloud, "
            "add GROQ_API_KEY to .env - a free key is at console.groq.com/keys.)"
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "groq is not installed. Run: pip install -r requirements.txt"
        ) from exc

    groq_model = model_name[len(GROQ_MODEL_PREFIX):]
    client = Groq(api_key=api_key)
    kwargs: dict[str, Any] = {"model": groq_model, "response_format": "verbose_json"}
    if language:
        kwargs["language"] = language

    try:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(file=(audio_path.name, f.read()), **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with user-facing copy
        message = str(exc).lower()
        if "connection" in message or "timed out" in message or "network" in message:
            raise RuntimeError(
                "No internet connection - Cloud needs one. Switch to Faster or "
                "Best to keep working offline."
            ) from exc
        raise RuntimeError(
            "Cloud transcription failed on Groq's end. Try again in a moment, "
            "or switch to Faster or Best."
        ) from exc

    segments = []
    for seg in (getattr(result, "segments", None) or []):
        get = seg.get if isinstance(seg, dict) else lambda k, d=None: getattr(seg, k, d)
        text = (get("text") or "").strip()
        if text:
            segments.append(TranscribedSegment(start=float(get("start")), end=float(get("end")), text=text))

    if progress_callback is not None:
        progress_callback(1.0)
    if chunk_sink is not None:
        chunk_sink(0, segments)
    return segments
