from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.dialogue import Word

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
    # This segment's own word-level timing, when the backend can give it (see
    # the per-backend extraction helpers below). Empty when it can't -
    # `dialogue._words_for_segment`'s synthetic-whole-segment fallback is the
    # one place that's resolved, so callers never need to check for this.
    words: list[Word] = field(default_factory=list)


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


# Cloud backend #2: ElevenLabs' Scribe v2 - a different model, not just a
# different host, and markedly better on Czech than stock Whisper (3-5% WER
# on FLEURS/Common Voice Czech vs. Whisper-class scoring far worse - see the
# cloud-model-research spike in this update's wiki). Kept alongside Groq, not
# swapped in for it, so cost/speed (Groq) and Czech accuracy (ElevenLabs) are
# each one click away.
ELEVENLABS_MODEL_PREFIX = "elevenlabs:"


def _is_elevenlabs_model(model_name: str) -> bool:
    return model_name.startswith(ELEVENLABS_MODEL_PREFIX)


ELEVENLABS_ID = f"{ELEVENLABS_MODEL_PREFIX}scribe_v2"


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


def _whisper_segment_words(seg: dict, offset: float = 0.0) -> list[Word]:
    """A Whisper segment's own word-level timing (`word_timestamps=True`).

    Each word dict's `"word"` text carries Whisper's own leading-space
    convention (a leading space means it's a new word; punctuation is already
    merged into the previous word with none, by `mlx_whisper`'s own
    `merge_punctuations`), so that's the direct source for `lead_space`.
    `offset` shifts into whole-file time for the VAD path, where `seg`'s own
    times are chunk-relative."""
    out: list[Word] = []
    for w in seg.get("words", []):
        start, end = w.get("start"), w.get("end")
        if start is None or end is None:
            continue
        raw = w.get("word", "")
        out.append(Word(
            start=offset + float(start), end=offset + float(end),
            text=raw.strip(), lead_space=raw.startswith(" "),
        ))
    return out


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

    if _is_groq_model(model_name) or _is_elevenlabs_model(model_name):
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

    if _is_elevenlabs_model(model_name):
        return _transcribe_with_elevenlabs(model_name, audio_path, language, progress_callback, chunk_sink)

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
            try:
                words = _whisper_segment_words(seg)
            except Exception:  # noqa: BLE001 - one segment's own words are optional
                words = []
            segments.append(TranscribedSegment(start=start, end=end, text=text, words=words))
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
                try:
                    words = _whisper_segment_words(seg, offset=offset)
                except Exception:  # noqa: BLE001 - one segment's own words are optional
                    words = []
                chunk_segments.append(TranscribedSegment(
                    start=offset + seg_start,
                    end=offset + seg_end,
                    text=text,
                    words=words,
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


def _parakeet_sentence_words(tokens: list[Any]) -> list[Word]:
    """Group a Parakeet sentence's sub-word tokens into whole words.

    `token.start`/`.end` are already absolute-to-the-whole-file (parakeet_mlx
    applies the chunk offset before cross-chunk merging), so no extra offset
    math is needed. A token starts a new word iff its own `.text` has a
    leading space - the same check `parakeet_mlx.alignment.tokens_to_sentences`
    uses internally to find word boundaries (confirmed by reading that
    source); punctuation tokens have no leading space and attach to the
    previous word, e.g. ' r', 'ô', 'z', 'ne', '.' -> 'rôzne.'."""
    words: list[Word] = []
    parts: list[str] = []
    start = end = 0.0
    lead_space = True
    for tok in tokens:
        if parts and tok.text.startswith(" "):
            words.append(Word(start, end, "".join(parts).strip(), lead_space))
            parts = []
        if not parts:
            start = tok.start
            lead_space = tok.text.startswith(" ")
        parts.append(tok.text)
        end = tok.end
    if parts:
        words.append(Word(start, end, "".join(parts).strip(), lead_space))
    return words


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
    segments = []
    for sentence in result.sentences:
        if not sentence.text.strip():
            continue
        try:
            words = _parakeet_sentence_words(sentence.tokens)
        except Exception:  # noqa: BLE001 - one segment's own words are optional
            words = []
        segments.append(TranscribedSegment(
            start=float(sentence.start), end=float(sentence.end),
            text=sentence.text.strip(), words=words,
        ))
    # Parakeet's chunking is internal and opaque, so the whole file resumes as a
    # single unit: emit it all as chunk 0 once the one call returns.
    if chunk_sink is not None:
        chunk_sink(0, segments)
    return segments


# 64 kbps mono keeps a 25-minute call around ~12 MB - well under Groq's
# per-request size cap (a 413 "request_too_large" on an uncompressed
# multi-track WAV, confirmed in testing, is what this exists to prevent).
# Local backends read `audio_path` directly - mp3 for single-track, or an
# uncompressed 16kHz WAV for an isolated multi-track unit - since on-device
# decoding never cared about file size; only the upload to Groq does.
_GROQ_UPLOAD_BITRATE = "64k"


def _compact_mp3_for_upload(audio_path: Path) -> Path:
    """Re-encode to a small mono mp3 in a temp file before uploading to Groq.

    Always re-encodes, even if `audio_path` is already a small mp3 - simpler
    and more robust than special-casing "is this already small enough", and
    the extra ffmpeg pass is seconds against an upload+inference call that's
    already tens of seconds.
    """
    ensure_ffmpeg_available()
    import subprocess
    import tempfile

    fd, tmp_name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    tmp_path = Path(tmp_name)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-vn", "-ac", "1", "-b:a", _GROQ_UPLOAD_BITRATE, str(tmp_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return tmp_path


def ensure_ffmpeg_available() -> None:
    from src.converter import check_ffmpeg_installed

    if not check_ffmpeg_installed():
        raise RuntimeError("ffmpeg is not installed (needed for Cloud transcription). brew install ffmpeg")


def _groq_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _bucket_groq_words_by_segment(segments: list[Any], words: list[Any]) -> list[list[Any]]:
    """Assign Groq's flat per-file word list to the segments it belongs to.

    Groq returns `words` as one flat list across the whole file, not nested
    per segment (confirmed via this update's own
    `spike/groq_word_granularity.py`), so there's no index to zip against
    directly. Bucketing by matching each word's `start` against segment time
    boundaries was tried first and rejected: on a real ~60s call, word and
    segment timestamps come from independent alignment passes that drift
    apart over the file, and by the 7th segment the drift was a whole word
    wide, shifting every later bucket. Word COUNT doesn't drift - Groq's own
    segment `text`, split on whitespace, has exactly as many tokens as the
    `words` entries that make it up (verified on that same real call: 116
    either way), since punctuation attaches to its word in both views - so
    segments are filled by consuming that many words off the front of the
    flat list, in order."""
    buckets: list[list[Any]] = []
    i = 0
    for seg in segments:
        n = len((_groq_get(seg, "text") or "").split())
        buckets.append(words[i:i + n])
        i += n
    return buckets


def _groq_bucket_words(bucket: list[Any]) -> list[Word]:
    """Groq's word objects (confirmed shape: `word`/`start`/`end`, no leading-
    space marker) read as plain whitespace-separated text - same convention
    the segment `text` itself already joins on - so every word simply gets a
    separating space before it."""
    out: list[Word] = []
    for w in bucket:
        start, end = _groq_get(w, "start"), _groq_get(w, "end")
        if start is None or end is None:
            continue
        text = (_groq_get(w, "word") or "").strip()
        if text:
            out.append(Word(start=float(start), end=float(end), text=text, lead_space=True))
    return out


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
            "Fast or Slow to transcribe this one on-device. (To enable Cloud, "
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
    # Word granularity costs no extra latency over segment-only (confirmed via
    # this update's own `spike/groq_word_granularity.py`), so it's always
    # requested alongside the segment timestamps the rest of this pipeline
    # already depends on.
    kwargs: dict[str, Any] = {
        "model": groq_model, "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if language:
        kwargs["language"] = language

    upload_path = _compact_mp3_for_upload(audio_path)
    try:
        with open(upload_path, "rb") as f:
            result = client.audio.transcriptions.create(file=(upload_path.name, f.read()), **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with user-facing copy
        message = str(exc).lower()
        # The underlying message is always appended (not swallowed) - without
        # it, a real failure (rate limit, payload-too-large, auth) is
        # indistinguishable from a transient one and unfixable from the UI's
        # error text alone.
        if "413" in message or "too large" in message or "request_too_large" in message:
            raise RuntimeError(
                "This call is too long for Cloud - even compressed, it's over "
                f"Groq's upload limit. Use Fast or Slow for this one. ({exc})"
            ) from exc
        if "connection" in message or "timed out" in message or "network" in message:
            raise RuntimeError(
                "No internet connection - Cloud needs one. Switch to Fast or "
                f"Slow to keep working offline. ({exc})"
            ) from exc
        raise RuntimeError(
            "Cloud transcription failed on Groq's end. Try again in a moment, "
            f"or switch to Fast or Slow. ({exc})"
        ) from exc
    finally:
        upload_path.unlink(missing_ok=True)

    raw_segments = getattr(result, "segments", None) or []
    raw_words = getattr(result, "words", None) or []
    word_buckets = _bucket_groq_words_by_segment(raw_segments, raw_words)
    segments = []
    for seg, bucket in zip(raw_segments, word_buckets):
        text = (_groq_get(seg, "text") or "").strip()
        if text:
            try:
                words = _groq_bucket_words(bucket)
            except Exception:  # noqa: BLE001 - one segment's own words are optional
                words = []
            segments.append(TranscribedSegment(
                start=float(_groq_get(seg, "start")), end=float(_groq_get(seg, "end")),
                text=text, words=words,
            ))

    if progress_callback is not None:
        progress_callback(1.0)
    if chunk_sink is not None:
        chunk_sink(0, segments)
    return segments


# ElevenLabs returns only a flat per-word stream (no sentence/segment grouping
# like Whisper/Groq give), so this reconstructs segments the `dialogue`
# pipeline can consume: a new segment starts on a pause longer than this gap,
# or right after sentence-ending punctuation. The gap is kept just under
# `dialogue.DEFAULT_PAUSE_SECONDS` (0.8s) so one speaker's natural mid-sentence
# breath still gets re-merged into a single block downstream instead of
# staying split into two.
_ELEVENLABS_SEGMENT_GAP_SECONDS = 0.6


def _elevenlabs_tokens_to_words(tokens: list[Any]) -> list[Word]:
    """Real words only - `spacing`/`audio_event` entries aren't spoken words,
    but a `spacing` entry's own text IS this codebase's one real signal for
    whether a space actually separates the words on either side of it (e.g.
    none before attached punctuation), so it's read here, not discarded."""
    out: list[Word] = []
    pending_space = False
    for t in tokens:
        ttype = getattr(t, "type", None)
        if ttype == "spacing":
            if getattr(t, "text", None):
                pending_space = True
            continue
        if ttype != "word" or t.start is None or t.end is None:
            continue
        out.append(Word(start=float(t.start), end=float(t.end), text=t.text, lead_space=pending_space))
        pending_space = False
    return out


def _coalesce_elevenlabs_words(tokens: list[Any]) -> list[TranscribedSegment]:
    """Group ElevenLabs' flat per-word stream into segments the `dialogue`
    pipeline can consume: a new segment starts on a pause longer than this
    gap, or right after sentence-ending punctuation. The gap is kept just
    under `dialogue.DEFAULT_PAUSE_SECONDS` (0.8s) so one speaker's natural
    mid-sentence breath still gets re-merged into a single block downstream
    instead of staying split into two."""
    words = _elevenlabs_tokens_to_words(tokens)
    segments: list[TranscribedSegment] = []
    current: list[Word] = []
    for w in words:
        if current and (
            w.start - current[-1].end > _ELEVENLABS_SEGMENT_GAP_SECONDS
            or current[-1].text.rstrip().endswith((".", "!", "?"))
        ):
            segments.append(_elevenlabs_segment_from_words(current))
            current = []
        current.append(w)
    if current:
        segments.append(_elevenlabs_segment_from_words(current))
    return segments


def _elevenlabs_segment_from_words(words: list[Word]) -> TranscribedSegment:
    parts: list[str] = []
    for i, w in enumerate(words):
        if i and w.lead_space:
            parts.append(" ")
        parts.append(w.text)
    return TranscribedSegment(
        start=words[0].start, end=words[-1].end, text="".join(parts).strip(), words=list(words),
    )


def _transcribe_with_elevenlabs(
    model_name: str,
    audio_path: Path,
    language: str | None,
    progress_callback: ProgressCallback | None = None,
    chunk_sink: ChunkSink | None = None,
) -> list[TranscribedSegment]:
    """Transcribe via ElevenLabs' hosted Scribe API (one HTTP call, whole file).

    Needs `ELEVENLABS_API_KEY` in the environment (same `.env` loading as
    Groq's key). Scribe's 5GB upload limit is generous enough that, unlike
    Groq, the source file is sent as-is - no re-encode/compaction pass needed.

    Shaped like `_transcribe_with_groq`: one blocking call, the whole result
    lands as chunk 0, `progress_callback` fires once at completion.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Cloud transcription isn't set up - missing API key. Switch to "
            "Fast or Slow to transcribe this one on-device. (To enable "
            "ElevenLabs Cloud, add ELEVENLABS_API_KEY to .env - get one at "
            "elevenlabs.io.)"
        )
    try:
        from elevenlabs import ElevenLabs
    except ImportError as exc:
        raise RuntimeError(
            "elevenlabs is not installed. Run: pip install -r requirements.txt"
        ) from exc

    eleven_model = model_name[len(ELEVENLABS_MODEL_PREFIX):]
    client = ElevenLabs(api_key=api_key)
    kwargs: dict[str, Any] = {"model_id": eleven_model, "timestamps_granularity": "word"}
    if language:
        kwargs["language_code"] = language

    try:
        with open(audio_path, "rb") as f:
            result = client.speech_to_text.convert(file=f, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with user-facing copy
        message = str(exc).lower()
        # Same reasoning as Groq's handler above: the underlying message is
        # always appended, never swallowed.
        if "413" in message or "too large" in message or "payload" in message:
            raise RuntimeError(
                "This call is too long for Cloud - over ElevenLabs' upload "
                f"limit. Use Fast or Slow for this one. ({exc})"
            ) from exc
        if "connection" in message or "timed out" in message or "network" in message:
            raise RuntimeError(
                "No internet connection - Cloud needs one. Switch to Fast or "
                f"Slow to keep working offline. ({exc})"
            ) from exc
        raise RuntimeError(
            "Cloud transcription failed on ElevenLabs' end. Try again in a "
            f"moment, or switch to Fast or Slow. ({exc})"
        ) from exc

    words = getattr(result, "words", None) or []
    segments = _coalesce_elevenlabs_words(words)

    if progress_callback is not None:
        progress_callback(1.0)
    if chunk_sink is not None:
        chunk_sink(0, segments)
    return segments
