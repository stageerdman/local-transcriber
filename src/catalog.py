from __future__ import annotations

from src.transcription import CZECH_FT_ID, ELEVENLABS_ID, GROQ_ID

# (display label, model id)
# Czech-finetuned large-v3 (run with Silero VAD) is the default: it won a blind
# quality shootout on real Czech call audio for both clean and isolated tracks
# (see updates/2026-10-01 transcription-quality-research). Parakeet is the fast
# alternative - robust (never hallucinates) but garbles some Czech words.
MODEL_CHOICES: list[tuple[str, str]] = [
    ("Czech - finetuned large-v3, best quality (default)", CZECH_FT_ID),
    ("Cloud - Groq Whisper large-v3-turbo, fastest (needs internet)", GROQ_ID),
    ("Cloud - ElevenLabs Scribe v2, best Czech accuracy (needs internet)", ELEVENLABS_ID),
    ("Parakeet TDT v3 - fastest, robust", "mlx-community/parakeet-tdt-0.6b-v3"),
    ("Whisper large-v3 - multilingual", "mlx-community/whisper-large-v3-mlx"),
    ("Whisper large-v3-turbo - fast + accurate", "mlx-community/whisper-large-v3-turbo"),
    ("Whisper medium", "mlx-community/whisper-medium-mlx"),
    ("Whisper small", "mlx-community/whisper-small-mlx"),
    ("Whisper tiny - fastest, lowest accuracy", "mlx-community/whisper-tiny-mlx"),
]

DEFAULT_MODEL = CZECH_FT_ID

# (display label, language code or None for auto-detect)
LANGUAGE_CHOICES: list[tuple[str, str | None]] = [
    ("Czech", "cs"),
    ("Auto-detect", None),
    ("English", "en"),
    ("Slovak", "sk"),
    ("German", "de"),
]

DEFAULT_LANGUAGE: str | None = "cs"

# LLM polishing (optional cleanup pass via the Claude Code CLI). Default on.
DEFAULT_POLISH_ENABLED = True

# Polish-time estimate, so the polish phase shows as a moving tail of the same
# progress bar instead of a frozen 0. One fast `claude -p` call covers the whole
# transcript; audio duration is a cheap proxy for transcript length.
POLISH_BASE_SECONDS = 8.0
POLISH_RTF = 0.05
# Fraction of the overall bar reserved for the polish tail is clamped to this
# range so neither phase can swallow the whole bar.
POLISH_SHARE_MIN = 0.05
POLISH_SHARE_MAX = 0.40

# Rough real-time factor (elapsed_seconds / audio_duration_seconds) per model,
# used only until real history exists for that model.
DEFAULT_REAL_TIME_FACTOR: dict[str, float] = {
    "mlx-community/whisper-tiny-mlx": 0.12,
    "mlx-community/whisper-base-mlx": 0.18,
    "mlx-community/whisper-small-mlx": 0.28,
    "mlx-community/whisper-medium-mlx": 0.45,
    "mlx-community/whisper-large-v3-mlx": 0.75,
    "mlx-community/whisper-large-v3-turbo": 0.35,
    # Czech finetune is large-v3 run through Silero-VAD-chunked decoding;
    # measured ~0.24x on clean call audio (silence skipped keeps it down).
    CZECH_FT_ID: 0.30,
    # A different architecture (NVIDIA's TDT, not an encoder-decoder like
    # Whisper) - real-world reports put it around 60x realtime on Apple
    # Silicon, well past every Whisper size here.
    "mlx-community/parakeet-tdt-0.6b-v3": 0.05,
    # Hosted on Groq's LPUs, not this Mac's GPU - measured ~0.03-0.04x
    # (25-35x realtime, including network) in the transcription-quality-
    # research update's benchmarking.
    GROQ_ID: 0.03,
    # Placeholder, not yet benchmarked (ElevenLabs was only just added - see
    # the cloud-model-research spike). Ordinary cloud GPU inference rather
    # than Groq's LPUs, so assumed meaningfully slower than Groq but still
    # well under realtime; revisit once a real call has run through it.
    ELEVENLABS_ID: 0.15,
}
FALLBACK_REAL_TIME_FACTOR = 0.35


def estimate_polish_seconds(
    audio_duration_seconds: float | None, polish_enabled: bool
) -> float:
    """Rough wall-clock for the polish pass, used only to animate its progress
    tail. One `claude -p` call covers the whole transcript; audio duration is a
    cheap proxy for transcript length. 0 when polish is off or duration unknown."""
    if not polish_enabled or not audio_duration_seconds:
        return 0.0
    return POLISH_BASE_SECONDS + audio_duration_seconds * POLISH_RTF


def model_label(model: str) -> str:
    for label, value in MODEL_CHOICES:
        if value == model:
            return label
    return model


def language_label(language: str | None) -> str:
    for label, value in LANGUAGE_CHOICES:
        if value == language:
            return label
    return language or "Auto-detect"
