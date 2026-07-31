from __future__ import annotations


# (display label, mlx-community repo id)
MODEL_CHOICES: list[tuple[str, str]] = [
    ("Tiny - fastest, lowest accuracy", "mlx-community/whisper-tiny-mlx"),
    ("Base", "mlx-community/whisper-base-mlx"),
    ("Small - default", "mlx-community/whisper-small-mlx"),
    ("Medium - recommended for Czech", "mlx-community/whisper-medium-mlx"),
    ("Large-v3 - best accuracy, slowest", "mlx-community/whisper-large-v3-mlx"),
    ("Large-v3-turbo - fast + accurate", "mlx-community/whisper-large-v3-turbo"),
]

DEFAULT_MODEL = "mlx-community/whisper-small-mlx"

# (display label, language code or None for auto-detect)
LANGUAGE_CHOICES: list[tuple[str, str | None]] = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Czech", "cs"),
    ("Slovak", "sk"),
    ("German", "de"),
]

DEFAULT_LANGUAGE: str | None = "en"

# Rough real-time factor (elapsed_seconds / audio_duration_seconds) per model,
# used only until real history exists for that model.
DEFAULT_REAL_TIME_FACTOR: dict[str, float] = {
    "mlx-community/whisper-tiny-mlx": 0.12,
    "mlx-community/whisper-base-mlx": 0.18,
    "mlx-community/whisper-small-mlx": 0.28,
    "mlx-community/whisper-medium-mlx": 0.45,
    "mlx-community/whisper-large-v3-mlx": 0.75,
    "mlx-community/whisper-large-v3-turbo": 0.35,
}
FALLBACK_REAL_TIME_FACTOR = 0.35


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
