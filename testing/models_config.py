"""Registry of transcription "services" (models/backends) to benchmark.

Each entry is a small config describing how to run one transcription
backend against an audio file. Today everything here is a local MLX model
(reusing local-transcriber's own engine code), matching what the app
already ships in src/catalog.py. To add a cloud API later, add a new
TranscriptionBackend with a `transcribe` function that shells out to that
API and returns plain text - the benchmark runner and rating UI don't care
how the text was produced.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# local-transcriber's src/ package holds the actual mlx-whisper / parakeet
# integration - reuse it instead of duplicating model-loading logic.
LOCAL_TRANSCRIBER_ROOT = Path(__file__).resolve().parent.parent / "local-transcriber"
if str(LOCAL_TRANSCRIBER_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_TRANSCRIBER_ROOT))


@dataclass
class TranscribeResult:
    text: str
    load_seconds: float
    transcribe_seconds: float


@dataclass
class TranscriptionBackend:
    id: str  # stable key, used as filename / results.json key
    label: str  # human-readable name shown in the rating UI
    engine: str  # short tag: "mlx-whisper", "parakeet", "api", ...
    run: Callable[[Path, str | None], TranscribeResult]


def _make_local_mlx_backend(model_repo: str, label: str) -> TranscriptionBackend:
    def run(audio_path: Path, language: str | None) -> TranscribeResult:
        from src.transcription import ensure_model_loaded, transcribe_mp3_segments

        load_start = time.perf_counter()
        ensure_model_loaded(model_repo)
        load_seconds = time.perf_counter() - load_start

        transcribe_start = time.perf_counter()
        segments = transcribe_mp3_segments(model_repo, audio_path, language=language)
        transcribe_seconds = time.perf_counter() - transcribe_start

        text = "\n".join(seg.text for seg in segments)
        return TranscribeResult(text=text, load_seconds=load_seconds, transcribe_seconds=transcribe_seconds)

    safe_id = model_repo.split("/")[-1]
    engine = "parakeet-mlx" if "parakeet" in model_repo.lower() else "mlx-whisper"
    return TranscriptionBackend(id=safe_id, label=label, engine=engine, run=run)


def _make_silicon_asr_backend(runner_name: str, label: str) -> TranscriptionBackend:
    """Apple Neural Engine (CoreML) runners via the `silicon-asr` package.

    https://pypi.org/project/silicon-asr/ - FluidInference's CoreML
    conversions of NVIDIA's Parakeet models, run on the ANE instead of the
    GPU (which is what mlx-whisper/parakeet-mlx use). Distinct hardware
    path, so it's worth benchmarking separately even though the underlying
    Parakeet weights are the same family already covered above.

    `create_runner()` downloads (first run only) and loads the model, so
    that's what we time as "load"; `transcribe_file()` is "transcribe".
    """

    def run(audio_path: Path, language: str | None) -> TranscribeResult:
        try:
            from silicon_asr.runners import create_runner
        except ImportError as exc:
            raise RuntimeError(
                "silicon-asr is not installed. Run: pip install -r requirements-testing.txt"
            ) from exc

        load_start = time.perf_counter()
        runner = create_runner(runner_name)
        load_seconds = time.perf_counter() - load_start

        transcribe_start = time.perf_counter()
        result = runner.transcribe_file(str(audio_path))
        transcribe_seconds = time.perf_counter() - transcribe_start

        return TranscribeResult(text=result.text, load_seconds=load_seconds, transcribe_seconds=transcribe_seconds)

    return TranscriptionBackend(id=runner_name, label=label, engine="silicon-asr", run=run)


def _make_groq_backend(model_name: str, label: str) -> TranscriptionBackend:
    """Groq isn't a model - it's an inference host running open models
    (Whisper Large-v3 / Large-v3-turbo) on their LPU hardware over an API.
    Included specifically to see how much "just run it on faster hardware"
    buys over the same weights running locally in MLX.

    Requires `pip install groq` and a GROQ_API_KEY environment variable.
    There's no local model to load, so load_seconds is always 0 and the
    whole request (upload + inference) counts as "transcribe".
    """

    def run(audio_path: Path, language: str | None) -> TranscribeResult:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq is not installed. Run: pip install -r requirements-testing.txt"
            ) from exc

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")

        client = Groq(api_key=api_key)

        kwargs: dict = {"model": model_name, "response_format": "text"}
        if language:
            kwargs["language"] = language

        transcribe_start = time.perf_counter()
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(file=(audio_path.name, f.read()), **kwargs)
        transcribe_seconds = time.perf_counter() - transcribe_start

        text = transcription if isinstance(transcription, str) else getattr(transcription, "text", str(transcription))
        return TranscribeResult(text=text, load_seconds=0.0, transcribe_seconds=transcribe_seconds)

    return TranscriptionBackend(id=f"groq-{model_name}", label=label, engine="groq-api", run=run)


# Same models already offered in local-transcriber/src/catalog.py's
# MODEL_CHOICES, plus a couple of other "ways" to run comparable models:
# silicon-asr (same Parakeet weights, but on the Apple Neural Engine instead
# of the GPU) and Groq (same Whisper weights, hosted on their LPU hardware).
# Trim this list (or pass --models on the CLI) to skip slower/unavailable
# ones during a quick pass.
BACKENDS: list[TranscriptionBackend] = [
    _make_local_mlx_backend("mlx-community/whisper-tiny-mlx", "Whisper Tiny (MLX)"),
    _make_local_mlx_backend("mlx-community/whisper-base-mlx", "Whisper Base (MLX)"),
    _make_local_mlx_backend("mlx-community/whisper-small-mlx", "Whisper Small (MLX)"),
    _make_local_mlx_backend("mlx-community/whisper-medium-mlx", "Whisper Medium (MLX)"),
    _make_local_mlx_backend("mlx-community/whisper-large-v3-mlx", "Whisper Large-v3 (MLX)"),
    _make_local_mlx_backend("mlx-community/whisper-large-v3-turbo", "Whisper Large-v3-turbo (MLX)"),
    _make_local_mlx_backend("mlx-community/parakeet-tdt-0.6b-v3", "Parakeet TDT v3 (MLX, GPU)"),
    _make_silicon_asr_backend("parakeet-coreml", "Parakeet TDT v3 (CoreML, ANE)"),
    _make_silicon_asr_backend("parakeet-v2-coreml", "Parakeet TDT v2 (CoreML, ANE)"),
    _make_silicon_asr_backend("parakeet-ctc-coreml", "Parakeet CTC 110M (CoreML, ANE)"),
    _make_groq_backend("whisper-large-v3", "Whisper Large-v3 (Groq API)"),
    _make_groq_backend("whisper-large-v3-turbo", "Whisper Large-v3-turbo (Groq API)"),
]

BACKENDS_BY_ID: dict[str, TranscriptionBackend] = {b.id: b for b in BACKENDS}
