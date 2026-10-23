"""ASR shootout over the prepped per-speaker Czech clips.

Runs each config against each wav, writes the transcript and records timing +
objective quality proxies (repetition, compression ratio, known-hallucination
phrase hits). Quality judging of the actual Czech is done separately by an
LLM-judge reading these transcripts; these metrics only flag the mechanical
failure modes (looping, silence-hallucination).

Usage:
    python benchmark_asr.py <config_id> [<config_id> ...]   # subset
    python benchmark_asr.py                                  # all configs
"""
from __future__ import annotations

import json
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path("/Users/stage/dev/transcriber")
sys.path.insert(0, str(ROOT))

WORK = Path(
    "/private/tmp/claude-501/-Users-stage-dev-transcriber/"
    "7a0eee61-b76e-48e9-b9a4-bfca94046233/scratchpad/asr_research"
)
SPK = WORK / "speakers"
OUT = WORK / "transcripts"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "results.json"

CLIP_SECONDS = 420.0  # all clips are ~7 min

# A short Czech bias prompt: sales/business call, two speakers, correct diacritics.
CS_PROMPT = (
    "Přepis obchodního telefonního hovoru v češtině. Dva lidé spolu mluví "
    "o penězích, investování, byznysu, klientech a produktu."
)

# Tuned Whisper decode options (per pipeline research): no context carryover,
# short temperature ladder, silence guard. Forced Czech.
TUNED = dict(
    language="cs",
    condition_on_previous_text=False,
    temperature=(0.0, 0.2, 0.4),
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    verbose=False,
)
# Default-ish options: forced Czech but otherwise library defaults (approximates
# what the app does today).
DEFAULT = dict(language="cs", verbose=False)


def _whisper_cfg(repo, extra=None):
    opts = dict(extra or {})
    return {"engine": "mlx-whisper", "repo": repo, "opts": opts}


CONFIGS: dict[str, dict] = {
    # current app behaviour (baselines)
    "small_cs_default": _whisper_cfg("mlx-community/whisper-small-mlx", DEFAULT),
    "medium_cs_default": _whisper_cfg("mlx-community/whisper-medium-mlx", DEFAULT),
    # tuned Whisper family
    "medium_cs_tuned": _whisper_cfg("mlx-community/whisper-medium-mlx", TUNED),
    "largev3_cs_tuned": _whisper_cfg("mlx-community/whisper-large-v3-mlx", TUNED),
    "largev3_cs_tuned_prompt": _whisper_cfg(
        "mlx-community/whisper-large-v3-mlx", {**TUNED, "initial_prompt": CS_PROMPT}
    ),
    "turbo_cs_tuned": _whisper_cfg("mlx-community/whisper-large-v3-turbo", TUNED),
    # parakeet (auto language; no biasing)
    "parakeet_mlx": {"engine": "parakeet-mlx", "repo": "mlx-community/parakeet-tdt-0.6b-v3"},
    "parakeet_coreml": {"engine": "silicon-asr", "repo": "parakeet-coreml"},
    # Czech-finetuned Whisper large-v3 via faster-whisper (CTranslate2) + Silero VAD
    "czech_ft_vad": {
        "engine": "faster-whisper",
        "repo": "ThinkHome/whisper-large-v3-czech-cv13-ct2-fp16",
        "opts": {"language": "cs", "condition_on_previous_text": False,
                 "temperature": [0.0, 0.2, 0.4], "no_speech_threshold": 0.6,
                 "vad_filter": True, "beam_size": 5},
    },
    "czech_ft_novad": {
        "engine": "faster-whisper",
        "repo": "ThinkHome/whisper-large-v3-czech-cv13-ct2-fp16",
        "opts": {"language": "cs", "condition_on_previous_text": False,
                 "temperature": [0.0, 0.2, 0.4], "no_speech_threshold": 0.6,
                 "vad_filter": False, "beam_size": 5},
    },
    # vanilla large-v3 via faster-whisper + VAD, to separate the VAD effect from the finetune
    "largev3_fw_vad": {
        "engine": "faster-whisper",
        "repo": "mlx-community/whisper-large-v3-mlx",  # placeholder; overridden below
        "opts": {"language": "cs", "condition_on_previous_text": False,
                 "temperature": [0.0, 0.2, 0.4], "no_speech_threshold": 0.6,
                 "vad_filter": True, "beam_size": 5},
    },
}
# vanilla large-v3 for faster-whisper needs a CTranslate2 build:
CONFIGS["largev3_fw_vad"]["repo"] = "Systran/faster-whisper-large-v3"

_FW_MODELS: dict[str, object] = {}


def run_faster_whisper(repo: str, wav: Path, opts: dict) -> str:
    from faster_whisper import WhisperModel
    if repo not in _FW_MODELS:
        _FW_MODELS[repo] = WhisperModel(repo, device="cpu", compute_type="int8")
    model = _FW_MODELS[repo]
    segments, _info = model.transcribe(str(wav), **opts)
    return "\n".join(s.text.strip() for s in segments if s.text.strip())

HALLUCINATION_PHRASES = [
    "titulky", "amara.org", "děkuji za pozornost", "thank you for watching",
    "přepis", "www.", ".cz/titulky", "produkce", "český rozhlas",
]


def quality_metrics(text: str) -> dict:
    words = text.split()
    n = len(words)
    # fraction of word-5-grams that repeat (loop detector)
    grams = [" ".join(words[i:i + 5]) for i in range(max(0, n - 4))]
    rep = 1.0 - (len(set(grams)) / len(grams)) if grams else 0.0
    comp = len(text) / max(1, len(zlib.compress(text.encode("utf-8"))))
    low = text.lower()
    hits = [p for p in HALLUCINATION_PHRASES if p in low]
    return {
        "chars": len(text),
        "words": n,
        "repeat_5gram_frac": round(rep, 4),
        "compression_ratio": round(comp, 3),
        "hallucination_hits": hits,
    }


def run_whisper(repo: str, wav: Path, opts: dict) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, **opts)
    segs = [s.get("text", "").strip() for s in result.get("segments", [])]
    return "\n".join(s for s in segs if s)


def run_parakeet_mlx(repo: str, wav: Path) -> str:
    from src.transcription import transcribe_mp3_segments
    segs = transcribe_mp3_segments(repo, wav)
    return "\n".join(s.text for s in segs if s.text.strip())


def run_silicon(runner: str, wav: Path) -> str:
    from silicon_asr.runners import create_runner
    r = create_runner(runner)
    return r.transcribe_file(str(wav)).text


def main() -> None:
    import os
    wanted = sys.argv[1:] or list(CONFIGS)
    track_filter = os.environ.get("ASR_TRACKS")  # comma-separated substrings
    wavs = sorted(SPK.glob("*.wav"))
    if track_filter:
        subs = [s for s in track_filter.split(",") if s]
        wavs = [w for w in wavs if any(s in w.stem for s in subs)]
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

    for cid in wanted:
        cfg = CONFIGS[cid]
        for wav in wavs:
            key = f"{wav.stem}__{cid}"
            t0 = time.perf_counter()
            try:
                if cfg["engine"] == "mlx-whisper":
                    text = run_whisper(cfg["repo"], wav, cfg["opts"])
                elif cfg["engine"] == "parakeet-mlx":
                    text = run_parakeet_mlx(cfg["repo"], wav)
                elif cfg["engine"] == "faster-whisper":
                    text = run_faster_whisper(cfg["repo"], wav, cfg["opts"])
                else:
                    text = run_silicon(cfg["repo"], wav)
                err = None
            except Exception as exc:  # noqa: BLE001
                text, err = "", f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - t0

            (OUT / f"{key}.txt").write_text(text)
            entry = {
                "config": cid, "track": wav.stem, "engine": cfg["engine"],
                "repo": cfg.get("repo"), "elapsed_s": round(elapsed, 2),
                "rtf": round(elapsed / CLIP_SECONDS, 4), "error": err,
                **quality_metrics(text),
            }
            results[key] = entry
            RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            flag = f" ERROR {err}" if err else (
                f" rep={entry['repeat_5gram_frac']} comp={entry['compression_ratio']}"
                f" halluc={entry['hallucination_hits']}"
            )
            print(f"[{cid}] {wav.stem}: {elapsed:.1f}s ({entry['rtf']:.2f}x) "
                  f"words={entry['words']}{flag}", flush=True)


if __name__ == "__main__":
    main()
