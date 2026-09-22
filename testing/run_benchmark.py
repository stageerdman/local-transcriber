#!/usr/bin/env python3
"""Benchmark transcription backends on a single sample video/audio file.

Usage:
    python run_benchmark.py path/to/video.mp4
    python run_benchmark.py path/to/video.mp4 --models whisper-small-mlx,parakeet-tdt-0.6b-v3
    python run_benchmark.py --list-models

Drop a sample file in testing/videos/ and it'll be picked up automatically
if you don't pass a path. Results (timing + transcript text) are written to
testing/results/ and merged into results/results.json, which the rating UI
(rate_server.py) reads from. Re-running is safe - existing ratings for a
model are preserved, only that model's timing/transcript gets refreshed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models_config import BACKENDS, BACKENDS_BY_ID, LOCAL_TRANSCRIBER_ROOT

TESTING_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = TESTING_ROOT / "videos"
RESULTS_DIR = TESTING_ROOT / "results"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
AUDIO_DIR = RESULTS_DIR / "audio"
RESULTS_JSON = RESULTS_DIR / "results.json"


def find_sample_video() -> Path:
    candidates = sorted(
        p for p in VIDEOS_DIR.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if not candidates:
        raise SystemExit(
            f"No video found in {VIDEOS_DIR}. Pass a path explicitly or drop a file there."
        )
    if len(candidates) > 1:
        print(f"Multiple files in {VIDEOS_DIR}, using the first: {candidates[0].name}")
    return candidates[0]


def prepare_audio(video_path: Path, force: bool = False) -> Path:
    from src.converter import check_ffmpeg_installed, prepare_mp3

    if not check_ffmpeg_installed():
        raise SystemExit("ffmpeg is not installed (needed to extract audio). brew install ffmpeg")

    audio_path = AUDIO_DIR / f"{video_path.stem}.mp3"
    if force or not audio_path.exists():
        print(f"Extracting audio -> {audio_path.name}")
        prepare_mp3(video_path, audio_path)
    else:
        print(f"Reusing cached audio: {audio_path.name}")
    return audio_path


def load_results() -> dict:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {"video": None, "audio_duration_seconds": None, "runs": {}}


def save_results(data: dict) -> None:
    RESULTS_JSON.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, help="Path to a video/audio file")
    parser.add_argument(
        "--models",
        help="Comma-separated backend ids to run (default: all). See --list-models.",
    )
    parser.add_argument("--language", default="auto", help="Language code, or 'auto' (default: auto)")
    parser.add_argument("--force-audio", action="store_true", help="Re-extract audio even if cached")
    parser.add_argument("--list-models", action="store_true", help="List available backend ids and exit")
    args = parser.parse_args()

    if args.list_models:
        for b in BACKENDS:
            print(f"{b.id:30s} {b.label}")
        return

    video_path = args.video or find_sample_video()
    if not video_path.exists():
        raise SystemExit(f"File not found: {video_path}")

    language = None if args.language == "auto" else args.language

    if args.models:
        ids = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in ids if m not in BACKENDS_BY_ID]
        if unknown:
            raise SystemExit(f"Unknown model id(s): {unknown}. Run --list-models to see options.")
        backends = [BACKENDS_BY_ID[m] for m in ids]
    else:
        backends = BACKENDS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = prepare_audio(video_path, force=args.force_audio)

    from src.duration import probe_duration_seconds

    duration = probe_duration_seconds(audio_path)
    print(f"Audio duration: {duration:.1f}s\n")

    data = load_results()
    data["video"] = video_path.name
    data["audio_duration_seconds"] = duration
    data.setdefault("runs", {})

    for backend in backends:
        print(f"--- {backend.label} ({backend.id}) ---")
        try:
            start = time.perf_counter()
            result = backend.run(audio_path, language)
            total_seconds = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001 - keep benchmarking the rest
            print(f"FAILED: {exc}\n")
            data["runs"][backend.id] = {
                **data["runs"].get(backend.id, {}),
                "id": backend.id,
                "label": backend.label,
                "engine": backend.engine,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_results(data)
            continue

        transcript_path = TRANSCRIPTS_DIR / f"{backend.id}.txt"
        transcript_path.write_text(result.text)

        realtime_factor = result.transcribe_seconds / duration if duration else None
        speedup = duration / result.transcribe_seconds if result.transcribe_seconds else None

        existing = data["runs"].get(backend.id, {})
        data["runs"][backend.id] = {
            "id": backend.id,
            "label": backend.label,
            "engine": backend.engine,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "load_seconds": round(result.load_seconds, 3),
            "transcribe_seconds": round(result.transcribe_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "realtime_factor": round(realtime_factor, 4) if realtime_factor else None,
            "speedup_vs_realtime": round(speedup, 2) if speedup else None,
            "transcript_file": str(transcript_path.relative_to(TESTING_ROOT)),
            "error": None,
            # Preserve any rating/notes a previous run already collected.
            "rating": existing.get("rating"),
            "notes": existing.get("notes", ""),
        }
        save_results(data)

        print(f"load: {result.load_seconds:.2f}s  transcribe: {result.transcribe_seconds:.2f}s  "
              f"({speedup:.1f}x realtime)\n" if speedup else "")

    print(f"\nDone. Results: {RESULTS_JSON}")
    print(f"Rate them: python rate_server.py")


if __name__ == "__main__":
    main()
