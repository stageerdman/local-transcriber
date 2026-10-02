"""Extract a ~7-min middle clip from each source call and render per-speaker WAVs.

Reuses the real app modules (src.audio_tracks, src.track_separation) so the test
input matches what the app actually feeds the transcriber. Writes:

    <work>/clips/<slug>.mkv              (middle clip, all audio streams copied)
    <work>/speakers/<slug>__<role>.wav   (one clean mono 16k wav per kept speaker)

and prints the separation plan per clip so we know which tracks mattered.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/stage/dev/transcriber")
sys.path.insert(0, str(ROOT))

from src.audio_tracks import probe_audio_streams, track_speaker_names  # noqa: E402
from src.track_separation import (  # noqa: E402
    ANALYSIS_RATE,
    TrackSignal,
    classify_tracks,
    decode_track_pcm,
    render_speaker_wav,
)

WORK = Path(
    "/private/tmp/claude-501/-Users-stage-dev-transcriber/"
    "7a0eee61-b76e-48e9-b9a4-bfca94046233/scratchpad/asr_research"
)
CLIP_SECONDS = 420  # 7 minutes

BASE = Path(
    "/Users/stage/Library/CloudStorage/OneDrive-Personal/Library/My Sales/My Call Analysis"
)

# (slug, folder, filename)
SOURCES = [
    ("vaclav", "2026-09-30 Václav", "Sales Call - Václav (difficult problem).mov"),
    ("tomas", "2026-09-30 Tomáš Král", "Sales Call - Tomáš Král.mov"),
    ("marek", "2026-09-25 Marek Harvan", "Sales Call - Marek Havran.mov"),
    ("roleplay", "2026-09-08 Roleplay", "Role Play - Vlado .mp4"),
]


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def extract_middle_clip(src: Path, dst: Path, clip_seconds: int) -> None:
    dur = probe_duration(src)
    start = max(0.0, dur / 2 - clip_seconds / 2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
         "-t", str(clip_seconds), "-map", "0:a", "-c", "copy", str(dst)],
        check=True,
    )


def main() -> None:
    import os
    only = os.environ.get("ONLY_SLUG")
    for slug, folder, fname in SOURCES:
        if only and slug != only:
            continue
        src = BASE / folder / fname
        if not src.exists():
            print(f"[SKIP] {slug}: missing {src}")
            continue
        clip = WORK / "clips" / f"{slug}.mkv"
        print(f"\n=== {slug} === ({src.name})")
        extract_middle_clip(src, clip, CLIP_SECONDS)

        streams = probe_audio_streams(clip)
        names = track_speaker_names(streams)
        signals = [
            TrackSignal(index=s.index, name=nm,
                        samples=decode_track_pcm(clip, s.index))
            for s, nm in zip(streams, names)
        ]
        plan = classify_tracks(signals)
        by_index = {s.index: s.samples for s in signals}

        for role in plan.roles:
            tag = role.kind
            if role.kind == "isolate":
                tag += f"(-{role.reference_indices})"
            elif role.kind == "duplicate":
                tag += f"(of {role.duplicate_of})"
            print(f"  stream {role.index} [{role.name}] -> {tag}")

        kept = plan.speaker_roles()
        for i, role in enumerate(kept, start=1):
            out = WORK / "speakers" / f"{slug}__spk{i}_s{role.index}.wav"
            render_speaker_wav(role, by_index, out)
            secs = len(by_index[role.index]) / ANALYSIS_RATE
            print(f"  -> {out.name}  ({secs:.0f}s)")


if __name__ == "__main__":
    main()
