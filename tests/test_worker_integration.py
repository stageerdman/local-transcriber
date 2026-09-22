"""End-to-end check of the real multi-track pipeline wiring (not just the
underlying primitives): real ffmpeg-extracted per-track audio through the
actual `src.audio_tracks` + `src.dialogue` functions the worker uses,
skipping only the mlx-whisper model call itself (which needs a downloaded
model and real speech, not available in CI).

Single-track audio has no speaker detection or pause-based blocking - see
updates/multi-speaker-transcripts.md - so there's nothing to integration-test
there beyond what test_dialogue.py and test_audio_tracks.py already cover.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from src import dialogue
from src.audio_tracks import extract_track_to_mp3, probe_audio_streams, track_speaker_names

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_multitrack_extraction_and_merge_produces_interruption_flagged_dialogue(tmp_path: Path) -> None:
    """Multi-track sources don't have the single-mic overlap problem - each
    speaker is on an independent signal - so overlapping speech should be
    reliably flagged as an interruption end to end."""
    media = tmp_path / "multi.mkv"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-map", "0:a", "-map", "1:a",
            "-c:a", "libmp3lame", str(media), "-loglevel", "error",
        ],
        check=True,
    )
    streams = probe_audio_streams(media)
    names = track_speaker_names(streams)
    assert names == ["Person 1", "Person 2"]

    track0 = tmp_path / "track0.mp3"
    track1 = tmp_path / "track1.mp3"
    extract_track_to_mp3(media, streams[0].index, track0)
    extract_track_to_mp3(media, streams[1].index, track1)
    assert track0.exists() and track1.exists()

    # Simulate what per-track whisper transcription would have returned:
    # Person 1 speaks 0-3s, Person 2 interrupts at 1.5-2.5s.
    labeled = [
        dialogue.Segment(speaker=names[0], start=0.0, end=3.0, text="So I wanted to go over the proposal"),
        dialogue.Segment(speaker=names[1], start=1.5, end=2.5, text="actually before that"),
    ]
    blocks = dialogue.merge_segments_into_blocks(labeled)
    rendered = dialogue.render_blocks_markdown(blocks)

    assert "(interrupting)" in rendered
    assert rendered.index("Person 1") < rendered.index("Person 2")
