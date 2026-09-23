"""End-to-end check of the real multi-track pipeline wiring (not just the
underlying primitives): real ffmpeg-extracted per-track audio through the
actual `src.audio_tracks` + `src.dialogue` functions the worker uses,
skipping only the mlx-whisper model call itself (which needs a downloaded
model and real speech, not available in CI).

Single-track audio has no speaker detection or pause-based blocking - see
updates/multi-speaker-transcripts.md - so there's nothing to integration-test
there beyond what test_dialogue.py and test_audio_tracks.py already cover.
"""

import queue
import shutil
import subprocess
from pathlib import Path

import pytest

from app.jobs import Job
from app.worker import SENTINEL, TranscriptionWorker
from src import dialogue
from src.audio_tracks import extract_track_to_mp3, probe_audio_streams, track_speaker_names
from src.transcription import TranscribedSegment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_three_track_file(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=3",
            "-map", "0:a", "-map", "1:a", "-map", "2:a",
            "-c:a", "libmp3lame", str(path), "-loglevel", "error",
        ],
        check=True,
    )


def test_multitrack_applies_names_language_and_subtraction(tmp_path: Path, monkeypatch) -> None:
    """The per-track panel controls (custom name, per-track language, voice
    subtraction) reach the transcription: custom names become speaker labels, a
    per-track language override is passed to the engine, and a track used only
    as a subtraction reference is decoded but never transcribed on its own."""
    media = tmp_path / "multi.mkv"
    _make_three_track_file(media)
    streams = probe_audio_streams(media)
    assert len(streams) == 3

    worker = TranscriptionWorker(queue.Queue(), queue.Queue(), db_conn=None)  # type: ignore[arg-type]

    calls: list[tuple[str, object]] = []

    def fake_engine(job, audio_path, on_progress=None, language=SENTINEL):  # noqa: ANN001
        calls.append((Path(audio_path).stem, language))
        if on_progress is not None:
            on_progress(1.0)
        return [TranscribedSegment(start=0.0, end=1.0, text=f"said-{Path(audio_path).stem}")]

    monkeypatch.setattr(worker, "_transcribe_via_engine", fake_engine)

    a, b, c = streams[0].index, streams[1].index, streams[2].index
    job = Job(
        source_path=media,
        model="whatever",
        language="en",
        tracks=streams,
        selected_track_indices={a, b},          # transcribe tracks a, b; c is reference-only
        track_names={a: "Alice", b: "Bob"},
        track_languages={b: "cs"},              # b overrides to Czech; a inherits "en"
        track_denoise={a: False, b: False, c: False},
        track_subtractions={a: [c]},            # remove track c's voice from track a
    )
    selected = [s for s in streams if s.index in job.selected_track_indices]

    segments = worker._transcribe_multi_track(job, streams, selected, tmp_path)

    # custom names became the speaker labels
    assert {s.speaker for s in segments} == {"Alice", "Bob"}
    # exactly two tracks transcribed (the reference-only track c was not)
    assert len(calls) == 2
    # per-track language override reached the engine (a -> en inherited, b -> cs)
    assert sorted(lang for _, lang in calls) == ["cs", "en"]


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
