"""Worker-level checkpoint persist / restore / resume-skip (no real engine).

Drives the seams the worker uses to make transcription resumable:
`_persist_chunk` (durably record a streamed chunk), `_finish_unit` (commit a
unit), and `_restore_unit` (rebuild segment objects from the log). The engine
and model are not involved - the transcription-layer chunk/start_chunk behavior
is covered in test_transcription_resume.py.
"""
import queue
from pathlib import Path

import numpy as np
import pytest

from app.worker import TranscriptionWorker
from src import dialogue, job_store
from src.transcription import TranscribedSegment


@pytest.fixture(autouse=True)
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_TRANSCRIBER_STATE_DIR", str(tmp_path))
    return tmp_path


def _worker() -> TranscriptionWorker:
    return TranscriptionWorker(queue.Queue(), queue.Queue(), db_conn=None)  # type: ignore[arg-type]


def _seg(s: float, e: float, t: str) -> TranscribedSegment:
    return TranscribedSegment(start=s, end=e, text=t)


def test_persist_then_restore_single_track_in_order() -> None:
    w = _worker()
    cp = job_store.Checkpoint(key="j1", config={}, status="running")
    w._cp = cp
    w._unit, w._unit_speaker, w._unit_track_pcm = "single", None, None

    w._persist_chunk("single", 0, [_seg(0, 1, "hello")])
    w._persist_chunk("single", 1, [_seg(1, 2, "there")])
    assert cp.partial_unit == "single" and cp.partial_done_chunks == 2

    w._finish_unit("single")
    assert cp.completed_units == ["single"] and cp.partial_unit is None

    restored = w._restore_unit("single", None)
    assert [(s.text, s.start) for s in restored] == [("hello", 0), ("there", 1)]
    assert all(isinstance(s, TranscribedSegment) for s in restored)


def test_interrupted_then_resumed_matches_full_run() -> None:
    chunks = [[_seg(0, 1, "a")], [_seg(1, 2, "b")], [_seg(2, 3, "c")]]

    # --- uninterrupted run ---
    w_full = _worker()
    cp_full = job_store.Checkpoint(key="full", config={}, status="running")
    w_full._cp = cp_full
    w_full._unit, w_full._unit_speaker, w_full._unit_track_pcm = "single", None, None
    for i, c in enumerate(chunks):
        w_full._persist_chunk("single", i, c)
    w_full._finish_unit("single")
    full = [s.text for s in w_full._restore_unit("single", None)]

    # --- interrupted after 2 chunks, then resumed in a fresh worker ---
    w1 = _worker()
    cp1 = job_store.Checkpoint(key="part", config={}, status="running")
    w1._cp = cp1
    w1._unit, w1._unit_speaker, w1._unit_track_pcm = "single", None, None
    w1._persist_chunk("single", 0, chunks[0])
    w1._persist_chunk("single", 1, chunks[1])
    # Simulate a crash: a stale, uncommitted chunk-2 line slips into the log
    # before the counter could be bumped.
    job_store.append_segments("part", "single", 2, [{"start": 2, "end": 3, "text": "c-STALE"}])
    cp1.status = "paused"
    job_store.save(cp1)

    # resume
    w2 = _worker()
    cp2 = job_store.load("part")
    assert cp2.partial_unit == "single" and cp2.partial_done_chunks == 2
    job_store.compact(cp2)  # drops the stale chunk-2 line
    w2._cp = cp2
    w2._unit, w2._unit_speaker, w2._unit_track_pcm = "single", None, None
    start_chunk = cp2.partial_done_chunks
    assert start_chunk == 2  # resume exactly where we stopped
    for i in range(start_chunk, len(chunks)):
        w2._persist_chunk("single", i, chunks[i])
    w2._finish_unit("single")

    resumed = [s.text for s in w2._restore_unit("single", None)]
    assert resumed == full == ["a", "b", "c"]  # no dupes, no gaps


def test_persist_attaches_speaker_and_energy_for_multitrack() -> None:
    w = _worker()
    cp = job_store.Checkpoint(key="mt", config={}, status="running")
    w._cp = cp
    # a loud track -> energy for a segment covering the whole track is ~1.0
    pcm = np.ones(16000, dtype=np.float32)
    w._unit, w._unit_speaker, w._unit_track_pcm, w._unit_track_rms = "track0", "Alice", pcm, 1.0

    w._persist_chunk("track0", 0, [_seg(0.0, 1.0, "hi")])
    w._finish_unit("track0")

    restored = w._restore_unit("track0", "Alice")
    assert isinstance(restored[0], dialogue.Segment)
    assert restored[0].speaker == "Alice"
    assert restored[0].energy is not None and restored[0].energy > 0


def test_restore_uses_label_when_segment_has_no_speaker() -> None:
    w = _worker()
    cp = job_store.Checkpoint(key="lbl", config={}, status="running")
    w._cp = cp
    # a line persisted without a speaker key still resolves to the track label
    job_store.append_segments("lbl", "track0", 0, [{"start": 0, "end": 1, "text": "x"}])
    cp.completed_units = ["track0"]
    job_store.save(cp)
    restored = w._restore_unit("track0", "Bob")
    assert restored[0].speaker == "Bob"
