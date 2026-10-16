"""Phase 5: on launch the bridge reconstructs interrupted jobs from their
checkpoints and re-queues them, skipping any whose source file is gone."""
import queue
from pathlib import Path

import pytest

from app.bridge import Bridge
from src import job_store


@pytest.fixture(autouse=True)
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_TRANSCRIBER_STATE_DIR", str(tmp_path))
    return tmp_path


def _bare_bridge() -> Bridge:
    # Skip Bridge.__init__ (webview window, worker thread, db) - we only exercise
    # the resume methods, which need just these attributes.
    b = Bridge.__new__(Bridge)
    b._pending = {}
    b._resumed = []
    b.job_queue = queue.Queue()
    return b


def _save_checkpoint(key: str, source: Path, **config) -> None:
    cfg = {
        "source_path": str(source), "model": "m", "language": "cs", "polish": True,
        "audio_duration_seconds": 60.0, "tracks": None, "selected_track_indices": None,
        "separation_done": False,
    }
    cfg.update(config)
    job_store.save(job_store.Checkpoint(
        key=key, config=cfg, status="paused",
        partial_unit="single", partial_done_chunks=3,
        started_at="2026-10-16T10:00:00",
    ))


def test_resume_requeues_job_with_same_key(tmp_path: Path) -> None:
    source = tmp_path / "call.mp3"
    source.write_bytes(b"fake")
    _save_checkpoint("key-1", source)

    b = _bare_bridge()
    b._resume_interrupted_jobs()

    assert len(b._resumed) == 1
    job = b._resumed[0]
    assert job.key == "key-1"                 # same on-disk identity -> finds checkpoint
    assert job.source_path == source
    assert job.language == "cs"
    assert job.status == "queued"
    assert b._pending[job.id] is job          # discoverable for stop/resume by id
    assert b.job_queue.get_nowait() is job     # actually enqueued for the worker


def test_resume_skips_and_discards_job_whose_source_is_gone(tmp_path: Path) -> None:
    _save_checkpoint("key-missing", tmp_path / "deleted.mp3")  # file never created

    b = _bare_bridge()
    b._resume_interrupted_jobs()

    assert b._resumed == []
    assert job_store.load("key-missing") is None  # discarded
    with pytest.raises(queue.Empty):
        b.job_queue.get_nowait()


def test_resuming_jobs_reports_in_flight_only(tmp_path: Path) -> None:
    source = tmp_path / "call.mp3"
    source.write_bytes(b"fake")
    _save_checkpoint("k", source)
    b = _bare_bridge()
    b._resume_interrupted_jobs()

    payloads = b.resuming_jobs()
    assert len(payloads) == 1
    assert payloads[0]["resuming"] is True
    assert payloads[0]["name"] == "call"

    # once the job settles it drops out of the resuming list
    b._resumed[0].status = "done"
    assert b.resuming_jobs() == []


def test_resume_restores_multitrack_plan(tmp_path: Path) -> None:
    source = tmp_path / "multi.mkv"
    source.write_bytes(b"fake")
    _save_checkpoint(
        "mt", source,
        tracks=[{"index": 0, "channels": 1, "title": "A"}, {"index": 1, "channels": 1, "title": None}],
        selected_track_indices=[0, 1],
        track_names={"0": "Alice", "1": "Bob"},
        track_subtractions={"1": [0]},
        separation_done=True,
    )
    b = _bare_bridge()
    b._resume_interrupted_jobs()
    job = b._resumed[0]

    assert job.separation_done is True                 # won't re-run detection
    assert job.selected_track_indices == {0, 1}
    assert job.track_names == {0: "Alice", 1: "Bob"}   # int keys restored
    assert job.track_subtractions == {1: [0]}
    assert [t.index for t in job.tracks] == [0, 1]
