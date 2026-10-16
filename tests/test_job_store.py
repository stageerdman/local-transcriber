from pathlib import Path

import pytest

from src import job_store
from src.audio_tracks import AudioStream


@pytest.fixture(autouse=True)
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_TRANSCRIBER_STATE_DIR", str(tmp_path))
    return tmp_path


class _FakeJob:
    """Duck-typed stand-in for app.jobs.Job (job_store never imports Job)."""

    def __init__(self, **kw):
        self.source_path = Path("/calls/meeting.mp4")
        self.model = "mlx-community/whisper-small-mlx"
        self.language = "cs"
        self.polish = True
        self.audio_duration_seconds = 123.0
        self.tracks = None
        self.selected_track_indices = None
        self.selected_channel_by_track = None
        self.track_names = None
        self.track_languages = None
        self.track_denoise = None
        self.track_subtractions = None
        self.separation_done = False
        for k, v in kw.items():
            setattr(self, k, v)


def test_config_round_trips_through_serialize_deserialize() -> None:
    job = _FakeJob(
        tracks=[AudioStream(0, 1, "Alice"), AudioStream(1, 2, None)],
        selected_track_indices={0, 1},
        track_names={0: "Alice", 1: "Bob"},
        track_subtractions={1: [0]},
        selected_channel_by_track={1: 0},
        separation_done=True,
    )
    restored = job_store.deserialize_config(job_store.serialize_config(job))

    assert restored["source_path"] == Path("/calls/meeting.mp4")
    assert restored["model"] == "mlx-community/whisper-small-mlx"
    assert restored["language"] == "cs"
    assert restored["polish"] is True
    assert restored["audio_duration_seconds"] == 123.0
    assert restored["separation_done"] is True
    assert restored["selected_track_indices"] == {0, 1}
    # int keys survive the JSON string-key round trip
    assert restored["track_names"] == {0: "Alice", 1: "Bob"}
    assert restored["track_subtractions"] == {1: [0]}
    assert restored["selected_channel_by_track"] == {1: 0}
    assert [(t.index, t.channels, t.title) for t in restored["tracks"]] == [
        (0, 1, "Alice"),
        (1, 2, None),
    ]


def test_config_handles_single_track_job_with_no_track_metadata() -> None:
    restored = job_store.deserialize_config(job_store.serialize_config(_FakeJob()))
    assert restored["tracks"] is None
    assert restored["selected_track_indices"] is None
    assert restored["track_names"] is None


def test_save_and_load_checkpoint() -> None:
    cp = job_store.Checkpoint(
        key="abc123",
        config=job_store.serialize_config(_FakeJob()),
        status="paused",
        phase="transcribing",
        partial_unit="single",
        partial_done_chunks=4,
        started_at="2026-10-16T10:00:00",
    )
    job_store.save(cp)

    loaded = job_store.load("abc123")
    assert loaded is not None
    assert loaded.status == "paused"
    assert loaded.phase == "transcribing"
    assert loaded.partial_unit == "single"
    assert loaded.partial_done_chunks == 4
    assert loaded.updated_at is not None  # stamped on save


def test_load_missing_returns_none() -> None:
    assert job_store.load("nope") is None


def test_list_resumable_filters_terminal_and_sorts_oldest_first() -> None:
    job_store.save(job_store.Checkpoint(key="k_done", config={}, status="done", started_at="2026-10-16T09:00:00"))
    job_store.save(job_store.Checkpoint(key="k_new", config={}, status="running", started_at="2026-10-16T12:00:00"))
    job_store.save(job_store.Checkpoint(key="k_old", config={}, status="paused", started_at="2026-10-16T08:00:00"))

    resumable = job_store.list_resumable()
    assert [c.key for c in resumable] == ["k_old", "k_new"]


def test_discard_removes_work_dir() -> None:
    cp = job_store.Checkpoint(key="gone", config={}, status="paused")
    job_store.save(cp)
    job_store.append_segments("gone", "single", 0, [{"start": 0.0, "end": 1.0, "text": "hi"}])
    assert job_store.work_dir("gone").exists()

    job_store.discard("gone")
    assert not job_store.work_dir("gone").exists()


def test_read_segments_keeps_only_committed_work() -> None:
    key = "seg"
    # single-track unit, 3 chunks written to the log...
    job_store.append_segments(key, "single", 0, [{"start": 0.0, "end": 1.0, "text": "a"}])
    job_store.append_segments(key, "single", 1, [{"start": 1.0, "end": 2.0, "text": "b"}])
    # chunk 2 was appended but the app crashed before the counter bump:
    job_store.append_segments(key, "single", 2, [{"start": 2.0, "end": 3.0, "text": "c-STALE"}])

    cp = job_store.Checkpoint(
        key=key, config={}, status="paused",
        partial_unit="single", partial_done_chunks=2,  # only chunks 0,1 committed
    )
    texts = [s["text"] for s in job_store.read_segments(cp)]
    assert texts == ["a", "b"]  # the stale chunk-2 line is filtered out


def test_read_segments_keeps_all_completed_units() -> None:
    key = "multi"
    job_store.append_segments(key, "track0", 0, [{"start": 0.0, "end": 1.0, "text": "alice-0"}])
    job_store.append_segments(key, "track0", 1, [{"start": 1.0, "end": 2.0, "text": "alice-1"}])
    job_store.append_segments(key, "track1", 0, [{"start": 0.5, "end": 1.5, "text": "bob-0"}])
    job_store.append_segments(key, "track1", 1, [{"start": 2.0, "end": 3.0, "text": "bob-1-STALE"}])

    cp = job_store.Checkpoint(
        key=key, config={}, status="paused",
        completed_units=["track0"],           # track0 fully done -> keep all its chunks
        partial_unit="track1", partial_done_chunks=1,  # track1 only chunk 0 committed
    )
    texts = [s["text"] for s in job_store.read_segments(cp)]
    assert texts == ["alice-0", "alice-1", "bob-0"]
    assert job_store.segments_for_unit(cp, "track0") == [
        s for s in job_store.read_segments(cp) if s["unit"] == "track0"
    ]


def test_read_segments_empty_when_nothing_committed() -> None:
    key = "fresh"
    job_store.append_segments(key, "single", 0, [{"start": 0.0, "end": 1.0, "text": "pending"}])
    cp = job_store.Checkpoint(key=key, config={}, status="running", partial_unit="single", partial_done_chunks=0)
    assert job_store.read_segments(cp) == []
