"""Worker auto-applies the validated shared-voice plan so a plain Start produces
a clean transcript without a manual UI step (see mic-bleed-removal update)."""
import queue

from app.jobs import Job
from app.worker import TranscriptionWorker
from src.audio_tracks import AudioStream
from src.track_separation import SharedVoiceWarning


def _worker() -> TranscriptionWorker:
    return TranscriptionWorker(queue.Queue(), queue.Queue(), db_conn=None)  # type: ignore[arg-type]


def _job() -> Job:
    return Job(id="j1", source_path="/tmp/call.mkv", model="m", language="cs")


# The canonical case: track 3 clean, track 1 = mix (3 + another voice),
# track 2 = duplicate mix. Plan: isolate 1-3, keep 3, drop 2.
_WARNINGS = [
    SharedVoiceWarning(mix_index=1, component_index=3, kind="contains", drop_db=16.0),
    SharedVoiceWarning(mix_index=2, component_index=1, kind="duplicate", drop_db=-0.1),
    SharedVoiceWarning(mix_index=3, component_index=2, kind="duplicate", drop_db=-1.2),
]
_STREAMS = [AudioStream(1, 2), AudioStream(2, 2), AudioStream(3, 2)]


def test_auto_apply_isolates_the_mix_and_drops_the_duplicate() -> None:
    job = _job()
    job.shared_voice_warnings = _WARNINGS  # injected -> no audio decode
    kept = _worker()._auto_apply_separation(job, _STREAMS, duration=600.0)

    assert [s.index for s in kept] == [1, 3]  # duplicate track 2 dropped
    assert job.track_subtractions == {1: [3]}  # mix 1 isolated against clean 3


def test_user_configured_subtraction_is_respected() -> None:
    job = _job()
    job.shared_voice_warnings = _WARNINGS
    job.track_subtractions = {3: [1]}  # user's own (different) choice
    kept = _worker()._auto_apply_separation(job, _STREAMS, duration=600.0)

    assert [s.index for s in kept] == [1, 2, 3]  # nothing dropped
    assert job.track_subtractions == {3: [1]}  # untouched


def test_no_shared_voice_leaves_selection_and_subtractions_alone() -> None:
    job = _job()
    job.shared_voice_warnings = []  # analyzed, nothing shared
    kept = _worker()._auto_apply_separation(job, _STREAMS, duration=600.0)

    assert [s.index for s in kept] == [1, 2, 3]
    assert not job.track_subtractions


def test_never_drops_the_last_track() -> None:
    job = _job()
    # Two tracks that are pure duplicates of each other -> resolve would drop one;
    # make sure at least one survives to transcribe.
    job.shared_voice_warnings = [
        SharedVoiceWarning(mix_index=2, component_index=1, kind="duplicate", drop_db=0.0),
    ]
    streams = [AudioStream(1, 2), AudioStream(2, 2)]
    kept = _worker()._auto_apply_separation(job, streams, duration=600.0)

    assert len(kept) >= 1
