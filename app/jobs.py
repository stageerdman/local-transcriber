from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.audio_tracks import AudioStream

_id_counter = itertools.count(1)

# "ready" (added, not started) | "queued" (start clicked, waiting for the
# worker) | "converting" | "loading model" | "transcribing" | "done" |
# "error" | "stopped"
JobStatus = str


@dataclass
class Job:
    source_path: Path
    model: str
    language: str | None
    id: int = field(default_factory=lambda: next(_id_counter))
    status: JobStatus = "ready"
    status_detail: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    audio_duration_seconds: float | None = None
    estimated_seconds: float | None = None
    started_at: datetime | None = None
    elapsed_seconds: float | None = None
    output_path: Path | None = None
    error: str | None = None

    # Populated lazily (on first UI expand of this job's row, or by the
    # worker if the row was never expanded before Start was clicked).
    # `selected_track_indices` holds `AudioStream.index` values; `None` means
    # "not decided yet - use every track", same as the pre-selection default
    # behavior.
    tracks: list[AudioStream] | None = None
    selected_track_indices: set[int] | None = None
    # `track_envelopes` is keyed by (AudioStream.index, channel_index) so a
    # multi-channel track's channels each get their own waveform preview -
    # a stream's channels aren't assumed to carry the same signal.
    track_envelopes: dict[tuple[int, int], list[float]] | None = None
    # Maps AudioStream.index -> a single channel index to use for that
    # track instead of the default full-stream downmix. Absent/no entry
    # means "mix every channel of this stream together", same as before
    # per-channel selection existed.
    selected_channel_by_track: dict[int, int] | None = None

    # Set once every track's channel envelope(s) have loaded and the
    # cross-track/cross-channel correlation pass has run (see
    # MainWindow._maybe_analyze_tracks) - guards against re-running the
    # analysis (and re-clobbering a manual selection) on every envelope
    # that trickles in.
    duplicates_analyzed: bool = False
    # AudioStream.index -> True if that stream's channels are near-identical
    # (e.g. a mono mic duplicated across stereo) - collapses the per-channel
    # waveform display to one row instead of showing two redundant ones.
    channel_pairs_identical: dict[int, bool] | None = None
    # AudioStream.index -> the earlier AudioStream.index it's a near-exact
    # duplicate of (same source routed to more than one track) - surfaced in
    # the UI and excluded from the default selection.
    duplicate_of: dict[int, int] | None = None


@dataclass
class WorkerEvent:
    job: Job
