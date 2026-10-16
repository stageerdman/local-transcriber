"""Durable job checkpoints so a transcription can be stopped and resumed.

A transcription job gets one **work dir** on disk, keyed by a restart-stable
`key`:

    ~/Library/Application Support/LocalTranscriber/jobs/<key>/
        job.json        - config + checkpoint (status, phase, progress index)
        segments.jsonl  - append-only log of completed segments
        *.mp3 / *.wav   - rendered audio artifacts (survive a restart)

The **segment log is the source of truth** for completed work; the counters in
`job.json` (`completed_units`, `partial_unit`, `partial_done_chunks`) are the
authoritative index into it. On read we keep a segment line iff its unit is in
`completed_units`, or it belongs to `partial_unit` with `chunk <
partial_done_chunks`. That tolerates a crash *between* appending segments and
bumping the counter: the stale trailing lines are ignored and that chunk is
redone deterministically.

Write order is always: append segments (flushed) -> then atomically rewrite
`job.json` (temp file + os.replace). Never the reverse.

This module is deliberately ignorant of `app.jobs.Job`: it serializes/restores a
plain config dict, so it stays an isolated, independently-testable src module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.audio_tracks import AudioStream

CHECKPOINT_VERSION = 1

# Lifecycle values stored in job.json. "running" and "paused" are both
# resumable; the rest are terminal (the work dir is normally discarded on
# success, but a terminal checkpoint may briefly exist before cleanup).
RESUMABLE_STATUSES = ("running", "paused")

# The single-track "unit" key (one engine call). Multi-track jobs use
# "track<stream index>" per selected stream instead.
SINGLE_UNIT = "single"


def state_root() -> Path:
    """Root dir holding every job's work dir.

    Overridable via `LOCAL_TRANSCRIBER_STATE_DIR` (tests point it at a tmp dir;
    could also relocate the store). Defaults under macOS Application Support,
    next to history.db.
    """
    override = os.environ.get("LOCAL_TRANSCRIBER_STATE_DIR")
    if override:
        return Path(override) / "jobs"
    return Path.home() / "Library" / "Application Support" / "LocalTranscriber" / "jobs"


def work_dir(key: str) -> Path:
    return state_root() / key


def ensure_work_dir(key: str) -> Path:
    d = work_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_json_path(key: str) -> Path:
    return work_dir(key) / "job.json"


def _segments_path(key: str) -> Path:
    return work_dir(key) / "segments.jsonl"


def track_unit(stream_index: int) -> str:
    return f"track{stream_index}"


# -- config serialization ------------------------------------------------------
# Only the fields that define *what* to transcribe are persisted. Transient
# progress/UI fields (bar fractions, envelopes, detection caches) are not - they
# are either recomputed or irrelevant to reproducing the run.

_INT_KEY_DICTS = (
    "track_names",
    "track_languages",
    "track_denoise",
    "track_subtractions",
    "selected_channel_by_track",
)


def serialize_config(job) -> dict:
    """Read the transcribe-defining attributes off a Job into a JSON-able dict.

    Duck-typed on purpose (no Job import). `selected_track_indices` is stored as
    the *resolved* set (post speaker-separation) and `track_subtractions` as the
    resolved plan, together with `separation_done`, so resume never re-runs the
    best-effort shared-voice detection.
    """
    tracks = getattr(job, "tracks", None)
    selected = getattr(job, "selected_track_indices", None)
    config: dict = {
        "source_path": str(job.source_path),
        "model": job.model,
        "language": job.language,
        "polish": bool(job.polish),
        "audio_duration_seconds": getattr(job, "audio_duration_seconds", None),
        "tracks": [
            {"index": s.index, "channels": s.channels, "title": s.title}
            for s in tracks
        ]
        if tracks is not None
        else None,
        "selected_track_indices": sorted(selected) if selected is not None else None,
        "separation_done": bool(getattr(job, "separation_done", False)),
    }
    for name in _INT_KEY_DICTS:
        value = getattr(job, name, None)
        config[name] = (
            {str(k): v for k, v in value.items()} if value else None
        )
    return config


def deserialize_config(config: dict) -> dict:
    """Turn a stored config dict back into kwargs/attrs for a fresh Job.

    Reconstructs `AudioStream`s and re-ints the string-keyed dicts. Returned as a
    plain dict so the caller (bridge) owns Job construction.
    """
    out: dict = {
        "source_path": Path(config["source_path"]),
        "model": config["model"],
        "language": config.get("language"),
        "polish": bool(config.get("polish", True)),
        "audio_duration_seconds": config.get("audio_duration_seconds"),
        "separation_done": bool(config.get("separation_done", False)),
    }
    tracks = config.get("tracks")
    out["tracks"] = (
        [AudioStream(index=t["index"], channels=t["channels"], title=t.get("title")) for t in tracks]
        if tracks is not None
        else None
    )
    sel = config.get("selected_track_indices")
    out["selected_track_indices"] = set(sel) if sel is not None else None
    for name in _INT_KEY_DICTS:
        value = config.get(name)
        out[name] = {int(k): v for k, v in value.items()} if value else None
    return out


# -- checkpoint ----------------------------------------------------------------


@dataclass
class Checkpoint:
    key: str
    config: dict
    status: str = "running"            # running | paused | done | error
    phase: str = "queued"              # mirrors Job.status for the UI
    multi_track: bool | None = None
    completed_units: list[str] = field(default_factory=list)
    partial_unit: str | None = None
    partial_done_chunks: int = 0
    artifacts: dict = field(default_factory=dict)  # {"mp3": path} / {"tracks": {"<idx>": path}}
    started_at: str | None = None
    updated_at: str | None = None
    version: int = CHECKPOINT_VERSION

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "version": self.version,
            "status": self.status,
            "phase": self.phase,
            "multi_track": self.multi_track,
            "completed_units": self.completed_units,
            "partial_unit": self.partial_unit,
            "partial_done_chunks": self.partial_done_chunks,
            "artifacts": self.artifacts,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "config": self.config,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Checkpoint":
        return cls(
            key=data["key"],
            config=data.get("config", {}),
            status=data.get("status", "running"),
            phase=data.get("phase", "queued"),
            multi_track=data.get("multi_track"),
            completed_units=list(data.get("completed_units", [])),
            partial_unit=data.get("partial_unit"),
            partial_done_chunks=int(data.get("partial_done_chunks", 0)),
            artifacts=dict(data.get("artifacts", {})),
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at"),
            version=int(data.get("version", CHECKPOINT_VERSION)),
        )


def save(checkpoint: Checkpoint) -> None:
    """Atomically (temp file + os.replace) write job.json for this checkpoint."""
    ensure_work_dir(checkpoint.key)
    checkpoint.updated_at = datetime.now().isoformat(timespec="seconds")
    path = _job_json_path(checkpoint.key)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(checkpoint.to_json(), fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load(key: str) -> Checkpoint | None:
    path = _job_json_path(key)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return Checkpoint.from_json(json.load(fh))
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def list_resumable() -> list[Checkpoint]:
    """Every on-disk job whose checkpoint is in a resumable state, oldest first.

    Oldest-first (by `started_at`) so a queue of interrupted jobs resumes in the
    order they were originally started.
    """
    root = state_root()
    if not root.exists():
        return []
    found: list[Checkpoint] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        cp = load(child.name)
        if cp is not None and cp.status in RESUMABLE_STATUSES:
            found.append(cp)
    found.sort(key=lambda c: c.started_at or "")
    return found


def discard(key: str) -> None:
    """Remove a job's entire work dir (artifacts + checkpoint + segment log)."""
    import shutil

    shutil.rmtree(work_dir(key), ignore_errors=True)


# -- segment log ---------------------------------------------------------------


def append_segments(key: str, unit: str, chunk: int, segments: list[dict]) -> None:
    """Append one chunk's worth of completed segments to the log (flushed).

    Each segment dict carries at least start/end/text; speaker/energy are added
    for multi-track. The `unit` and `chunk` are stamped on so the filter-on-read
    rule can tell completed work from stale trailing lines after a crash. Always
    called *before* the matching counter bump in `save()`.
    """
    ensure_work_dir(key)
    path = _segments_path(key)
    with open(path, "a", encoding="utf-8") as fh:
        for seg in segments:
            line = {"unit": unit, "chunk": chunk, **seg}
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_segments(checkpoint: Checkpoint) -> list[dict]:
    """All *committed* segments for a checkpoint, in log order.

    A line is kept iff its unit is fully done (`unit in completed_units`) or it
    belongs to the in-flight unit at a committed chunk (`unit == partial_unit and
    chunk < partial_done_chunks`). Everything else is an uncommitted/stale
    trailing write and is dropped.
    """
    path = _segments_path(checkpoint.key)
    if not path.exists():
        return []
    completed = set(checkpoint.completed_units)
    kept: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue  # torn final line from a crash mid-write
            unit = line.get("unit")
            if unit in completed:
                kept.append(line)
            elif (
                unit == checkpoint.partial_unit
                and int(line.get("chunk", 0)) < checkpoint.partial_done_chunks
            ):
                kept.append(line)
    return kept


def segments_for_unit(checkpoint: Checkpoint, unit: str) -> list[dict]:
    """Committed segments belonging to one unit, in log order."""
    return [s for s in read_segments(checkpoint) if s.get("unit") == unit]
