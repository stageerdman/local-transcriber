from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_id_counter = itertools.count(1)

JobStatus = str  # "queued" | "converting" | "transcribing" | "done" | "error"


@dataclass
class Job:
    source_path: Path
    model: str
    language: str | None
    id: int = field(default_factory=lambda: next(_id_counter))
    status: JobStatus = "queued"
    audio_duration_seconds: float | None = None
    estimated_seconds: float | None = None
    started_at: datetime | None = None
    elapsed_seconds: float | None = None
    output_path: Path | None = None
    error: str | None = None


@dataclass
class WorkerEvent:
    job: Job
