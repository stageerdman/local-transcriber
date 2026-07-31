from __future__ import annotations

import queue
import shutil
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from app.jobs import Job, WorkerEvent
from src import history_db, md_writer
from src.converter import prepare_mp3
from src.duration import probe_duration_seconds
from src.transcription import transcribe_mp3

SENTINEL = object()


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        job_queue: "queue.Queue[Job | object]",
        event_queue: "queue.Queue[WorkerEvent]",
        db_conn: sqlite3.Connection,
    ) -> None:
        super().__init__(daemon=True)
        self.job_queue = job_queue
        self.event_queue = event_queue
        self.db_conn = db_conn

    def run(self) -> None:
        while True:
            job = self.job_queue.get()
            if job is SENTINEL:
                return
            self._process(job)  # type: ignore[arg-type]

    def _emit(self, job: Job) -> None:
        self.event_queue.put(WorkerEvent(job=job))

    def _process(self, job: Job) -> None:
        started_at = datetime.now()
        job.started_at = started_at
        start_monotonic = time.monotonic()
        temp_dir: Path | None = None

        try:
            job.audio_duration_seconds = probe_duration_seconds(job.source_path)
            job.estimated_seconds = history_db.estimate_seconds(
                self.db_conn, job.model, job.language, job.audio_duration_seconds
            )

            if job.source_path.suffix.lower() == ".mp3":
                mp3_path = job.source_path
            else:
                job.status = "converting"
                self._emit(job)
                temp_dir = Path(tempfile.mkdtemp(prefix="local-transcriber-"))
                mp3_path = temp_dir / f"{job.source_path.stem}.mp3"
                prepare_mp3(job.source_path, mp3_path)

            job.status = "transcribing"
            self._emit(job)
            transcript_text = transcribe_mp3(job.model, mp3_path, language=job.language)

            elapsed_seconds = time.monotonic() - start_monotonic
            md_path = md_writer.write_transcript_markdown(
                job.source_path,
                transcript_text,
                model=job.model,
                language=job.language,
                elapsed_seconds=elapsed_seconds,
                transcribed_at=started_at,
            )

            job.elapsed_seconds = elapsed_seconds
            job.output_path = md_path
            job.status = "done"
            history_db.record_run(
                self.db_conn,
                source_path=str(job.source_path),
                output_path=str(md_path),
                model=job.model,
                language=job.language,
                audio_duration_seconds=job.audio_duration_seconds,
                elapsed_seconds=elapsed_seconds,
                started_at=started_at.isoformat(timespec="seconds"),
                finished_at=datetime.now().isoformat(timespec="seconds"),
                status="success",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            elapsed_seconds = time.monotonic() - start_monotonic
            job.elapsed_seconds = elapsed_seconds
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            history_db.record_run(
                self.db_conn,
                source_path=str(job.source_path),
                output_path=None,
                model=job.model,
                language=job.language,
                audio_duration_seconds=job.audio_duration_seconds,
                elapsed_seconds=elapsed_seconds,
                started_at=started_at.isoformat(timespec="seconds"),
                finished_at=datetime.now().isoformat(timespec="seconds"),
                status="error",
                error=job.error,
            )
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self._emit(job)
