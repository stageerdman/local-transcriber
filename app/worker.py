from __future__ import annotations

import multiprocessing
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.engine import run_engine
from app.jobs import Job, WorkerEvent
from src import dialogue, history_db, md_writer
from src.audio_tracks import (
    AudioStream,
    extract_channel_to_mp3,
    extract_track_to_mp3,
    safe_probe_audio_streams,
    track_speaker_names,
)
from src.converter import prepare_mp3
from src.duration import probe_duration_seconds
from src.transcription import TranscribedSegment

SENTINEL = object()

ENGINE_POLL_SECONDS = 0.2


class _StopRequested(Exception):
    pass


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

        self.mp_ctx = multiprocessing.get_context("spawn")
        self.engine_process: multiprocessing.process.BaseProcess | None = None
        self.request_queue: multiprocessing.Queue | None = None
        self.response_queue: multiprocessing.Queue | None = None

    def run(self) -> None:
        while True:
            job = self.job_queue.get()
            if job is SENTINEL:
                self._kill_engine()
                return
            self._process(job)  # type: ignore[arg-type]

    def _emit(self, job: Job) -> None:
        self.event_queue.put(WorkerEvent(job=job))

    # -- transcription engine (separate process, so it can be killed) --------

    def _ensure_engine_running(self) -> None:
        if self.engine_process is not None and self.engine_process.is_alive():
            return
        self.request_queue = self.mp_ctx.Queue()
        self.response_queue = self.mp_ctx.Queue()
        self.engine_process = self.mp_ctx.Process(
            target=run_engine, args=(self.request_queue, self.response_queue), daemon=True
        )
        self.engine_process.start()

    def _kill_engine(self) -> None:
        if self.engine_process is not None and self.engine_process.is_alive():
            self.engine_process.terminate()
            self.engine_process.join(timeout=2)
            if self.engine_process.is_alive():
                self.engine_process.kill()
                self.engine_process.join(timeout=2)
        self.engine_process = None

    def _transcribe_via_engine(
        self,
        job: Job,
        mp3_path: Path,
        on_progress: "Callable[[float], None] | None" = None,
    ) -> list[TranscribedSegment]:
        """Transcribe one mp3 via the engine process.

        `on_progress` receives a fraction in [0, 1] of *this mp3* processed so
        far, as the engine reports it. Callers translate that into the job's
        overall bar value (identity for single-track; (done + frac)/N for
        multi-track).
        """
        self._ensure_engine_running()
        self.request_queue.put((job.id, job.model, mp3_path, job.language))

        while True:
            if job.stop_event.is_set():
                self._kill_engine()
                raise _StopRequested()
            try:
                message = self.response_queue.get(timeout=ENGINE_POLL_SECONDS)
            except queue.Empty:
                if not self.engine_process.is_alive():
                    raise RuntimeError("Transcription engine process exited unexpectedly")
                continue

            kind = message[0]
            if kind == "phase":
                job.status = message[2]
                self._emit(job)
            elif kind == "progress":
                if on_progress is not None:
                    on_progress(float(message[2]))
            elif kind == "result":
                return message[2]
            elif kind == "error":
                raise RuntimeError(f"{message[2]}: {message[3]}")

    # -- job processing --------------------------------------------------------

    def _process(self, job: Job) -> None:
        started_at = datetime.now()
        job.started_at = started_at
        job.transcribe_fraction = None
        job.transcribe_position_fraction = None
        start_monotonic = time.monotonic()
        temp_dir: Path | None = None

        try:
            if job.stop_event.is_set():
                raise _StopRequested()

            # The UI probes duration as soon as a file is added (to show an
            # estimate before Start is clicked) - reuse that if present
            # instead of re-running ffprobe. The estimate itself is always
            # recomputed here since the model/language may have changed
            # since that pre-probe.
            if job.audio_duration_seconds is None:
                job.audio_duration_seconds = probe_duration_seconds(job.source_path)
            job.estimated_seconds = history_db.estimate_seconds(
                self.db_conn, job.model, job.language, job.audio_duration_seconds
            )

            if job.stop_event.is_set():
                raise _StopRequested()

            # A container with more than one audio stream (e.g. Zoom's
            # "record a separate audio file for each participant") is
            # ground truth for who's speaking - prefer it over guessing.
            # Speaker separation and pause-based blocking only apply here;
            # a single mixed-down track has no reliable way to tell speakers
            # apart, so it's transcribed as plain text like before.
            #
            # `job.tracks`/`job.selected_track_indices` are set by the UI if
            # the user expanded this row before clicking Start (to preview
            # tracks and optionally deselect some); fall back to probing and
            # "every track" here if they never did.
            streams = job.tracks if job.tracks is not None else safe_probe_audio_streams(job.source_path)
            if job.selected_track_indices is not None:
                selected_streams = [s for s in streams if s.index in job.selected_track_indices]
                if not selected_streams:
                    selected_streams = streams
            else:
                selected_streams = streams
            container_has_multiple_streams = len(streams) > 1
            temp_dir = Path(tempfile.mkdtemp(prefix="local-transcriber-"))

            if container_has_multiple_streams:
                # Each selected track is transcribed as its own pass, so
                # total work scales with how many tracks were picked, not
                # the container's total track count.
                if job.estimated_seconds:
                    job.estimated_seconds *= len(selected_streams)
                labeled_segments = self._transcribe_multi_track(job, streams, selected_streams, temp_dir)
                blocks = dialogue.merge_segments_into_blocks(labeled_segments)
                transcript_text = dialogue.render_blocks_markdown(blocks)
                # A single selected track out of a multi-track container has
                # nothing to diarize - `render_blocks_markdown` already
                # collapses to plain text for one speaker, so keep the
                # frontmatter plain-text-shaped too.
                multi_speaker = len(selected_streams) > 1
                speakers_detected = len(selected_streams) if multi_speaker else None
                diarization_method = "multitrack" if multi_speaker else None
            else:
                if job.source_path.suffix.lower() == ".mp3":
                    mp3_path = job.source_path
                else:
                    job.status = "converting"
                    self._emit(job)
                    mp3_path = temp_dir / f"{job.source_path.stem}.mp3"
                    prepare_mp3(job.source_path, mp3_path)

                if job.stop_event.is_set():
                    raise _StopRequested()

                job.status = "loading model"
                self._emit(job)

                def on_progress(fraction: float) -> None:
                    job.transcribe_fraction = fraction
                    job.transcribe_position_fraction = fraction
                    self._emit(job)

                segments = self._transcribe_via_engine(job, mp3_path, on_progress)
                text = " ".join(seg.text for seg in segments).strip()
                transcript_text = text + ("\n" if text else "")
                speakers_detected = None
                diarization_method = None

            elapsed_seconds = time.monotonic() - start_monotonic
            md_path = md_writer.write_transcript_markdown(
                job.source_path,
                transcript_text,
                model=job.model,
                language=job.language,
                elapsed_seconds=elapsed_seconds,
                transcribed_at=started_at,
                speakers=speakers_detected,
                diarization=diarization_method,
            )

            job.elapsed_seconds = elapsed_seconds
            job.output_path = md_path
            job.status = "done"
            job.status_detail = None
            self._record_history(job, status="success", elapsed_seconds=elapsed_seconds, started_at=started_at)
        except _StopRequested:
            elapsed_seconds = time.monotonic() - start_monotonic
            job.elapsed_seconds = elapsed_seconds
            job.status = "stopped"
            job.status_detail = None
            self._record_history(job, status="stopped", elapsed_seconds=elapsed_seconds, started_at=started_at)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            elapsed_seconds = time.monotonic() - start_monotonic
            job.elapsed_seconds = elapsed_seconds
            job.status = "error"
            job.status_detail = None
            job.error = f"{type(exc).__name__}: {exc}"
            self._record_history(
                job, status="error", elapsed_seconds=elapsed_seconds, started_at=started_at, error=job.error
            )
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self._emit(job)

    def _transcribe_multi_track(
        self,
        job: Job,
        all_streams: list[AudioStream],
        streams_to_transcribe: list[AudioStream],
        temp_dir: Path,
    ) -> list[dialogue.Segment]:
        # Names are derived from the container's full track list so a track's
        # label ("Person 2") stays stable regardless of which subset got
        # selected for transcription.
        name_by_index = dict(zip((s.index for s in all_streams), track_speaker_names(all_streams)))
        labeled_segments: list[dialogue.Segment] = []
        total = len(streams_to_transcribe)

        for i, stream in enumerate(streams_to_transcribe):
            name = name_by_index[stream.index]
            if job.stop_event.is_set():
                raise _StopRequested()

            job.status = "converting"
            job.status_detail = f"track {i + 1}/{total} - {name}"
            self._emit(job)
            track_mp3 = temp_dir / f"track{i}.mp3"
            channel_index = (job.selected_channel_by_track or {}).get(stream.index)
            if channel_index is not None:
                extract_channel_to_mp3(job.source_path, stream.index, channel_index, track_mp3)
            else:
                extract_track_to_mp3(job.source_path, stream.index, track_mp3)

            if job.stop_event.is_set():
                raise _StopRequested()

            job.status = "loading model" if i == 0 else "transcribing"
            job.status_detail = f"track {i + 1}/{total} - {name}"
            self._emit(job)

            def on_progress(fraction: float, i=i) -> None:
                # Overall bar spans all tracks; position is within this track.
                job.transcribe_fraction = (i + fraction) / total
                job.transcribe_position_fraction = fraction
                self._emit(job)

            segments = self._transcribe_via_engine(job, track_mp3, on_progress)
            labeled_segments.extend(
                dialogue.Segment(speaker=name, start=seg.start, end=seg.end, text=seg.text)
                for seg in segments
            )

        return labeled_segments

    def _record_history(
        self,
        job: Job,
        *,
        status: str,
        elapsed_seconds: float,
        started_at: datetime,
        error: str | None = None,
    ) -> None:
        history_db.record_run(
            self.db_conn,
            source_path=str(job.source_path),
            output_path=str(job.output_path) if job.output_path else None,
            model=job.model,
            language=job.language,
            audio_duration_seconds=job.audio_duration_seconds,
            elapsed_seconds=elapsed_seconds,
            started_at=started_at.isoformat(timespec="seconds"),
            finished_at=datetime.now().isoformat(timespec="seconds"),
            status=status,
            error=error,
        )
