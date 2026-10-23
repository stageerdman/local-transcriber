from __future__ import annotations

import json
import multiprocessing
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.engine import run_engine
from app.jobs import Job, WorkerEvent
from src import catalog, dialogue, history_db, job_store, md_writer
from src.audio_tracks import (
    AudioStream,
    safe_probe_audio_streams,
    track_speaker_names,
)
from src.converter import prepare_mp3
from src.duration import probe_duration_seconds
from src.transcription import TranscribedSegment
from src import track_separation

SENTINEL = object()

ENGINE_POLL_SECONDS = 0.2

# Representative segment (35% into the call) used to detect which tracks share a
# voice - the same window the UI analyzes. 90s is long enough for both speakers
# to appear.
SHARED_VOICE_SEGMENT_S = 90.0


def _track_rms(track_pcm) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(track_pcm * track_pcm))) if len(track_pcm) else 0.0


def _segment_energy(track_pcm, start: float, end: float, track_rms: float) -> float | None:
    """How prominent a segment is on its own track: segment RMS / track RMS.

    RELATIVE, not absolute, so it is comparable ACROSS tracks of different gain.
    A speaker is prominent on their own mic (ratio around/above 1) and faint
    where they only bleed into another track (well below 1) - even when the
    bleed's absolute level rivals a quiet clean track's. This is the signal the
    echo filter uses to keep the real copy and drop the bleed. Measured on the
    raw pre-subtraction PCM. Returns None if the span is empty (can't rank it)."""
    import numpy as np

    rate = track_separation.ANALYSIS_RATE
    s0 = max(0, int(start * rate))
    s1 = min(len(track_pcm), int(end * rate))
    if s1 <= s0 or track_rms <= 0:
        return None
    window = track_pcm[s0:s1]
    return float(np.sqrt(np.mean(window * window)) / track_rms)


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

        # Checkpoint of the job currently being processed, and the audio "unit"
        # (one engine call) in flight - set so `_await_engine_result` can persist
        # streamed chunks under the right unit with the right speaker/energy.
        self._cp: job_store.Checkpoint | None = None
        self._unit: str | None = None
        self._unit_speaker: str | None = None
        self._unit_track_pcm = None
        self._unit_track_rms: float = 0.0

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

    def _transcribe_unit(
        self,
        job: Job,
        unit: str,
        audio_path: Path,
        on_progress: "Callable[[float], None] | None" = None,
        *,
        speaker: str | None = None,
        track_pcm=None,
        track_rms: float = 0.0,
        language: str | None = SENTINEL,  # type: ignore[assignment]
    ) -> list:
        """Transcribe one audio file ("unit") via the engine, resumably.

        A *unit* is "single" for a single-track job or "track<index>" per
        selected stream for multi-track. The engine streams each completed chunk
        back; `_await_engine_result` persists them to the segment log tagged with
        this unit (and `speaker`/`energy` for multi-track). On resume it skips the
        chunks already persisted and the segments come back from the log, so the
        result is identical whether or not the unit was interrupted.

        Returns `TranscribedSegment`s when `speaker is None` (single-track), or
        `dialogue.Segment`s (speaker-labeled, with energy) otherwise.

        `on_progress` receives a fraction in [0, 1] of *this unit* processed so
        far. `language` overrides the job's global language (per-track); the
        sentinel falls back to `job.language`.
        """
        cp = self._cp
        self._unit = unit
        self._unit_speaker = speaker
        self._unit_track_pcm = track_pcm
        self._unit_track_rms = track_rms

        if unit not in cp.completed_units:
            start_chunk = cp.partial_done_chunks if cp.partial_unit == unit else 0
            if cp.partial_unit != unit:
                cp.partial_unit = unit
                cp.partial_done_chunks = 0
                job_store.save(cp)
            if language is SENTINEL:
                language = job.language
            self._ensure_engine_running()
            self.request_queue.put(
                ("transcribe", job.id, job.model, audio_path, language, True, start_chunk)
            )
            self._await_engine_result(job, on_progress)  # persists chunks as they stream in
            self._finish_unit(unit)

        self._unit = None
        return self._restore_unit(unit, speaker)

    # -- checkpoint persistence of streamed chunks ----------------------------

    def _persist_chunk(self, unit: str, chunk_index: int, segments: list) -> None:
        """Durably record one completed chunk, then advance the committed counter.

        Segments arrive as `TranscribedSegment`s; we attach the unit's speaker
        and (multi-track) per-segment energy here, where the track PCM is on
        hand, so a restored segment is byte-for-byte what a fresh run produces.
        Segments are written (and fsync'd) BEFORE the counter bump so a crash in
        between leaves them as harmless stale lines, not committed duplicates."""
        cp = self._cp
        rows = []
        for seg in segments:
            row = {
                "start": seg.start, "end": seg.end, "text": seg.text,
                "words": [dialogue.word_to_dict(w) for w in seg.words],
            }
            if self._unit_speaker is not None:
                row["speaker"] = self._unit_speaker
            if self._unit_track_pcm is not None:
                row["energy"] = _segment_energy(
                    self._unit_track_pcm, seg.start, seg.end, self._unit_track_rms
                )
            rows.append(row)
        job_store.append_segments(cp.key, unit, chunk_index, rows)
        cp.partial_unit = unit
        cp.partial_done_chunks = chunk_index + 1
        job_store.save(cp)

    def _finish_unit(self, unit: str) -> None:
        cp = self._cp
        if unit not in cp.completed_units:
            cp.completed_units.append(unit)
        cp.partial_unit = None
        cp.partial_done_chunks = 0
        job_store.save(cp)

    def _restore_unit(self, unit: str, speaker: str | None) -> list:
        """Rebuild a unit's segment objects from the persisted log.

        `r.get("words") or []` guards checkpoints written before this field
        existed - they restore with no words, same as a fresh segment a
        backend couldn't supply any for (`dialogue._words_for_segment`'s
        synthetic fallback covers it downstream)."""
        rows = job_store.segments_for_unit(self._cp, unit)
        if speaker is None:
            return [
                TranscribedSegment(
                    start=r["start"], end=r["end"], text=r["text"],
                    words=[dialogue.word_from_dict(w) for w in (r.get("words") or [])],
                )
                for r in rows
            ]
        return [
            dialogue.Segment(
                speaker=r.get("speaker") or speaker,
                start=r["start"], end=r["end"], text=r["text"], energy=r.get("energy"),
                words=[dialogue.word_from_dict(w) for w in (r.get("words") or [])],
            )
            for r in rows
        ]

    def _polish_via_engine(self, job: Job, payload, is_blocks: bool):
        """Run the LLM cleanup pass (Claude CLI) via the engine process.

        `is_blocks=True`: `payload` is a list of per-speaker block texts, polished
        in one batched call (structure preserved); returns a list. `False`:
        `payload` is one plain-text passage, chunked internally; returns a str.

        The polish bar is a time-animated tail of the same overall bar (the UI
        drives it from `polish_started_at` + `estimated_polish_seconds`), so we
        don't feed per-call fractions back here - that would clobber the
        transcription-end bar value. We just mark when polishing started.
        """
        self._ensure_engine_running()
        self._unit = None  # polish streams no chunks; don't persist anything
        job.polish_started_at = datetime.now()
        job.status_detail = None  # drop any "track N/M" detail from transcription
        self.request_queue.put(("polish", job.id, payload, is_blocks))
        return self._await_engine_result(job, on_progress=None)

    def _await_engine_result(self, job: Job, on_progress):
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
            elif kind == "chunk":
                # One transcription chunk finished - persist it immediately so a
                # stop/crash after this point keeps its work.
                _, _, chunk_index, segments = message
                self._persist_chunk(self._unit, int(chunk_index), segments)
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

        # Load an existing checkpoint (this job was interrupted and is being
        # resumed) or start a fresh one. Either way the checkpoint goes
        # "running" and its work dir survives a stop/crash; only a clean finish
        # or an unrecoverable error removes it.
        cp = job_store.load(job.key)
        resuming = cp is not None and cp.status in job_store.RESUMABLE_STATUSES
        if resuming:
            job_store.compact(cp)  # drop any stale uncommitted trailing segments
            cp.status = "running"
        else:
            cp = job_store.Checkpoint(
                key=job.key,
                config=job_store.serialize_config(job),
                status="running",
                phase="queued",
                started_at=started_at.isoformat(timespec="seconds"),
            )
        self._cp = cp
        job_store.save(cp)
        work_dir = job_store.ensure_work_dir(job.key)

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
            job.estimated_polish_seconds = catalog.estimate_polish_seconds(
                job.audio_duration_seconds, job.polish
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

            if container_has_multiple_streams:
                # Auto-apply speaker separation unless the user configured it
                # themselves. One voice landing on several tracks is the mic-bleed
                # doubling case; this subtracts/drops redundant tracks by default
                # so a plain Start is clean, no manual UI step. (The text-level
                # echo filter after merge is the backstop for whatever survives.)
                # Skipped on resume: the resolved plan is already in the config.
                if len(selected_streams) > 1:
                    selected_streams = self._auto_apply_separation(
                        job, selected_streams, job.audio_duration_seconds
                    )
                # Each selected track is transcribed as its own pass, so
                # total work scales with how many tracks were picked, not
                # the container's total track count.
                if job.estimated_seconds:
                    job.estimated_seconds *= len(selected_streams)
                labeled_segments = self._transcribe_multi_track(job, streams, selected_streams, work_dir)
                blocks = dialogue.merge_segments_into_blocks(labeled_segments)
                # Mic-bleed safety net: a voice that leaked onto a second track
                # gets transcribed twice, under two speakers. Drop the quiet echo
                # turn (keeping the louder original) before rendering, then re-flag
                # interruptions since a removed echo may have falsely marked one.
                if len(selected_streams) > 1:
                    blocks = dialogue.drop_cross_speaker_echoes(blocks)
                    dialogue.flag_interruptions(blocks)
                raw_transcript_text = dialogue.render_blocks_markdown(blocks)
                # Polish one turn at a time, then re-render so speaker labels and
                # timestamps (added by render) are never exposed to the LLM.
                if job.polish and blocks:
                    polished = self._polish_via_engine(job, [b.text for b in blocks], is_blocks=True)
                    for block, text in zip(blocks, polished):
                        block.words = dialogue.realign_words_to_text(block.words, text)
                        block.text = text
                    transcript_text = dialogue.render_blocks_markdown(blocks)
                else:
                    transcript_text = raw_transcript_text
                job.turns = dialogue.blocks_to_dicts(blocks)
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
                    mp3_path = work_dir / f"{job.source_path.stem}.mp3"
                    # Reuse a conversion from a prior run; only re-encode if it's
                    # not already sitting in the (durable) work dir.
                    if not mp3_path.exists():
                        job.status = "converting"
                        self._emit(job)
                        prepare_mp3(job.source_path, mp3_path)

                if job.stop_event.is_set():
                    raise _StopRequested()

                job.status = "loading model"
                self._emit(job)

                def on_progress(fraction: float) -> None:
                    job.transcribe_fraction = fraction
                    job.transcribe_position_fraction = fraction
                    self._emit(job)

                segments = self._transcribe_unit(
                    job, job_store.SINGLE_UNIT, mp3_path, on_progress, speaker=None
                )
                text = " ".join(seg.text for seg in segments).strip()
                raw_transcript_text = text + ("\n" if text else "")
                if job.polish and text:
                    polished = self._polish_via_engine(job, text, is_blocks=False)
                    transcript_text = polished.strip() + "\n"
                else:
                    transcript_text = raw_transcript_text
                speakers_detected = None
                diarization_method = None

            polished_applied = job.polish and transcript_text != raw_transcript_text
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
                polished=polished_applied,
                raw_transcript_text=raw_transcript_text if polished_applied else None,
            )

            job.elapsed_seconds = elapsed_seconds
            job.output_path = md_path
            job.status = "done"
            job.status_detail = None
            self._record_history(job, status="success", elapsed_seconds=elapsed_seconds, started_at=started_at)
            # Clean finish: the transcript is written, so the checkpoint and its
            # artifacts have served their purpose.
            job_store.discard(job.key)
        except _StopRequested:
            # Clean stop within a session: keep the checkpoint as "paused" so the
            # job can be resumed (automatically on next launch, or via Resume).
            elapsed_seconds = time.monotonic() - start_monotonic
            job.elapsed_seconds = elapsed_seconds
            job.status = "stopped"
            job.status_detail = None
            cp.status = "paused"
            cp.phase = "transcribing"
            job_store.save(cp)
            self._record_history(job, status="stopped", elapsed_seconds=elapsed_seconds, started_at=started_at)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            elapsed_seconds = time.monotonic() - start_monotonic
            job.elapsed_seconds = elapsed_seconds
            job.status = "error"
            job.status_detail = None
            job.error = f"{type(exc).__name__}: {exc}"
            # A real error would just re-fail on resume, so don't keep the job
            # around to be auto-resumed; drop its work dir.
            job_store.discard(job.key)
            self._record_history(
                job, status="error", elapsed_seconds=elapsed_seconds, started_at=started_at, error=job.error
            )
        finally:
            self._cp = None
            self._emit(job)

    def _auto_apply_separation(
        self, job: Job, selected_streams: list[AudioStream], duration: float | None
    ) -> list[AudioStream]:
        """Apply the validated shared-voice plan (isolate / keep / drop) by default.

        Returns the streams still to transcribe (dropped ones removed) and sets
        `job.track_subtractions` for the mixes to isolate. No-op when the user
        already configured subtractions (their choice wins), when detection finds
        nothing to fix, or on any detection failure - the post-merge echo filter
        is the backstop either way. Never drops the last remaining track.

        On resume `job.separation_done` is already set (the resolved plan was
        baked into the checkpoint config), so this no-ops and we never re-run the
        best-effort detection - which could decide differently and desync from
        the segments already saved under the old plan.
        """
        if job.separation_done:  # resolved in a prior run; use the saved plan
            return selected_streams
        if job.track_subtractions:  # user override (or UI "Fix automatically")
            self._mark_separation_resolved(job, selected_streams)
            return selected_streams

        warnings = job.shared_voice_warnings
        if warnings is None:
            warnings = self._detect_shared_voice(job.source_path, selected_streams, duration)
        selected_indices = [s.index for s in selected_streams]
        subtractions, drop = track_separation.resolve_shared_voice(selected_indices, warnings)
        if not subtractions and not drop:
            self._mark_separation_resolved(job, selected_streams)
            return selected_streams

        job.status = "separating speakers"
        job.status_detail = None
        self._emit(job)
        job.track_subtractions = subtractions
        kept = [s for s in selected_streams if s.index not in drop]
        kept = kept or selected_streams
        self._mark_separation_resolved(job, kept)
        return kept

    def _mark_separation_resolved(self, job: Job, kept: list[AudioStream]) -> None:
        """Freeze the resolved separation plan into the checkpoint for resume."""
        job.separation_done = True
        job.selected_track_indices = {s.index for s in kept}
        if self._cp is not None:
            self._cp.config = job_store.serialize_config(job)
            self._cp.multi_track = True
            job_store.save(self._cp)

    def _detect_shared_voice(
        self, source: Path, streams: list[AudioStream], duration: float | None
    ) -> list:
        """Decode a representative window of each stream and detect shared voices.
        Best-effort: returns [] on any failure (auto-separation then no-ops)."""
        try:
            start = max(0.0, (duration * 0.35) if duration else 0.0)
            if duration and start + SHARED_VOICE_SEGMENT_S > duration:
                start = max(0.0, duration - SHARED_VOICE_SEGMENT_S)
            signals = [
                track_separation.TrackSignal(
                    index=s.index,
                    name="",
                    samples=track_separation.decode_track_pcm(
                        source, s.index, start_s=start, duration_s=SHARED_VOICE_SEGMENT_S
                    ),
                )
                for s in streams
            ]
            return track_separation.detect_shared_voice(signals)
        except Exception:  # noqa: BLE001 - detection is best-effort; backstop covers us
            return []

    def _transcribe_multi_track(
        self,
        job: Job,
        all_streams: list[AudioStream],
        streams_to_transcribe: list[AudioStream],
        work_dir: Path,
    ) -> list[dialogue.Segment]:
        # Default labels come from the container's full track list so a track's
        # name stays stable regardless of which subset got selected; the user's
        # custom names (from the panel) override them.
        default_names = dict(zip((s.index for s in all_streams), track_speaker_names(all_streams)))
        names = job.track_names or {}
        languages = job.track_languages or {}
        denoise_flags = job.track_denoise or {}
        subtractions = job.track_subtractions or {}
        channels = job.selected_channel_by_track or {}
        cp = self._cp

        def label_for(index: int) -> str:
            return names.get(index) or default_names.get(index, f"Speaker {index}")

        # Decode only the PCM still needed: tracks not yet completed, plus any
        # track used as a subtraction reference by one of those. A track finished
        # in a prior run is restored from the segment log, so it needs no PCM.
        # Tracks share the container's timeline, so PCM decoded here is already
        # sample-aligned.
        pending = [s for s in streams_to_transcribe if job_store.track_unit(s.index) not in cp.completed_units]
        needed: set[int] = {s.index for s in pending}
        for s in pending:
            needed.update(subtractions.get(s.index, []))
        pcm: dict[int, object] = {}
        for index in needed:
            if job.stop_event.is_set():
                raise _StopRequested()
            pcm[index] = track_separation.decode_track_pcm(
                job.source_path, index, channels.get(index)
            )

        labeled_segments: list[dialogue.Segment] = []
        total = len(streams_to_transcribe)

        for i, stream in enumerate(streams_to_transcribe):
            name = label_for(stream.index)
            unit = job_store.track_unit(stream.index)

            # Already transcribed in a prior run: pull its labeled segments
            # straight from the log and advance the overall bar.
            if unit in cp.completed_units:
                labeled_segments.extend(self._restore_unit(unit, name))
                job.transcribe_fraction = (i + 1) / total
                self._emit(job)
                continue

            if job.stop_event.is_set():
                raise _StopRequested()

            job.status = "converting"
            job.status_detail = f"track {i + 1}/{total} - {name}"
            self._emit(job)

            refs = [pcm[r] for r in subtractions.get(stream.index, []) if r in pcm]
            denoise = denoise_flags.get(stream.index, True)
            # Track wavs are keyed by stream index (stable across runs) and kept
            # in the durable work dir, so a resumed job reuses the rendered wav.
            track_wav = work_dir / f"{unit}.wav"
            if not track_wav.exists():
                track_separation.render_transcription_wav(
                    pcm[stream.index], refs, track_wav, denoise=denoise
                )

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

            language = languages.get(stream.index) or job.language
            track_pcm = pcm[stream.index]
            labeled_segments.extend(
                self._transcribe_unit(
                    job, unit, track_wav, on_progress,
                    speaker=name, track_pcm=track_pcm, track_rms=_track_rms(track_pcm),
                    language=language,
                )
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
            turns_json=json.dumps(job.turns, ensure_ascii=False) if job.turns else None,
        )
