from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path

from src.transcription import ensure_model_loaded, transcribe_mp3_segments

SHUTDOWN = None


def run_engine(request_queue: "Queue", response_queue: "Queue") -> None:
    """Long-lived child process: loads/caches models and does the heavy work.

    Runs in a separate OS process (not just a thread) so the worker thread can
    forcibly kill it to cancel an in-progress job - mlx_whisper's transcribe()
    and mlx-lm's generate() are single blocking calls with no cooperative
    cancellation hook. Kept alive across jobs so loaded models stay warm; only
    killed (and respawned fresh) when the user stops a job mid-run.

    Requests are tagged tuples:
      ("transcribe", job_id, model_name, audio_path, language, use_vad, start_chunk)
      ("polish",     job_id, payload, is_blocks)
    Transcribe replies stream ("chunk", job_id, chunk_index, segments) as each
    chunk finishes (so the worker can persist it before any kill), then a final
    ("result", job_id, n_segments). Both also emit ("phase"|"progress"|"error").
    """
    while True:
        request = request_queue.get()
        if request is SHUTDOWN:
            return

        kind = request[0]
        if kind == "transcribe":
            _, job_id, model_name, audio_path, language, use_vad, start_chunk = request
            try:
                ensure_model_loaded(model_name)
                response_queue.put(("phase", job_id, "transcribing"))
                segments = transcribe_mp3_segments(
                    model_name,
                    Path(audio_path),
                    language=language,
                    progress_callback=lambda fraction: response_queue.put(("progress", job_id, fraction)),
                    use_vad=use_vad,
                    chunk_sink=lambda idx, segs: response_queue.put(("chunk", job_id, idx, segs)),
                    start_chunk=start_chunk,
                )
                response_queue.put(("result", job_id, len(segments)))
            except Exception as exc:  # noqa: BLE001 - reported back, not swallowed
                response_queue.put(("error", job_id, type(exc).__name__, str(exc)))

        elif kind == "polish":
            _, job_id, payload, is_blocks = request
            try:
                from src.polish import polish_blocks, polish_text

                response_queue.put(("phase", job_id, "polishing"))
                on_progress = lambda fraction: response_queue.put(("progress", job_id, fraction))
                if is_blocks:
                    result = polish_blocks(payload, progress_callback=on_progress)
                else:
                    result = polish_text(payload, progress_callback=on_progress)
                response_queue.put(("result", job_id, result))
            except Exception as exc:  # noqa: BLE001 - reported back, not swallowed
                response_queue.put(("error", job_id, type(exc).__name__, str(exc)))
