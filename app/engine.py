from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path

from src.transcription import ensure_model_loaded, transcribe_mp3_segments

SHUTDOWN = None


def run_engine(request_queue: "Queue", response_queue: "Queue") -> None:
    """Long-lived child process: loads/caches a model and transcribes files.

    Runs in a separate OS process (not just a thread) so the worker thread can
    forcibly kill it to cancel an in-progress transcription - mlx_whisper's
    transcribe() is a single blocking call with no cooperative cancellation
    hook. Kept alive across jobs so the loaded model stays warm; only killed
    (and respawned fresh) when the user stops a job mid-transcription.
    """
    while True:
        request = request_queue.get()
        if request is SHUTDOWN:
            return

        job_id, model_name, mp3_path, language = request
        try:
            ensure_model_loaded(model_name)
            response_queue.put(("phase", job_id, "transcribing"))
            segments = transcribe_mp3_segments(model_name, Path(mp3_path), language=language)
            response_queue.put(("result", job_id, segments))
        except Exception as exc:  # noqa: BLE001 - reported back, not swallowed
            response_queue.put(("error", job_id, type(exc).__name__, str(exc)))
