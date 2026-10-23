"""End-to-end verify: drive the real TranscriptionWorker._process on a short
multi-track Czech clip with the new default engine (Czech finetune + VAD) and
LLM polishing ON, then print the produced transcript + raw sibling.

Exercises the whole chain: track separation -> VAD-chunked Whisper (engine
subprocess) -> dialogue blocking -> Gemma polish -> markdown + .raw.md.
"""
from __future__ import annotations

import queue
import sys
import tempfile
from pathlib import Path

ROOT = Path("/Users/stage/dev/transcriber")
sys.path.insert(0, str(ROOT))

from src import history_db, md_writer  # noqa: E402
from src.transcription import CZECH_FT_ID  # noqa: E402


def main() -> None:
    clip = Path(
        "/private/tmp/claude-501/-Users-stage-dev-transcriber/"
        "7a0eee61-b76e-48e9-b9a4-bfca94046233/scratchpad/asr_research/verify_vaclav_90s.mkv"
    )
    out_dir = Path(tempfile.mkdtemp(prefix="verify-out-"))
    md_writer.default_output_dir = lambda: out_dir  # keep output out of ~/Downloads

    from app.jobs import Job
    from app.worker import TranscriptionWorker

    conn = history_db.init_db(Path(tempfile.mkstemp(suffix=".db")[1]))
    worker = TranscriptionWorker(queue.Queue(), queue.Queue(), conn)

    job = Job(
        source_path=clip,
        model=CZECH_FT_ID,
        language="cs",
        polish=True,
        selected_track_indices={0, 2},            # 0 = mic mix (isolate), 2 = clean remote
        track_subtractions={0: [2]},              # remove the remote voice from the mic
        track_names={0: "Prodejce", 2: "Klient"},
    )

    print("Running _process (this loads Whisper + Gemma; takes a few minutes)...", flush=True)
    worker._process(job)
    worker._kill_engine()

    print("\n=== job status:", job.status, "| elapsed:", round(job.elapsed_seconds or 0, 1), "s")
    if job.error:
        print("ERROR:", job.error)
        return
    md = job.output_path
    print("=== output:", md)
    print("\n----- POLISHED (.md) -----\n")
    print(md.read_text()[:2500])
    raw = md.with_name(f"{md.stem}.raw.md")
    print("\n----- RAW sibling exists:", raw.exists(), "-----\n")
    if raw.exists():
        print(raw.read_text()[:1200])


if __name__ == "__main__":
    main()
