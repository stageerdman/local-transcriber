# Resumable transcription — durable decisions & lessons

## Durable decisions

- **Segment log is the source of truth.** Completed work lives in an
  append-only `segments.jsonl`; `job.json` counters (`completed_units`,
  `partial_unit`, `partial_done_chunks`) are the authoritative index. On read,
  keep a line iff `unit ∈ completed_units` or (`unit == partial_unit` and
  `chunk < partial_done_chunks`). This tolerates a crash between "segments
  appended" and "counter bumped": the stale trailing lines are filtered out and
  the chunk is redone deterministically.

- **Write order for crash-safety:** append segments (flush) → *then* bump the
  counter and atomically rewrite `job.json` (temp file + `os.replace`). Never
  the other way around.

- **Everything after transcription is recomputed, not checkpointed.** Dialogue
  blocking, echo-drop, interruption flagging, polish, markdown-write are pure
  functions of the segment set. Resuming during/after transcription just re-runs
  them. Keeps the checkpoint surface tiny.

- **Unit = one engine call.** `"single"` for single-track; `"track<idx>"` per
  selected stream for multi-track. Whisper streams a chunk per VAD region;
  Parakeet is one chunk (whole file, no mid-file resume — matches the existing
  "Parakeet self-chunking" known issue).

- **Durable work dir, not `mkdtemp`.** `~/Library/Application
  Support/LocalTranscriber/jobs/<key>/`. The unconditional `rmtree` in
  `_process`'s `finally` is removed; cleanup happens only on success or an
  explicit discard, so artifacts survive a stop/crash.

- **Persistent job `key` (uuid), separate from the in-memory `Job.id`.** The id
  counter resets each launch and would collide after a restart; the key is the
  stable on-disk identity.

## Lessons / gotchas

- **Don't assert byte-identical transcripts across a resume.** The engine path
  (mlx-whisper in the spawned process) is NOT deterministic across process
  invocations — two uninterrupted full runs of the same 60s clip through the
  worker gave 1797 vs 633 chars (whisper-base + spawn + the temperature ladder
  firing differently). Verified resume correctness the right way instead: the
  segment log shows each VAD chunk transcribed exactly once, in order, with no
  gaps or duplicates (scratch `diag.py`), plus deterministic unit tests with a
  fake engine (`test_worker_resume.py`). Because `condition_on_previous_text=
  False`, a resumed chunk is independent of whether the previous chunk ran in
  the same process, so resume adds no seam error beyond the VAD chunk split a
  full run already makes.

- **`compact()` on resume is essential.** Without it, a stale uncommitted
  chunk line left by a crash would be re-appended and then, once the unit lands
  in `completed_units` (where every line is kept), survive as a duplicate. End-
  to-end stop→resume confirmed a single chunk-0 line, no dup. (scratch
  `e2e_resume.py`, `diag.py`.)

- **Engine message protocol change is backward-incompatible within the app**:
  the transcribe request tuple grew a `start_chunk` field and the engine now
  streams `("chunk", …)` before `("result", …)`. Worker and engine must be
  updated together.
