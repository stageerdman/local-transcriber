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

(fill in as we build)
