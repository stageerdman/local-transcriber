# Resumable transcription — update v1

## Goal

A transcription job can be **stopped at any moment, in any phase, and restored
later** — including after quitting the app, or a full reboot. You come back and
it picks up where it left off, losing at most a few seconds of in-flight work.

Two decisions from the user set the shape:

- **Auto-resume on launch.** On app start, any interrupted/paused job silently
  re-queues and continues from its last checkpoint. Zero clicks. (Within a
  running session, a *Stop* leaves the job paused with a *Resume* affordance;
  only across an app restart is continuation automatic.)
- **Finest granularity the backend allows.** Whisper transcribes in VAD chunks
  and resumes *mid-file* (≤~28 s of work lost). Parakeet runs a whole track in
  one opaque call and resumes at *track* boundaries (re-runs the in-flight
  track). Multi-track jobs skip already-completed tracks.

## Why this is non-trivial today

- `Job` state is 100% in-memory (`app/jobs.py`); nothing but a terminal history
  row is persisted. A stop sets `stop_event`, kills the engine process, records
  status `"stopped"`, and **throws all work away**.
- Intermediate artifacts live in a random `tempfile.mkdtemp` dir that is
  `rmtree`'d in a `finally` on every exit (success/error/stop) — unaddressable
  and gone after the process ends.
- The transcription engine is a **separate process that is killed to cancel**,
  so anything in flight inside a single `transcribe_mp3_segments` call is lost
  unless completed chunks are streamed out and persisted as they finish.

## Architecture

**One durable work dir per job**, keyed by a persistent `key` (uuid), under
`~/Library/Application Support/LocalTranscriber/jobs/<key>/`:

- `job.json` — the serialized `Job` config **plus** a checkpoint: lifecycle
  status, current phase, `completed_units`, `partial_unit`,
  `partial_done_chunks`, and durable artifact paths. Rewritten atomically
  (temp + rename) on each checkpoint.
- `segments.jsonl` — **append-only** log of completed segments, each line tagged
  with its `unit` and `chunk` index. Append-only ⇒ a crash loses at most the
  last line.
- Rendered audio artifacts (converted mp3, per-track wavs) so they survive a
  restart and aren't recomputed.

**The segment log is the single source of truth** for completed work; the
`job.json` counters are the authoritative index. On read we keep only lines
whose `unit ∈ completed_units`, or `unit == partial_unit and chunk <
partial_done_chunks` — so a crash that persisted segments but not the counter
bump is self-healing (the stale trailing lines are ignored and re-done
deterministically).

**Units.** A *unit* is one audio file handed to the engine:
- single-track job → one unit `"single"`.
- multi-track job → one unit `"track<stream index>"` per selected track.

Within a unit, Whisper streams one *chunk* per VAD speech region; Parakeet is a
single chunk 0 (whole file). The engine always streams completed chunks back via
a `("chunk", …)` message so the worker can persist them before any kill; a
`("result", …)` message just signals unit completion.

Dialogue blocking, echo-drop, interruption flagging, polish, and markdown-write
are all **pure functions of the collected segments**, so they are simply re-run
from the restored segment set — never checkpointed mid-way. A job interrupted
during polish resumes straight to re-polish + write.

## Phased roadmap

- **Phase 1 — Durable job store (`src/job_store.py`).** Job key + work dir;
  serialize/restore `Job` config (incl. `AudioStream`); checkpoint read/write
  (atomic); append-only segment log with the filter-on-read rule; `list_resumable()`
  scan; `discard(key)`. Replace `mkdtemp` with the durable dir and stop the
  unconditional `rmtree`. _Tests: config round-trip, segment log append/filter,
  resumable scan._  **STATUS: done**

- **Phase 2 — Checkpoint writes in the worker.** Write a checkpoint at every
  phase boundary and persist each completed track/chunk's segments. On success:
  write md, mark done, discard work dir. On stop: mark `"paused"`, keep work
  dir. _Tests: a stopped multi-track job leaves a valid checkpoint with the
  completed tracks' segments._  **STATUS: done**

- **Phase 3 — Resume execution (coarse).** `_process` detects a resumable job,
  loads persisted segments, skips completed units/phases, continues. _Tests:
  resuming a checkpoint yields the same final transcript as an uninterrupted
  run (determinism fixture)._  **STATUS: done**

- **Phase 4 — Mid-file Whisper chunk resume (finest).** `transcription.py`
  gains a `chunk_sink` callback + `start_chunk` skip over the deterministic VAD
  chunk list; `engine.py` streams `("chunk", …)`; the worker persists per chunk
  and passes `start_chunk` on resume. Parakeet unchanged (unit-level). _Tests:
  persisted head + skipped-tail resume == full run._  **STATUS: done**

- **Phase 5 — Auto-resume on launch.** Bridge startup scans resumable jobs,
  reconstructs the `Job`s, re-enqueues them, and emits a "resuming" event.
  _Tests: reconstruct → enqueue at the bridge level._  **STATUS: done**

- **Phase 6 — Stop / Resume UX (web UI).** UX-expert agents design the surface;
  bridge gets `stop_job` / `resume_job` / `discard_job`; the payload carries
  paused/resumable state; `app/web/index.html` grows a Stop control on a working
  call, a Paused + Resume state, and a "resuming…" cue on launch. _Verify in the
  real app._  **STATUS: done**

- **Phase 7 — End-to-end verification + docs.** Real app: start → stop mid-run →
  inspect checkpoint → relaunch → auto-resume → completes → transcript matches a
  clean run. Also a hard-kill (reboot proxy) → relaunch → resumes. Update
  `issues.txt` and `wiki.md`.  **STATUS: done**

## ▶ RESUME HERE

All phases shipped and verified (unit suite + real-engine stop/resume e2e +
full app auto-resume-on-launch integration). Update can be closed once the
follow-ups below are no longer wanted.

## Deferred follow-ups (optional)

- **File-missing resume UX.** Today a resumable job whose source file moved/was
  deleted is silently discarded at launch. UX review recommended instead keeping
  the row with a "Can't resume — source file moved or deleted" state + a "Locate
  file…" action that continues from the still-valid checkpoint. Not built.
- **Home cards have no inline Stop/Resume/Discard.** Those controls live in the
  always-visible sidebar Calls list; home cards only show status text. Fine for
  now (the sidebar is the queue surface); revisit if users act from Home.
- **Two history rows per interrupted-then-resumed job** (one `stopped`, one
  `success`). Honest but slightly noisy; could coalesce if it bothers.
