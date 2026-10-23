# Word-level dialogue blocks — update v1

## Goal

Today a speaker's uninterrupted turn gets chopped into many small blocks
whenever two ASR segments are more than `DEFAULT_PAUSE_SECONDS` (0.8s) apart —
even mid-turn, with nobody else talking. In the live UI this reads as
scattered, over-fragmented dialogue, and each fragment only seeks to its own
block start.

Target behavior: **one block per uninterrupted speaker turn, however long**
(confirmed with user — no auto-splitting of long turns), while every
word/sentence inside that block still carries its own real timestamp, so
clicking anywhere in the text seeks the audio to that exact point (confirmed —
best granularity available per backend, not just block-level).

Two decisions from the user set the shape:
- **Always one block per turn.** Same speaker always merges regardless of
  pause length; only a speaker change starts a new block. (Single-speaker
  content, where there's no other voice to define a turn boundary, keeps the
  existing pause-based paragraphing — unaffected, out of scope.)
- **Realign words after polish.** Polish (on by default) rewrites each
  block's text — fixes punctuation, drops filler words, corrects a misheard
  word. Rather than only getting word-level seeking on unpolished transcripts,
  a realignment step maps the polished text back onto the original word
  timestamps (kept word → its real timestamp, corrected word → inherits the
  old one, dropped filler → just disappears).

## Why this is non-trivial today

- `TranscribedSegment`/`dialogue.Segment`/`Block` carry only a start/end/text
  per sentence-ish chunk — word-level timing exists transiently inside two of
  the four backends (Whisper's `word_timestamps=True`, ElevenLabs' native
  word stream) and is thrown away right after being used to set the
  segment's own start/end (`transcription._seg_times`).
- `merge_segments_into_blocks`'s pause-based split is the *only* thing
  bounding block size today; removing it changes block granularity
  end-to-end, not just the number of blocks.
- The written `.md` transcript is the one durable artifact — word-level data
  can't be crammed into it without wrecking readability, so it has to live in
  a separate structured channel (an in-memory `Job` field for the
  just-finished case, a DB column for reopening history later) that doesn't
  exist yet.
- Polish (`src/polish.py`) replaces a block's whole `text` wholesale. Nothing
  today re-derives word timing after that rewrite, so without a realignment
  step the feature would silently not work for the default (`polish=True`)
  path.
- Three of the four backends (Parakeet, Groq, ElevenLabs-as-currently-used)
  don't currently expose word-level timing through this codebase's wrapper,
  even though the underlying API/library can give it.

## Architecture

**`src/dialogue.py` gains a `Word(start, end, text, lead_space)`.** `text` is
whitespace-stripped; `lead_space` says whether a separator renders before it —
pushes each backend's own "does punctuation attach to the previous word"
convention down to where it's already known, so `dialogue`/the UI never need
backend-specific spacing logic. `_join_words(words)` is the single place block
text is reconstructed from words (first word's `lead_space` is ignored).
`Segment`/`Block` both gain `words: list[Word]`; when a segment has none
(shouldn't happen post-rollout, but keeps old resumed data safe), a synthetic
single `Word` covering the whole segment stands in.

**Merge rule** (`merge_segments_into_blocks`): when 2+ distinct speakers
appear anywhere in the input, a same-speaker segment always merges into the
current block regardless of gap — only a speaker change starts a new block.
When there's only ever one speaker label (nothing to "interrupt" the turn),
the existing `pause_seconds`-gated paragraph splitting is unchanged (that
path renders as plain prose with no timestamps/seeking either way, so it's
not part of this feature).

**Polish realignment**: `realign_words_to_text(words, new_text)` —
`difflib.SequenceMatcher` over accent-folded word tokens maps the old words
onto the new text's whitespace-delimited tokens: `equal`/`replace` → inherit
the matched old word's timing (proportionally for uneven replace blocks),
`delete` → dropped, `insert` → interpolated between neighbors. Called right
after `block.text = text` in the worker, before anything downstream reads
`block.words`.

**Carrying real data to the UI instead of re-parsing markdown**:
`dialogue.blocks_to_dicts(blocks)` serializes the finished blocks
(`speaker, start, end, interrupting, text, words`). `app/worker.py` stashes
this on `job.turns` right after the final render. `app/bridge.py`'s
`_job_payload` uses `job.turns` directly for a just-finished job instead of
re-parsing the written `.md` (lossy, block-level only). The same JSON is
persisted into `history_db` (`turns_json` column — this codebase's first
schema migration, so it gets its own dedicated test against a pre-migration
DB fixture) so reopening a past call later also gets word-level data via a
new output-path lookup. Legacy rows / single-track jobs with no blocks keep
falling back to today's `_parse_transcript(md_path)` regex path — same
coarse, block-level, no-word-spans behavior as now, intentional degradation,
not a bug.

**UI** (`app/web/index.html`): deferred to its own phase, starting with a
UX-expert agent pass per this project's working agreement — scope is
rendering clickable word/phrase spans without visual noise, moving the
read-along highlight from "whole block lights up" to a progressive
within-block highlight, and the legacy/no-word-spans degraded case.

## Phased roadmap

- **Phase 1 — `Word`, `_join_words`, new merge rule (`src/dialogue.py`).**
  `Word` dataclass; `words` field on `Segment`/`Block`; `_join_words`;
  rewrite `merge_segments_into_blocks`'s same-speaker gap check to branch on
  distinct-speaker-count. _Tests: always-merge-same-speaker-with-2+-speakers,
  speaker-change-always-splits, single-speaker-pause-fallback-unchanged,
  `_join_words` leading-space/boundary behavior, synthetic-`Word`
  fallback._ **STATUS: done** (commit `4f3a423`)

- **Phase 2 — Polish realignment + serialization (`src/dialogue.py`).**
  `realign_words_to_text`; `blocks_to_dicts`. _Tests: equal/replace/delete/
  insert realignment cases, identity case (polish no-op), serialization
  shape._ **STATUS: done** (commit `4f3a423`)

- **Phase 3 — Per-backend word population (`src/transcription.py`).**
  Whisper (keep+use `seg["words"]`), Parakeet (token→word grouping via the
  leading-space convention — confirmed `AlignedToken.start` is already
  absolute-to-file, no offset math needed), Groq (`timestamp_granularities=
  ["word","segment"]` + bucket words into segments by word COUNT, not time —
  see wiki.md, a real end-to-end run showed segment/word timestamps drift
  apart over a file), ElevenLabs (stop discarding `type=="spacing"`, use it
  for `lead_space`). Per-segment try/except fallback to a synthetic `Word`,
  not backend-wide. _Tests: Parakeet token-grouping, Groq word-bucketing
  (including the drift case that sank the first design), ElevenLabs
  spacing→`lead_space`, Whisper word extraction incl. VAD offset - each a
  pure function (`tests/test_transcription_words.py`,
  `tests/test_transcription_elevenlabs.py`)._ **STATUS: done** — also
  verified end-to-end on a real ~60s call through all four backends
  (`dialogue._join_words(seg.words) == seg.text` for every segment: Parakeet
  3/3, Whisper-tiny 6/6, Groq 25/25; ElevenLabs blocked from this sandbox by
  an IP allowlist on the API key, verified via the installed SDK's pydantic
  model instead - see wiki.md).

- **Phase 4 — Resumability (`app/worker.py`).** `src/job_store.py` needed no
  changes - its segment log is schema-agnostic JSON lines, so a new `words`
  key just rides along. Added `dialogue.word_to_dict`/`word_from_dict` (also
  refactored `blocks_to_dicts` onto `word_to_dict` - one shape, one place).
  `_persist_chunk` serializes `seg.words`; `_restore_unit` reconstructs them,
  `r.get("words") or []` guarding old-shaped checkpoint rows. _Tests: resume
  round-trip preserves words; old-shaped row (no `words` key) still restores
  with an empty list, same as any segment a backend couldn't supply words
  for._ **STATUS: done**

- **Phase 5 — Worker wiring (`app/worker.py`, `app/jobs.py`).** `Job` gains a
  `turns: list[dict] | None` field. In the multi-track branch, each polished
  block's words are realigned to the new text right before `block.text =
  text`; `job.turns = dialogue.blocks_to_dicts(blocks)` is set unconditionally
  after that branch's render (polished or not), so a non-polished job still
  gets structured data - only single-track jobs (no blocks at all) leave
  `job.turns` at its `None` default. _Tests:
  `test_a_polished_jobs_turns_words_still_join_back_to_its_text`
  (`tests/test_dialogue.py`) reproduces the worker's exact realign-then-render
  sequence and confirms `job.turns[i]["words"]` joins back to
  `job.turns[i]["text"]` via `_join_words` — the actual regression this phase
  exists to prevent._ **STATUS: done**

- **Phase 6 — History DB schema (`src/history_db.py`).** `_migrate()` adds
  `turns_json TEXT` via a guarded `ALTER TABLE` (checked against `PRAGMA
  table_info` first, so it's safe to call on every `init_db()`) — this
  codebase's first schema migration. Threaded through `record_run`/
  `fetch_history`/`HistoryRow`; new `turns_json_for_output_path` (most-
  recent-match, the column isn't unique). `app/worker.py`'s
  `_record_history` passes `json.dumps(job.turns)` when set. _Tests:
  pre-migration DB fixture (written with the bare old `SCHEMA`, bypassing
  `init_db`) migrates on open and reads back `turns_json=None` on the old
  row; fresh DB round-trips `turns_json`; most-recent-match and
  no-match-returns-None cases._ **STATUS: done**

- **Phase 7 — Bridge wiring (`app/bridge.py`).** New `_turns_payload(turns)`
  builds the same `{"speakers", "turns"}` shape `_parse_transcript` always
  returned, but from real data - `start`/`end`/`words` now ride along too.
  `_job_payload` uses it from `job.turns` when set; `open_call` looks up
  `history_db.turns_json_for_output_path` and uses it when present. Both
  fall back to `_parse_transcript(md_path)` otherwise (legacy rows,
  single-track jobs, or a malformed `turns_json` row - degrade, don't fail
  the open). _Tests: `tests/test_bridge_turns.py` - both paths for
  `_job_payload` and `open_call`, plus the missing-file error case._
  **STATUS: done** — also smoke-tested with a real `Bridge()` construction
  (worker thread + real sqlite history.db), not just the unit tests.

- **Phase 8 — UX pass + `app/web/index.html`.** A dedicated UX-expert agent
  pass designed and implemented all three pieces: `.say .w` word spans
  (click-to-seek, hover-underline only - no per-word tab stop, the existing
  `.ts` timestamp stays the one keyboard-reachable seek point), a cumulative
  `.said`-class sweep for the progressive within-block highlight (chosen
  over a single-word spotlight - forgiving of the 300ms poll cadence, no
  jitter), and the degraded case falls out structurally (`sayHtml` returns
  plain escaped text when `words` is empty - no separate code path for
  legacy/single-track rows). Mock data augmented so the standalone
  browser-preview mode exercises the new path too.
  **STATUS: implemented, NOT visually verified** — this sandbox has no
  Chrome extension connection (confirmed via `tabs_context_mcp`, same
  limitation Phase 7 hit) and no native-window access, so neither the fork
  nor I could screenshot/click-through the real rendering. Verified instead
  via a Node script re-implementing `sayHtml` + the word-index scan against
  10 cases (spacing, escaping/XSS, boundaries) and `node --check` on the
  extracted script - confirms the *logic*, not the *feel* (hover affordance,
  `.25s` transition smoothness against real audio cadence). **Needs one
  real look in the running app before this update closes.**

## ▶ RESUME HERE (2026-10-23)

All 8 phases implemented, 176 tests passing (up from 150 at the start of
this update). **Only remaining gate before closing:** open a real
multi-track call in the running app and confirm word-click-to-seek and the
progressive read-along highlight actually feel right — this sandbox has no
Chrome extension or native-window access, so Phase 8's visual feel was never
seen, only logic-verified (see Phase 8 above). Everything else is done.

## Deferred follow-ups (optional)

- Single-track (undiarized) transcripts still have no per-paragraph timing or
  click-to-seek at all — untouched by this update, a separate ask if ever
  wanted.
