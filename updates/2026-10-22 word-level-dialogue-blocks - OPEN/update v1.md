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
  fallback._ **STATUS: not started**

- **Phase 2 — Polish realignment + serialization (`src/dialogue.py`).**
  `realign_words_to_text`; `blocks_to_dicts`. _Tests: equal/replace/delete/
  insert realignment cases, identity case (polish no-op), serialization
  shape._ **STATUS: not started**

- **Phase 3 — Per-backend word population (`src/transcription.py`).**
  Whisper (keep+use `seg["words"]`), Parakeet (token→word grouping via the
  leading-space convention — confirmed `AlignedToken.start` is already
  absolute-to-file, no offset math needed), Groq (`timestamp_granularities=
  ["word","segment"]` + bucket words into segments by time), ElevenLabs
  (stop discarding `type=="spacing"`, use it for `lead_space`). Per-segment
  try/except fallback to a synthetic `Word`, not backend-wide. _Research
  spike first: one real Groq `verbose_json` call with word granularity
  against a short local fixture to confirm the actual field names before
  writing that backend's code — record in `wiki.md`._ _Tests: Parakeet
  token-grouping, Groq word-bucketing, ElevenLabs spacing→`lead_space`, each
  a pure function._ **STATUS: not started**

- **Phase 4 — Resumability (`src/job_store.py`, `app/worker.py`).**
  `_persist_chunk` serializes `seg.words`; `_restore_unit` reconstructs them,
  `r.get("words") or []` guarding old in-flight checkpoints. _Tests: resume
  round-trip preserves words; old-shaped row still restores via the Phase 1
  synthetic fallback._ **STATUS: not started**

- **Phase 5 — Worker wiring (`app/worker.py`, `app/jobs.py`).** Call
  `realign_words_to_text` right after each polish `block.text = text`;
  capture `job.turns = dialogue.blocks_to_dicts(blocks)` after final render.
  _Tests: a polished job's `job.turns[i]["words"]` joins back to
  `job.turns[i]["text"]` via `_join_words` — the actual regression this
  phase exists to prevent._ **STATUS: not started**

- **Phase 6 — History DB schema (`src/history_db.py`).** `turns_json`
  column + guarded `ALTER TABLE` migration (first one in this codebase —
  treat as novel); thread through `record_run`/`fetch_history`/`HistoryRow`;
  new lookup-by-`output_path` (most-recent-match, column isn't unique).
  _Tests: pre-migration DB fixture (no column) migrates and reads back NULL
  on old rows; fresh DB round-trips `turns_json`._ **STATUS: not started**

- **Phase 7 — Bridge wiring (`app/bridge.py`).** `_job_payload` builds the
  transcript payload from `job.turns` (shared helper with the history path)
  instead of re-parsing the `.md`; `open_call` looks up `turns_json` by
  output path, falling back to `_parse_transcript` for legacy rows /
  single-track jobs. _Tests: extend the bridge test suite for both
  paths._ **STATUS: not started**

- **Phase 8 — UX pass + `app/web/index.html`.** UX-expert agent(s) design
  clickable word/phrase rendering, progressive within-block read-along
  highlight, and the legacy-degradation look; then implement. _Verify in the
  real running app (not just tests) per this project's working
  agreement._ **STATUS: not started**

## ▶ RESUME HERE

Starting Phase 1.

## Deferred follow-ups (optional)

- Single-track (undiarized) transcripts still have no per-paragraph timing or
  click-to-seek at all — untouched by this update, a separate ask if ever
  wanted.
