# Word-level dialogue blocks — wiki

## Decisions

- **Merge rule is keyed on distinct-speaker-count, not a new flag or mode.**
  2+ speakers anywhere in the segment set → same-speaker segments always
  merge, regardless of gap; only a speaker change splits. Exactly 1 speaker →
  unchanged pause-based paragraphing (`tests/test_dialogue.py`'s existing
  `test_starts_new_block_after_a_long_pause_same_speaker` already covers this
  and needed no rewrite). Chosen because the single-speaker path renders as
  plain prose with no timestamps/seeking either way — there's no turn concept
  to merge around, so the old behavior is simply the right one there, and a
  separate parameter/mode would be unearned complexity for a case the merge
  rule doesn't actually need to touch.

- **Polish gets realigned, not bypassed.** Polish is on by default
  (`catalog.DEFAULT_POLISH_ENABLED = True`), so "only word-level when
  unpolished" would mean the headline feature doesn't fire for most real
  transcripts. Chosen the realignment step (`difflib.SequenceMatcher` over
  accent-folded tokens, same normalization idea as `polish._fold`) instead,
  even though it's net-new logic, because it's the only option that doesn't
  quietly downgrade the default path.

- **Word timing lives in a structured channel, never inline in the `.md`.**
  The written transcript is the durable, user-facing artifact; per-word
  timestamp markup would wreck its readability, especially for literal
  word-level granularity. Structured data (`job.turns`, then a
  `history_db.turns_json` column) is additive and never changes what's
  written to disk.

## Verified (not assumed) during design

- `src/dialogue.py` imports nothing from `src/transcription.py` — safe to
  import `dialogue.Word` into `transcription.py` one-directionally.
- Every `TranscribedSegment(...)` call site (prod + tests) uses keyword args
  only, so adding a `words` field with a `field(default_factory=list)`
  default breaks nothing.
- Parakeet: `parakeet_mlx/parakeet.py` applies `token.start += chunk_offset`
  *before* the cross-chunk merge functions run — `AlignedToken.start/end` on
  `result.sentences[*].tokens` are already absolute-to-whole-file. No extra
  offset math needed. The merge functions also dedupe overlapping tokens by
  `id`, so no double-counted word at a chunk boundary.
- Parakeet token→word boundary convention: a token starts a new word iff its
  `text` has a leading space (confirmed by reading
  `parakeet_mlx/alignment.py:tokens_to_sentences`, which uses exactly this
  check to find word/sentence boundaries).
- ElevenLabs' real word/spacing shape (via the installed SDK's pydantic
  model, not guessed): `type: Literal["word","spacing","audio_event"]`,
  `text`, `start`, `end` (start/end optional).
- Groq's `timestamp_granularities` param is real
  (`groq/types/audio/transcription_create_params.py`,
  `List[Literal["word","segment"]]`) — SDK docstring confirms word timestamps
  cost extra latency vs. segment (no latency cost), matching the plan to
  request both.
- `history_db.py` has never had a schema migration — one `CREATE TABLE IF NOT
  EXISTS` since initial creation, `record_run` is insert-only. The
  `turns_json` column is the first `ALTER TABLE` this codebase has ever
  needed — treated as novel ground in the roadmap (dedicated migration test
  against a pre-migration DB fixture), not assumed to be a well-trodden path.
- `_persist_chunk`/`_restore_unit` in `app/worker.py` is the only place
  segments cross the dataclass↔dict boundary; `_detect_shared_voice` and the
  echo-filter path only ever touch in-memory `Segment`/`Block` objects.

## Phase 3 research spike (`spike/groq_word_granularity.py`) — real findings

- Groq's `verbose_json` + `timestamp_granularities=["word","segment"]`
  returns `words` as a **flat list across the whole file**: each entry is
  `{"word": str, "start": float, "end": float}` - no leading-space marker (a
  word's own text has no spacing convention baked in, unlike Whisper's) and
  no per-segment nesting. `segments` keeps its existing shape
  (`id, seek, start, end, text, tokens, ...`) with no `words` field added -
  confirmed via one real call against `testing/videos/martin_track3_60s.mp3`.
- **First design (bucket words into segments by matching `start` against
  segment time boundaries) was built, tested on synthetic data, then thrown
  out after a real end-to-end run exposed it:** on that same 60s real call,
  segment-level and word-level timestamps come from two independent
  alignment passes that drift apart over the file - by the 7th segment the
  drift was a whole word wide, so every bucket from there on grabbed the
  *next* segment's first word and dropped its own first word, compounding for
  the rest of the file. Caught by comparing `dialogue._join_words(seg.words)`
  against `seg.text` end-to-end - they matched in the unit tests (synthetic,
  no drift) but diverged on 23/25 real segments.
- **Second design (bucket by word COUNT, not time): consume
  `len(segment.text.split())` words off the front of the flat list, in
  order.** Verified on the same real call: summing `len(seg.text.split())`
  across all 25 segments equals `len(words)` exactly (116 either way) -
  punctuation attaches to its word in both views, so counts never diverge
  even though absolute timestamps do. Re-ran the same end-to-end check after
  switching: 0/25 mismatches.

## Phase 8 UX decisions

- **Word-click is a mouse-only enhancement, not a keyboard-parity feature.**
  The existing per-turn `.ts` timestamp stays the one keyboard-reachable seek
  point; giving every word its own tab stop would make a 50+ word turn
  unusable with Tab. Conscious trade-off, not an oversight.
- **Progressive highlight is a cumulative reveal (`.said` class up to the
  current word index), not a moving single-word spotlight.** At the existing
  300ms playback-position poll cadence, a spotlight that jumps between
  exactly one word at a time would read as jittery/laggy on any polling gap.
  A monotonic "already-read fades to full ink" sweep tolerates gaps for free
  (state only ever needs `wIdx >= previous`) and matches familiar read-along
  UIs (podcast transcripts, teleprompters).
- **Degradation is structural, not a parallel code path.** `sayHtml(t)`
  returns plain escaped text when `t.words` is empty; the CSS rules only
  ever target `.say .w` spans, which simply don't exist on a degraded turn.
  Single-track jobs, legacy history rows, and most of the mock demo data all
  render exactly as they did before this phase, with zero special-casing in
  `renderTranscript`.
- **Not verified visually in this environment.** No Chrome extension
  connection and no native-window access in this sandbox (confirmed via
  `tabs_context_mcp`) — logic was checked with a standalone Node
  reimplementation of `sayHtml` + the word-index scan against 10 cases
  (spacing, XSS-escaping, boundaries) plus `node --check` on the extracted
  script, but the actual feel (hover affordance, `.25s` transition
  smoothness against real audio) needs one real look in the running app.
