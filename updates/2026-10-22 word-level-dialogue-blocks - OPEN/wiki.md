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

## Not yet verified (resolve in Phase 3's research spike)

- Groq's actual `verbose_json` word-object field names when
  `timestamp_granularities=["word","segment"]` is set — the installed SDK's
  `Transcription` model only statically declares `text: str`; `segments` and
  `words` are consumed via duck-typing today (same pattern the existing
  `segments` handling already uses). Needs one real call against a short
  local fixture before writing that backend's extraction code.
