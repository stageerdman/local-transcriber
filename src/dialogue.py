from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

DEFAULT_PAUSE_SECONDS = 0.8


@dataclass
class Word:
    """One word (or punctuation-attached token) with its own real timestamp.

    `text` is stripped of surrounding whitespace. `lead_space` says whether a
    separator renders before this word when prose is reconstructed from a word
    list (see `_join_words`) - each backend knows its own "does punctuation
    attach to the previous word" convention (Whisper/Parakeet encode it as a
    leading space on the token itself; ElevenLabs as a separate spacing token),
    so that knowledge stays local to extraction and never leaks into dialogue
    merging or the UI.
    """

    start: float
    end: float
    text: str
    lead_space: bool = True


def _join_words(words: list[Word]) -> str:
    """Reconstruct prose from a word list - the one place block/segment text is
    built, so the text and its time-indexed words can never drift apart."""
    parts: list[str] = []
    for i, w in enumerate(words):
        if i and w.lead_space:
            parts.append(" ")
        parts.append(w.text)
    return "".join(parts).strip()


def _words_for_segment(seg: "Segment") -> list[Word]:
    """A segment's own words, falling back to one synthetic whole-segment word
    when the backend didn't supply any - keeps old/resumed data safe without
    the merge logic needing to know the difference."""
    if seg.words:
        return seg.words
    text = seg.text.strip()
    return [Word(seg.start, seg.end, text, lead_space=True)] if text else []


@dataclass
class Segment:
    """A transcribed chunk of speech attributed to a speaker.

    `speaker` is a display label ("Person 1", or a name pulled from track
    metadata) - not a stable identity across files.

    `energy` is this segment's acoustic level (RMS) on its OWN source track,
    measured from the raw decoded PCM before any subtraction/normalization. It
    is the signal the cross-speaker echo filter uses to decide which copy of a
    doubled utterance is the real speaker (loud on their own mic) and which is
    mic bleed (a quiet leak into another track). `None` when unknown (e.g. a
    single-track job, where there is no bleed to resolve).

    `words` is this segment's own word-level timing, when the backend can give
    it (see `src.transcription`); empty when it can't - `_words_for_segment`
    is the one place that fallback is resolved.
    """

    speaker: str
    start: float
    end: float
    text: str
    energy: float | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class Block:
    """One or more consecutive same-speaker segments merged for display."""

    speaker: str
    start: float
    end: float
    text: str
    interrupting: bool = False
    # Loudest the speaker got during this turn, on their own track (the max of
    # the merged segments' `Segment.energy`). The cross-speaker echo filter uses
    # it to tell the real turn from a quiet bleed copy. `None` when unknown.
    energy: float | None = None
    # Word-level timing across the whole block, in order - lets the UI seek to
    # any point in the text, not just the block's start. `text` above is always
    # `_join_words(words)`: this is the single source of truth for the block's
    # prose, never maintained independently.
    words: list[Word] = field(default_factory=list)


# --- cross-speaker echo (mic-bleed) removal -----------------------------------
# One person's voice can land on more than one track: their own mic AND, as a
# quiet leak, a second participant's mic (or an un-subtracted mix track). The
# transcriber then emits the same utterance twice, under two speakers, a second
# or so apart. This is the final, output-level safety net for that doubling -
# it runs no matter how (or whether) acoustic separation was configured, because
# the duplication is defined by what actually reached the transcript.
#
# This runs on merged BLOCKS, not raw segments: each speaker's turn is one block,
# so a whole doubled turn lines up against its echo even when the two tracks
# chopped the audio at different segment boundaries (they routinely do). Matching
# raw segments misses those, since the boundaries don't coincide.
#
# A pair is an echo when, across two DIFFERENT speakers, the turns overlap in
# time (or sit within a short gap - ASR start times drift) and their text is
# substantially the same. The real speaker is the louder copy on its own track
# (`energy`); the quieter one is the bleed and is dropped. Short utterances are
# left alone: genuine simultaneous backchannels ("jo", "ano") are common and
# must not be collapsed.
ECHO_MAX_GAP_SECONDS = 1.5
ECHO_SIM_THRESHOLD = 0.60
ECHO_MIN_CHARS = 12


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _texts_match(a: str, b: str, threshold: float) -> bool:
    na, nb = _normalize_for_match(a), _normalize_for_match(b)
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _is_echo_pair(a: Segment, b: Segment, max_gap: float) -> bool:
    """True when a and b are the same utterance heard on two speakers' tracks."""
    if a.speaker == b.speaker:
        return False
    # time-aligned: spans overlap, or the gap between them is within ASR drift
    gap = max(a.start, b.start) - min(a.end, b.end)
    return gap <= max_gap


def _weaker(a: Segment, b: Segment) -> Segment:
    """Pick the bleed copy of an echo pair: the quieter track, else the shorter
    (more fragmented) text, else the later-starting one. Deterministic."""
    if a.energy is not None and b.energy is not None and a.energy != b.energy:
        return a if a.energy < b.energy else b
    la, lb = len(a.text.strip()), len(b.text.strip())
    if la != lb:
        return a if la < lb else b
    return a if a.start >= b.start else b


def drop_cross_speaker_echoes(
    blocks: list[Block],
    *,
    max_gap: float = ECHO_MAX_GAP_SECONDS,
    sim_threshold: float = ECHO_SIM_THRESHOLD,
    min_chars: int = ECHO_MIN_CHARS,
) -> list[Block]:
    """Remove mic-bleed duplicates: the same turn transcribed under two speakers.
    Keeps the louder original, drops the quiet echo. Pure function.

    Expects merged blocks (see `merge_segments_into_blocks`); also works on raw
    `Segment`s, which share the needed fields. Only turns at least `min_chars`
    long are eligible, so real short backchannels spoken over each other are
    never collapsed. Each echo pair drops exactly one side; a loud original
    survives any number of faint copies. Callers that care about interruption
    flags should re-run `flag_interruptions` on the result.
    """
    ordered = sorted(
        (b for b in blocks if b.text.strip()),
        key=lambda b: (b.start, b.end),
    )
    dropped: set[int] = set()
    n = len(ordered)
    for i in range(n):
        if i in dropped:
            continue
        a = ordered[i]
        if len(a.text.strip()) < min_chars:
            continue
        for j in range(i + 1, n):
            if j in dropped:
                continue
            b = ordered[j]
            # sorted by start: once b starts beyond a's reach, so does the rest
            if b.start - a.end > max_gap:
                break
            if len(b.text.strip()) < min_chars:
                continue
            if not _is_echo_pair(a, b, max_gap):
                continue
            if not _texts_match(a.text, b.text, sim_threshold):
                continue
            loser = _weaker(a, b)
            if loser is a:
                dropped.add(i)
                break  # a is gone; move to the next segment
            dropped.add(j)
    return [s for k, s in enumerate(ordered) if k not in dropped]


def merge_segments_into_blocks(
    segments: list[Segment], pause_seconds: float = DEFAULT_PAUSE_SECONDS
) -> list[Block]:
    """Merge segments (any speaker, any order) into chronological dialogue blocks.

    A speaker change always starts a new block. Within one speaker's turn, how
    segments merge depends on whether anyone else is even in the conversation:

    - **2+ speakers present** (a real dialogue): same-speaker segments always
      merge into the current block, regardless of the gap between them - one
      block per uninterrupted turn, however long. There's an actual "someone
      else could have interrupted but didn't" signal here, which is what a
      turn means.
    - **Only ever one speaker** (nothing to interrupt the turn - no diarized
      voice, or everyone else was filtered out): falls back to the original
      pause-based paragraph splitting, merging only when the gap is within
      `pause_seconds`. This path renders as plain prose with no per-paragraph
      timestamps anyway (`render_blocks_markdown`), so it's unaffected by (and
      not part of) the turn-based merge rule above.

    A block that starts before the previous block finished is flagged as an
    interruption - the previous speaker was talked over.
    """
    ordered = sorted((s for s in segments if s.text.strip()), key=lambda s: s.start)
    multi_speaker = len({s.speaker for s in ordered}) > 1

    blocks: list[Block] = []
    for seg in ordered:
        seg_words = _words_for_segment(seg)
        if blocks:
            last = blocks[-1]
            gap = seg.start - last.end
            same_speaker = seg.speaker == last.speaker
            if same_speaker and (multi_speaker or gap <= pause_seconds):
                if seg_words:
                    # Force a separating space at the segment join, matching
                    # the old plain-text-concat behavior, without mutating the
                    # segment's own word (another block may still reference it
                    # via `_words_for_segment`'s fallback).
                    first = seg_words[0]
                    seg_words = [Word(first.start, first.end, first.text, True), *seg_words[1:]]
                last.words.extend(seg_words)
                last.text = _join_words(last.words)
                last.end = max(last.end, seg.end)
                last.energy = _max_energy(last.energy, seg.energy)
                continue

        blocks.append(
            Block(
                speaker=seg.speaker, start=seg.start, end=seg.end,
                text=_join_words(seg_words), words=list(seg_words), energy=seg.energy,
            )
        )

    flag_interruptions(blocks)
    return blocks


def _max_energy(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


def flag_interruptions(blocks: list[Block]) -> None:
    """Set each block's `interrupting` flag in place: a block that starts before
    the previous one ended means the previous speaker was talked over.

    Kept separate from merging so it can be re-run after echo removal, which can
    delete the block that caused a false interruption flag on its neighbour.
    """
    for pos, block in enumerate(blocks):
        block.interrupting = pos > 0 and block.start < blocks[pos - 1].end


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_blocks_markdown(blocks: list[Block]) -> str:
    """Render blocks as the transcript body written to the .md file.

    With a single speaker (or no speaker info at all), renders plain
    pause-delimited paragraphs - no invented "Person 1:" label on a solo
    recording. With 2+ speakers, renders labeled dialogue with timestamps and
    an "(interrupting)" tag where speech overlapped.
    """
    if not blocks:
        return ""

    speakers = {b.speaker for b in blocks}
    if len(speakers) <= 1:
        return "\n\n".join(b.text for b in blocks) + "\n"

    parts = []
    for block in blocks:
        tag = " (interrupting)" if block.interrupting else ""
        parts.append(f"**{block.speaker}** [{format_timestamp(block.start)}]{tag}\n{block.text}")
    return "\n\n".join(parts) + "\n"


# --- polish realignment -------------------------------------------------------
# Polish (`src.polish`) replaces a block's whole `text` with an LLM-cleaned
# version - fixed punctuation, dropped filler words, a corrected misheard word.
# That text no longer lines up positionally with the word-level timing the
# block was built with. `realign_words_to_text` re-times the new text against
# the old words so word-level seeking keeps working on what the user actually
# reads, instead of only on raw (usually never-shown) ASR output.


def _fold(text: str) -> str:
    """Casefold and strip diacritics for matching, so polish adding/removing a
    Czech diacritic doesn't read as the word having changed."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def realign_words_to_text(words: list[Word], new_text: str) -> list[Word]:
    """Re-time `new_text` against `words`' original timing.

    Matches `new_text`'s whitespace-delimited tokens to `words` by
    accent-folded text similarity (`difflib.SequenceMatcher`):
    - a kept or lightly-corrected token inherits its matched old word's
      timing,
    - a dropped filler word has no corresponding output token at all - no
      orphaned timestamp left behind,
    - a token with no match (polish essentially never adds content - see the
      containment guard in `src.polish` - but handle it rather than assume)
      gets a timestamp interpolated between its timed neighbors.

    Returns a fresh `Word` list; `_join_words(...)` of the result is
    `new_text`. Call this right after any `block.text = <polished text>`
    reassignment, before anything downstream reads `block.words`.
    """
    new_tokens = new_text.split()
    if not new_tokens:
        return []
    if not words:
        # Nothing to realign against - keep this pure function total rather
        # than raise; spread evenly-untimed words rather than invent a time.
        return [Word(0.0, 0.0, t, lead_space=i > 0) for i, t in enumerate(new_tokens)]

    old_folded = [_fold(w.text) for w in words]
    new_folded = [_fold(t) for t in new_tokens]
    matcher = SequenceMatcher(None, old_folded, new_folded, autojunk=False)

    out: list[Word | None] = [None] * len(new_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, j in enumerate(range(j1, j2)):
                src = words[i1 + offset]
                out[j] = Word(src.start, src.end, new_tokens[j], lead_space=j > 0)
        elif tag == "replace":
            old_span = words[i1:i2]
            span_len = j2 - j1
            for offset, j in enumerate(range(j1, j2)):
                # Distribute an unevenly-sized replace block proportionally
                # across the old span rather than assuming a 1:1 count match.
                src = old_span[min(len(old_span) - 1, (offset * len(old_span)) // span_len)]
                out[j] = Word(src.start, src.end, new_tokens[j], lead_space=j > 0)
        # "delete": old words with no new token contribute nothing.
        # pure "insert" (no old span at all) is resolved by interpolation below.

    for j, w in enumerate(out):
        if w is not None:
            continue
        prev_w = next((out[k] for k in range(j - 1, -1, -1) if out[k] is not None), None)
        next_w = next((out[k] for k in range(j + 1, len(out)) if out[k] is not None), None)
        if prev_w is not None and next_w is not None:
            start = end = (prev_w.end + next_w.start) / 2
        elif prev_w is not None:
            start = end = prev_w.end
        elif next_w is not None:
            start = end = next_w.start
        else:
            start = end = 0.0
        out[j] = Word(start, end, new_tokens[j], lead_space=j > 0)

    return out  # type: ignore[return-value]  # every slot filled above


# --- serialization for the UI / history --------------------------------------


def blocks_to_dicts(blocks: list[Block]) -> list[dict]:
    """Plain-dict (JSON-ready) form of `blocks`, for the live UI payload and
    for persisting into history - the structured alternative to re-parsing the
    written markdown, which only ever carried block-level timing. Lane/color
    assignment is a presentation concern and stays with the caller."""
    return [
        {
            "speaker": b.speaker,
            "start": b.start,
            "end": b.end,
            "interrupting": b.interrupting,
            "text": b.text,
            "words": [
                {"start": w.start, "end": w.end, "text": w.text, "lead_space": w.lead_space}
                for w in b.words
            ],
        }
        for b in blocks
    ]
