from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

DEFAULT_PAUSE_SECONDS = 0.8


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
    """

    speaker: str
    start: float
    end: float
    text: str
    energy: float | None = None


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

    Consecutive segments from the same speaker separated by a gap no longer
    than `pause_seconds` are merged into one block (this is the "pause-based
    blocks of text" behavior). A speaker change always starts a new block. A
    block that starts before the previous block finished is flagged as an
    interruption - the previous speaker was talked over.
    """
    ordered = sorted((s for s in segments if s.text.strip()), key=lambda s: s.start)

    blocks: list[Block] = []
    for seg in ordered:
        text = seg.text.strip()
        if blocks:
            last = blocks[-1]
            gap = seg.start - last.end
            if seg.speaker == last.speaker and gap <= pause_seconds:
                last.text = f"{last.text} {text}".strip()
                last.end = max(last.end, seg.end)
                last.energy = _max_energy(last.energy, seg.energy)
                continue

        blocks.append(
            Block(speaker=seg.speaker, start=seg.start, end=seg.end, text=text, energy=seg.energy)
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
