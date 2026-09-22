from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PAUSE_SECONDS = 0.8


@dataclass
class Segment:
    """A transcribed chunk of speech attributed to a speaker.

    `speaker` is a display label ("Person 1", or a name pulled from track
    metadata) - not a stable identity across files.
    """

    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Block:
    """One or more consecutive same-speaker segments merged for display."""

    speaker: str
    start: float
    end: float
    text: str
    interrupting: bool = False


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
                continue

        interrupting = bool(blocks) and seg.start < blocks[-1].end
        blocks.append(
            Block(speaker=seg.speaker, start=seg.start, end=seg.end, text=text, interrupting=interrupting)
        )

    return blocks


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
