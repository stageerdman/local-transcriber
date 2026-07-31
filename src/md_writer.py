from __future__ import annotations

from datetime import datetime
from pathlib import Path


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def write_transcript_markdown(
    source_path: Path,
    transcript_text: str,
    *,
    model: str,
    language: str | None,
    elapsed_seconds: float,
    transcribed_at: datetime | None = None,
) -> Path:
    transcribed_at = transcribed_at or datetime.now()
    md_path = next_available_path(source_path.with_suffix(".md"))

    frontmatter = "\n".join(
        [
            "---",
            f"source: {source_path.name}",
            f"model: {model}",
            f"language: {language or 'auto'}",
            f"transcribed_at: {transcribed_at.isoformat(timespec='seconds')}",
            f"elapsed_seconds: {round(elapsed_seconds)}",
            "---",
            "",
        ]
    )
    md_path.write_text(frontmatter + "\n" + transcript_text, encoding="utf-8")
    return md_path
