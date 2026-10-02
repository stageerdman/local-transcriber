from __future__ import annotations

from datetime import datetime
from pathlib import Path


def default_output_dir() -> Path:
    return Path.home() / "Downloads"


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
    output_dir: Path | None = None,
    speakers: int | None = None,
    diarization: str | None = None,
    polished: bool = False,
    raw_transcript_text: str | None = None,
) -> Path:
    transcribed_at = transcribed_at or datetime.now()
    output_dir = output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = next_available_path(output_dir / f"{source_path.stem}.md")

    def _frontmatter(is_raw: bool) -> str:
        fields = [
            "---",
            f"source: {source_path}",
            f"model: {model}",
            f"language: {language or 'auto'}",
            f"transcribed_at: {transcribed_at.isoformat(timespec='seconds')}",
            f"elapsed_seconds: {round(elapsed_seconds)}",
        ]
        if speakers is not None:
            fields.append(f"speakers: {speakers}")
        if diarization is not None:
            fields.append(f"diarization: {diarization}")
        if polished:
            fields.append(f"polished: {'false' if is_raw else 'true'}")
        fields += ["---", ""]
        return "\n".join(fields)

    md_path.write_text(_frontmatter(is_raw=False) + "\n" + transcript_text, encoding="utf-8")

    # Keep the raw (un-polished) transcript alongside the cleaned one so the
    # LLM pass is always a non-destructive, reviewable derivation.
    if raw_transcript_text is not None:
        raw_path = md_path.with_name(f"{md_path.stem}.raw.md")
        raw_path.write_text(_frontmatter(is_raw=True) + "\n" + raw_transcript_text, encoding="utf-8")

    return md_path
