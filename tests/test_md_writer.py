from datetime import datetime
from pathlib import Path

from src.md_writer import next_available_path, write_transcript_markdown


def test_next_available_path_returns_input_when_free(tmp_path: Path) -> None:
    target = tmp_path / "call.md"
    assert next_available_path(target) == target


def test_next_available_path_appends_suffix_on_collision(tmp_path: Path) -> None:
    (tmp_path / "call.md").touch()
    assert next_available_path(tmp_path / "call.md") == tmp_path / "call (2).md"

    (tmp_path / "call (2).md").touch()
    assert next_available_path(tmp_path / "call.md") == tmp_path / "call (3).md"


def test_write_transcript_markdown_writes_frontmatter_and_text(tmp_path: Path) -> None:
    source = tmp_path / "Sales Call.mp4"
    source.touch()

    md_path = write_transcript_markdown(
        source,
        "Hello there.\n",
        model="mlx-community/whisper-small-mlx",
        language="en",
        elapsed_seconds=12.4,
        transcribed_at=datetime(2026, 7, 31, 10, 0, 0),
    )

    assert md_path == tmp_path / "Sales Call.md"
    content = md_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "source: Sales Call.mp4" in content
    assert "model: mlx-community/whisper-small-mlx" in content
    assert "language: en" in content
    assert "elapsed_seconds: 12" in content
    assert content.endswith("Hello there.\n")


def test_write_transcript_markdown_defaults_language_to_auto(tmp_path: Path) -> None:
    source = tmp_path / "call.mp4"
    source.touch()

    md_path = write_transcript_markdown(
        source, "Text.\n", model="m", language=None, elapsed_seconds=1.0
    )

    assert "language: auto" in md_path.read_text(encoding="utf-8")
