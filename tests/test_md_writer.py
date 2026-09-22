from datetime import datetime
from pathlib import Path

from src.md_writer import default_output_dir, next_available_path, write_transcript_markdown


def test_default_output_dir_is_downloads() -> None:
    assert default_output_dir() == Path.home() / "Downloads"


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
        output_dir=tmp_path,
    )

    assert md_path == tmp_path / "Sales Call.md"
    content = md_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert f"source: {source}" in content
    assert "model: mlx-community/whisper-small-mlx" in content
    assert "language: en" in content
    assert "elapsed_seconds: 12" in content
    assert content.endswith("Hello there.\n")


def test_write_transcript_markdown_defaults_language_to_auto(tmp_path: Path) -> None:
    source = tmp_path / "call.mp4"
    source.touch()

    md_path = write_transcript_markdown(
        source, "Text.\n", model="m", language=None, elapsed_seconds=1.0, output_dir=tmp_path
    )

    assert "language: auto" in md_path.read_text(encoding="utf-8")


def test_write_transcript_markdown_includes_speakers_and_diarization_when_given(tmp_path: Path) -> None:
    source = tmp_path / "call.mp4"
    source.touch()

    md_path = write_transcript_markdown(
        source,
        "**Person 1** [00:00]\nHi.\n",
        model="m",
        language="en",
        elapsed_seconds=1.0,
        output_dir=tmp_path,
        speakers=2,
        diarization="multitrack",
    )

    content = md_path.read_text(encoding="utf-8")
    assert "speakers: 2" in content
    assert "diarization: multitrack" in content


def test_write_transcript_markdown_omits_speakers_and_diarization_when_not_given(tmp_path: Path) -> None:
    source = tmp_path / "call.mp4"
    source.touch()

    md_path = write_transcript_markdown(
        source, "Text.\n", model="m", language="en", elapsed_seconds=1.0, output_dir=tmp_path
    )

    content = md_path.read_text(encoding="utf-8")
    assert "speakers:" not in content
    assert "diarization:" not in content


def test_write_transcript_markdown_writes_to_output_dir_not_next_to_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "call.mp4"
    source.touch()

    output_dir = tmp_path / "downloads"
    md_path = write_transcript_markdown(
        source, "Text.\n", model="m", language=None, elapsed_seconds=1.0, output_dir=output_dir
    )

    assert md_path == output_dir / "call.md"
    assert not (source_dir / "call.md").exists()
