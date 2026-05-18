from pathlib import Path

from src.media_finder import find_media_files


def test_find_media_files_recursively_and_only_supported_extensions(tmp_path: Path) -> None:
    supported = [
        tmp_path / "call.mp3",
        tmp_path / "nested" / "call.MP4",
        tmp_path / "nested" / "deeper" / "call.wav",
    ]
    unsupported = [
        tmp_path / "notes.txt",
        tmp_path / "nested" / "image.jpg",
    ]

    for path in supported + unsupported:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    found = find_media_files(tmp_path)

    assert found == sorted(supported)
