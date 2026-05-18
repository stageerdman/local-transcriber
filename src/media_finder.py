from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}


def find_media_files(root_folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in root_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)
