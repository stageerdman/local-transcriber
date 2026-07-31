from __future__ import annotations

from pathlib import Path
from tkinter import Tk, filedialog

from src.media_finder import SUPPORTED_EXTENSIONS


def _media_filetypes() -> list[tuple[str, str]]:
    patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
    return [("Media files", patterns), ("All files", "*.*")]


def pick_folder() -> Path | None:
    root = Tk()
    root.withdraw()
    root.update()
    try:
        selected = filedialog.askdirectory(
            title="Select folder containing audio or video files",
            mustexist=True,
        )
    finally:
        root.destroy()

    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def pick_files() -> list[Path]:
    root = Tk()
    root.withdraw()
    root.update()
    try:
        selected = filedialog.askopenfilenames(
            title="Select audio or video file(s)",
            filetypes=_media_filetypes(),
        )
    finally:
        root.destroy()

    return [Path(path).expanduser().resolve() for path in selected]
