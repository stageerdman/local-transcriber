from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

from src.media_finder import SUPPORTED_EXTENSIONS


def _media_filetypes() -> list[tuple[str, str]]:
    patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
    return [("Media files", patterns), ("All files", "*.*")]


def pick_folder() -> Path | None:
    # Uses the app's existing Tk root (tkinter's filedialog falls back to the
    # default root); creating a second Tk() instance alongside a running
    # mainloop is unsupported and causes crashes/hangs on macOS.
    selected = filedialog.askdirectory(
        title="Select folder containing audio or video files",
        mustexist=True,
    )
    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def pick_files() -> list[Path]:
    selected = filedialog.askopenfilenames(
        title="Select audio or video file(s)",
        filetypes=_media_filetypes(),
    )
    return [Path(path).expanduser().resolve() for path in selected]
