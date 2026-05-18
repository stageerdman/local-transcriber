from __future__ import annotations

from pathlib import Path
from tkinter import Tk, filedialog


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
