"""Native-window entry point for the reimagined Local Transcriber UI.

Renders the HTML/CSS/JS front-end (app/web/) in a real WebKit window via
pywebview, backed by the existing Python transcription engine through `Bridge`.
This replaces the Tkinter UI (app/main.py); the engine in `src/` is unchanged.

Run:  .venv/bin/python -m app.webview_main
"""
from __future__ import annotations

from pathlib import Path

import webview

from app.bridge import Bridge

WEB_DIR = Path(__file__).resolve().parent / "web"


def main() -> int:
    bridge = Bridge()
    window = webview.create_window(
        "Local Transcriber",
        url=str(WEB_DIR / "index.html"),
        js_api=bridge,
        width=1040,
        height=720,
        min_size=(780, 540),
    )
    bridge.set_window(window)
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
