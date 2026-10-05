"""Default entry point for Local Transcriber.

Launches the reimagined native UI (pywebview WebKit window over the Python
engine). The previous Tkinter UI is still available at `python -m app.main_tk`
until the new app is fully retired.
"""
from __future__ import annotations

from app.webview_main import main

if __name__ == "__main__":
    raise SystemExit(main())
