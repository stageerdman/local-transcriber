"""Python↔JS bridge for the pywebview front-end.

Methods on this object are exposed to the web UI as `window.pywebview.api.<name>()`
(each returns a value or a Promise on the JS side). The UI is pure presentation;
all real work - probing tracks, detecting speakers, transcribing, history, settings
- happens here, reusing the existing `src/` engine.

Phase 1 is the shell only: this holds the window handle and a couple of smoke
methods. Later phases add add_files / probe / transcribe / history / settings and
push progress to the UI via `self._emit(...)`.
"""
from __future__ import annotations

import json
from typing import Any


class Bridge:
    def __init__(self) -> None:
        self._window = None

    def set_window(self, window) -> None:
        self._window = window

    # -- UI → Python ---------------------------------------------------------
    def ping(self) -> str:
        """Round-trip check the web UI can call to confirm the bridge is live."""
        return "ok"

    # -- Python → UI ---------------------------------------------------------
    def _emit(self, event: str, payload: Any = None) -> None:
        """Push an event to the web UI: calls window.appEvent(event, payload)."""
        if self._window is None:
            return
        data = json.dumps(payload)
        self._window.evaluate_js(f"window.appEvent && window.appEvent({event!r}, {data})")
