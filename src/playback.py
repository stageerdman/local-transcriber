from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable


class TrackPlayer:
    """Plays one audio file at a time via macOS `afplay`.

    Owns a single subprocess: starting a new playback stops whatever was
    already playing, so at most one track is ever audible. Playback runs
    detached; when it finishes on its own the optional `on_finish` callback
    fires from a watcher thread - callers driving a UI toolkit must marshal
    that back onto their own main thread. An explicit `stop()` (or being
    superseded by a new `play()`) does NOT fire `on_finish`: the caller that
    asked to stop already knows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        # Identifies the current playback. Bumped on every play()/stop() so a
        # watcher thread whose process was superseded can tell it's stale and
        # skip both clearing state and firing on_finish.
        self._token = 0

    @staticmethod
    def available() -> bool:
        return shutil.which("afplay") is not None

    def play(self, path: Path, on_finish: Callable[[], None] | None = None) -> None:
        self.stop()
        with self._lock:
            self._token += 1
            token = self._token
            process = subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._process = process

        def watch() -> None:
            process.wait()
            with self._lock:
                if self._token != token:
                    return  # superseded by a newer play()/stop()
                self._process = None
            if on_finish is not None:
                on_finish()

        threading.Thread(target=watch, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._token += 1  # invalidate any watcher so its on_finish won't fire
            self._process = None
        if process is not None and process.poll() is None:
            process.terminate()

    def is_playing(self) -> bool:
        with self._lock:
            return self._process is not None
