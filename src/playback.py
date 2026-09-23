from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable


class TrackPlayer:
    """Plays (and seeks within) one audio track at a time via `ffplay`.

    Unlike `afplay`, ffplay can seek (`-ss`) and select a specific audio
    stream/channel straight from the source container, so there's no need to
    extract a track to a temp file first - playback starts effectively
    instantly and seeking is just relaunching at a new offset. At most one
    track is ever audible: starting a new play (or a seek) stops the previous
    one. `on_finish` fires only on natural end, never on an explicit stop or a
    supersede - the caller that asked to stop already knows.

    `position()` is tracked by wall clock from the seek point rather than
    parsed from ffplay: ffplay's status timestamp is absolute for some
    containers and relative for others, so parsing it isn't reliable. Wall
    clock can lead the real audio by ffplay's short startup latency, which is
    negligible for a preview scrubber.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        # Bumped on every play()/stop() so a watcher whose process was
        # superseded knows it's stale and skips clearing state / on_finish.
        self._token = 0
        self._start_seconds = 0.0
        self._launch_monotonic = 0.0

    @staticmethod
    def available() -> bool:
        return shutil.which("ffplay") is not None

    def play(
        self,
        source: Path,
        *,
        audio_index: int = 0,
        channel_index: int | None = None,
        start_seconds: float = 0.0,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        """Play `source`'s `audio_index`-th audio stream from `start_seconds`.

        `audio_index` is the position among the file's audio streams (0-based),
        i.e. ffplay's `-ast a:N`. `channel_index`, if given, isolates a single
        channel of that stream (e.g. right-only) via a pan downmix.
        """
        self.stop()
        start_seconds = max(0.0, start_seconds)
        command = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(source),
            "-ast",
            f"a:{audio_index}",
        ]
        if channel_index is not None:
            command += ["-af", f"pan=mono|c0=c{channel_index}"]

        with self._lock:
            self._token += 1
            token = self._token
            self._start_seconds = start_seconds
            self._launch_monotonic = time.monotonic()
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._process = process

        threading.Thread(target=self._watch, args=(process, token, on_finish), daemon=True).start()

    def _watch(
        self, process: subprocess.Popen, token: int, on_finish: Callable[[], None] | None
    ) -> None:
        process.wait()
        with self._lock:
            if self._token != token:
                return  # superseded by a newer play()/stop()
            self._process = None
        if on_finish is not None:
            on_finish()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._token += 1  # invalidate the watcher so its on_finish won't fire
            self._process = None
        if process is not None and process.poll() is None:
            process.terminate()

    def is_playing(self) -> bool:
        with self._lock:
            return self._process is not None

    def position(self) -> float | None:
        """Current playback position in seconds, or None if nothing is playing."""
        with self._lock:
            if self._process is None:
                return None
            return self._start_seconds + (time.monotonic() - self._launch_monotonic)
