from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

# ffplay's -nodisp status line, e.g. "  39.83 M-A:  0.000 fd=...": the leading
# float is the master clock (absolute stream time, in seconds). Lines are
# carriage-return-updated several times a second.
_CLOCK_RE = re.compile(rb"(\d+\.\d+)\s+[AM]-[AV]:")


class TrackPlayer:
    """Plays (and seeks within) one audio track at a time via `ffplay`.

    Unlike `afplay`, ffplay can seek (`-ss`) and select a specific audio
    stream/channel straight from the source container, so there's no need to
    extract a track to a temp file first - playback starts effectively
    instantly and seeking is just relaunching at a new offset. At most one
    track is ever audible: starting a new play (or a seek) stops the previous
    one. `on_finish` fires only on natural end, never on an explicit stop or a
    supersede - the caller that asked to stop already knows.

    `position()` reports ffplay's own master clock, parsed from its `-nodisp`
    status line (the leading float is absolute stream time). That tracks the
    *audible* position, so a read-along highlight stays in sync instead of
    leading the sound by ffplay's startup latency - which a wall-clock estimate
    does. Until the first status line arrives (a brief startup gap) it falls
    back to wall clock from the seek point so the scrubber still moves.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        # Bumped on every play()/stop() so a watcher whose process was
        # superseded knows it's stale and skips clearing state / on_finish.
        self._token = 0
        self._start_seconds = 0.0
        self._launch_monotonic = 0.0
        # Last position ffplay reported for the current play, or None until its
        # first status line lands (then position() uses the wall-clock fallback).
        self._reported_seconds: float | None = None

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
            # `error` (not `quiet`) keeps ffplay's status line, whose leading
            # float position() parses; `-stats` forces it on even when piped.
            "-loglevel",
            "error",
            "-stats",
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
            self._reported_seconds = None
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._process = process

        threading.Thread(target=self._watch, args=(process, token, on_finish), daemon=True).start()

    def _watch(
        self, process: subprocess.Popen, token: int, on_finish: Callable[[], None] | None
    ) -> None:
        # Follow ffplay's status line (carriage-return-updated on stderr) and
        # keep the latest reported clock. Reading stderr to EOF also serves as
        # the process wait - EOF means ffplay has exited.
        buf = b""
        stream = process.stderr
        if stream is not None:
            while True:
                chunk = stream.read(256)
                if not chunk:
                    break
                buf = (buf + chunk)[-512:]  # bounded: a status line is short
                matches = _CLOCK_RE.findall(buf)
                if matches:
                    try:
                        seconds = float(matches[-1])
                    except ValueError:
                        seconds = None
                    if seconds is not None:
                        with self._lock:
                            if self._token == token:
                                self._reported_seconds = seconds
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
        """Current playback position in seconds, or None if nothing is playing.

        Prefers ffplay's reported clock (the audible position); falls back to
        wall clock from the seek point for the brief gap before ffplay's first
        status line arrives.
        """
        with self._lock:
            if self._process is None:
                return None
            if self._reported_seconds is not None:
                return self._reported_seconds
            return self._start_seconds + (time.monotonic() - self._launch_monotonic)
