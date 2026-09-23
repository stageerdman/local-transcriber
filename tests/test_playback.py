import threading
import time
from pathlib import Path

import src.playback as playback
from src.playback import TrackPlayer


class FakeProcess:
    def __init__(self, cmd):
        self.cmd = cmd
        self._done = threading.Event()
        self.terminated = False

    def wait(self):
        self._done.wait(timeout=5)
        return 0

    def poll(self):
        return 0 if self._done.is_set() else None

    def terminate(self):
        self.terminated = True
        self._done.set()

    def finish(self):
        self._done.set()


def _patch_popen(monkeypatch):
    created = []

    def fake_popen(cmd, **kwargs):
        proc = FakeProcess(cmd)
        created.append(proc)
        return proc

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)
    return created


def _wait_until(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not predicate():
        time.sleep(0.01)


def test_play_builds_ffplay_command_and_marks_playing(monkeypatch):
    created = _patch_popen(monkeypatch)
    player = TrackPlayer()
    player.play(Path("a.mov"), audio_index=1, channel_index=0, start_seconds=12.5)
    assert len(created) == 1
    cmd = created[0].cmd
    assert cmd[0] == "ffplay"
    assert "-ast" in cmd and "a:1" in cmd            # audio-relative stream select
    assert "-ss" in cmd and "12.500" in cmd          # seek offset
    assert "pan=mono|c0=c0" in cmd                    # channel isolation
    assert player.is_playing()


def test_natural_finish_fires_on_finish_once(monkeypatch):
    created = _patch_popen(monkeypatch)
    calls = []
    player = TrackPlayer()
    player.play(Path("a.mov"), on_finish=lambda: calls.append(1))

    created[0].finish()
    _wait_until(lambda: not player.is_playing())

    assert calls == [1]
    assert player.position() is None


def test_stop_terminates_and_suppresses_on_finish(monkeypatch):
    created = _patch_popen(monkeypatch)
    calls = []
    player = TrackPlayer()
    player.play(Path("a.mov"), on_finish=lambda: calls.append(1))

    player.stop()

    assert created[0].terminated
    assert not player.is_playing()
    time.sleep(0.05)
    assert calls == []


def test_seek_supersedes_without_firing_previous_callback(monkeypatch):
    created = _patch_popen(monkeypatch)
    first, second = [], []
    player = TrackPlayer()
    player.play(Path("a.mov"), start_seconds=0.0, on_finish=lambda: first.append(1))
    player.play(Path("a.mov"), start_seconds=30.0, on_finish=lambda: second.append(1))

    assert created[0].terminated
    assert len(created) == 2
    assert "30.000" in created[1].cmd

    created[1].finish()
    _wait_until(lambda: not player.is_playing())

    assert first == []
    assert second == [1]


def test_position_tracks_from_seek_point(monkeypatch):
    _patch_popen(monkeypatch)
    player = TrackPlayer()
    player.play(Path("a.mov"), start_seconds=10.0)
    # Position starts at the seek offset and advances by wall clock.
    assert player.position() is not None and player.position() >= 10.0
    time.sleep(0.2)
    assert player.position() >= 10.2
    player.stop()
    assert player.position() is None
