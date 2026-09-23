import threading
import time
from pathlib import Path

import src.playback as playback
from src.playback import TrackPlayer


class FakeProcess:
    """Stand-in for subprocess.Popen whose exit is driven by the test."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self.terminated = False

    def wait(self) -> int:
        self._done.wait(timeout=5)
        return 0

    def poll(self) -> int | None:
        return 0 if self._done.is_set() else None

    def terminate(self) -> None:
        self.terminated = True
        self._done.set()

    def finish(self) -> None:
        """Simulate the audio playing to its natural end."""
        self._done.set()


def _patch_popen(monkeypatch) -> list[FakeProcess]:
    created: list[FakeProcess] = []

    def fake_popen(cmd, **kwargs):
        proc = FakeProcess()
        created.append(proc)
        return proc

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)
    return created


def test_play_starts_a_process(monkeypatch):
    created = _patch_popen(monkeypatch)
    player = TrackPlayer()
    player.play(Path("a.mp3"))
    assert len(created) == 1
    assert player.is_playing()


def test_natural_finish_fires_on_finish_once(monkeypatch):
    created = _patch_popen(monkeypatch)
    calls = []
    player = TrackPlayer()
    player.play(Path("a.mp3"), on_finish=lambda: calls.append(1))

    created[0].finish()
    _wait_until(lambda: not player.is_playing())

    assert calls == [1]
    assert not player.is_playing()


def test_stop_terminates_and_suppresses_on_finish(monkeypatch):
    created = _patch_popen(monkeypatch)
    calls = []
    player = TrackPlayer()
    player.play(Path("a.mp3"), on_finish=lambda: calls.append(1))

    player.stop()

    assert created[0].terminated
    assert not player.is_playing()
    # An explicit stop must not fire the natural-finish callback - give the
    # watcher thread a moment to run and confirm it stays silent.
    time.sleep(0.05)
    assert calls == []


def test_second_play_supersedes_first_without_firing_its_callback(monkeypatch):
    created = _patch_popen(monkeypatch)
    first_calls = []
    second_calls = []
    player = TrackPlayer()
    player.play(Path("a.mp3"), on_finish=lambda: first_calls.append(1))
    player.play(Path("b.mp3"), on_finish=lambda: second_calls.append(1))

    # Starting the second playback should have stopped the first.
    assert created[0].terminated
    assert len(created) == 2

    created[1].finish()
    _wait_until(lambda: not player.is_playing())

    assert first_calls == []  # superseded playback never fires
    assert second_calls == [1]


def _wait_until(predicate, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end and not predicate():
        time.sleep(0.01)
