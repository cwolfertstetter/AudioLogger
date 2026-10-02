"""Surviving a microphone dropout mid-recording.

On 2026-09-23 the Thunderbolt controller cycled its power state (RTD3) and
briefly dropped every device behind it. stream.read() raised "Unanticipated host
error", cleaning up the vanished stream raised again, and the mic thread died --
the recording ran on for four more hours without the user's voice.

The capture now reopens the default microphone and pads the gap with silence so
the mic track stays on the same timeline as the system track.
"""
import threading
import wave
from pathlib import Path

import numpy as np

from audiologger.mic_capture import MicDropout, record_microphone

SR = 10  # samples per one-second chunk


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def chunk(value: int) -> bytes:
    return np.full(SR, value, dtype=np.int16).tobytes()


class Stream:
    """Plays a script: bytes are read() results (one second each), exceptions are raised."""

    def __init__(self, script, clock: Clock, stop: threading.Event, broken_cleanup: bool):
        self._script = list(script)
        self._clock = clock
        self._stop = stop
        self._broken_cleanup = broken_cleanup

    def read(self, nframes, exception_on_overflow=True):
        if not self._script:
            self._stop.set()
            return b""
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        self._clock.now += 1.0
        return item

    def stop_stream(self):
        if self._broken_cleanup:
            raise OSError("[Errno -9999] Unanticipated host error")

    def close(self):
        if self._broken_cleanup:
            raise OSError("[Errno -9999] Unanticipated host error")


class PortAudio:
    """One PortAudio initialisation; `name=None` means no default input device."""

    def __init__(self, name, script, clock, stop, broken_cleanup=False):
        self._name, self._script = name, script
        self._clock, self._stop = clock, stop
        self._broken_cleanup = broken_cleanup

    def get_host_api_info_by_type(self, api_type):
        return {"defaultInputDevice": 0 if self._name else -1}

    def get_device_info_by_index(self, index):
        return {"index": 0, "name": self._name, "maxInputChannels": 1, "defaultSampleRate": SR}

    def open(self, **kw):
        return Stream(self._script, self._clock, self._stop, self._broken_cleanup)

    def terminate(self):
        pass


def factory(clock: Clock, stop: threading.Event, *inits, stop_after=None):
    """Hands out one PortAudio per call. Every call after the first is a
    reconnect attempt and costs one second. `stop_after` ends the recording on
    that attempt, for a device that never comes back."""
    inits = list(inits)
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        if calls["n"] > 1:
            clock.now += 1.0
        if stop_after is not None and calls["n"] - 1 >= stop_after:
            stop.set()
        name, script, broken = inits.pop(0) if inits else (None, [], False)
        return PortAudio(name, script, clock, stop, broken)

    return make


def record(tmp_path, make, clock, stop):
    return record_microphone(
        tmp_path / "mic.wav", SR, stop,
        pa_factory=make, clock=clock, retry_delay_s=0, settle_s=0,
    )


def samples(tmp_path) -> list[int]:
    with wave.open(str(tmp_path / "mic.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).tolist()


def test_a_dropout_is_bridged_with_silence_and_recording_carries_on(tmp_path):
    clock, stop = Clock(), threading.Event()
    lost = OSError("[Errno -9999] Unanticipated host error")
    make = factory(
        clock, stop,
        ("KLIM", [chunk(1), lost], False),  # first second fine, then the device vanishes
        (None, [], False),                   # attempt 1: nothing there yet
        (None, [], False),                   # attempt 2: still nothing
        ("KLIM", [chunk(2)], False),         # attempt 3: back
    )

    result = record(tmp_path, make, clock, stop)

    assert samples(tmp_path) == [1] * SR + [0] * (3 * SR) + [2] * SR
    assert result.dropouts == (MicDropout(at_s=1.0, gap_s=3.0, resumed_on="KLIM"),)
    assert result.frames == 2 * SR, "padding is not counted as captured audio"


def test_recording_continues_on_whatever_microphone_is_default_afterwards(tmp_path):
    clock, stop = Clock(), threading.Event()
    make = factory(
        clock, stop,
        ("KLIM", [chunk(1), OSError("gone")], False),
        ("Microphone Array (Realtek(R) Audio)", [chunk(3)], False),
    )

    result = record(tmp_path, make, clock, stop)

    assert result.dropouts[0].resumed_on == "Microphone Array (Realtek(R) Audio)"
    assert samples(tmp_path)[-SR:] == [3] * SR


def test_a_microphone_that_never_returns_is_padded_until_the_recording_stops(tmp_path):
    clock, stop = Clock(), threading.Event()
    make = factory(
        clock, stop,
        ("KLIM", [chunk(1), OSError("gone")], False),
        stop_after=4,
    )

    result = record(tmp_path, make, clock, stop)

    assert samples(tmp_path) == [1] * SR + [0] * (4 * SR)
    assert result.dropouts == (MicDropout(at_s=1.0, gap_s=4.0, resumed_on=None),)


def test_errors_while_releasing_the_vanished_stream_do_not_end_the_recording(tmp_path):
    """On 2026-09-23 it was the cleanup that raised the second, fatal error."""
    clock, stop = Clock(), threading.Event()
    make = factory(
        clock, stop,
        ("KLIM", [chunk(1), OSError("gone")], True),
        ("KLIM", [chunk(2)], False),
    )

    result = record(tmp_path, make, clock, stop)

    assert samples(tmp_path)[-SR:] == [2] * SR
    assert len(result.dropouts) == 1


def test_an_uneventful_recording_reports_no_dropouts(tmp_path):
    clock, stop = Clock(), threading.Event()
    make = factory(clock, stop, ("KLIM", [chunk(1), chunk(1)], False))

    result = record(tmp_path, make, clock, stop)

    assert result.dropouts == ()
    assert samples(tmp_path) == [1] * (2 * SR)
