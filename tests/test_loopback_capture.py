"""Recording the loopback of every output device, not just the default one.

On 2026-09-28 a Slack call played on a headset while the default device was the
laptop speakers. The single loopback of the default device recorded 20 minutes
of silence and the other side of the call was lost. Calls pick their own
device, so every output device is recorded and the results are mixed.
"""
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audiologger import audio_capture
from audiologger.audio_capture import AudioCaptureThread, LoopbackPartWriter
from audiologger.mic_capture import MicCaptureResult

SR = 10  # samples per one-second chunk; tiny keeps everything fast


def read(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def parts(session: Path) -> list[Path]:
    return sorted(session.glob("_loopback_*.wav"))


# --- LoopbackPartWriter --------------------------------------------------------

def test_a_device_that_stays_silent_never_writes_a_file(tmp_path):
    w = LoopbackPartWriter(tmp_path, index=0, sample_rate=SR)
    for _ in range(5):
        w.write(np.zeros(SR, dtype=np.int16))
    w.close()

    assert parts(tmp_path) == []
    assert w.path is None


def test_the_part_starts_at_the_first_sound_and_says_where(tmp_path):
    w = LoopbackPartWriter(tmp_path, index=3, sample_rate=SR)
    w.write(np.zeros(SR, dtype=np.int16))
    w.write(np.zeros(SR, dtype=np.int16))
    w.write(np.full(SR, 9, dtype=np.int16))
    w.close()

    assert w.path is not None
    assert w.path.name == "_loopback_03_at000000000020.wav"
    assert read(w.path).tolist() == [9] * SR


def test_silence_after_the_first_sound_is_kept_to_hold_the_timeline(tmp_path):
    w = LoopbackPartWriter(tmp_path, index=0, sample_rate=SR)
    w.write(np.full(SR, 4, dtype=np.int16))
    w.write(np.zeros(SR, dtype=np.int16))
    w.write(np.full(SR, 4, dtype=np.int16))
    w.close()

    assert read(w.path).tolist() == [4] * SR + [0] * SR + [4] * SR


# --- AudioCaptureThread in "all" mode ------------------------------------------

class FakeRecorder:
    """Plays back `chunks` one record() at a time, then idles in silence."""

    def __init__(self, chunks, drained: threading.Event):
        self._chunks = list(chunks)
        self._drained = drained

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        if self._chunks:
            return self._chunks.pop(0)
        self._drained.set()
        time.sleep(0.005)
        return np.zeros((numframes, 1), dtype=np.float32)


def chunk(value: float) -> np.ndarray:
    return np.full((SR, 1), value, dtype=np.float32)


@pytest.fixture
def fake_devices(monkeypatch):
    """Install a fake soundcard. Returns a dict to describe devices with."""
    devices: dict[str, dict] = {}

    def all_speakers():
        return [SimpleNamespace(name=name, id=f"id-{name}") for name in devices]

    def get_microphone(id, include_loopback=False):
        name = id.removeprefix("id-")
        spec = devices[name]
        if spec.get("broken"):
            raise AssertionError("mix format not supported")
        return SimpleNamespace(
            recorder=lambda samplerate, channels: FakeRecorder(spec["chunks"], spec["drained"])
        )

    monkeypatch.setattr(audio_capture, "sc", SimpleNamespace(
        all_speakers=all_speakers, get_microphone=get_microphone,
    ))

    def silent_mic(out_path, sample_rate, stop_event, **kw):
        stop_event.wait()
        return MicCaptureResult("Fake Mic", 1)

    monkeypatch.setattr(audio_capture, "record_microphone", silent_mic)

    def add(name, chunks=(), broken=False):
        devices[name] = {"chunks": list(chunks), "drained": threading.Event(), "broken": broken}

    add.devices = devices
    return add


def run_capture(session: Path, fake_devices) -> AudioCaptureThread:
    cap = AudioCaptureThread(session, SR, audio_source="all")
    cap.start()
    for spec in fake_devices.devices.values():
        if not spec["broken"]:
            assert spec["drained"].wait(5), "fake device never finished its chunks"
    cap.stop()
    return cap


def test_the_call_is_captured_even_when_it_plays_on_a_non_default_device(tmp_path, fake_devices):
    fake_devices("Speakers (Realtek(R) Audio)", [chunk(0.0)] * 3)
    fake_devices("Speakers (G432 Gaming Headset)", [chunk(0.0), chunk(0.5), chunk(0.5)])

    cap = run_capture(tmp_path, fake_devices)

    system = read(tmp_path / "system.wav")
    assert not system[:SR].any(), "nothing played during the first second"
    assert system[SR:3 * SR].min() > 16000, "the headset's audio sits from second 1 on"
    assert parts(tmp_path) == [], "parts are merged away"
    assert cap.system_device_name == "Speakers (G432 Gaming Headset)"


def test_audio_on_several_devices_is_mixed_together(tmp_path, fake_devices):
    fake_devices("Speakers (KLIM Mantis Audio 7.1)", [chunk(0.25)])
    fake_devices("Speakers (Realtek(R) Audio)", [chunk(0.25)])

    run_capture(tmp_path, fake_devices)

    system = read(tmp_path / "system.wav")
    assert abs(int(system[0]) - 16383) <= 2, "two quarters add up to one half"


def test_a_recording_with_no_system_sound_at_all_says_so(tmp_path, fake_devices):
    fake_devices("Speakers (Realtek(R) Audio)", [chunk(0.0)] * 2)

    cap = run_capture(tmp_path, fake_devices)

    assert not (tmp_path / "system.wav").exists()
    assert any("no system audio" in w.lower() for w in cap.warnings), cap.warnings


def test_one_device_that_cannot_be_opened_does_not_stop_the_others(tmp_path, fake_devices):
    fake_devices("Odd Device", broken=True)
    fake_devices("Speakers (KLIM Mantis Audio 7.1)", [chunk(0.5)])

    cap = run_capture(tmp_path, fake_devices)

    assert read(tmp_path / "system.wav")[0] > 16000
    assert cap.system_device_name == "Speakers (KLIM Mantis Audio 7.1)"


# --- crash recovery ------------------------------------------------------------

def test_recovery_after_a_crash_merges_the_parts_left_behind(tmp_path):
    """A freeze mid-call leaves parts but no system.wav; recovery must merge them."""
    from audiologger.audio_mix import loopback_part_name
    from audiologger.config import Config
    from audiologger.paths import MARKER_FILENAME
    from audiologger.tray_app import TrayApp

    recs = tmp_path / "recs"
    session = recs / "2026-09-28_10-32-26"
    session.mkdir(parents=True)
    (session / MARKER_FILENAME).touch()
    for name, value in (("mic.wav", 1), (loopback_part_name(4, 2), 300)):
        with wave.open(str(session / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.full(3, value, dtype=np.int16).tobytes())

    app = TrayApp.__new__(TrayApp)
    app.cfg = Config(output_dir=recs)
    enqueued = []
    app.queue = SimpleNamespace(enqueue=enqueued.append)

    app._handle_orphaned_sessions()

    assert read(session / "system.wav").tolist() == [0, 0, 300, 300, 300]
    assert parts(session) == []
    assert enqueued == [session]
