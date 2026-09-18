"""Mic capture via pyaudiowpatch (PortAudio/WASAPI).

Regression context: soundcard 0.4.6 asserts that a device reports its WASAPI
mix format as WAVE_FORMAT_EXTENSIBLE.  Headsets that report plain
WAVE_FORMAT_IEEE_FLOAT (e.g. KLIM Mantis 7.1) raise AssertionError, which
killed the mic thread before a single sample was written and left a 44-byte
mic.wav behind.
"""
import threading
import wave
from pathlib import Path

import pytest

from audiologger.mic_capture import MicrophoneNotAvailable, record_microphone


WASAPI_DEFAULT_INDEX = 31


class FakeStream:
    """Yields `chunks` one read() at a time, then silence."""

    def __init__(self, chunks: list[bytes], stop_event: threading.Event):
        self._chunks = list(chunks)
        self._stop = stop_event
        self.closed = False
        self.stopped = False

    def read(self, nframes: int, exception_on_overflow: bool = True) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        self._stop.set()  # nothing left: end the capture loop
        return b"\x00\x00" * nframes

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakePyAudio:
    def __init__(self, chunks=None, stop_event=None, default_input=WASAPI_DEFAULT_INDEX,
                 name="Microphone (KLIM Mantis Audio 7.1)", channels=2, rate=48000.0):
        self._chunks = chunks or []
        self._stop = stop_event
        self._default_input = default_input
        self._name = name
        self._channels = channels
        self._rate = rate
        self.open_kwargs: dict | None = None
        self.terminated = False
        self.stream: FakeStream | None = None

    def get_host_api_info_by_type(self, api_type):
        return {"name": "Windows WASAPI", "defaultInputDevice": self._default_input}

    def get_device_info_by_index(self, index):
        if index != self._default_input:
            raise OSError(f"unexpected device index {index}")
        return {
            "index": index,
            "name": self._name,
            "maxInputChannels": self._channels,
            "defaultSampleRate": self._rate,
        }

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        self.stream = FakeStream(self._chunks, self._stop)
        return self.stream

    def terminate(self):
        self.terminated = True


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sampwidth": w.getsampwidth(),
            "rate": w.getframerate(),
            "frames": w.getnframes(),
            "data": w.readframes(w.getnframes()),
        }


def test_writes_captured_audio_as_mono_16bit_wav(tmp_path):
    stop = threading.Event()
    payload = b"\x11\x22" * 48000
    pa = FakePyAudio(chunks=[payload], stop_event=stop, channels=1)

    record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    got = read_wav(tmp_path / "mic.wav")
    assert got["channels"] == 1
    assert got["sampwidth"] == 2
    assert got["rate"] == 48000
    assert got["data"].startswith(payload)


def test_records_from_the_wasapi_default_input_device(tmp_path):
    stop = threading.Event()
    pa = FakePyAudio(chunks=[b"\x00\x00" * 10], stop_event=stop)

    record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    assert pa.open_kwargs["input_device_index"] == WASAPI_DEFAULT_INDEX
    assert pa.open_kwargs["input"] is True
    assert pa.open_kwargs["rate"] == 48000
    assert pa.open_kwargs["channels"] == 2, (
        "must open at the device's own channel count -- asking PortAudio for mono "
        "on a stereo WASAPI device returns the interleaved stereo frames as if "
        "they were mono, which stretches the audio 2x and drops it an octave"
    )


def test_returns_device_name_and_frame_count(tmp_path):
    stop = threading.Event()
    pa = FakePyAudio(chunks=[b"\x01\x02" * 4800], stop_event=stop)

    result = record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    assert result.device_name == "Microphone (KLIM Mantis Audio 7.1)"
    assert result.frames >= 4800


def test_raises_when_no_default_input_device(tmp_path):
    stop = threading.Event()
    pa = FakePyAudio(default_input=-1, stop_event=stop)

    with pytest.raises(MicrophoneNotAvailable):
        record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)


def test_stops_and_releases_the_stream_when_stop_event_is_set(tmp_path):
    stop = threading.Event()
    stop.set()
    pa = FakePyAudio(chunks=[b"\x00\x00" * 10], stop_event=stop)

    record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    assert pa.stream.stopped and pa.stream.closed
    assert pa.terminated


def test_terminates_portaudio_even_when_opening_fails(tmp_path):
    stop = threading.Event()
    pa = FakePyAudio(stop_event=stop)
    pa.open = lambda **kw: (_ for _ in ()).throw(OSError("Invalid sample rate"))

    with pytest.raises(MicrophoneNotAvailable):
        record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    assert pa.terminated


# --- stereo devices -----------------------------------------------------------
# Asking PortAudio for channels=1 on a 2-channel WASAPI device hands back the
# interleaved stereo frames as a mono buffer.  Measured with a 777 Hz tone
# through the same device: channels=1 recorded it at 377.9 Hz, channels=2 at
# 776.4 Hz.  Every recording made this way plays an octave too low.

def test_downmixes_the_device_channels_to_mono(tmp_path):
    import numpy as np

    stop = threading.Event()
    # one stereo frame per column: left 1000, right 3000 -> mono 2000
    frame = np.array([1000, 3000], dtype=np.int16)
    chunk = np.tile(frame, 480).tobytes()
    pa = FakePyAudio(chunks=[chunk], stop_event=stop, channels=2)

    record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    got = read_wav(tmp_path / "mic.wav")
    samples = np.frombuffer(got["data"], dtype=np.int16)
    assert got["channels"] == 1
    assert samples[0] == 2000, "left and right must be averaged, not interleaved"
    assert len(samples) == got["frames"]


def test_a_mono_device_is_recorded_as_is(tmp_path):
    import numpy as np

    stop = threading.Event()
    chunk = np.full(960, 1234, dtype=np.int16).tobytes()
    pa = FakePyAudio(chunks=[chunk], stop_event=stop, channels=1)

    record_microphone(tmp_path / "mic.wav", 48000, stop, pa_factory=lambda: pa)

    assert pa.open_kwargs["channels"] == 1
    samples = np.frombuffer(read_wav(tmp_path / "mic.wav")["data"], dtype=np.int16)
    assert samples[0] == 1234
