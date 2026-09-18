"""Microphone capture via pyaudiowpatch (PortAudio/WASAPI).

soundcard 0.4.6 hard-asserts that a capture device reports its WASAPI mix
format as WAVE_FORMAT_EXTENSIBLE (0xFFFE).  Devices that report a plain
WAVE_FORMAT_IEEE_FLOAT header instead -- many USB headsets do -- raise
AssertionError inside soundcard, which killed the mic thread before a single
sample was written and left a 44-byte (header-only) mic.wav behind.

PortAudio negotiates both formats, so mic capture goes through pyaudiowpatch --
the same library process_loopback.py already uses.

It will not do the stereo->mono downmix, though: asking it for channels=1 on a
2-channel WASAPI device hands back the interleaved stereo frames as a mono
buffer, stretching one second of audio into two and dropping everything an
octave.  So the stream is opened at the device's own channel count and mixed
down here.
"""
import logging
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


log = logging.getLogger(__name__)

CHUNK_SECONDS = 1


class MicrophoneNotAvailable(Exception):
    pass


@dataclass(frozen=True)
class MicCaptureResult:
    """Outcome of one capture run.  `frames` is 0 when the mic stayed silent."""
    device_name: str
    frames: int


try:
    import pyaudiowpatch as pyaudio  # noqa: F401
    _HAS_PYAUDIOWPATCH = True
except Exception:  # pragma: no cover
    _HAS_PYAUDIOWPATCH = False


def _to_mono(data: bytes, channels: int) -> bytes:
    """Average interleaved int16 channels down to one."""
    if channels <= 1:
        return data
    samples = np.frombuffer(data, dtype=np.int16)
    usable = len(samples) - (len(samples) % channels)
    if usable <= 0:
        return b""
    frames = samples[:usable].reshape(-1, channels)
    return frames.mean(axis=1).round().astype(np.int16).tobytes()


def _default_input_device(pa: Any, pyaudio_mod: Any) -> dict:
    """Resolve the Windows default recording device via the WASAPI host API."""
    try:
        api = pa.get_host_api_info_by_type(pyaudio_mod.paWASAPI)
    except Exception as e:
        raise MicrophoneNotAvailable(f"WASAPI host API not available: {e}") from e

    index = api.get("defaultInputDevice", -1)
    if index is None or index < 0:
        raise MicrophoneNotAvailable(
            "No default recording device — check Windows Sound settings."
        )
    try:
        info = pa.get_device_info_by_index(index)
    except Exception as e:
        raise MicrophoneNotAvailable(f"Default input device {index} unreadable: {e}") from e

    if info.get("maxInputChannels", 0) < 1:
        raise MicrophoneNotAvailable(
            f"Default device {info.get('name')!r} has no input channels"
        )
    return info


def record_microphone(
    out_path: Path,
    sample_rate: int,
    stop_event: threading.Event,
    *,
    pa_factory: Callable[[], Any] | None = None,
) -> MicCaptureResult:
    """Record the default microphone to `out_path` (16-bit mono PCM).

    PortAudio is asked for mono int16 at `sample_rate` directly; WASAPI shared
    mode refuses rates the device does not run at, so `sample_rate` must match
    the device (48000 for every current endpoint).
    """
    if pa_factory is None:
        if not _HAS_PYAUDIOWPATCH:
            raise MicrophoneNotAvailable(
                "pyaudiowpatch is not installed — install with `uv pip install pyaudiowpatch`"
            )
        pa_factory = pyaudio.PyAudio

    import pyaudiowpatch as pyaudio_mod  # constants; cheap, already imported

    pa = pa_factory()
    try:
        device = _default_input_device(pa, pyaudio_mod)
        # Open at the device's own channel count and mix down ourselves.  Asking
        # PortAudio for mono on a 2-channel WASAPI device hands back the
        # interleaved stereo frames as if they were a mono buffer, so one second
        # of audio is written as two and everything plays an octave too low.
        # Measured through one device with a 777 Hz tone: channels=1 recorded it
        # at 377.9 Hz, channels=2 at 776.4 Hz.
        channels = max(1, int(device.get("maxInputChannels", 1) or 1))
        stream = pa.open(
            format=pyaudio_mod.paInt16,
            channels=channels,
            rate=sample_rate,
            frames_per_buffer=sample_rate * CHUNK_SECONDS,
            input=True,
            input_device_index=device["index"],
        )
    except MicrophoneNotAvailable:
        pa.terminate()
        raise
    except Exception as e:
        pa.terminate()
        raise MicrophoneNotAvailable(f"Failed to open microphone stream: {e}") from e

    frames = 0
    try:
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            while not stop_event.is_set():
                try:
                    data = stream.read(
                        sample_rate * CHUNK_SECONDS, exception_on_overflow=False
                    )
                except Exception:
                    log.exception("Microphone read failed")
                    break
                mono = _to_mono(data, channels)
                wav.writeframes(mono)
                frames += len(mono) // 2
    finally:
        try:
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()

    return MicCaptureResult(str(device.get("name") or "unknown"), frames)
