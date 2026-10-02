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

A microphone can also vanish mid-recording: on 2026-09-23 the Thunderbolt
controller cycled its power state and briefly dropped every device behind it,
and the mic thread gave up for the remaining four hours of a call.  A dropout
now reopens the default microphone and pads the gap with silence, so the file
stays on the wall-clock timeline the system track follows.
"""
import logging
import threading
import time
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
class MicDropout:
    """The microphone vanished mid-recording; the gap was filled with silence."""
    at_s: float              # seconds into the recording
    gap_s: float             # how long it was gone
    resumed_on: str | None   # device recording continued on; None if it never came back


@dataclass(frozen=True)
class MicCaptureResult:
    """Outcome of one capture run.

    `frames` counts what the microphone actually delivered -- 0 when it stayed
    silent -- not the silence padded into dropouts.
    """
    device_name: str
    frames: int
    dropouts: tuple[MicDropout, ...] = ()


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


def _quietly(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception:
        log.debug("Ignoring error releasing a stream whose device is gone", exc_info=True)


def _release(pa: Any, stream: Any) -> None:
    """Close a stream and its PortAudio.  After a dropout the device is already
    gone and these calls raise too -- on 2026-09-23 it was exactly such a
    cleanup error that killed the mic thread for good."""
    if stream is not None:
        _quietly(stream.stop_stream)
        _quietly(stream.close)
    if pa is not None:
        _quietly(pa.terminate)


def _open_default_stream(pa_factory: Callable[[], Any], pyaudio_mod: Any, sample_rate: int):
    """Initialise PortAudio and open the default microphone.

    A fresh PortAudio every time: it reads the device list only when it
    initialises, so after a dropout an old instance would still point at the
    vanished device.  Returns (pa, stream, device_info, channels).
    """
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
        _quietly(pa.terminate)
        raise
    except Exception as e:
        _quietly(pa.terminate)
        raise MicrophoneNotAvailable(f"Failed to open microphone stream: {e}") from e
    return pa, stream, device, channels


def _reconnect(
    pa_factory: Callable[[], Any],
    pyaudio_mod: Any,
    sample_rate: int,
    stop_event: threading.Event,
    *,
    settle_s: float,
    retry_delay_s: float,
):
    """Reopen the default microphone, retrying until it is back or recording stops.

    Waits `settle_s` first, so Windows can re-add a device that only blinked off
    and make it the default again before we grab whatever stands in for it.
    Returns what _open_default_stream returns, or None if recording ended first.
    """
    if stop_event.wait(settle_s):
        return None
    attempts = 0
    while not stop_event.is_set():
        attempts += 1
        try:
            opened = _open_default_stream(pa_factory, pyaudio_mod, sample_rate)
        except MicrophoneNotAvailable as e:
            if attempts == 1 or attempts % 30 == 0:
                log.info("Microphone still unavailable (attempt %d): %s", attempts, e)
            stop_event.wait(retry_delay_s)
            continue
        log.info("Microphone back after %d attempt(s): %r", attempts, opened[2].get("name"))
        return opened
    return None


def record_microphone(
    out_path: Path,
    sample_rate: int,
    stop_event: threading.Event,
    *,
    pa_factory: Callable[[], Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
    settle_s: float = 2.0,
    retry_delay_s: float = 1.0,
) -> MicCaptureResult:
    """Record the default microphone to `out_path` (16-bit mono PCM).

    `sample_rate` must be one the device runs at -- WASAPI shared mode refuses
    others (48000 for every current endpoint).  Raises MicrophoneNotAvailable if
    no microphone can be opened at the start.  A dropout later on is survived:
    the default microphone is reopened and the gap padded with silence.
    """
    if pa_factory is None:
        if not _HAS_PYAUDIOWPATCH:
            raise MicrophoneNotAvailable(
                "pyaudiowpatch is not installed — install with `uv pip install pyaudiowpatch`"
            )
        pa_factory = pyaudio.PyAudio

    import pyaudiowpatch as pyaudio_mod  # constants; cheap, already imported

    pa, stream, device, channels = _open_default_stream(pa_factory, pyaudio_mod, sample_rate)
    first_name = current_name = str(device.get("name") or "unknown")
    start = clock()
    written = 0    # samples in the file, padding included
    captured = 0   # samples the microphone actually delivered
    dropouts: list[MicDropout] = []

    try:
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)

            def pad_to_now() -> None:
                nonlocal written
                missing = int(round((clock() - start) * sample_rate)) - written
                if missing > 0:
                    wav.writeframes(bytes(2 * missing))
                    written += missing

            while not stop_event.is_set():
                try:
                    data = stream.read(
                        sample_rate * CHUNK_SECONDS, exception_on_overflow=False
                    )
                except Exception as e:
                    lost_at = clock()
                    log.warning(
                        "Microphone %r dropped out %.0f s in (%s) -- reconnecting",
                        current_name, lost_at - start, e,
                    )
                    _release(pa, stream)
                    pa = stream = None
                    opened = _reconnect(
                        pa_factory, pyaudio_mod, sample_rate, stop_event,
                        settle_s=settle_s, retry_delay_s=retry_delay_s,
                    )
                    if opened is not None:
                        pa, stream, device, channels = opened
                        current_name = str(device.get("name") or "unknown")
                    pad_to_now()
                    dropouts.append(MicDropout(
                        at_s=lost_at - start,
                        gap_s=clock() - lost_at,
                        resumed_on=current_name if opened is not None else None,
                    ))
                    if opened is None:
                        break
                    continue
                mono = _to_mono(data, channels)
                wav.writeframes(mono)
                written += len(mono) // 2
                captured += len(mono) // 2
    finally:
        _release(pa, stream)

    return MicCaptureResult(first_name, captured, tuple(dropouts))
