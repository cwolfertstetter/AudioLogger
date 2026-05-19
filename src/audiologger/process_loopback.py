"""Per-app loopback via Windows ActivateAudioInterfaceAsync.

Implements process-loopback capture for one or more processes selected by
executable name. Requires Windows 10 21H2 (Build 19044) or newer.
"""
import logging
import threading
import wave
from pathlib import Path

import psutil  # for pid lookup by name


log = logging.getLogger(__name__)


class ProcessLoopbackNotAvailable(Exception):
    pass


# --- COM/WinRT constants -------------------------------------------------
PROCESS_LOOPBACK_MODE_INCLUDE = 0
PROCESS_LOOPBACK_MODE_EXCLUDE = 1

try:
    import pyaudiowpatch as pyaudio  # noqa: F401
    _HAS_PYAUDIOWPATCH = True
except Exception:  # pragma: no cover
    _HAS_PYAUDIOWPATCH = False


def _resolve_pids(app_names: list[str]) -> list[int]:
    """Map exe basenames (case-insensitive) to current PIDs."""
    wanted = {n.lower() for n in app_names}
    pids: list[int] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name in wanted:
                pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def record_app_loopback(
    out_path: Path,
    app_names: list[str],
    sample_rate: int,
    stop_event: threading.Event,
) -> None:
    """Record per-app loopback to `out_path` (16-bit mono PCM)."""
    if not _HAS_PYAUDIOWPATCH:
        raise ProcessLoopbackNotAvailable(
            "pyaudiowpatch is not installed — install with `uv pip install pyaudiowpatch`"
        )

    pids = _resolve_pids(app_names)
    if not pids:
        raise ProcessLoopbackNotAvailable(
            f"None of the requested apps are running: {app_names}"
        )
    if len(pids) > 1:
        raise ProcessLoopbackNotAvailable(
            f"Per-app loopback currently supports only one process at a time; "
            f"found {len(pids)} matching PIDs for {app_names}. "
            f"Reduce filtered_app_names to a single app or use audio_source='all'."
        )

    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        device_info = None
        for info in pa.get_loopback_device_info_generator():
            device_info = info
            break
        if device_info is None:
            raise ProcessLoopbackNotAvailable("No loopback device available")

        stream_kwargs: dict = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            frames_per_buffer=sample_rate,  # 1 s
            input=True,
            input_device_index=device_info["index"],
        )
        if hasattr(pyaudio, "PaWasapiStreamInfo"):
            stream_kwargs["input_host_api_specific_stream_info"] = pyaudio.PaWasapiStreamInfo(
                flags=pyaudio.paWinWasapiProcessLoopback,
                process_id=pids[0],
                process_loopback_mode=PROCESS_LOOPBACK_MODE_INCLUDE,
            )

        stream = pa.open(**stream_kwargs)
    except ProcessLoopbackNotAvailable:
        pa.terminate()
        raise
    except Exception as e:
        pa.terminate()
        raise ProcessLoopbackNotAvailable(f"Failed to open process-loopback stream: {e}") from e

    try:
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            while not stop_event.is_set():
                try:
                    data = stream.read(sample_rate, exception_on_overflow=False)
                    wav.writeframes(data)
                except Exception:
                    log.exception("Process-loopback read failed")
                    break
    finally:
        try:
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()
