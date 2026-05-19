"""Mix two 16-bit PCM mono WAV files into one, padding the shorter one."""
import wave
from pathlib import Path

import numpy as np


def _read_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit"
        assert w.getnchannels() == 1, "expected mono"
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    return np.frombuffer(frames, dtype=np.int16), sr


def _write_int16(path: Path, samples: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())


def mix_to_file(mic_path: Path, system_path: Path, out_path: Path) -> None:
    """Sum-mix mic + system to `out_path`.

    - If both files exist: pad shorter with zeros, sum, clip to int16.
    - If one file exists: copy it to out_path.
    - If neither exists: FileNotFoundError.
    """
    mic_exists = mic_path.exists()
    sys_exists = system_path.exists()
    if not mic_exists and not sys_exists:
        raise FileNotFoundError(f"Neither {mic_path} nor {system_path} exists")

    if not mic_exists:
        data, sr = _read_int16(system_path)
        _write_int16(out_path, data, sr)
        return
    if not sys_exists:
        data, sr = _read_int16(mic_path)
        _write_int16(out_path, data, sr)
        return

    mic, sr_mic = _read_int16(mic_path)
    sys_, sr_sys = _read_int16(system_path)
    assert sr_mic == sr_sys, "sample rates must match"

    n = max(len(mic), len(sys_))
    mic_padded = np.zeros(n, dtype=np.int32)
    sys_padded = np.zeros(n, dtype=np.int32)
    mic_padded[: len(mic)] = mic
    sys_padded[: len(sys_)] = sys_
    summed = mic_padded + sys_padded
    clipped = np.clip(summed, -32768, 32767).astype(np.int16)
    _write_int16(out_path, clipped, sr_mic)
