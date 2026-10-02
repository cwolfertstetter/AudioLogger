"""Mixing per-device loopback parts into system.wav.

Each output device is recorded into its own part file, created only once that
device first makes a sound.  The part's start position (in samples) is part of
its file name, so the parts can be laid back onto one timeline -- also after a
crash, when nothing else survives.
"""
import wave
from pathlib import Path

import numpy as np

from audiologger.audio_mix import loopback_part_name, mix_loopback_parts

SR = 10  # tiny rate keeps the files small; the arithmetic is the same


def write_part(session: Path, index: int, offset: int, samples) -> Path:
    p = session / loopback_part_name(index, offset)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return p


def read(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def test_lays_each_part_down_at_its_own_offset(tmp_path):
    write_part(tmp_path, 0, 0, [1, 1, 1])
    write_part(tmp_path, 1, 5, [7, 7])

    out = mix_loopback_parts(tmp_path)

    assert out == tmp_path / "system.wav"
    assert read(out).tolist() == [1, 1, 1, 0, 0, 7, 7]


def test_overlapping_parts_are_summed_and_clipped(tmp_path):
    write_part(tmp_path, 0, 0, [30000, 100, -30000])
    write_part(tmp_path, 1, 0, [30000, 100, -30000])

    out = mix_loopback_parts(tmp_path)

    assert read(out).tolist() == [32767, 200, -32768]


def test_the_parts_are_removed_once_system_wav_is_written(tmp_path):
    a = write_part(tmp_path, 0, 0, [1])
    b = write_part(tmp_path, 1, 3, [2])

    mix_loopback_parts(tmp_path)

    assert not a.exists() and not b.exists()


def test_without_any_part_nothing_is_written(tmp_path):
    assert mix_loopback_parts(tmp_path) is None
    assert not (tmp_path / "system.wav").exists()


def test_unrelated_files_are_left_alone(tmp_path):
    (tmp_path / "mic.wav").write_bytes(b"not touched")
    write_part(tmp_path, 0, 0, [5])

    mix_loopback_parts(tmp_path)

    assert (tmp_path / "mic.wav").read_bytes() == b"not touched"


def test_block_boundaries_do_not_change_the_result(tmp_path):
    """Mixing streams in blocks so a four-hour call never sits in memory whole."""
    write_part(tmp_path, 0, 0, list(range(1, 12)))  # 1..11
    write_part(tmp_path, 1, 6, [100] * 4)            # lands on positions 6..9

    out = mix_loopback_parts(tmp_path, block_frames=3)

    assert read(out).tolist() == [1, 2, 3, 4, 5, 6, 107, 108, 109, 110, 11]
