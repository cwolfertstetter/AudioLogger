import wave
from pathlib import Path

import numpy as np
import pytest

from audiologger.audio_mix import mix_to_file, append_wav


def write_wav(path: Path, samples: np.ndarray, sr: int = 48000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    return np.frombuffer(frames, dtype=np.int16), sr


def test_mix_equal_length(tmp_path):
    mic = tmp_path / "mic.wav"
    sys = tmp_path / "system.wav"
    out = tmp_path / "mixed.wav"
    write_wav(mic, np.full(48000, 1000, dtype=np.int16))
    write_wav(sys, np.full(48000, 2000, dtype=np.int16))
    mix_to_file(mic, sys, out)
    data, sr = read_wav(out)
    assert sr == 48000
    assert len(data) == 48000
    # Mix is sum, clipped to int16 range. 1000 + 2000 = 3000.
    assert np.all(data == 3000)


def test_mix_different_lengths_pads_shorter(tmp_path):
    mic = tmp_path / "mic.wav"
    sys = tmp_path / "system.wav"
    out = tmp_path / "mixed.wav"
    write_wav(mic, np.full(48000, 1000, dtype=np.int16))      # 1s
    write_wav(sys, np.full(96000, 2000, dtype=np.int16))      # 2s
    mix_to_file(mic, sys, out)
    data, _ = read_wav(out)
    assert len(data) == 96000
    # First second: 1000+2000=3000. Second second: 0+2000=2000.
    assert data[0] == 3000
    assert data[-1] == 2000


def test_mix_clips_to_int16_range(tmp_path):
    mic = tmp_path / "mic.wav"
    sys = tmp_path / "system.wav"
    out = tmp_path / "mixed.wav"
    write_wav(mic, np.full(100, 30000, dtype=np.int16))
    write_wav(sys, np.full(100, 30000, dtype=np.int16))
    mix_to_file(mic, sys, out)
    data, _ = read_wav(out)
    assert data.max() == 32767  # clipped, no wraparound


def test_mix_missing_input_falls_back_to_present(tmp_path):
    """If one file is missing, mixed = the other file (copy)."""
    mic = tmp_path / "mic.wav"  # not created
    sys = tmp_path / "system.wav"
    out = tmp_path / "mixed.wav"
    write_wav(sys, np.full(48000, 1500, dtype=np.int16))
    mix_to_file(mic, sys, out)
    data, _ = read_wav(out)
    assert np.all(data == 1500)


def test_mix_both_missing_raises(tmp_path):
    out = tmp_path / "mixed.wav"
    with pytest.raises(FileNotFoundError):
        mix_to_file(tmp_path / "mic.wav", tmp_path / "sys.wav", out)


# --- append_wav -----------------------------------------------------------

def test_append_wav_concatenates_samples(tmp_path):
    target = tmp_path / "target.wav"
    addition = tmp_path / "addition.wav"
    write_wav(target, np.full(48000, 100, dtype=np.int16))     # 1 second
    write_wav(addition, np.full(24000, 200, dtype=np.int16))   # 0.5 second
    append_wav(target, addition)
    data, sr = read_wav(target)
    assert sr == 48000
    assert len(data) == 72000
    assert np.all(data[:48000] == 100)
    assert np.all(data[48000:] == 200)


def test_mix_treats_an_empty_file_as_missing(tmp_path):
    """A crash in the first second leaves mic.wav without even a header."""
    (tmp_path / "mic.wav").touch()
    write_wav(tmp_path / "sys.wav", np.array([5, 6, 7], dtype=np.int16))
    out = tmp_path / "mixed.wav"

    mix_to_file(tmp_path / "mic.wav", tmp_path / "sys.wav", out)

    assert read_wav(out)[0].tolist() == [5, 6, 7]


def test_mix_works_in_blocks_across_unequal_lengths(tmp_path):
    mic = np.arange(25, dtype=np.int16)
    sys_ = np.full(12, 100, dtype=np.int16)
    write_wav(tmp_path / "mic.wav", mic)
    write_wav(tmp_path / "sys.wav", sys_)
    out = tmp_path / "mixed.wav"

    mix_to_file(tmp_path / "mic.wav", tmp_path / "sys.wav", out, block_frames=5)

    expected = mic.astype(np.int32)
    expected[:12] += 100
    assert read_wav(out)[0].tolist() == expected.tolist()
