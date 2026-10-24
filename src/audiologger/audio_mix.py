"""Mix two 16-bit PCM mono WAV files into one, padding the shorter one."""
import logging
import re
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# One part file per output device, created only once that device first makes a
# sound.  The start position in samples is part of the name so the parts can be
# laid back onto one timeline -- also by crash recovery, when nothing else of
# the capture survives.
_LOOPBACK_PART = re.compile(r"^_loopback_(\d+)_at(\d+)\.wav$")


def loopback_part_name(index: int, offset_samples: int) -> str:
    return f"_loopback_{index:02d}_at{offset_samples:012d}.wav"


def mix_loopback_parts(
    session_dir: Path,
    out_name: str = "system.wav",
    *,
    block_frames: int = 48000 * 10,
) -> Path | None:
    """Sum the loopback parts in `session_dir` into one file, then delete them.

    Each part is placed at the offset in its name.  Works block by block so a
    multi-hour call with several active devices never sits in memory whole.
    Returns the written path, or None when there were no parts to mix.
    """
    parts = []
    for p in session_dir.iterdir():
        m = _LOOPBACK_PART.match(p.name)
        if m:
            parts.append((int(m.group(2)), p))
    if not parts:
        return None

    readers = []
    try:
        rate = None
        for offset, p in sorted(parts):
            r = wave.open(str(p), "rb")
            readers.append((offset, r.getnframes(), r))
            if r.getsampwidth() != 2 or r.getnchannels() != 1:
                raise ValueError(f"{p.name}: expected 16-bit mono")
            if rate is None:
                rate = r.getframerate()
            elif r.getframerate() != rate:
                raise ValueError(f"{p.name}: sample rate {r.getframerate()} != {rate}")

        total = max(offset + n for offset, n, _ in readers)
        out_path = session_dir / out_name
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            pos = 0
            while pos < total:
                n = min(block_frames, total - pos)
                acc = np.zeros(n, dtype=np.int32)
                for offset, nframes, r in readers:
                    lo, hi = max(pos, offset), min(pos + n, offset + nframes)
                    if hi <= lo:
                        continue
                    r.setpos(lo - offset)
                    chunk = np.frombuffer(r.readframes(hi - lo), dtype=np.int16)
                    acc[lo - pos: lo - pos + len(chunk)] += chunk
                out.writeframes(np.clip(acc, -32768, 32767).astype(np.int16).tobytes())
                pos += n
    finally:
        for _, _, r in readers:
            r.close()

    for _, p in parts:
        try:
            p.unlink()
        except OSError:
            # Still held open by a capture thread that outlived stop()'s wait;
            # system.wav is complete, so a leftover part costs nothing.
            log.warning("Could not delete %s", p.name, exc_info=True)
    return out_path


def read_int16(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono WAV file.  Returns (samples, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit WAV")
        if w.getnchannels() != 1:
            raise ValueError("expected mono WAV")
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    return np.frombuffer(frames, dtype=np.int16), sr


# Keep private aliases so any callers (including mix_to_file itself) still work.
_read_int16 = read_int16


def write_int16(path: Path, samples: np.ndarray, sr: int) -> None:
    """Write samples as a 16-bit mono WAV file."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())


# Keep private alias.
_write_int16 = write_int16


def append_wav(target: Path, addition: Path, out_path: Path | None = None) -> None:
    """Concatenate *addition* onto *target* (both must be 16-bit mono).

    Writes the result to *out_path*, or back over *target* when omitted.
    Raises ValueError if sample rates differ.
    """
    target_samples, sr_target = read_int16(target)
    addition_samples, sr_addition = read_int16(addition)
    if sr_target != sr_addition:
        raise ValueError(
            f"sample rates differ: target={sr_target}, addition={sr_addition}"
        )
    concatenated = np.concatenate([target_samples, addition_samples])
    write_int16(out_path if out_path is not None else target, concatenated, sr_target)


def _open_track(path: Path) -> wave.Wave_read | None:
    """Open a 16-bit mono WAV for mixing; None if it is missing or empty.

    A crash in the first second of a recording leaves a 0-byte file: `wave`
    writes the header with the first frames.  That track simply has no audio.
    """
    if not path.exists():
        return None
    try:
        r = wave.open(str(path), "rb")
    except EOFError:
        return None
    if r.getsampwidth() != 2 or r.getnchannels() != 1:
        r.close()
        raise ValueError(f"{path.name}: expected 16-bit mono")
    return r


def mix_to_file(
    mic_path: Path,
    system_path: Path,
    out_path: Path,
    *,
    block_frames: int = 48000 * 10,
) -> None:
    """Sum-mix mic + system to `out_path`.

    - If both files have audio: pad shorter with zeros, sum, clip to int16.
    - If one has: copy it to out_path.
    - If neither has: FileNotFoundError.

    Works block by block: a four-hour call would otherwise need over 10 GB.
    """
    readers = []
    try:
        for p in (mic_path, system_path):
            r = _open_track(p)
            if r is not None:
                readers.append(r)
        if not readers:
            raise FileNotFoundError(f"Neither {mic_path} nor {system_path} has audio")
        rate = readers[0].getframerate()
        for r in readers[1:]:
            if r.getframerate() != rate:
                raise ValueError(f"sample rates differ: mic={rate}, sys={r.getframerate()}")

        total = max(r.getnframes() for r in readers)
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            pos = 0
            while pos < total:
                n = min(block_frames, total - pos)
                acc = np.zeros(n, dtype=np.int32)
                for r in readers:
                    chunk = np.frombuffer(r.readframes(n), dtype=np.int16)
                    acc[: len(chunk)] += chunk
                out.writeframes(np.clip(acc, -32768, 32767).astype(np.int16).tobytes())
                pos += n
    finally:
        for r in readers:
            r.close()
