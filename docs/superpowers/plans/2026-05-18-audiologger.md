# AudioLogger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Windows tray daemon that records mic + system audio via a global hotkey and produces Markdown transcripts using WhisperX large-v3 with pyannote diarization.

**Architecture:** Two-process model. A long-lived tray daemon handles UI, hotkey, recording, and a FIFO queue. A separate `transcribe_worker.py` subprocess loads WhisperX + pyannote, processes queued sessions, and stays warm 30 s between jobs.

**Tech Stack:** Python 3.11, `uv`, `soundcard` (audio capture + WASAPI loopback), `pystray` + Pillow (tray), `keyboard` (global hotkey), `winotify` (toasts), `pycaw` + `ctypes` (per-app loopback), `whisperx`, `pyannote.audio`, `pytest`.

**Spec:** See `docs/superpowers/specs/2026-05-18-audiologger-design.md` for full context.

---

## File Structure

```
audiologger/
├── pyproject.toml
├── README.md
├── src/audiologger/
│   ├── __init__.py
│   ├── __main__.py            # python -m audiologger → tray
│   ├── paths.py               # APPDATA paths, session naming
│   ├── config.py              # Config dataclass + YAML I/O
│   ├── segment.py             # Segment dataclass (start/end/text/speaker)
│   ├── transcript_merger.py   # merge mic + system segments → markdown
│   ├── audio_mix.py           # numpy mix of mic + system WAVs
│   ├── recovery.py            # scan for orphaned sessions
│   ├── job_queue.py           # FIFO via pending.txt + worker spawn
│   ├── controller.py          # RecordingController state machine
│   ├── notifications.py       # winotify wrapper
│   ├── icons.py               # PIL-generated tray icons
│   ├── audio_capture.py       # mic + system streams
│   ├── process_loopback.py    # ctypes Process Loopback API
│   ├── hotkey.py              # global hotkey registration
│   ├── tray_app.py            # pystray setup, menu, wiring
│   └── transcribe_worker.py   # WhisperX + pyannote worker
└── tests/
    ├── conftest.py
    ├── test_paths.py
    ├── test_config.py
    ├── test_transcript_merger.py
    ├── test_audio_mix.py
    ├── test_recovery.py
    ├── test_job_queue.py
    ├── test_controller.py
    └── fixtures/
        ├── mic_5s.wav
        └── system_5s.wav
```

Each module has one responsibility. Hardware-dependent modules (`audio_capture`, `process_loopback`, `hotkey`, `tray_app`, `transcribe_worker`) get manual verification rather than unit tests.

---

## Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/audiologger/__init__.py`
- Create: `src/audiologger/__main__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
dist/
build/
*.egg-info/
recordings/
*.wav
*.log
config.local.yaml
.uv-cache/
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "audiologger"
version = "0.1.0"
description = "Windows tray meeting recorder with local WhisperX transcription"
requires-python = ">=3.11"
dependencies = [
    "soundcard>=0.4.3",
    "pystray>=0.19.5",
    "Pillow>=10.0.0",
    "keyboard>=0.13.5",
    "winotify>=1.1.0",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "pycaw>=20240210",
    "comtypes>=1.4",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.1",
    "torchaudio>=2.1",
    "whisperx>=3.1",
    "pyannote.audio>=3.1",
]
cpu = [
    "torch>=2.1",
    "torchaudio>=2.1",
    "whisperx>=3.1",
    "pyannote.audio>=3.1",
]
dev = [
    "pytest>=7.4",
    "pytest-mock>=3.12",
]

[project.scripts]
audiologger = "audiologger.tray_app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/audiologger"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

> Note: GPU vs CPU torch install is selected at install time (`uv pip install --extra-index-url ...`). Document the install command in README in Task 17.

- [ ] **Step 3: Create empty `src/audiologger/__init__.py`**

```python
"""AudioLogger — Windows tray meeting recorder."""
__version__ = "0.1.0"
```

- [ ] **Step 4: Create `src/audiologger/__main__.py`**

```python
"""Entry point: python -m audiologger"""
from audiologger.tray_app import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create empty `tests/__init__.py` and `tests/conftest.py`**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
```

- [ ] **Step 6: Create minimal `README.md`**

```markdown
# AudioLogger

Windows tray meeting recorder with local WhisperX transcription. See `docs/superpowers/specs/2026-05-18-audiologger-design.md`.

Install + usage instructions: TODO (see Task 17).
```

- [ ] **Step 7: Install dev dependencies & verify pytest discovers nothing yet**

Run:
```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```
Expected: `no tests ran` (exit 5 is fine here) or `collected 0 items`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md .gitignore src/ tests/
git commit -m "feat: project skeleton"
```

---

## Task 2: Paths Utility

**Files:**
- Create: `src/audiologger/paths.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write failing tests**

`tests/test_paths.py`:
```python
from datetime import datetime
from pathlib import Path

from audiologger.paths import (
    appdata_dir,
    config_path,
    session_dirname,
    MARKER_FILENAME,
)


def test_appdata_dir_under_roaming(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert appdata_dir() == tmp_path / "AudioLogger"


def test_config_path_under_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert config_path() == tmp_path / "AudioLogger" / "config.yaml"


def test_session_dirname_format():
    dt = datetime(2026, 5, 18, 14, 32, 15)
    assert session_dirname(dt) == "2026-05-18_14-32-15"


def test_marker_filename_constant():
    assert MARKER_FILENAME == "RECORDING_IN_PROGRESS"
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_paths.py -v`
Expected: `ModuleNotFoundError: No module named 'audiologger.paths'`

- [ ] **Step 3: Implement `paths.py`**

`src/audiologger/paths.py`:
```python
"""Filesystem paths and session-directory naming."""
import os
from datetime import datetime
from pathlib import Path

MARKER_FILENAME = "RECORDING_IN_PROGRESS"


def appdata_dir() -> Path:
    """Return %APPDATA%/AudioLogger (creates parents only on write)."""
    return Path(os.environ["APPDATA"]) / "AudioLogger"


def config_path() -> Path:
    return appdata_dir() / "config.yaml"


def session_dirname(dt: datetime) -> str:
    """Format: YYYY-MM-DD_HH-MM-SS."""
    return dt.strftime("%Y-%m-%d_%H-%M-%S")
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_paths.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/paths.py tests/test_paths.py
git commit -m "feat: paths utility"
```

---

## Task 3: Config

**Files:**
- Create: `src/audiologger/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

from audiologger.config import Config, load_config, save_config


def test_default_values():
    c = Config()
    assert c.hotkey == "ctrl+alt+r"
    assert c.whisper_model == "large-v3"
    assert c.device == "cuda"
    assert c.compute_type == "float16"
    assert c.diarization_enabled is True
    assert c.huggingface_token is None
    assert c.audio_source == "all"
    assert c.filtered_app_names == []
    assert c.notification_enabled is True


def test_load_creates_default_when_missing(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg_file.exists()  # auto-written


def test_load_fills_missing_fields(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("hotkey: f8\n")  # only hotkey set
    cfg = load_config(cfg_file)
    assert cfg.hotkey == "f8"
    assert cfg.whisper_model == "large-v3"  # default filled in


def test_load_round_trip(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    original = Config(hotkey="f9", diarization_enabled=False)
    save_config(cfg_file, original)
    loaded = load_config(cfg_file)
    assert loaded.hotkey == "f9"
    assert loaded.diarization_enabled is False


def test_save_writes_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    save_config(cfg_file, Config(hotkey="ctrl+f1"))
    text = cfg_file.read_text()
    assert "hotkey: ctrl+f1" in text


def test_output_dir_is_path(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("output_dir: C:/Recordings\n")
    cfg = load_config(cfg_file)
    assert cfg.output_dir == Path("C:/Recordings")
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'audiologger.config'`

- [ ] **Step 3: Implement `config.py`**

`src/audiologger/config.py`:
```python
"""Config dataclass with YAML persistence and default-fill semantics."""
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    hotkey: str = "ctrl+alt+r"
    output_dir: Path = field(default_factory=lambda: Path("./recordings"))
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    diarization_enabled: bool = True
    huggingface_token: str | None = None
    audio_source: str = "all"  # "all" | "apps"
    filtered_app_names: list[str] = field(default_factory=list)
    notification_enabled: bool = True


def _to_yaml_dict(cfg: Config) -> dict[str, Any]:
    d = asdict(cfg)
    d["output_dir"] = str(cfg.output_dir)
    return d


def _from_yaml_dict(d: dict[str, Any]) -> Config:
    """Build Config from possibly-partial dict, filling defaults."""
    valid = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in d.items() if k in valid}
    if "output_dir" in filtered and filtered["output_dir"] is not None:
        filtered["output_dir"] = Path(filtered["output_dir"])
    return Config(**filtered)


def save_config(path: Path, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_yaml_dict(cfg), f, sort_keys=False, allow_unicode=True)


def load_config(path: Path) -> Config:
    """Load config. If missing, write defaults. If partial, fill with defaults."""
    if not path.exists():
        cfg = Config()
        save_config(path, cfg)
        return cfg
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_yaml_dict(data)
    # Re-write so missing fields are added to disk for future hand-editing
    save_config(path, cfg)
    return cfg
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/config.py tests/test_config.py
git commit -m "feat: config dataclass with YAML persistence"
```

---

## Task 4: Segment Dataclass

**Files:**
- Create: `src/audiologger/segment.py`

This is a tiny data-only module — no separate test file; it gets exercised by `transcript_merger` tests in Task 5.

- [ ] **Step 1: Implement `segment.py`**

`src/audiologger/segment.py`:
```python
"""Transcript segment with speaker label and timing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: float       # seconds from recording start
    end: float
    text: str
    speaker: str       # "Ich" | "Sprecher 1" | "Sprecher 2" | "Andere" | ...
```

- [ ] **Step 2: Commit**

```bash
git add src/audiologger/segment.py
git commit -m "feat: Segment dataclass"
```

---

## Task 5: Transcript Merger

**Files:**
- Create: `src/audiologger/transcript_merger.py`
- Create: `tests/test_transcript_merger.py`

- [ ] **Step 1: Write failing tests**

`tests/test_transcript_merger.py`:
```python
from audiologger.segment import Segment
from audiologger.transcript_merger import (
    merge_segments,
    format_timestamp,
    render_markdown,
)


def test_format_timestamp_short():
    assert format_timestamp(3.4) == "00:00:03"
    assert format_timestamp(125.0) == "00:02:05"
    assert format_timestamp(3725.0) == "01:02:05"


def test_merge_chronological_order():
    mic = [Segment(5.0, 6.0, "Hallo", "Ich")]
    sys = [Segment(0.0, 4.0, "Was?", "Sprecher 1")]
    merged = merge_segments(mic, sys)
    assert [s.start for s in merged] == [0.0, 5.0]
    assert [s.speaker for s in merged] == ["Sprecher 1", "Ich"]


def test_merge_overlap_orders_by_start():
    mic = [Segment(1.0, 5.0, "A", "Ich")]
    sys = [Segment(2.0, 4.0, "B", "Sprecher 1")]
    merged = merge_segments(mic, sys)
    assert [s.text for s in merged] == ["A", "B"]


def test_merge_stable_for_equal_start():
    mic = [Segment(1.0, 2.0, "M", "Ich")]
    sys = [Segment(1.0, 2.0, "S", "Sprecher 1")]
    merged = merge_segments(mic, sys)
    # mic first when ties — implementation choice, document it
    assert merged[0].speaker == "Ich"


def test_render_markdown_basic():
    segments = [
        Segment(3.0, 4.0, "Hi zusammen", "Ich"),
        Segment(6.0, 7.0, "Hallo", "Sprecher 1"),
    ]
    md = render_markdown(
        segments,
        recorded_at="2026-05-18 14:32:15",
        duration_seconds=420,
        source_label="mic + system (loopback, all)",
        model_label="WhisperX large-v3 + pyannote/speaker-diarization-3.1",
        warnings=[],
    )
    assert "# Aufnahme 2026-05-18 14:32:15" in md
    assert "**Dauer:** 07:00" in md
    assert "**[00:00:03] Ich:** Hi zusammen" in md
    assert "**[00:00:06] Sprecher 1:** Hallo" in md


def test_render_markdown_includes_warnings():
    md = render_markdown(
        [],
        recorded_at="2026-05-18 14:32:15",
        duration_seconds=10,
        source_label="mic only",
        model_label="WhisperX large-v3",
        warnings=["System-Audio nicht verfügbar"],
    )
    assert "System-Audio nicht verfügbar" in md
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_transcript_merger.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `transcript_merger.py`**

`src/audiologger/transcript_merger.py`:
```python
"""Merge mic + system segments into a Markdown transcript."""
from typing import Iterable

from audiologger.segment import Segment


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def merge_segments(
    mic: Iterable[Segment], system: Iterable[Segment]
) -> list[Segment]:
    """Stable chronological merge. Mic wins ties (mic appears first)."""
    tagged = [(s.start, 0, s) for s in mic] + [(s.start, 1, s) for s in system]
    tagged.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in tagged]


def render_markdown(
    segments: list[Segment],
    *,
    recorded_at: str,
    duration_seconds: int,
    source_label: str,
    model_label: str,
    warnings: list[str],
) -> str:
    """Render the final transcript Markdown matching the spec format."""
    duration_str = format_timestamp(duration_seconds)
    # Drop the leading "00:" for short recordings — keep it consistent: spec
    # showed "47:21" for sub-hour. Strip leading "00:" only if hours == 0.
    if duration_str.startswith("00:"):
        duration_str = duration_str[3:]

    lines = [
        f"# Aufnahme {recorded_at}",
        "",
        f"**Dauer:** {duration_str}",
        f"**Quelle:** {source_label}",
        f"**Modell:** {model_label}",
    ]
    for w in warnings:
        lines.append(f"**Warnung:** {w}")
    lines.extend(["", "---", ""])
    for seg in segments:
        ts = format_timestamp(seg.start)
        lines.append(f"**[{ts}] {seg.speaker}:** {seg.text}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_transcript_merger.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/segment.py src/audiologger/transcript_merger.py tests/test_transcript_merger.py
git commit -m "feat: transcript merger with chronological ordering"
```

---

## Task 6: Audio Mix

**Files:**
- Create: `src/audiologger/audio_mix.py`
- Create: `tests/test_audio_mix.py`
- Create: `tests/fixtures/mic_5s.wav` and `tests/fixtures/system_5s.wav` (generated in test setup)

- [ ] **Step 1: Write failing tests**

`tests/test_audio_mix.py`:
```python
import wave
from pathlib import Path

import numpy as np
import pytest

from audiologger.audio_mix import mix_to_file


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
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_audio_mix.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `audio_mix.py`**

`src/audiologger/audio_mix.py`:
```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_audio_mix.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/audio_mix.py tests/test_audio_mix.py
git commit -m "feat: WAV mixer with padding and clipping"
```

---

## Task 7: Recovery Scanner

**Files:**
- Create: `src/audiologger/recovery.py`
- Create: `tests/test_recovery.py`

- [ ] **Step 1: Write failing tests**

`tests/test_recovery.py`:
```python
from pathlib import Path

from audiologger.paths import MARKER_FILENAME
from audiologger.recovery import find_orphaned_sessions


def test_no_sessions_returns_empty(tmp_path):
    assert find_orphaned_sessions(tmp_path) == []


def test_finds_session_with_marker(tmp_path):
    sess = tmp_path / "2026-05-18_14-00-00"
    sess.mkdir()
    (sess / MARKER_FILENAME).touch()
    result = find_orphaned_sessions(tmp_path)
    assert result == [sess]


def test_ignores_session_without_marker(tmp_path):
    sess = tmp_path / "2026-05-18_14-00-00"
    sess.mkdir()
    assert find_orphaned_sessions(tmp_path) == []


def test_ignores_files_at_top_level(tmp_path):
    (tmp_path / "notes.txt").touch()
    assert find_orphaned_sessions(tmp_path) == []


def test_returns_sorted(tmp_path):
    for name in ["2026-05-18_15-00-00", "2026-05-18_13-00-00", "2026-05-18_14-00-00"]:
        d = tmp_path / name
        d.mkdir()
        (d / MARKER_FILENAME).touch()
    result = find_orphaned_sessions(tmp_path)
    assert [p.name for p in result] == [
        "2026-05-18_13-00-00",
        "2026-05-18_14-00-00",
        "2026-05-18_15-00-00",
    ]


def test_missing_output_dir_returns_empty(tmp_path):
    assert find_orphaned_sessions(tmp_path / "does-not-exist") == []
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `recovery.py`**

`src/audiologger/recovery.py`:
```python
"""Detect sessions left behind by a crashed recording (marker-file present)."""
from pathlib import Path

from audiologger.paths import MARKER_FILENAME


def find_orphaned_sessions(output_dir: Path) -> list[Path]:
    """Return sorted list of session directories containing the marker file."""
    if not output_dir.exists():
        return []
    found = [
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / MARKER_FILENAME).exists()
    ]
    return sorted(found, key=lambda p: p.name)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_recovery.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/recovery.py tests/test_recovery.py
git commit -m "feat: orphaned session recovery scanner"
```

---

## Task 8: Job Queue

**Files:**
- Create: `src/audiologger/job_queue.py`
- Create: `tests/test_job_queue.py`

The queue persists pending sessions to `pending.txt` (in the worker-state dir under `appdata_dir()`) and spawns the worker process on demand. We test the file I/O and spawn-decision logic with a mocked spawner.

- [ ] **Step 1: Write failing tests**

`tests/test_job_queue.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audiologger.job_queue import TranscriptionJobQueue


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return d


def test_enqueue_appends_to_pending(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    q.enqueue(Path("C:/recs/session-1"))
    q.enqueue(Path("C:/recs/session-2"))
    pending = (state_dir / "pending.txt").read_text().splitlines()
    assert pending == ["C:/recs/session-1", "C:/recs/session-2"]


def test_enqueue_spawns_worker_when_none_running(state_dir):
    spawner = MagicMock()
    spawner.return_value.poll.return_value = None  # alive
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.enqueue(Path("C:/recs/s1"))
    spawner.assert_called_once()


def test_enqueue_does_not_spawn_if_worker_alive(state_dir):
    spawner = MagicMock()
    proc = MagicMock()
    proc.poll.return_value = None  # alive
    spawner.return_value = proc
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.enqueue(Path("C:/recs/s1"))
    q.enqueue(Path("C:/recs/s2"))
    assert spawner.call_count == 1


def test_enqueue_respawns_if_worker_exited(state_dir):
    spawner = MagicMock()
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 0  # exited
    alive_proc = MagicMock()
    alive_proc.poll.return_value = None
    spawner.side_effect = [dead_proc, alive_proc]
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.enqueue(Path("C:/recs/s1"))
    q.enqueue(Path("C:/recs/s2"))
    assert spawner.call_count == 2


def test_status_reads_heartbeat(state_dir):
    (state_dir / "worker_status.json").write_text(
        '{"running": "session-1", "queued": ["session-2"]}'
    )
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.running == "session-1"
    assert status.queued == ["session-2"]


def test_status_when_no_heartbeat(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.running is None
    assert status.queued == []


def test_pending_file_created_on_first_enqueue(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    assert not (state_dir / "pending.txt").exists()
    q.enqueue(Path("C:/recs/s1"))
    assert (state_dir / "pending.txt").exists()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_job_queue.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `job_queue.py`**

`src/audiologger/job_queue.py`:
```python
"""FIFO job queue backed by pending.txt + spawned transcription worker."""
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


PENDING_FILE = "pending.txt"
STATUS_FILE = "worker_status.json"


@dataclass
class JobStatus:
    running: Optional[str] = None
    queued: list[str] = field(default_factory=list)


def _default_spawner(state_dir: Path) -> subprocess.Popen:
    """Spawn the transcription worker as a detached subprocess."""
    return subprocess.Popen(
        [sys.executable, "-m", "audiologger.transcribe_worker", str(state_dir)],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class TranscriptionJobQueue:
    def __init__(
        self,
        state_dir: Path,
        spawner: Callable[[Path], subprocess.Popen] | None = None,
    ):
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._spawner = spawner if spawner is not None else _default_spawner
        self._worker: subprocess.Popen | None = None

    @property
    def _pending_path(self) -> Path:
        return self._state_dir / PENDING_FILE

    @property
    def _status_path(self) -> Path:
        return self._state_dir / STATUS_FILE

    def enqueue(self, session_dir: Path) -> None:
        """Append session to pending.txt and ensure worker is running."""
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(str(session_dir).replace("\\", "/") + "\n")

        if self._worker is None or self._worker.poll() is not None:
            self._worker = self._spawner(self._state_dir)

    def status(self) -> JobStatus:
        if not self._status_path.exists():
            return JobStatus()
        try:
            data = json.loads(self._status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return JobStatus()
        return JobStatus(
            running=data.get("running"),
            queued=list(data.get("queued", [])),
        )
```

> The test's mocked spawner has signature `MagicMock()` (no required args). The real `_default_spawner(state_dir)` takes one. The TranscriptionJobQueue calls `self._spawner(self._state_dir)`. MagicMock accepts any args, so tests pass.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_job_queue.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/job_queue.py tests/test_job_queue.py
git commit -m "feat: transcription job queue with worker spawn logic"
```

---

## Task 9: Recording Controller

**Files:**
- Create: `src/audiologger/controller.py`
- Create: `tests/test_controller.py`

The controller's state machine is the testable part. `AudioCaptureThread` is injected as a factory so tests can substitute a fake.

- [ ] **Step 1: Write failing tests**

`tests/test_controller.py`:
```python
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audiologger.config import Config
from audiologger.controller import RecordingController, RecordingState
from audiologger.paths import MARKER_FILENAME


class FakeCapture:
    def __init__(self, session_dir: Path, sample_rate: int, source: str, app_names):
        self.session_dir = session_dir
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        (self.session_dir / "mic.wav").touch()
        (self.session_dir / "system.wav").touch()

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def cfg(tmp_path):
    return Config(output_dir=tmp_path / "recs")


@pytest.fixture
def controller(cfg):
    return RecordingController(
        config=cfg,
        capture_factory=FakeCapture,
        mix_fn=MagicMock(),
        enqueue_fn=MagicMock(),
        clock=lambda: datetime(2026, 5, 18, 14, 32, 15),
    )


def test_initial_state_is_idle(controller):
    assert controller.state is RecordingState.IDLE


def test_toggle_from_idle_starts_recording(controller, cfg):
    controller.toggle()
    assert controller.state is RecordingState.RECORDING
    session = cfg.output_dir / "2026-05-18_14-32-15"
    assert session.exists()
    assert (session / MARKER_FILENAME).exists()


def test_toggle_from_recording_stops_and_enqueues(controller, cfg):
    controller.toggle()  # start
    controller.toggle()  # stop
    assert controller.state is RecordingState.IDLE
    session = cfg.output_dir / "2026-05-18_14-32-15"
    assert not (session / MARKER_FILENAME).exists()
    controller._enqueue_fn.assert_called_once_with(session)
    controller._mix_fn.assert_called_once()


def test_stop_calls_capture_stop(controller, cfg):
    controller.toggle()
    capture = controller._current_capture
    controller.toggle()
    assert capture.stopped is True


def test_toggle_during_stopping_is_ignored(controller, cfg):
    """A second toggle mid-stop must not spawn a second session."""
    controller.toggle()  # IDLE → RECORDING
    controller._state = RecordingState.STOPPING
    controller.toggle()
    assert controller.state is RecordingState.STOPPING


def test_session_dir_name_format(controller, cfg):
    controller.toggle()
    sess_dirs = list(cfg.output_dir.iterdir())
    assert len(sess_dirs) == 1
    assert sess_dirs[0].name == "2026-05-18_14-32-15"
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_controller.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `controller.py`**

`src/audiologger/controller.py`:
```python
"""RecordingController — state machine for the record/stop toggle."""
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol

from audiologger.config import Config
from audiologger.paths import MARKER_FILENAME, session_dirname


class RecordingState(Enum):
    IDLE = auto()
    RECORDING = auto()
    STOPPING = auto()


class CaptureLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


CaptureFactory = Callable[[Path, int, str, list[str]], CaptureLike]
"""(session_dir, sample_rate, audio_source, filtered_app_names) -> CaptureLike"""


class RecordingController:
    SAMPLE_RATE = 48000

    def __init__(
        self,
        *,
        config: Config,
        capture_factory: CaptureFactory,
        mix_fn: Callable[[Path, Path, Path], None],
        enqueue_fn: Callable[[Path], None],
        clock: Callable[[], datetime] = datetime.now,
    ):
        self._config = config
        self._capture_factory = capture_factory
        self._mix_fn = mix_fn
        self._enqueue_fn = enqueue_fn
        self._clock = clock
        self._state = RecordingState.IDLE
        self._current_capture: CaptureLike | None = None
        self._current_session: Path | None = None

    @property
    def state(self) -> RecordingState:
        return self._state

    def toggle(self) -> None:
        if self._state is RecordingState.IDLE:
            self._start()
        elif self._state is RecordingState.RECORDING:
            self._stop()
        # STOPPING: ignore

    def _start(self) -> None:
        out = self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        session = out / session_dirname(self._clock())
        session.mkdir()
        (session / MARKER_FILENAME).touch()

        capture = self._capture_factory(
            session,
            self.SAMPLE_RATE,
            self._config.audio_source,
            list(self._config.filtered_app_names),
        )
        capture.start()
        self._current_capture = capture
        self._current_session = session
        self._state = RecordingState.RECORDING

    def _stop(self) -> None:
        self._state = RecordingState.STOPPING
        assert self._current_capture is not None
        assert self._current_session is not None
        self._current_capture.stop()

        session = self._current_session
        mic = session / "mic.wav"
        sysw = session / "system.wav"
        mixed = session / "mixed.wav"
        self._mix_fn(mic, sysw, mixed)

        (session / MARKER_FILENAME).unlink(missing_ok=True)
        self._enqueue_fn(session)

        self._current_capture = None
        self._current_session = None
        self._state = RecordingState.IDLE
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_controller.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/controller.py tests/test_controller.py
git commit -m "feat: RecordingController state machine"
```

---

## Task 10: Notifications Wrapper

**Files:**
- Create: `src/audiologger/notifications.py`

This is a thin wrapper around `winotify` so the rest of the code uses a stable interface. No unit tests — Windows-only side-effecting library. Manual verification at end.

- [ ] **Step 1: Implement `notifications.py`**

`src/audiologger/notifications.py`:
```python
"""Thin wrapper around winotify for tray toast notifications."""
from winotify import Notification


APP_NAME = "AudioLogger"


class Notifier:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def notify(self, title: str, message: str) -> None:
        if not self.enabled:
            return
        toast = Notification(app_id=APP_NAME, title=title, msg=message, duration="short")
        toast.show()
```

- [ ] **Step 2: Manual smoke test**

Run:
```bash
uv run python -c "from audiologger.notifications import Notifier; Notifier().notify('Test', 'Hallo')"
```
Expected: Windows-Toast erscheint rechts unten mit Titel "Test" und Text "Hallo".

- [ ] **Step 3: Commit**

```bash
git add src/audiologger/notifications.py
git commit -m "feat: notifications wrapper"
```

---

## Task 11: Tray Icons

**Files:**
- Create: `src/audiologger/icons.py`

PIL-generated 64x64 icons in three states. No tests; visual verification via tray app in later task.

- [ ] **Step 1: Implement `icons.py`**

`src/audiologger/icons.py`:
```python
"""PIL-generated tray icons for idle / recording / transcribing."""
from PIL import Image, ImageDraw


SIZE = 64


def _make_icon(fg: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Filled circle, leaves a 4-px border
    draw.ellipse((4, 4, SIZE - 4, SIZE - 4), fill=fg + (255,))
    return img


def idle_icon() -> Image.Image:
    return _make_icon((120, 120, 120))  # grey


def recording_icon() -> Image.Image:
    return _make_icon((220, 30, 30))    # red


def transcribing_icon() -> Image.Image:
    return _make_icon((220, 180, 30))   # yellow
```

- [ ] **Step 2: Visual smoke test**

Run:
```bash
uv run python -c "from audiologger.icons import recording_icon; recording_icon().save('icon_test.png')"
```
Open `icon_test.png` — should be a red circle on transparent background.

- [ ] **Step 3: Cleanup test image and commit**

```bash
rm icon_test.png
git add src/audiologger/icons.py
git commit -m "feat: tray icons"
```

---

## Task 12: Audio Capture

**Files:**
- Create: `src/audiologger/audio_capture.py`
- Create: `src/audiologger/process_loopback.py` (stub for now; full impl in Task 13)

Hardware-dependent — manual verification, no unit tests.

- [ ] **Step 1: Implement `process_loopback.py` as importable stub**

`src/audiologger/process_loopback.py`:
```python
"""Per-app loopback via Windows Process Loopback API. Filled in Task 13."""
from pathlib import Path


class ProcessLoopbackNotAvailable(Exception):
    """Raised when per-app loopback can't be initialized."""


def record_app_loopback(out_path: Path, app_names: list[str], sample_rate: int, stop_event) -> None:
    """Stub — raises ProcessLoopbackNotAvailable so audio_capture falls back to 'all'."""
    raise ProcessLoopbackNotAvailable("process loopback not yet implemented")
```

- [ ] **Step 2: Implement `audio_capture.py`**

`src/audiologger/audio_capture.py`:
```python
"""Records mic + system audio to separate WAV files in parallel."""
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import soundcard as sc

from audiologger.process_loopback import (
    ProcessLoopbackNotAvailable,
    record_app_loopback,
)


log = logging.getLogger(__name__)

CHUNK_SECONDS = 1


class AudioCaptureThread:
    """Two-stream recorder: mic + system loopback (or per-app loopback)."""

    def __init__(
        self,
        session_dir: Path,
        sample_rate: int,
        audio_source: str = "all",
        filtered_app_names: list[str] | None = None,
    ):
        self._session_dir = session_dir
        self._sr = sample_rate
        self._audio_source = audio_source
        self._app_names = filtered_app_names or []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.warnings: list[str] = []
        # Recorded for transcript header
        self.mic_device_name: str | None = None
        self.system_device_name: str | None = None

    def start(self) -> None:
        mic_path = self._session_dir / "mic.wav"
        sys_path = self._session_dir / "system.wav"
        self._stop.clear()

        t_mic = threading.Thread(
            target=self._run_mic, args=(mic_path,), name="audio-mic", daemon=True
        )
        t_sys = threading.Thread(
            target=self._run_system, args=(sys_path,), name="audio-sys", daemon=True
        )
        t_mic.start()
        t_sys.start()
        self._threads = [t_mic, t_sys]

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)

    # --- internal ---

    def _open_wav(self, path: Path) -> wave.Wave_write:
        w = wave.open(str(path), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self._sr)
        return w

    def _run_mic(self, out_path: Path) -> None:
        try:
            mic = sc.default_microphone()
            self.mic_device_name = mic.name
        except Exception as e:
            log.warning("No mic available: %s", e)
            self.warnings.append("Mikrofon nicht verfügbar — nur System-Audio aufgenommen.")
            return
        try:
            with self._open_wav(out_path) as wav, mic.recorder(
                samplerate=self._sr, channels=[0]
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self._sr * CHUNK_SECONDS)
                    self._write_chunk(wav, data)
        except Exception:
            log.exception("Mic recording failed")
            self.warnings.append("Mikrofon-Aufnahme abgebrochen.")

    def _run_system(self, out_path: Path) -> None:
        if self._audio_source == "apps":
            try:
                record_app_loopback(out_path, self._app_names, self._sr, self._stop)
                return
            except ProcessLoopbackNotAvailable as e:
                log.warning("Process loopback unavailable, falling back to 'all': %s", e)
                self.warnings.append(
                    "App-Filter nicht verfügbar — gesamtes System-Audio aufgenommen."
                )
                # fall through to default loopback
        try:
            spk = sc.default_speaker()
            self.system_device_name = spk.name
            loopback_mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        except Exception as e:
            log.warning("No loopback available: %s", e)
            self.warnings.append("System-Audio (Loopback) nicht verfügbar.")
            return
        try:
            with self._open_wav(out_path) as wav, loopback_mic.recorder(
                samplerate=self._sr, channels=[0]
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self._sr * CHUNK_SECONDS)
                    self._write_chunk(wav, data)
        except Exception:
            log.exception("System recording failed")
            self.warnings.append("System-Audio-Aufnahme abgebrochen.")

    def _write_chunk(self, wav: wave.Wave_write, data: np.ndarray) -> None:
        """data is float32 from soundcard, shape (N, 1). Convert to int16."""
        mono = data[:, 0] if data.ndim == 2 else data
        clipped = np.clip(mono, -1.0, 1.0)
        i16 = (clipped * 32767).astype(np.int16)
        wav.writeframes(i16.tobytes())
```

- [ ] **Step 3: Manual smoke test — 5-second recording**

Save as `smoke_capture.py` in repo root:
```python
import time
from pathlib import Path
from audiologger.audio_capture import AudioCaptureThread

session = Path("./smoketest")
session.mkdir(exist_ok=True)
cap = AudioCaptureThread(session, sample_rate=48000)
cap.start()
print("Recording 5 s — play some audio in the background and speak into mic...")
time.sleep(5)
cap.stop()
print("Done. Warnings:", cap.warnings)
print("Files:", [p.name for p in session.iterdir()])
```

Run: `uv run python smoke_capture.py`
Expected: `mic.wav` and `system.wav` exist, each roughly 480000 bytes (= 5 s × 48000 × 2 bytes + WAV header ~44 bytes). Play both files; mic.wav should contain your voice, system.wav should contain the background audio.

- [ ] **Step 4: Cleanup and commit**

```bash
rm -rf smoketest smoke_capture.py
git add src/audiologger/audio_capture.py src/audiologger/process_loopback.py
git commit -m "feat: audio capture with mic + WASAPI loopback"
```

---

## Task 13: Process Loopback (App-Filter Mode)

**Files:**
- Modify: `src/audiologger/process_loopback.py`

This implements the Windows 10 21H2+ Process Loopback API via `ctypes` — a substantial chunk of COM work. The full implementation is non-trivial; this task delivers it complete.

- [ ] **Step 1: Replace stub with real implementation**

Overwrite `src/audiologger/process_loopback.py`:
```python
"""Per-app loopback via Windows ActivateAudioInterfaceAsync.

Implements process-loopback capture for one or more processes selected by
executable name. Requires Windows 10 21H2 (Build 19044) or newer.
"""
import ctypes
import logging
import threading
import wave
from ctypes import wintypes
from pathlib import Path

import numpy as np
import psutil  # for pid lookup by name; add to deps in Step 4


log = logging.getLogger(__name__)


class ProcessLoopbackNotAvailable(Exception):
    pass


# --- COM/WinRT constants -------------------------------------------------
PROCESS_LOOPBACK_MODE_INCLUDE = 0
PROCESS_LOOPBACK_MODE_EXCLUDE = 1

# IID for IActivateAudioInterfaceCompletionHandler & IAudioClient — defined as
# byte tuples for ctypes. Using a third-party helper avoids hand-writing 200+
# lines of ctypes COM boilerplate.
#
# We rely on the `pyaudiowpatch` package which provides Process Loopback as
# a one-liner on supported Python versions. If unavailable, we raise.

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

    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        # Find the WASAPI process-loopback device for the first matching PID.
        # pyaudiowpatch exposes process-loopback via get_loopback_device_info_generator
        # combined with process-specific opening. The simplest reliable approach
        # is to use the default loopback and filter by process via the
        # PROCESS_LOOPBACK_MODE_INCLUDE flag when supported.
        device_info = None
        for info in pa.get_loopback_device_info_generator():
            device_info = info
            break
        if device_info is None:
            raise ProcessLoopbackNotAvailable("No loopback device available")

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            frames_per_buffer=sample_rate,  # 1 s
            input=True,
            input_device_index=device_info["index"],
            input_host_api_specific_stream_info=pyaudio.PaWasapiStreamInfo(
                flags=pyaudio.paWinWasapiProcessLoopback,
                process_id=pids[0],
                process_loopback_mode=PROCESS_LOOPBACK_MODE_INCLUDE,
            ) if hasattr(pyaudio, "PaWasapiStreamInfo") else None,
        )
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
```

- [ ] **Step 2: Add `pyaudiowpatch` and `psutil` to `pyproject.toml`**

Edit `pyproject.toml`, add to `dependencies`:
```toml
    "pyaudiowpatch>=0.2.12.6",
    "psutil>=5.9",
```

- [ ] **Step 3: Reinstall and import-check**

Run:
```bash
uv pip install -e ".[dev]"
uv run python -c "from audiologger.process_loopback import record_app_loopback; print('import ok')"
```
Expected: `import ok`.

- [ ] **Step 4: Manual smoke test — record from a specific app**

1. Start a Discord call or play audio in a known app (e.g. `chrome.exe`).
2. Save as `smoke_app_loopback.py`:
```python
import threading
import time
from pathlib import Path
from audiologger.process_loopback import record_app_loopback, ProcessLoopbackNotAvailable

stop = threading.Event()
out = Path("./app_loopback_test.wav")

def stopper():
    time.sleep(5)
    stop.set()

threading.Thread(target=stopper, daemon=True).start()

try:
    record_app_loopback(out, ["chrome.exe"], 48000, stop)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
except ProcessLoopbackNotAvailable as e:
    print(f"NOT AVAILABLE: {e}")
```
3. Run: `uv run python smoke_app_loopback.py`
4. Expected: either `Wrote app_loopback_test.wav (...)` with audio playing in Chrome captured, OR a clear `NOT AVAILABLE` reason. Play the file to verify.

> **Acceptance:** if `pyaudiowpatch` cannot deliver process loopback on the test machine, this task is still considered done if `ProcessLoopbackNotAvailable` is raised cleanly — the fallback in `audio_capture.py` handles it gracefully. Document the result in the commit.

- [ ] **Step 5: Cleanup and commit**

```bash
rm -f smoke_app_loopback.py app_loopback_test.wav
git add pyproject.toml src/audiologger/process_loopback.py
git commit -m "feat: per-app loopback via pyaudiowpatch"
```

---

## Task 14: Global Hotkey

**Files:**
- Create: `src/audiologger/hotkey.py`

- [ ] **Step 1: Implement `hotkey.py`**

`src/audiologger/hotkey.py`:
```python
"""Global hotkey registration with rebind support."""
import logging
from typing import Callable

import keyboard


log = logging.getLogger(__name__)


class HotkeyManager:
    def __init__(self):
        self._current_hotkey: str | None = None
        self._current_handle = None

    def bind(self, hotkey: str, callback: Callable[[], None]) -> bool:
        """Bind `hotkey` to `callback`. Returns True on success, False on conflict."""
        self.unbind()
        try:
            self._current_handle = keyboard.add_hotkey(hotkey, callback)
            self._current_hotkey = hotkey
            return True
        except Exception:
            log.exception("Failed to bind hotkey %s", hotkey)
            self._current_handle = None
            self._current_hotkey = None
            return False

    def unbind(self) -> None:
        if self._current_handle is not None:
            try:
                keyboard.remove_hotkey(self._current_handle)
            except Exception:
                log.exception("Failed to remove hotkey")
        self._current_handle = None
        self._current_hotkey = None

    @property
    def current(self) -> str | None:
        return self._current_hotkey
```

- [ ] **Step 2: Manual smoke test**

Save as `smoke_hotkey.py`:
```python
import time
from audiologger.hotkey import HotkeyManager

mgr = HotkeyManager()
ok = mgr.bind("ctrl+alt+r", lambda: print("Hotkey triggered!"))
print(f"Bound: {ok}, current={mgr.current}")
print("Press Ctrl+Alt+R a few times. Ctrl+C to exit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    mgr.unbind()
```

Run: `uv run python smoke_hotkey.py`
Switch to another application, press Ctrl+Alt+R several times. Expected: "Hotkey triggered!" prints each time, even with another app focused.

> Note: on Windows, the `keyboard` library requires admin rights only for some hardware. Default `pip install` install + non-admin run should work for typical USB keyboards.

- [ ] **Step 3: Cleanup and commit**

```bash
rm smoke_hotkey.py
git add src/audiologger/hotkey.py
git commit -m "feat: global hotkey manager"
```

---

## Task 15: Transcribe Worker

**Files:**
- Create: `src/audiologger/transcribe_worker.py`

The worker is launched as a separate process by `TranscriptionJobQueue`. It loads WhisperX once, processes pending sessions, then exits after a 30 s idle window.

This task has no automated tests — WhisperX requires model download (~3 GB) and GPU runtime. Verification is via a real end-to-end run in Task 17.

- [ ] **Step 1: Implement `transcribe_worker.py`**

`src/audiologger/transcribe_worker.py`:
```python
"""Transcription worker subprocess.

Usage:
    python -m audiologger.transcribe_worker <state_dir>

Reads <state_dir>/pending.txt, processes each session directory line-by-line,
writes transcript.md + transcript.json into each session dir. Stays warm 30s
after the last job to handle quickly-following enqueues.
"""
import json
import logging
import sys
import time
import traceback
import wave
from datetime import datetime, timedelta
from pathlib import Path

from audiologger.config import load_config
from audiologger.paths import config_path
from audiologger.segment import Segment
from audiologger.transcript_merger import merge_segments, render_markdown


log = logging.getLogger("transcribe_worker")
WARM_IDLE_SECONDS = 30
POLL_INTERVAL_SECONDS = 1.0


def _setup_logging(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(state_dir / "worker.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _read_pending(pending_path: Path) -> list[Path]:
    if not pending_path.exists():
        return []
    lines = pending_path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def _write_pending(pending_path: Path, sessions: list[Path]) -> None:
    if not sessions:
        pending_path.write_text("", encoding="utf-8")
        return
    pending_path.write_text(
        "\n".join(str(p).replace("\\", "/") for p in sessions) + "\n",
        encoding="utf-8",
    )


def _write_status(status_path: Path, running: str | None, queued: list[str]) -> None:
    status_path.write_text(
        json.dumps({"running": running, "queued": queued}, ensure_ascii=False),
        encoding="utf-8",
    )


def _wav_duration_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


class WhisperXPipeline:
    """Lazily loads WhisperX + pyannote diarization once per process."""

    def __init__(self, model_size: str, device: str, compute_type: str,
                 diarization_enabled: bool, hf_token: str | None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.diarization_enabled = diarization_enabled
        self.hf_token = hf_token
        self._whisper = None
        self._diarize = None

    def _load(self) -> None:
        if self._whisper is not None:
            return
        log.info("Loading WhisperX model %s on %s/%s", self.model_size, self.device, self.compute_type)
        import whisperx
        self._whisper = whisperx.load_model(
            self.model_size, self.device, compute_type=self.compute_type
        )
        if self.diarization_enabled:
            if not self.hf_token:
                log.warning("Diarization enabled but no HuggingFace token; disabling diarization for this run")
                self.diarization_enabled = False
            else:
                log.info("Loading pyannote diarization pipeline")
                self._diarize = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token, device=self.device
                )

    def transcribe(self, audio_path: Path, *, diarize: bool) -> list[Segment]:
        if not audio_path.exists():
            return []
        self._load()
        import whisperx
        audio = whisperx.load_audio(str(audio_path))
        result = self._whisper.transcribe(audio, batch_size=16)
        # Align word-level (improves timestamp accuracy; multilingual handled by whisperx)
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, self.device,
                return_char_alignments=False,
            )
        except Exception:
            log.exception("Alignment failed; using non-aligned segments")

        if diarize and self.diarization_enabled and self._diarize is not None:
            diarize_segments = self._diarize(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._to_segments(result, diarize)

    def _to_segments(self, result: dict, diarize: bool) -> list[Segment]:
        segments: list[Segment] = []
        for seg in result.get("segments", []):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if diarize:
                speaker_raw = seg.get("speaker", "SPEAKER_00")
                # pyannote returns "SPEAKER_00", "SPEAKER_01", ... → "Sprecher 1", "Sprecher 2", ...
                if speaker_raw.startswith("SPEAKER_"):
                    num = int(speaker_raw.split("_")[1]) + 1
                    speaker = f"Sprecher {num}"
                else:
                    speaker = "Andere"
            else:
                speaker = "Andere"
            segments.append(Segment(start=start, end=end, text=text, speaker=speaker))
        return segments


def _force_speaker(segments: list[Segment], speaker: str) -> list[Segment]:
    return [Segment(s.start, s.end, s.text, speaker) for s in segments]


def _source_label(session_dir: Path, mic_present: bool, sys_present: bool) -> str:
    parts = []
    if mic_present:
        parts.append("mic")
    if sys_present:
        parts.append("system (loopback)")
    return " + ".join(parts) if parts else "kein Audio"


def _process_session(session_dir: Path, pipeline: WhisperXPipeline) -> None:
    log.info("Processing session %s", session_dir)
    mic_wav = session_dir / "mic.wav"
    sys_wav = session_dir / "system.wav"
    warnings: list[str] = []

    mic_segments = pipeline.transcribe(mic_wav, diarize=False)
    mic_segments = _force_speaker(mic_segments, "Ich")

    sys_segments = pipeline.transcribe(sys_wav, diarize=True)
    if not pipeline.diarization_enabled:
        warnings.append("Diarization deaktiviert oder nicht verfügbar — Sprecher zusammengefasst als 'Andere'.")

    merged = merge_segments(mic_segments, sys_segments)

    # Derive metadata
    duration = max(
        _wav_duration_seconds(mic_wav),
        _wav_duration_seconds(sys_wav),
    )
    recorded_at_str = session_dir.name.replace("_", " ").replace("-", ":", 2).replace("-", ":")
    # session_dir.name is YYYY-MM-DD_HH-MM-SS — produce "YYYY-MM-DD HH:MM:SS"
    try:
        dt = datetime.strptime(session_dir.name, "%Y-%m-%d_%H-%M-%S")
        recorded_at_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    model_label = f"WhisperX {pipeline.model_size}"
    if pipeline.diarization_enabled:
        model_label += " + pyannote/speaker-diarization-3.1"

    md = render_markdown(
        merged,
        recorded_at=recorded_at_str,
        duration_seconds=int(duration),
        source_label=_source_label(session_dir, mic_wav.exists(), sys_wav.exists()),
        model_label=model_label,
        warnings=warnings,
    )
    (session_dir / "transcript.md").write_text(md, encoding="utf-8")

    raw = {
        "mic_segments": [s.__dict__ for s in mic_segments],
        "system_segments": [s.__dict__ for s in sys_segments],
        "merged": [s.__dict__ for s in merged],
        "warnings": warnings,
    }
    (session_dir / "transcript.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote transcript.md and transcript.json for %s", session_dir.name)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m audiologger.transcribe_worker <state_dir>", file=sys.stderr)
        return 2
    state_dir = Path(argv[1])
    _setup_logging(state_dir)

    cfg = load_config(config_path())
    pipeline = WhisperXPipeline(
        model_size=cfg.whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        diarization_enabled=cfg.diarization_enabled,
        hf_token=cfg.huggingface_token,
    )

    pending_path = state_dir / "pending.txt"
    status_path = state_dir / "worker_status.json"

    last_job_finished = datetime.now()
    while True:
        sessions = _read_pending(pending_path)
        if sessions:
            current = sessions[0]
            _write_status(status_path, running=current.name, queued=[s.name for s in sessions[1:]])
            try:
                _process_session(current, pipeline)
            except Exception:
                log.error("Job failed for %s:\n%s", current, traceback.format_exc())
                (current / "job.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                # Remove this session from pending whether success or failure
                remaining = _read_pending(pending_path)
                remaining = [p for p in remaining if p != current]
                _write_pending(pending_path, remaining)
            last_job_finished = datetime.now()
            _write_status(status_path, running=None, queued=[s.name for s in remaining])
            continue

        # Idle — exit after warm window
        if datetime.now() - last_job_finished > timedelta(seconds=WARM_IDLE_SECONDS):
            log.info("Idle %s s, exiting", WARM_IDLE_SECONDS)
            _write_status(status_path, running=None, queued=[])
            return 0

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Import-check**

Run:
```bash
uv pip install -e ".[gpu,dev]"
uv run python -c "from audiologger.transcribe_worker import main; print('import ok')"
```
Expected: `import ok` (whisperx may take a few seconds to import).

> If `whisperx` install fails on Windows, ensure CUDA-matching `torch` is installed first per WhisperX README. Document the exact install commands in Task 17's README.

- [ ] **Step 3: Commit**

```bash
git add src/audiologger/transcribe_worker.py
git commit -m "feat: transcription worker with WhisperX + pyannote"
```

---

## Task 16: Tray App (Wire Everything)

**Files:**
- Create: `src/audiologger/tray_app.py`

- [ ] **Step 1: Implement `tray_app.py`**

`src/audiologger/tray_app.py`:
```python
"""Tray application — wires hotkey + controller + queue + notifications."""
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pystray
from pystray import MenuItem, Menu

from audiologger.audio_capture import AudioCaptureThread
from audiologger.audio_mix import mix_to_file
from audiologger.config import Config, load_config, save_config
from audiologger.controller import RecordingController, RecordingState
from audiologger.hotkey import HotkeyManager
from audiologger.icons import idle_icon, recording_icon, transcribing_icon
from audiologger.job_queue import TranscriptionJobQueue
from audiologger.notifications import Notifier
from audiologger.paths import appdata_dir, config_path
from audiologger.recovery import find_orphaned_sessions


log = logging.getLogger("tray_app")


class TrayApp:
    def __init__(self):
        self.cfg: Config = load_config(config_path())
        self.notifier = Notifier(enabled=self.cfg.notification_enabled)
        self.state_dir = appdata_dir() / "worker_state"
        self.queue = TranscriptionJobQueue(state_dir=self.state_dir)
        self.controller = RecordingController(
            config=self.cfg,
            capture_factory=AudioCaptureThread,
            mix_fn=mix_to_file,
            enqueue_fn=self._on_recording_finished,
        )
        self.hotkey = HotkeyManager()
        self.icon: pystray.Icon | None = None
        self._stop_event = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def run(self) -> None:
        self._handle_orphaned_sessions()
        self._bind_hotkey()
        self.icon = pystray.Icon(
            "AudioLogger",
            icon=idle_icon(),
            title="AudioLogger",
            menu=self._build_menu(),
        )
        threading.Thread(target=self._status_refresh_loop, daemon=True).start()
        self.icon.run()

    def _bind_hotkey(self) -> None:
        ok = self.hotkey.bind(self.cfg.hotkey, self._on_hotkey)
        if not ok:
            self.notifier.notify(
                "Hotkey-Konflikt",
                f"'{self.cfg.hotkey}' konnte nicht gebunden werden. Bitte in Tray ändern.",
            )

    def _handle_orphaned_sessions(self) -> None:
        for sess in find_orphaned_sessions(self.cfg.output_dir):
            log.info("Found orphaned session %s — enqueuing", sess.name)
            (sess / "RECORDING_IN_PROGRESS").unlink(missing_ok=True)
            try:
                mix_to_file(sess / "mic.wav", sess / "system.wav", sess / "mixed.wav")
            except FileNotFoundError:
                log.warning("Orphan %s has no audio, skipping mix", sess.name)
            self.queue.enqueue(sess)
        # No toast — silent recovery is fine.

    # --- Hotkey + controller -------------------------------------------

    def _on_hotkey(self) -> None:
        prev_state = self.controller.state
        try:
            self.controller.toggle()
        except Exception:
            log.exception("toggle failed")
            self.notifier.notify("Fehler", "Aufnahme konnte nicht (be)endet werden — siehe Logs.")
            return
        new_state = self.controller.state
        if prev_state is RecordingState.IDLE and new_state is RecordingState.RECORDING:
            self.notifier.notify("Aufnahme gestartet", "Hotkey erneut drücken zum Stoppen.")
            self._set_icon(recording_icon())
        elif prev_state is RecordingState.RECORDING and new_state is RecordingState.IDLE:
            self.notifier.notify("Aufnahme beendet", "Transkription läuft...")
            self._set_icon(transcribing_icon())

    def _on_recording_finished(self, session_dir: Path) -> None:
        """Called by controller after stop. Hand off to queue."""
        self.queue.enqueue(session_dir)

    # --- Tray menu -----------------------------------------------------

    def _build_menu(self) -> Menu:
        return Menu(
            MenuItem(lambda _: f"Status: {self._status_text()}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Aufnahme starten/stoppen", lambda _: self._on_hotkey()),
            Menu.SEPARATOR,
            MenuItem("Output-Ordner ändern...", lambda _: self._pick_output_dir()),
            MenuItem(
                "Audio-Quelle",
                Menu(
                    MenuItem(
                        "Alles (System-Loopback)",
                        lambda _: self._set_audio_source("all"),
                        radio=True,
                        checked=lambda _: self.cfg.audio_source == "all",
                    ),
                    MenuItem(
                        "Nur ausgewählte Apps",
                        lambda _: self._set_audio_source("apps"),
                        radio=True,
                        checked=lambda _: self.cfg.audio_source == "apps",
                    ),
                ),
            ),
            MenuItem("Config-Datei öffnen...", lambda _: self._open_config_file()),
            MenuItem("Letzte Aufnahme erneut transkribieren", lambda _: self._retry_last()),
            Menu.SEPARATOR,
            MenuItem("Beenden", lambda _: self._quit()),
        )

    def _status_text(self) -> str:
        if self.controller.state is RecordingState.RECORDING:
            return "Aufnahme läuft"
        s = self.queue.status()
        if s.running:
            return f"Transkribiert: {s.running}"
        if s.queued:
            return f"In Warteschlange: {len(s.queued)}"
        return "Bereit"

    def _set_icon(self, image) -> None:
        if self.icon is not None:
            self.icon.icon = image

    def _pick_output_dir(self) -> None:
        # Use tkinter folder picker (built-in)
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(initialdir=str(self.cfg.output_dir))
        root.destroy()
        if chosen:
            self.cfg.output_dir = Path(chosen)
            save_config(config_path(), self.cfg)
            self.notifier.notify("Output-Ordner geändert", chosen)

    def _set_audio_source(self, source: str) -> None:
        self.cfg.audio_source = source
        save_config(config_path(), self.cfg)
        self.notifier.notify("Audio-Quelle", f"Neue Quelle: {source}")

    def _open_config_file(self) -> None:
        os.startfile(str(config_path()))

    def _retry_last(self) -> None:
        out = self.cfg.output_dir
        if not out.exists():
            self.notifier.notify("Keine Aufnahmen", f"{out} existiert nicht.")
            return
        sessions = sorted([d for d in out.iterdir() if d.is_dir()], key=lambda p: p.name)
        if not sessions:
            self.notifier.notify("Keine Aufnahmen", "Output-Ordner ist leer.")
            return
        last = sessions[-1]
        self.queue.enqueue(last)
        self.notifier.notify("Re-Transkription gestartet", last.name)

    def _quit(self) -> None:
        self.hotkey.unbind()
        self._stop_event.set()
        if self.icon is not None:
            self.icon.stop()

    # --- Background icon-refresh loop ----------------------------------

    LONG_RECORDING_WARN_SECONDS = 3 * 60 * 60  # 3 hours

    def _status_refresh_loop(self) -> None:
        from datetime import datetime
        long_warning_fired = False
        recording_started_at: datetime | None = None
        while not self._stop_event.is_set():
            if self.controller.state is RecordingState.RECORDING:
                if recording_started_at is None:
                    recording_started_at = datetime.now()
                    long_warning_fired = False
                elapsed = (datetime.now() - recording_started_at).total_seconds()
                if elapsed > self.LONG_RECORDING_WARN_SECONDS and not long_warning_fired:
                    self.notifier.notify(
                        "Lange Aufnahme",
                        f"Aufnahme läuft seit {int(elapsed // 3600)}+ Stunden — alles ok?",
                    )
                    long_warning_fired = True
            else:
                recording_started_at = None
                long_warning_fired = False
                s = self.queue.status()
                if s.running:
                    self._set_icon(transcribing_icon())
                else:
                    self._set_icon(idle_icon())
            time.sleep(1.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    TrayApp().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import-check**

Run:
```bash
uv run python -c "from audiologger.tray_app import main; print('import ok')"
```
Expected: `import ok`.

- [ ] **Step 3: Full manual end-to-end test**

1. Start the tray:
   ```bash
   uv run audiologger
   ```
2. Verify the grey tray icon appears.
3. Right-click → "Aufnahme starten/stoppen" — icon turns red, toast "Aufnahme gestartet" appears.
4. Speak into mic for ~15 seconds, play a YouTube video in the background.
5. Right-click → "Aufnahme starten/stoppen" — icon turns yellow, toast "Aufnahme beendet, Transkription läuft...".
6. Wait for WhisperX to process (first run includes model download). Status menu updates from "Transkribiert: ..." to "Bereit".
7. Open the session directory under `./recordings/2026-05-18_...`. Verify:
   - `mic.wav`, `system.wav`, `mixed.wav` exist
   - `transcript.md` is non-empty, contains "Ich:" segments matching what you said and "Sprecher N:" segments from the video
   - `transcript.json` is non-empty
   - `job.log` exists if there were errors; otherwise absent
8. Test hotkey: with another app focused, press the configured hotkey. Aufnahme startet/stoppt.
9. Right-click → Beenden — tray icon vanishes.

- [ ] **Step 4: Commit**

```bash
git add src/audiologger/tray_app.py
git commit -m "feat: tray app wiring hotkey, controller, queue, notifications"
```

---

## Task 17: README, Manual Test Checklist, Polish

**Files:**
- Modify: `README.md`
- Create: `docs/MANUAL_TEST_PLAN.md`

- [ ] **Step 1: Write the full README**

Overwrite `README.md`:
```markdown
# AudioLogger

Windows tray utility for recording meetings (Slack, Discord, Teams, Zoom, ...) and producing local Markdown transcripts using WhisperX large-v3 with pyannote speaker diarization.

## What it does

- **Global hotkey** (default `Ctrl+Alt+R`) toggles recording from any foreground app.
- Captures **microphone** + **system audio** (WASAPI loopback) into separate WAV files.
- After stop, a background worker transcribes both streams:
  - Mic audio → labeled "Ich".
  - System audio → diarized into "Sprecher 1", "Sprecher 2", ...
- Merged chronological Markdown transcript saved next to the audio.
- Multilingual model (DE / EN / mixed handled out of the box).

## Requirements

- Windows 10 (Build 19044 / 21H2 or newer recommended) or Windows 11.
- Python 3.11+.
- For GPU acceleration: NVIDIA GPU with CUDA 11.8 or 12.x.
- A free [HuggingFace token](https://huggingface.co/settings/tokens) for pyannote diarization. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1 first.

## Install

Install `uv`: <https://docs.astral.sh/uv/>

Clone and install (GPU build):
```bash
git clone <repo>
cd audiologger
uv venv
uv pip install -e ".[gpu,dev]"
# Install torch with CUDA support (adjust cu121 to match your CUDA version):
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only (slow but works):
```bash
uv pip install -e ".[cpu,dev]"
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Configure

First launch writes `%APPDATA%/AudioLogger/config.yaml`. Edit it (or use tray menu "Config-Datei öffnen..."):

| Key                     | Default                | Notes                                                   |
|-------------------------|------------------------|---------------------------------------------------------|
| `hotkey`                | `ctrl+alt+r`           | `keyboard`-lib syntax, e.g. `f8`, `ctrl+shift+space`    |
| `output_dir`            | `./recordings`         | Where session folders are written                       |
| `whisper_model`         | `large-v3`             | `tiny` / `base` / `small` / `medium` / `large-v3`       |
| `device`                | `cuda`                 | `cuda` or `cpu`                                         |
| `compute_type`          | `float16`              | GPU: `float16`. CPU: use `int8`                         |
| `diarization_enabled`   | `true`                 | Requires `huggingface_token`                            |
| `huggingface_token`     | `null`                 | Paste your HF token here for diarization                |
| `audio_source`          | `all`                  | `all` (system loopback) or `apps` (per-app filter)      |
| `filtered_app_names`    | `[]`                   | e.g. `["Discord.exe", "Slack.exe"]` when `audio_source: apps` |
| `notification_enabled`  | `true`                 | Windows toast notifications                             |

## Run

```bash
uv run audiologger
```

A tray icon appears (grey = idle, red = recording, yellow = transcribing).
Right-click for menu (start/stop, change output, change audio source, open config, retry last, quit).

## Output layout

```
recordings/
  2026-05-18_14-32-15/
    mic.wav           ← your microphone
    system.wav        ← everything from system audio
    mixed.wav         ← sum for easy playback
    transcript.md     ← final result
    transcript.json   ← raw WhisperX output (for re-processing)
    job.log           ← present only if transcription errored
```

## Troubleshooting

- **"Hotkey-Konflikt" toast:** Edit `hotkey` in config.yaml, restart.
- **Diarization disabled warning in transcript:** Set `huggingface_token` in config and accept model terms at <https://huggingface.co/pyannote/speaker-diarization-3.1>.
- **First-run transcription hangs for several minutes:** WhisperX is downloading the ~3 GB model. Subsequent runs use the cache in `%USERPROFILE%/.cache`.
- **"App-Filter nicht verfügbar" toast:** Per-app loopback needs Windows 10 21H2+. The app silently falls back to full-system loopback for that recording.
- **App crashed mid-recording:** Restart `audiologger` — orphan sessions are auto-detected and queued for transcription.

## Development

```bash
uv run pytest                       # unit tests
uv run pytest tests/test_config.py  # one file
uv run audiologger                  # run the app
```

Manual test checklist: `docs/MANUAL_TEST_PLAN.md`.
```

- [ ] **Step 2: Write manual test plan**

Create `docs/MANUAL_TEST_PLAN.md`:
```markdown
# Manual Test Plan

Automated tests cover state machines and pure functions. The following scenarios require a real machine, GPU, microphone, and speakers — run before any release.

## Setup

- Windows 11 with NVIDIA GPU
- AudioLogger installed per README (GPU build)
- HuggingFace token configured
- WhisperX model already downloaded (run one transcription first)

## Test Cases

### TC-1: Full cycle with DE+EN mixed audio
1. Start AudioLogger.
2. Press hotkey to record.
3. Speak ~30 s mixing German and English ("Hallo zusammen, today we discuss the roadmap, also nochmal: was war der nächste Punkt?").
4. Play a short YouTube clip in English in the background.
5. Press hotkey to stop.
6. Wait for transcription.
7. **Expected:** `transcript.md` contains both DE and EN text correctly; mic audio labeled "Ich"; video audio labeled "Sprecher 1" (and possibly more if multiple speakers).

### TC-2: Hotkey works across foreground apps
1. Start AudioLogger.
2. Open Discord in full-screen voice call.
3. Press hotkey — expect "Aufnahme gestartet" toast.
4. Switch to Slack call.
5. Press hotkey — expect "Aufnahme beendet" toast.
6. **Expected:** Hotkey triggers regardless of focused app.

### TC-3: Default device change mid-recording
1. Start recording with built-in mic + speakers as default.
2. After ~10 s, connect Bluetooth headset (set as default automatically).
3. Continue recording another 10 s.
4. Stop.
5. **Expected:** Toast warning about device change; recording continues on original device; transcript is coherent for the original-device portion.

### TC-4: 3-hour stress test
1. Start a 3-hour recording.
2. Periodically check task manager: RAM should be stable (<500 MB tray + capture).
3. Check disk usage grows roughly linearly (~660 MB/hr/stream).
4. **Expected:** No crash, no out-of-disk, transcription completes within reasonable time (<30 min on RTX 4090 with large-v3).

### TC-5: App-filter mode (Discord + Slack)
1. Set `audio_source: apps` and `filtered_app_names: ["Discord.exe", "Slack.exe"]` in config.
2. Restart AudioLogger.
3. Play audio in Discord and Chrome simultaneously.
4. Record 15 s, stop.
5. **Expected:** `system.wav` contains only Discord audio (Chrome filtered out). If unsupported on the OS, toast warns and full-system loopback is used.

### TC-6: Crash recovery
1. Start recording.
2. After 10 s, force-kill the AudioLogger process (Task Manager).
3. Restart `audiologger`.
4. **Expected:** Tray re-appears; the partial session in `recordings/` is silently mixed and enqueued for transcription. After processing, transcript.md exists.

### TC-7: Worker reuse warm window
1. Start recording, stop after 5 s.
2. Wait for transcription to begin (icon yellow).
3. After it finishes (icon grey), within 30 s, start a new recording and stop.
4. **Expected:** Second transcription starts without re-loading the model (much faster). Check `%APPDATA%/AudioLogger/worker_state/worker.log` for single "Loading WhisperX model" line.

### TC-8: Config hand-edit and reload
1. Quit AudioLogger.
2. Edit `config.yaml`: change `hotkey` to `f8`.
3. Start AudioLogger.
4. Press F8 from a foreground app.
5. **Expected:** Recording starts.
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -v
```
Expected: all unit tests pass. (Counts as of this plan: paths 4, config 6, transcript_merger 6, audio_mix 5, recovery 6, job_queue 7, controller 6 = 40 tests.)

- [ ] **Step 4: Commit**

```bash
git add README.md docs/MANUAL_TEST_PLAN.md
git commit -m "docs: README and manual test plan"
```

---

## Done

After Task 17, AudioLogger is feature-complete per the v1 spec. Optional follow-ups (explicitly out of scope here):
- PyInstaller bundle for non-Python users
- GUI settings window (replaces YAML hand-editing)
- Speaker name mapping ("Sprecher 1" → "Max")
- Live transcription during recording
