# Screenshots in Meeting Transcripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every image put on the clipboard during a meeting recording (Win+Shift+S, Alt+Print) is saved next to the recording and embedded in the transcript at the moment it was taken.

**Architecture:** A new module `screenshot_watch.py` owns the naming scheme, discovery of saved screenshots, and a clipboard watcher thread that polls `GetClipboardSequenceNumber()` and saves new images as PNG. The controller starts the watcher for meeting recordings only; the tray adds a toast; the worker finds the PNGs by file name and `render_markdown` interleaves them with the speech lines.

**Tech Stack:** Python 3.12, Pillow (`ImageGrab.grabclipboard`, already a dependency), ctypes (`user32.GetClipboardSequenceNumber`), pytest.

**Spec:** [docs/superpowers/specs/2026-10-02-screenshots-design.md](../specs/2026-10-02-screenshots-design.md)

---

## Conventions

- Work from the repo root `C:\Users\chris\Claude\AudioLogger` on `main`.
- Run tests with the project venv: `.venv/Scripts/python.exe -m pytest <path> -v`. The venv lives only in the main checkout; in a worktree call it by absolute path.
- Write code that contains regex backslashes with the Write/Edit tools, not shell heredocs — the shell here eats backslashes.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `src/audiologger/screenshot_watch.py` (new) | Naming scheme, `find_screenshots`, `WindowsClipboard` adapter, `ClipboardScreenshotWatcher` |
| `src/audiologger/transcript_merger.py` | `render_markdown(..., screenshots=())` interleaves screenshot lines |
| `src/audiologger/transcribe_worker.py` | Meeting path passes discovered screenshots to the Markdown and JSON |
| `src/audiologger/controller.py` | Starts/stops a watcher for meeting recordings via an optional factory |
| `src/audiologger/tray_app.py` | Supplies the factory and the "Screenshot N saved" toast |
| `tests/test_screenshot_watch.py` (new) | Naming, discovery, watcher |
| `tests/test_transcript_merger.py` | Rendering |
| `tests/test_worker_screenshots.py` (new) | Worker integration |
| `tests/test_controller.py`, `tests/test_tray_app.py` | Wiring |
| `README.md`, `docs/MANUAL_TEST_PLAN.md` | User-facing docs, manual test TC-10 |

---

### Task 1: Naming scheme and discovery

**Files:**
- Create: `src/audiologger/screenshot_watch.py`
- Create: `tests/test_screenshot_watch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_screenshot_watch.py`:

```python
"""Screenshots taken during a meeting recording: naming, discovery, and the
clipboard watcher. See docs/superpowers/specs/2026-10-02-screenshots-design.md.
"""
from pathlib import Path

from audiologger.screenshot_watch import Screenshot, find_screenshots, screenshot_file_name


def test_the_name_carries_the_offset_into_the_recording():
    assert screenshot_file_name(751.6, set()) == "screenshot_00-12-31.png"
    assert screenshot_file_name(3725.0, set()) == "screenshot_01-02-05.png"


def test_a_second_screenshot_in_the_same_second_gets_a_suffix():
    taken = {"screenshot_00-12-31.png"}
    assert screenshot_file_name(751.2, taken) == "screenshot_00-12-31_2.png"
    taken.add("screenshot_00-12-31_2.png")
    assert screenshot_file_name(751.9, taken) == "screenshot_00-12-31_3.png"


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"png")


def test_screenshots_are_found_by_name_in_time_order(tmp_path):
    _touch(tmp_path / "screenshots",
           "screenshot_00-12-31_2.png", "screenshot_01-02-05.png", "screenshot_00-12-31.png")

    assert find_screenshots(tmp_path) == [
        Screenshot(at_s=751, path="screenshots/screenshot_00-12-31.png"),
        Screenshot(at_s=751, path="screenshots/screenshot_00-12-31_2.png"),
        Screenshot(at_s=3725, path="screenshots/screenshot_01-02-05.png"),
    ]


def test_other_files_in_the_folder_are_ignored(tmp_path):
    _touch(tmp_path / "screenshots", "notes.txt", "screenshot.bmp", "screenshot_00-00-05.png")

    assert [s.at_s for s in find_screenshots(tmp_path)] == [5]


def test_a_recording_without_screenshots_has_none(tmp_path):
    assert find_screenshots(tmp_path) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_screenshot_watch.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'audiologger.screenshot_watch'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/audiologger/screenshot_watch.py` (use the Write tool — it contains a regex):

```python
"""Screenshots taken during a meeting recording.

The user captures with Win+Shift+S or Alt+Print, both of which put an image on
the clipboard. While a meeting is recorded, every new image that appears there
is saved next to the recording, and the transcript embeds it at the moment it
was taken. See docs/superpowers/specs/2026-10-02-screenshots-design.md.

The file name is the only record of when a screenshot was taken, so it survives
a crash and re-transcription finds screenshots without a separate list.
"""
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

SCREENSHOT_DIR = "screenshots"
_NAME = re.compile(r"^screenshot_(\d{2})-(\d{2})-(\d{2})(?:_(\d+))?\.png$")


@dataclass(frozen=True)
class Screenshot:
    at_s: int   # whole seconds from the start of the recording
    path: str   # relative to the session folder, forward slashes


def screenshot_file_name(offset_s: float, taken: set[str]) -> str:
    """`screenshot_HH-MM-SS.png`, with `_2`, `_3`... for the same second."""
    h, rest = divmod(int(offset_s), 3600)
    m, s = divmod(rest, 60)
    base = f"screenshot_{h:02d}-{m:02d}-{s:02d}"
    name, n = f"{base}.png", 2
    while name in taken:
        name, n = f"{base}_{n}.png", n + 1
    return name


def find_screenshots(session_dir: Path) -> list[Screenshot]:
    """Screenshots saved in `session_dir`, in the order they were taken."""
    folder = session_dir / SCREENSHOT_DIR
    if not folder.is_dir():
        return []
    found = []
    for p in folder.iterdir():
        m = _NAME.match(p.name)
        if m:
            h, mi, s = (int(g) for g in m.group(1, 2, 3))
            found.append((h * 3600 + mi * 60 + s, int(m.group(4) or 1), p.name))
    found.sort()
    return [Screenshot(at_s=at, path=f"{SCREENSHOT_DIR}/{name}") for at, _, name in found]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_screenshot_watch.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/screenshot_watch.py tests/test_screenshot_watch.py
git commit -m "feat(screenshots): naming scheme and discovery of saved screenshots"
```

---

### Task 2: Clipboard watcher

**Files:**
- Modify: `src/audiologger/screenshot_watch.py` (append)
- Modify: `tests/test_screenshot_watch.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_screenshot_watch.py`:

```python
# --- the clipboard watcher -----------------------------------------------------

import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from audiologger.screenshot_watch import MAX_READ_FAILURES, ClipboardScreenshotWatcher


class FakeClipboard:
    """put() changes the content like a copy would; `locked` makes reads fail."""

    def __init__(self):
        self.seq = 100
        self.content = None
        self.locked = 0

    def put(self, content):
        self.seq += 1
        self.content = content

    def sequence_number(self) -> int:
        return self.seq

    def read_image(self):
        if self.locked:
            self.locked -= 1
            raise OSError("clipboard is open in another program")
        return None if isinstance(self.content, str) else self.content


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def image():
    return Image.new("RGB", (4, 3), "red")


def files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.png")) if folder.exists() else []


@pytest.fixture
def watch(tmp_path):
    """An unstarted watcher whose thread never polls on its own (poll_s=3600):
    tests start it and drive poll_once() by hand."""
    clip, clock, saved = FakeClipboard(), Clock(), []
    w = ClipboardScreenshotWatcher(
        tmp_path, on_saved=lambda *a: saved.append(a), clock=clock, poll_s=3600, clipboard=clip,
    )
    yield SimpleNamespace(w=w, clip=clip, clock=clock, saved=saved, dir=tmp_path / "screenshots")
    w.stop()


def test_what_was_on_the_clipboard_before_the_recording_is_ignored(watch):
    watch.clip.put(image())
    watch.w.start()

    watch.w.poll_once()

    assert files(watch.dir) == []


def test_a_new_image_is_saved_under_its_offset(watch):
    watch.w.start()
    watch.clock.now += 751.6
    watch.clip.put(image())

    watch.w.poll_once()

    assert files(watch.dir) == ["screenshot_00-12-31.png"]
    index, offset, path = watch.saved[0]
    assert (index, round(offset, 1), path.name) == (1, 751.6, "screenshot_00-12-31.png")
    assert Image.open(path).size == (4, 3)


def test_each_change_is_saved_once(watch):
    watch.w.start()
    watch.clip.put(image())

    watch.w.poll_once()
    watch.w.poll_once()

    assert len(files(watch.dir)) == 1


def test_text_on_the_clipboard_is_ignored(watch):
    watch.w.start()
    watch.clip.put("some copied text")

    watch.w.poll_once()

    assert files(watch.dir) == []
    assert watch.saved == []


def test_a_locked_clipboard_is_retried_not_lost(watch):
    watch.w.start()
    watch.clip.put(image())
    watch.clip.locked = 2

    watch.w.poll_once()
    watch.w.poll_once()
    assert files(watch.dir) == []

    watch.w.poll_once()
    assert len(files(watch.dir)) == 1


def test_a_change_that_cannot_be_read_is_given_up_after_twenty_tries(watch):
    watch.w.start()
    watch.clip.put(image())
    watch.clip.locked = 10_000
    for _ in range(MAX_READ_FAILURES):
        watch.w.poll_once()

    watch.clip.locked = 0
    watch.w.poll_once()          # that change is abandoned, not read late
    assert files(watch.dir) == []

    watch.clip.put(image())      # the next change works again
    watch.w.poll_once()
    assert len(files(watch.dir)) == 1


def test_two_screenshots_in_one_second_both_survive(watch):
    watch.w.start()
    for _ in range(2):
        watch.clip.put(image())
        watch.w.poll_once()

    assert files(watch.dir) == ["screenshot_00-00-00.png", "screenshot_00-00-00_2.png"]


def test_a_failing_save_does_not_stop_the_watcher(watch):
    class Unsavable:
        def save(self, *a, **k):
            raise OSError("disk full")

    watch.w.start()
    watch.clip.put(Unsavable())
    watch.w.poll_once()
    watch.clip.put(image())
    watch.w.poll_once()

    assert files(watch.dir) == ["screenshot_00-00-00.png"]
    assert [s[0] for s in watch.saved] == [1]


def test_the_thread_picks_up_changes_by_itself_and_stops(tmp_path):
    clip = FakeClipboard()
    w = ClipboardScreenshotWatcher(tmp_path, poll_s=0.01, clipboard=clip)
    w.start()
    clip.put(image())

    deadline = time.monotonic() + 2
    while not files(tmp_path / "screenshots") and time.monotonic() < deadline:
        time.sleep(0.01)
    w.stop()

    assert len(files(tmp_path / "screenshots")) == 1
    assert not any(t.name == "screenshot-watch" for t in threading.enumerate())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_screenshot_watch.py -v`
Expected: collection error, `ImportError: cannot import name 'MAX_READ_FAILURES'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/audiologger/screenshot_watch.py`:

```python
# A change whose content cannot be read this many polls in a row (5 s at the
# default rate) is given up, so one stuck clipboard cannot pin the watcher.
MAX_READ_FAILURES = 20


class WindowsClipboard:
    """The real clipboard: a change counter and an image reader."""

    def sequence_number(self) -> int:
        """Windows bumps this on every clipboard change; reading it needs no lock."""
        import ctypes
        return int(ctypes.windll.user32.GetClipboardSequenceNumber())

    def read_image(self) -> Any:
        """The clipboard's image, or None when it holds text, files or nothing.
        Raises when another program holds the clipboard open."""
        from PIL import Image, ImageGrab
        content = ImageGrab.grabclipboard()
        return content if isinstance(content, Image.Image) else None


class ClipboardScreenshotWatcher:
    """Saves every image put on the clipboard while a meeting is recorded.

    Only reads the clipboard, never writes it -- dictation puts its result
    there, and the clipboard stays the user's. Every error inside a poll is
    logged and swallowed: a screenshot is never worth disturbing the recording.
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        on_saved: Callable[[int, float, Path], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 0.25,
        clipboard: Any = None,
    ):
        self._dir = session_dir / SCREENSHOT_DIR
        self._on_saved = on_saved
        self._clock = clock
        self._poll_s = poll_s
        self._clipboard = clipboard if clipboard is not None else WindowsClipboard()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._handled = 0
        self._failures = 0
        self._saved = 0
        self._taken: set[str] = set()

    def start(self) -> None:
        self._started_at = self._clock()
        # Whatever is on the clipboard already belongs to before the recording.
        self._handled = self._clipboard.sequence_number()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="screenshot-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def poll_once(self) -> None:
        try:
            seq = self._clipboard.sequence_number()
            if seq == self._handled:
                return
            try:
                image = self._clipboard.read_image()
            except Exception as e:
                # Usually another program holding the clipboard open: keep the
                # change pending and try again next poll.
                self._failures += 1
                if self._failures >= MAX_READ_FAILURES:
                    log.warning("Giving up on clipboard change after %d failed reads: %s",
                                self._failures, e)
                    self._handled, self._failures = seq, 0
                return
            self._handled, self._failures = seq, 0
            if image is not None:
                self._save(image)
        except Exception:
            log.exception("Screenshot watcher poll failed")

    def _save(self, image: Any) -> None:
        offset = self._clock() - self._started_at
        name = screenshot_file_name(offset, self._taken)
        path = self._dir / name
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            image.save(path, "PNG")
        except Exception:
            log.exception("Could not save %s", name)
            return
        self._taken.add(name)
        self._saved += 1
        log.info("Saved %s", name)
        if self._on_saved is not None:
            try:
                self._on_saved(self._saved, offset, path)
            except Exception:
                log.exception("Screenshot callback failed")

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            self.poll_once()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_screenshot_watch.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/screenshot_watch.py tests/test_screenshot_watch.py
git commit -m "feat(screenshots): clipboard watcher that saves new images as PNG"
```

---

### Task 3: Screenshot lines in the Markdown transcript

**Files:**
- Modify: `src/audiologger/transcript_merger.py` (imports; `render_markdown`, currently lines 23–50)
- Modify: `tests/test_transcript_merger.py` (append)

- [ ] **Step 1: Write the tests**

Append to `tests/test_transcript_merger.py`:

```python
# --- screenshots ------------------------------------------------------------------

from audiologger.screenshot_watch import Screenshot

HEADER = dict(
    recorded_at="2026-10-02 10:00:00",
    duration_seconds=900,
    source_label="mic + system (loopback)",
    model_label="WhisperX large-v3",
    warnings=[],
)
SPEECH = [
    Segment(748.0, 752.0, "Hier seht ihr die neue Rezeptstruktur.", "Speaker 1"),
    Segment(755.0, 757.0, "Okay, und wo kommt der Teig rein?", "Me"),
]


def body(md: str) -> list[str]:
    return md.split("---\n\n", 1)[1].splitlines()


def test_without_screenshots_the_transcript_is_exactly_as_before():
    assert render_markdown(SPEECH, **HEADER) == (
        "# Recording 2026-10-02 10:00:00\n"
        "\n"
        "**Duration:** 15:00\n"
        "**Source:** mic + system (loopback)\n"
        "**Model:** WhisperX large-v3\n"
        "\n"
        "---\n"
        "\n"
        "**[00:12:28] Speaker 1:** Hier seht ihr die neue Rezeptstruktur.\n"
        "**[00:12:35] Me:** Okay, und wo kommt der Teig rein?\n"
    )


def test_a_screenshot_sits_between_the_speech_lines_around_it():
    md = render_markdown(SPEECH, **HEADER,
                         screenshots=[Screenshot(751, "screenshots/screenshot_00-12-31.png")])

    assert body(md) == [
        "**[00:12:28] Speaker 1:** Hier seht ihr die neue Rezeptstruktur.",
        "**[00:12:31] Screenshot 1:** ![Screenshot 1](screenshots/screenshot_00-12-31.png)",
        "**[00:12:35] Me:** Okay, und wo kommt der Teig rein?",
    ]


def test_speech_starting_in_the_same_second_comes_first():
    md = render_markdown([Segment(751.0, 753.0, "Genau hier.", "Me")], **HEADER,
                         screenshots=[Screenshot(751, "screenshots/screenshot_00-12-31.png")])

    assert body(md)[0].startswith("**[00:12:31] Me:**")
    assert body(md)[1].startswith("**[00:12:31] Screenshot 1:**")


def test_screenshots_before_the_first_and_after_the_last_word_are_kept_in_order():
    md = render_markdown(SPEECH, **HEADER, screenshots=[
        Screenshot(800, "screenshots/screenshot_00-13-20.png"),
        Screenshot(10, "screenshots/screenshot_00-00-10.png"),
    ])

    assert body(md)[0] == (
        "**[00:00:10] Screenshot 1:** ![Screenshot 1](screenshots/screenshot_00-00-10.png)")
    assert body(md)[-1] == (
        "**[00:13:20] Screenshot 2:** ![Screenshot 2](screenshots/screenshot_00-13-20.png)")


def test_the_header_counts_the_screenshots():
    md = render_markdown(SPEECH, **HEADER, screenshots=[
        Screenshot(10, "screenshots/screenshot_00-00-10.png"),
        Screenshot(800, "screenshots/screenshot_00-13-20.png"),
    ])

    assert "**Model:** WhisperX large-v3\n**Screenshots:** 2\n" in md
```

- [ ] **Step 2: Run them**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transcript_merger.py -v`
Expected: `test_without_screenshots_the_transcript_is_exactly_as_before` **passes** — it pins today's output as a regression guard. The four screenshot tests fail with `TypeError: render_markdown() got an unexpected keyword argument 'screenshots'`.

- [ ] **Step 3: Write the implementation**

In `src/audiologger/transcript_merger.py`, change the imports at the top to:

```python
"""Merge mic + system segments into a Markdown transcript."""
from typing import Iterable, Sequence

from audiologger.screenshot_watch import Screenshot
from audiologger.segment import Segment
```

Replace the whole `render_markdown` function with:

```python
def render_markdown(
    segments: list[Segment],
    *,
    recorded_at: str,
    duration_seconds: int,
    source_label: str,
    model_label: str,
    warnings: list[str],
    screenshots: Sequence[Screenshot] = (),
) -> str:
    """Render the final transcript Markdown matching the spec format.

    Screenshots are placed among the speech lines by time; a speech line that
    starts in the same second comes first. Without screenshots the output is
    unchanged.
    """
    duration_str = format_timestamp(duration_seconds)
    # Drop the leading "00:" for short recordings — keep it consistent: spec
    # showed "47:21" for sub-hour. Strip leading "00:" only if hours == 0.
    if duration_str.startswith("00:"):
        duration_str = duration_str[3:]

    shots = sorted(screenshots, key=lambda s: s.at_s)
    lines = [
        f"# Recording {recorded_at}",
        "",
        f"**Duration:** {duration_str}",
        f"**Source:** {source_label}",
        f"**Model:** {model_label}",
    ]
    if shots:
        lines.append(f"**Screenshots:** {len(shots)}")
    for w in warnings:
        lines.append(f"**Warning:** {w}")
    lines.extend(["", "---", ""])
    next_shot = 0
    for seg in segments:
        while next_shot < len(shots) and shots[next_shot].at_s < seg.start:
            lines.append(_screenshot_line(next_shot + 1, shots[next_shot]))
            next_shot += 1
        ts = format_timestamp(seg.start)
        lines.append(f"**[{ts}] {seg.speaker}:** {seg.text}")
    for i in range(next_shot, len(shots)):
        lines.append(_screenshot_line(i + 1, shots[i]))
    return "\n".join(lines) + "\n"


def _screenshot_line(number: int, shot: Screenshot) -> str:
    return (f"**[{format_timestamp(shot.at_s)}] Screenshot {number}:** "
            f"![Screenshot {number}]({shot.path})")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transcript_merger.py -v`
Expected: all passed (the existing tests and the five new ones)

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/transcript_merger.py tests/test_transcript_merger.py
git commit -m "feat(screenshots): interleave screenshot lines in the Markdown transcript"
```

---

### Task 4: The worker picks screenshots up

**Files:**
- Modify: `src/audiologger/transcribe_worker.py` (imports around line 34; `_process_meeting_session`, the `render_markdown(` call and `raw = {` dict near the end of the function)
- Create: `tests/test_worker_screenshots.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker_screenshots.py`:

```python
"""The worker finds screenshots in the session folder by their file names."""
import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from audiologger.segment import Segment
from audiologger.transcribe_worker import TranscriptionResult, _process_meeting_session


class OneLinePipeline:
    """One short segment per track; no model involved."""

    model_size = "large-v3"
    diarization_enabled = False
    language = "de"

    def transcribe(self, audio_path, *, diarize, model_size=None, align=True, language=None):
        return TranscriptionResult(
            segments=[Segment(0.5, 1.0, f"aus {Path(audio_path).name}", "Others")],
            language="de",
        )


def make_session(tmp_path: Path) -> Path:
    d = tmp_path / "2026-10-02_10-00-00"
    d.mkdir()
    for name in ("mic.wav", "system.wav"):
        with wave.open(str(d / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(np.zeros(48000, dtype=np.int16).tobytes())
    return d


def test_screenshots_on_disk_land_in_both_transcript_files(tmp_path):
    session = make_session(tmp_path)
    (session / "screenshots").mkdir()
    Image.new("RGB", (2, 2)).save(session / "screenshots" / "screenshot_00-00-00.png")

    _process_meeting_session(session, OneLinePipeline())

    md = (session / "transcript.md").read_text(encoding="utf-8")
    assert "**Screenshots:** 1" in md
    assert ("**[00:00:00] Screenshot 1:** "
            "![Screenshot 1](screenshots/screenshot_00-00-00.png)") in md
    raw = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    assert raw["screenshots"] == [{"at_s": 0, "path": "screenshots/screenshot_00-00-00.png"}]


def test_a_recording_without_screenshots_lists_none(tmp_path):
    session = make_session(tmp_path)

    _process_meeting_session(session, OneLinePipeline())

    raw = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    assert raw["screenshots"] == []
    assert "Screenshots" not in (session / "transcript.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_worker_screenshots.py -v`
Expected: both FAIL — the first on the missing `**Screenshots:** 1`, the second with `KeyError: 'screenshots'`

- [ ] **Step 3: Write the implementation**

In `src/audiologger/transcribe_worker.py`, add next to the other `audiologger` imports:

```python
from audiologger.screenshot_watch import find_screenshots
```

In `_process_meeting_session`, replace the block from `md = render_markdown(` through the end of the `raw = {...}` dict with:

```python
    screenshots = find_screenshots(session_dir)
    md = render_markdown(
        merged,
        recorded_at=recorded_at_str,
        duration_seconds=int(duration),
        source_label=_source_label(session_dir, mic_wav.exists(), sys_wav.exists()),
        model_label=model_label,
        warnings=warnings,
        screenshots=screenshots,
    )
    (session_dir / "transcript.md").write_text(md, encoding="utf-8")

    raw = {
        "mic_segments": [asdict(s) for s in mic_segments],
        "system_segments": [asdict(s) for s in sys_segments],
        "merged": [asdict(s) for s in merged],
        "warnings": warnings,
        "screenshots": [asdict(s) for s in screenshots],
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_worker_screenshots.py tests/test_transcribe_language.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/transcribe_worker.py tests/test_worker_screenshots.py
git commit -m "feat(screenshots): worker embeds saved screenshots in transcript.md and .json"
```

---

### Task 5: The controller runs a watcher for meetings

**Files:**
- Modify: `src/audiologger/controller.py` (`__init__` lines 38–57, `_start` after `capture.start()` at line 111, `_stop` start at line 117)
- Modify: `tests/test_controller.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_controller.py`:

```python
# --- screenshot watcher ------------------------------------------------------------

class FakeWatcher:
    def __init__(self, session_dir: Path, fail_on_start: bool = False):
        self.session_dir = session_dir
        self.started = False
        self.stopped = False
        self._fail = fail_on_start

    def start(self) -> None:
        if self._fail:
            raise OSError("clipboard unavailable")
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def make_controller(cfg, factory):
    return RecordingController(
        config=cfg,
        capture_factory=FakeCapture,
        mix_fn=MagicMock(),
        enqueue_fn=MagicMock(),
        clock=lambda: datetime(2026, 5, 18, 14, 32, 15),
        screenshot_watcher_factory=factory,
    )


def test_a_meeting_recording_watches_for_screenshots(cfg):
    watchers = []
    c = make_controller(cfg, lambda d: watchers.append(FakeWatcher(d)) or watchers[-1])

    c.toggle()
    [w] = watchers
    assert w.started
    assert w.session_dir == cfg.output_dir / "2026-05-18_14-32-15"

    c.toggle()
    assert w.stopped


def test_dictation_does_not_watch_for_screenshots(cfg):
    watchers = []
    c = make_controller(cfg, lambda d: watchers.append(FakeWatcher(d)) or watchers[-1])

    c.toggle("dictation")
    c.toggle("dictation")

    assert watchers == []


def test_a_watcher_that_cannot_start_does_not_stop_the_recording(cfg):
    c = make_controller(cfg, lambda d: FakeWatcher(d, fail_on_start=True))

    c.toggle()
    assert c.state is RecordingState.RECORDING

    c.toggle()
    assert c.state is RecordingState.IDLE
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_controller.py -v`
Expected: the three new tests fail with `TypeError: RecordingController.__init__() got an unexpected keyword argument 'screenshot_watcher_factory'`

- [ ] **Step 3: Write the implementation**

In `src/audiologger/controller.py`, after the `CaptureLike` protocol add:

```python
class WatcherLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


ScreenshotWatcherFactory = Callable[[Path], WatcherLike]
"""(session_dir) -> a watcher that saves screenshots taken during the recording."""
```

Add the parameter to `__init__` after `notify_fn`:

```python
        notify_fn: NotifyFn | None = None,
        screenshot_watcher_factory: ScreenshotWatcherFactory | None = None,
    ):
```

and store it next to the other attributes:

```python
        self._screenshot_watcher_factory = screenshot_watcher_factory
        self._current_watcher: WatcherLike | None = None
```

In `_start`, directly after `capture.start()`:

```python
        if mode == "meeting" and self._screenshot_watcher_factory is not None:
            self._current_watcher = self._start_watcher(session)
```

In `_stop`, directly after the `RuntimeError` check:

```python
        watcher, self._current_watcher = self._current_watcher, None
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:
                log.exception("Screenshot watcher did not stop cleanly")
```

Add the helper method after `_start`:

```python
    def _start_watcher(self, session: Path) -> WatcherLike | None:
        """Screenshots are a bonus: if the watcher cannot start, record anyway."""
        try:
            watcher = self._screenshot_watcher_factory(session)
            watcher.start()
            return watcher
        except Exception:
            log.exception("Screenshot watcher could not start for %s", session.name)
            return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_controller.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/controller.py tests/test_controller.py
git commit -m "feat(screenshots): controller runs a screenshot watcher during meetings"
```

---

### Task 6: Tray wiring and toast

**Files:**
- Modify: `src/audiologger/tray_app.py` (imports; `RecordingController(...)` in `__init__` around line 37; new methods next to `_notify_capture_warnings`)
- Modify: `tests/test_tray_app.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tray_app.py`:

```python
# --- screenshots ---------------------------------------------------------------------

def test_screenshot_toast_names_the_screenshot_and_the_moment(app):
    app._notify_screenshot_saved(3, 751.6, Path("screenshots/screenshot_00-12-31.png"))

    assert app.notifier.calls == [
        {"title": "Screenshot 3 saved", "message": "at 12:31", "launch": "", "actions": []}
    ]


def test_screenshot_toast_shows_hours_in_long_recordings(app):
    app._notify_screenshot_saved(1, 3725.0, Path("screenshots/screenshot_01-02-05.png"))

    assert app.notifier.calls[0]["message"] == "at 01:02:05"


def test_meetings_get_a_clipboard_screenshot_watcher(tmp_path, monkeypatch):
    import audiologger.tray_app as ta
    from audiologger.screenshot_watch import ClipboardScreenshotWatcher

    cfg = Config(output_dir=tmp_path / "recs")
    monkeypatch.setattr(ta, "config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(ta, "load_config", lambda _p: cfg)
    monkeypatch.setattr(ta, "appdata_dir", lambda: tmp_path / "appdata")

    app = ta.TrayApp()

    watcher = app.controller._screenshot_watcher_factory(tmp_path)
    assert isinstance(watcher, ClipboardScreenshotWatcher)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tray_app.py -v`
Expected: the three new tests fail with `AttributeError` (`_notify_screenshot_saved`, `_screenshot_watcher_factory`)

- [ ] **Step 3: Write the implementation**

In `src/audiologger/tray_app.py` add to the imports (add `from pathlib import Path` too if it is not imported yet):

```python
from audiologger.screenshot_watch import ClipboardScreenshotWatcher
from audiologger.transcript_merger import format_timestamp
```

Pass the factory to the controller in `__init__`:

```python
        self.controller = RecordingController(
            config=self.cfg,
            capture_factory=AudioCaptureThread,
            mix_fn=mix_to_file,
            enqueue_fn=self._on_recording_finished,
            notify_fn=self._notify_capture_warnings,
            screenshot_watcher_factory=self._make_screenshot_watcher,
        )
```

Add the two methods next to `_notify_capture_warnings`:

```python
    def _make_screenshot_watcher(self, session_dir: Path) -> ClipboardScreenshotWatcher:
        return ClipboardScreenshotWatcher(session_dir, on_saved=self._notify_screenshot_saved)

    def _notify_screenshot_saved(self, index: int, offset_s: float, path: Path) -> None:
        ts = format_timestamp(offset_s)
        if ts.startswith("00:"):
            ts = ts[3:]
        self.notifier.notify(f"Screenshot {index} saved", f"at {ts}")
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all passed (177 before this feature + 27 new = 204)

- [ ] **Step 5: Commit**

```bash
git add src/audiologger/tray_app.py tests/test_tray_app.py
git commit -m "feat(screenshots): tray wires the watcher and shows a toast per screenshot"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md` ("What it does" list; "Output layout" block)
- Modify: `docs/MANUAL_TEST_PLAN.md` (append TC-10)

- [ ] **Step 1: README — feature bullet**

In `README.md`, under "## What it does", after the line `- Merged chronological Markdown transcript saved next to the audio.` add:

```markdown
- Screenshots during meetings: take them as usual with `Win+Shift+S` or `Alt+Print`.
  Every image put on the clipboard while a meeting is recorded is saved next to the
  audio and embedded in the transcript at the moment it was taken.
```

- [ ] **Step 2: README — output layout**

In the "## Output layout" code block, after the `transcript.json` line add:

```
    screenshots/      ← images copied to the clipboard during the meeting
```

- [ ] **Step 3: Manual test TC-10**

Append to `docs/MANUAL_TEST_PLAN.md`:

```markdown

### TC-10: Screenshots during a meeting
1. Copy some text to the clipboard, then start a meeting recording (`Ctrl+Alt+R`).
2. After ~10 s, take a region snip with `Win+Shift+S`.
3. **Expected:** A "Screenshot 1 saved" toast appears within a second, naming the moment.
4. After a few more seconds, press `Alt+Print` with any window focused.
5. **Expected:** A "Screenshot 2 saved" toast.
6. Copy some text, stop the recording, wait for the transcript.
7. **Expected:** `screenshots/` in the session folder holds two PNGs named after their
   offsets; `transcript.md` has `**Screenshots:** 2` in the header and two
   `Screenshot N` lines at the right places, rendering as images in a Markdown preview.
8. **Expected:** Nothing was saved for the text copied before the recording or after it.
9. Start a dictation (`Ctrl+Alt+D`), take a `Win+Shift+S` snip, stop.
10. **Expected:** No screenshot toast and no `screenshots/` folder for the dictation.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/MANUAL_TEST_PLAN.md
git commit -m "docs: screenshots during meetings in README and manual test plan"
```

---

### Task 8: Acceptance on real hardware and release

- [ ] **Step 1: Tell the user the next step overwrites their clipboard**

The check below puts a test image on the real clipboard. Say so before running it.

- [ ] **Step 2: Real clipboard, real watcher**

Write this script with the Write tool to `%TEMP%\audiologger-check\watcher_check.py` (outside the repo):

```python
import subprocess, sys, time
sys.path.insert(0, "src")
from pathlib import Path
from PIL import Image
from audiologger.screenshot_watch import ClipboardScreenshotWatcher, find_screenshots

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
saved = []
w = ClipboardScreenshotWatcher(out, on_saved=lambda *a: saved.append(a))
w.start()
time.sleep(1.0)
subprocess.run([
    "powershell", "-STA", "-NoProfile", "-Command",
    "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
    "$b = New-Object System.Drawing.Bitmap 320,200;"
    "[System.Drawing.Graphics]::FromImage($b).Clear([System.Drawing.Color]::Orange);"
    "[System.Windows.Forms.Clipboard]::SetImage($b)",
], check=True)
time.sleep(1.5)
w.stop()
shots = find_screenshots(out)
print("callbacks:", [(i, round(o, 1), p.name) for i, o, p in saved])
print("found:", shots)
print("size:", Image.open(out / shots[0].path).size if shots else None)
```

Run (bash, from the repo root): `.venv/Scripts/python.exe "$TEMP/audiologger-check/watcher_check.py" "$TEMP/audiologger-check/out"`
Expected: one callback around 1.x s, one `Screenshot(at_s=1, ...)`, size `(320, 200)`

- [ ] **Step 3: Push**

```bash
git push origin main
```

- [ ] **Step 4: Hand over to the user**

Ask the user to restart AudioLogger (tray → Quit, start again — the tray loads the capture code at start), then run TC-10 steps 1–8 once with a real `Win+Shift+S` snip. The Snipping Tool path cannot be triggered programmatically, so this is the proof.
