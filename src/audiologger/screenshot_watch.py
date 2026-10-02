"""Screenshots taken during a meeting recording.

The user captures with Win+Shift+S or Alt+Print, both of which put an image on
the clipboard. While a meeting is recorded, every new image that appears there
is saved next to the recording, and the transcript embeds it at the moment it
was taken. See docs/superpowers/specs/2026-10-02-screenshots-design.md.

The file name is the only record of when a screenshot was taken, so it survives
a crash and re-transcription finds screenshots without a separate list.
"""
import logging
import os
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
    """Screenshots saved for the session in `session_dir`, in the order they were taken."""
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
        # Written under a temporary name first: the file name is the only record
        # of a screenshot, so a half-written PNG must never carry a final name.
        partial = path.with_name(name + ".part")
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            image.save(partial, "PNG")
            os.replace(partial, path)
        except Exception:
            log.exception("Could not save %s", name)
            partial.unlink(missing_ok=True)
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
