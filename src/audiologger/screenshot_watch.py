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
from collections.abc import Container
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

# Win32 clipboard formats.
_CF_DIB = 8
_CF_UNICODETEXT = 13
_CF_DIBV5 = 17


def choose_image_format(available: Container[int], png_format: int) -> int | None:
    """The clipboard format to read a screenshot from, or None to skip the copy.

    A screenshot comes without text. A copy that also offers text -- Excel
    cells, a Word selection -- is somebody copying content, even though Office
    adds a picture of it, and is skipped. Among image formats PNG wins (it keeps
    real transparency), then plain DIB, which Windows always provides, also for
    the bitmap Alt+Print stores. DIBV5 comes last: Pillow reads its usually
    all-zero alpha channel literally and the screenshot would come out fully
    transparent.
    """
    if _CF_UNICODETEXT in available:
        return None
    for fmt in (png_format, _CF_DIB, _CF_DIBV5):
        if fmt in available:
            return fmt
    return None


class WindowsClipboard:
    """The real clipboard, read through the Win32 API with ctypes.

    Deliberately not Pillow's ImageGrab.grabclipboard(): when another program
    holds the clipboard it sleeps 500 ms and retries with the GIL held. That
    stalls every Python thread, including the soundcard loopback loops, whose
    ~10 ms WASAPI buffers then overflow and drop the other side of the call.
    ctypes releases the GIL around each Win32 call, so a busy clipboard costs
    the audio nothing; and the format check never opens the clipboard for text
    copies at all.
    """

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        # Private library instances, so these signatures cannot clash with
        # other modules that configure ctypes.windll.user32 differently.
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
        user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterClipboardFormatW.restype = wintypes.UINT
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.CloseClipboard.restype = wintypes.BOOL
        # Handles are pointer-sized; the default int restype would truncate them.
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = wintypes.HANDLE
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalSize.restype = ctypes.c_size_t
        self._ctypes = ctypes
        self._user32 = user32
        self._kernel32 = kernel32
        self._png = user32.RegisterClipboardFormatW("PNG")

    def sequence_number(self) -> int:
        """Windows bumps this on every clipboard change; reading it needs no lock."""
        return int(self._user32.GetClipboardSequenceNumber())

    def read_image(self) -> Any:
        """The clipboard's screenshot, or None when it holds anything else.
        Raises when another program holds the clipboard open."""
        user32, kernel32 = self._user32, self._kernel32
        candidates = (_CF_UNICODETEXT, self._png, _CF_DIBV5, _CF_DIB)
        fmt = choose_image_format(
            {f for f in candidates if user32.IsClipboardFormatAvailable(f)}, self._png
        )
        if fmt is None:
            return None
        if not user32.OpenClipboard(None):
            raise OSError(
                f"clipboard is open in another program (error {self._ctypes.get_last_error()})"
            )
        try:
            handle = user32.GetClipboardData(fmt)
            if not handle:
                # Replaced since the format check (the counter re-check will
                # notice), or the owner failed to render it: retry either way.
                raise OSError(
                    f"clipboard data unavailable (error {self._ctypes.get_last_error()})"
                )
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                raise OSError(
                    f"could not lock the clipboard data (error {self._ctypes.get_last_error()})"
                )
            try:
                data = self._ctypes.string_at(pointer, kernel32.GlobalSize(handle))
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()

        import io
        from PIL import BmpImagePlugin, PngImagePlugin

        buffer = io.BytesIO(data)
        if fmt == self._png:
            image = PngImagePlugin.PngImageFile(buffer)
        else:
            image = BmpImagePlugin.DibImageFile(buffer)
        image.load()  # decode now, so a bad image is a failed read, not a failed save
        return image


class ClipboardScreenshotWatcher:
    """Saves every image put on the clipboard while a meeting is recorded.

    Only reads the clipboard, never writes it -- dictation puts its result
    there, and the clipboard stays the user's. Every error inside a poll is
    logged and swallowed: a screenshot is never worth disturbing the recording.
    Single-use: one watcher per recording.
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
        self._pending = -1          # the change being read while reads fail
        self._pending_since = 0.0   # when that change was first noticed
        self._failures = 0
        self._saved = 0
        self._taken: set[str] = set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("a ClipboardScreenshotWatcher can only be started once")
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

    def poll_once(self) -> None:
        try:
            seq = self._clipboard.sequence_number()
            if seq == self._handled:
                return
            if seq != self._pending:
                # A new change: its moment is now, and earlier failures
                # belonged to a different one.
                self._pending, self._pending_since, self._failures = seq, self._clock(), 0
            try:
                image = self._clipboard.read_image()
            except Exception:
                # Usually another program holding the clipboard open: keep the
                # change pending and try again next poll.
                self._failures += 1
                if self._failures >= MAX_READ_FAILURES:
                    log.warning("Giving up on clipboard change after %d failed reads",
                                self._failures, exc_info=True)
                    self._handled = seq
                return
            if self._clipboard.sequence_number() != seq:
                # The writer was still adding formats while we read; read the
                # settled content next poll rather than saving the copy twice.
                return
            self._handled = seq
            if image is not None:
                self._save(image, self._pending_since - self._started_at)
        except Exception:
            log.exception("Screenshot watcher poll failed")

    def _save(self, image: Any, offset: float) -> None:
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
