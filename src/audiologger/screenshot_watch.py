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
