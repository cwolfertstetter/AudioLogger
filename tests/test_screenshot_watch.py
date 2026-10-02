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


def test_suffixes_sort_as_numbers_not_as_text(tmp_path):
    _touch(tmp_path / "screenshots", "screenshot_00-00-07_10.png", "screenshot_00-00-07_2.png")

    assert [s.path for s in find_screenshots(tmp_path)] == [
        "screenshots/screenshot_00-00-07_2.png",
        "screenshots/screenshot_00-00-07_10.png",
    ]


def test_other_files_in_the_folder_are_ignored(tmp_path):
    _touch(tmp_path / "screenshots", "notes.txt", "screenshot.bmp", "screenshot_00-00-05.png")

    assert [s.at_s for s in find_screenshots(tmp_path)] == [5]


def test_a_recording_without_screenshots_has_none(tmp_path):
    assert find_screenshots(tmp_path) == []


# --- the clipboard watcher -----------------------------------------------------

import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from audiologger.screenshot_watch import MAX_READ_FAILURES, ClipboardScreenshotWatcher, choose_image_format


class FakeClipboard:
    """put() changes the content like a copy would; `locked` makes reads fail."""

    def __init__(self):
        self.seq = 100
        self.content = None
        self.locked = 0
        self.bumps_during_read = 0

    def put(self, content):
        self.content = content
        self.seq += 1

    def sequence_number(self) -> int:
        return self.seq

    def read_image(self):
        if self.locked:
            self.locked -= 1
            raise OSError("clipboard is open in another program")
        content = None if isinstance(self.content, str) else self.content
        if self.bumps_during_read:
            self.bumps_during_read -= 1
            self.seq += 1   # the writer adds another format while we read
        return content


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


def test_a_save_that_fails_halfway_leaves_no_file_behind(watch):
    class HalfWritten:
        def save(self, path, fmt):
            Path(path).write_bytes(b"\x89PNG truncated")
            raise OSError("disk full")

    watch.w.start()
    watch.clip.put(HalfWritten())
    watch.w.poll_once()

    assert list(watch.dir.iterdir()) == []


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


def test_a_copy_still_being_written_is_read_again_not_saved_twice(watch):
    watch.w.start()
    watch.clip.put(image())
    watch.clip.bumps_during_read = 1

    watch.w.poll_once()
    assert files(watch.dir) == []

    watch.w.poll_once()
    watch.w.poll_once()
    assert files(watch.dir) == ["screenshot_00-00-00.png"]


def test_the_moment_is_when_the_copy_appeared_not_when_it_could_be_read(watch):
    watch.w.start()
    watch.clock.now += 60
    watch.clip.put(image())
    watch.clip.locked = 3
    for _ in range(4):
        watch.w.poll_once()
        watch.clock.now += 1

    assert files(watch.dir) == ["screenshot_00-01-00.png"]


def test_failures_are_counted_per_change(watch):
    watch.w.start()
    watch.clip.put(image())
    watch.clip.locked = 15
    for _ in range(15):
        watch.w.poll_once()

    watch.clip.put(image())                 # a new copy, also hard to read
    watch.clip.locked = MAX_READ_FAILURES - 1
    for _ in range(MAX_READ_FAILURES - 1):
        watch.w.poll_once()
    watch.w.poll_once()                     # its last allowed try succeeds

    assert len(files(watch.dir)) == 1


def test_a_failing_callback_does_not_stop_the_watcher(tmp_path):
    clip = FakeClipboard()

    def explode(*_):
        raise RuntimeError("toast failed")

    w = ClipboardScreenshotWatcher(tmp_path, on_saved=explode, poll_s=3600, clipboard=clip)
    w.start()
    for _ in range(2):
        clip.put(image())
        w.poll_once()
    w.stop()

    assert len(files(tmp_path / "screenshots")) == 2


def test_stop_without_start_is_harmless(tmp_path):
    ClipboardScreenshotWatcher(tmp_path, clipboard=FakeClipboard()).stop()


def test_a_watcher_is_single_use(watch):
    watch.w.start()
    with pytest.raises(RuntimeError):
        watch.w.start()


PNG = 0xC0F1  # stands in for the id Windows registers for "PNG"


def test_the_snipping_tools_png_is_preferred():
    assert choose_image_format({PNG, 17, 8}, PNG) == PNG


def test_an_alt_print_bitmap_is_read_through_its_synthesised_dib():
    assert choose_image_format({2, 17, 8}, PNG) == 17
    assert choose_image_format({8}, PNG) == 8


def test_a_copy_that_also_offers_text_is_not_a_screenshot():
    """Excel cells come with a picture of themselves; they are content."""
    assert choose_image_format({13, PNG, 8}, PNG) is None


def test_no_image_format_means_nothing_to_save():
    assert choose_image_format({13}, PNG) is None
    assert choose_image_format(set(), PNG) is None
