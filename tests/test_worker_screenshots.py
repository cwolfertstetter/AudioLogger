"""The worker finds screenshots in the session folder by their file names."""
import json
import logging
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


def test_a_failing_screenshot_lookup_never_costs_the_transcript(tmp_path, monkeypatch, caplog):
    """Screenshots are a bonus: if finding them fails (unreadable folder, a
    permission error), the transcript is written without them."""
    session = make_session(tmp_path)

    def unreadable(_session_dir):
        raise PermissionError("screenshots folder is not readable")

    monkeypatch.setattr("audiologger.transcribe_worker.find_screenshots", unreadable)

    with caplog.at_level(logging.ERROR, logger="transcribe_worker"):
        _process_meeting_session(session, OneLinePipeline())

    md = (session / "transcript.md").read_text(encoding="utf-8")
    assert "aus system.wav" in md
    assert "Screenshots" not in md
    raw = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    assert raw["screenshots"] == []
    assert any("screenshot" in r.getMessage().lower() and r.exc_info for r in caplog.records)


def test_a_recording_without_screenshots_lists_none(tmp_path):
    session = make_session(tmp_path)

    _process_meeting_session(session, OneLinePipeline())

    raw = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    assert raw["screenshots"] == []
    assert "Screenshots" not in (session / "transcript.md").read_text(encoding="utf-8")
