from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audiologger.config import Config
from audiologger.controller import RecordingController, RecordingState
from audiologger.paths import MARKER_FILENAME, MODE_FILENAME


class FakeCapture:
    def __init__(self, session_dir: Path, sample_rate: int, source: str, app_names, mic_only: bool = False):
        self.session_dir = session_dir
        self.started = False
        self.stopped = False
        self.warnings: list[str] = []
        self.mic_only = mic_only

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
    controller.toggle()  # IDLE -> RECORDING
    controller._state = RecordingState.STOPPING
    controller.toggle()
    assert controller.state is RecordingState.STOPPING


def test_session_dir_name_format(controller, cfg):
    controller.toggle()
    sess_dirs = list(cfg.output_dir.iterdir())
    assert len(sess_dirs) == 1
    assert sess_dirs[0].name == "2026-05-18_14-32-15"


# --- C1: capture warnings written to capture_warnings.txt ----------------

class FakeCaptureWithWarnings(FakeCapture):
    """FakeCapture that populates warnings after stop()."""

    def stop(self) -> None:
        self.stopped = True
        self.warnings = ["Mikrofon nicht verfügbar.",
                         "App-Filter nicht verfügbar — gesamtes System-Audio aufgenommen."]


def test_capture_warnings_written_to_file(cfg, tmp_path):
    """When capture.warnings is non-empty, _stop() writes capture_warnings.txt."""
    controller = RecordingController(
        config=cfg,
        capture_factory=FakeCaptureWithWarnings,
        mix_fn=MagicMock(),
        enqueue_fn=MagicMock(),
        clock=lambda: datetime(2026, 5, 18, 14, 32, 15),
    )
    controller.toggle()  # start
    session = cfg.output_dir / "2026-05-18_14-32-15"
    controller.toggle()  # stop
    warnings_file = session / "capture_warnings.txt"
    assert warnings_file.exists(), "capture_warnings.txt should be written"
    content = warnings_file.read_text(encoding="utf-8")
    assert "Mikrofon nicht verfügbar." in content
    assert "App-Filter nicht verfügbar" in content


def test_no_capture_warnings_file_when_no_warnings(controller, cfg):
    """When capture.warnings is empty, capture_warnings.txt must NOT be created."""
    controller.toggle()  # start
    session = cfg.output_dir / "2026-05-18_14-32-15"
    controller.toggle()  # stop
    assert not (session / "capture_warnings.txt").exists()


# --- C2: mix_fn error does not lock state in STOPPING --------------------

def test_mix_error_leaves_controller_idle(cfg):
    """If mix_fn raises, the controller must still be IDLE afterward."""
    failing_mix = MagicMock(side_effect=OSError("disk full"))
    controller = RecordingController(
        config=cfg,
        capture_factory=FakeCapture,
        mix_fn=failing_mix,
        enqueue_fn=MagicMock(),
        clock=lambda: datetime(2026, 5, 18, 14, 32, 15),
    )
    controller.toggle()  # start
    controller.toggle()  # stop — mix fails
    assert controller.state is RecordingState.IDLE


def test_mix_error_still_removes_marker(cfg):
    """If mix_fn raises, the marker file must still be cleaned up."""
    failing_mix = MagicMock(side_effect=OSError("disk full"))
    controller = RecordingController(
        config=cfg,
        capture_factory=FakeCapture,
        mix_fn=failing_mix,
        enqueue_fn=MagicMock(),
        clock=lambda: datetime(2026, 5, 18, 14, 32, 15),
    )
    controller.toggle()  # start
    session = cfg.output_dir / "2026-05-18_14-32-15"
    assert (session / MARKER_FILENAME).exists()
    controller.toggle()  # stop — mix fails
    assert not (session / MARKER_FILENAME).exists()


# --- Dictation mode tests -------------------------------------------------

def test_dictation_mode_writes_mode_marker(controller, cfg):
    """toggle(mode='dictation') writes mode.txt containing 'dictation'."""
    controller.toggle(mode="dictation")
    session = cfg.output_dir / "2026-05-18_14-32-15"
    mode_file = session / MODE_FILENAME
    assert mode_file.exists(), "mode.txt should be created"
    assert mode_file.read_text(encoding="utf-8") == "dictation"


def test_meeting_mode_writes_mode_marker(controller, cfg):
    """Default toggle() writes mode.txt containing 'meeting'."""
    controller.toggle()
    session = cfg.output_dir / "2026-05-18_14-32-15"
    mode_file = session / MODE_FILENAME
    assert mode_file.exists(), "mode.txt should be created"
    assert mode_file.read_text(encoding="utf-8") == "meeting"


def test_dictation_mode_passes_mic_only_true(cfg):
    """Dictation mode passes mic_only=True to capture factory; meeting passes False."""
    times = [datetime(2026, 5, 18, 14, 32, 15), datetime(2026, 5, 18, 14, 33, 0)]
    idx = 0

    def advancing_clock():
        nonlocal idx
        t = times[idx % len(times)]
        idx += 1
        return t

    ctrl = RecordingController(
        config=cfg,
        capture_factory=FakeCapture,
        mix_fn=MagicMock(),
        enqueue_fn=MagicMock(),
        clock=advancing_clock,
    )
    ctrl.toggle(mode="dictation")
    assert ctrl._current_capture.mic_only is True

    ctrl.toggle(mode="dictation")  # stop

    ctrl.toggle(mode="meeting")
    assert ctrl._current_capture.mic_only is False


def test_hotkey_of_different_mode_during_recording_ignored(controller, cfg):
    """Toggling a different mode during recording is silently ignored."""
    controller.toggle(mode="meeting")  # start meeting recording
    assert controller.state is RecordingState.RECORDING

    controller.toggle(mode="dictation")  # should be ignored
    assert controller.state is RecordingState.RECORDING
    assert controller._current_mode == "meeting"
    # No second session should have been created — still the same one
    sessions = list(cfg.output_dir.iterdir())
    assert len(sessions) == 1
