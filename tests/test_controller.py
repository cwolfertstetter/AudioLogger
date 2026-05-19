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
    controller.toggle()  # IDLE -> RECORDING
    controller._state = RecordingState.STOPPING
    controller.toggle()
    assert controller.state is RecordingState.STOPPING


def test_session_dir_name_format(controller, cfg):
    controller.toggle()
    sess_dirs = list(cfg.output_dir.iterdir())
    assert len(sess_dirs) == 1
    assert sess_dirs[0].name == "2026-05-18_14-32-15"
