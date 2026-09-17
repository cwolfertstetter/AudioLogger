"""Tray-side toast building. Constructed via __new__ so we never touch the
real config/appdata directories that TrayApp.__init__ reaches for."""
from pathlib import Path

import pytest

from audiologger.config import Config
from audiologger.tray_app import TrayApp


class RecordingNotifier:
    """Stands in for Notifier and records what would have been shown."""

    def __init__(self):
        self.calls: list[dict] = []

    def notify(self, title, message, *, launch="", actions=None):
        self.calls.append(
            {"title": title, "message": message, "launch": launch,
             "actions": list(actions or [])}
        )


@pytest.fixture
def app(tmp_path):
    a = TrayApp.__new__(TrayApp)
    a.cfg = Config(output_dir=tmp_path / "recs")
    a.notifier = RecordingNotifier()
    return a


def test_capture_warning_toast_shows_the_warning_text(app, tmp_path):
    """The dead-mic warning must appear in the toast body, not only on disk."""
    session = tmp_path / "recs" / "2026-09-17_09-00-00"
    session.mkdir(parents=True)

    app._notify_capture_warnings(session, ["Microphone recording aborted."])

    assert len(app.notifier.calls) == 1
    call = app.notifier.calls[0]
    assert "Microphone recording aborted." in call["message"]


def test_capture_warning_toast_opens_the_session_folder(app, tmp_path):
    """Clicking the toast, or its button, lands the user in the session folder."""
    session = tmp_path / "recs" / "2026-09-17_09-00-00"
    session.mkdir(parents=True)

    app._notify_capture_warnings(session, ["Microphone recording aborted."])

    call = app.notifier.calls[0]
    folder_uri = session.resolve().as_uri()
    assert call["launch"] == folder_uri
    assert folder_uri in [a.launch for a in call["actions"]]


def test_capture_warning_toast_links_the_warnings_file(app, tmp_path):
    """When capture_warnings.txt was written, offer to open it directly."""
    session = tmp_path / "recs" / "2026-09-17_09-00-00"
    session.mkdir(parents=True)
    wfile = session / "capture_warnings.txt"
    wfile.write_text("Microphone recording aborted.\n", encoding="utf-8")

    app._notify_capture_warnings(session, ["Microphone recording aborted."])

    launches = [a.launch for a in app.notifier.calls[0]["actions"]]
    assert wfile.resolve().as_uri() in launches


def test_capture_warning_toast_omits_missing_warnings_file(app, tmp_path):
    """No dead link when the warnings file could not be written."""
    session = tmp_path / "recs" / "2026-09-17_09-00-00"
    session.mkdir(parents=True)

    app._notify_capture_warnings(session, ["Microphone recording aborted."])

    launches = [a.launch for a in app.notifier.calls[0]["actions"]]
    assert launches == [session.resolve().as_uri()]


def test_controller_is_wired_to_the_warning_toast(tmp_path, monkeypatch):
    """The controller must actually be handed the notifier — the toast is
    worthless if nothing calls it."""
    import audiologger.tray_app as ta

    cfg = Config(output_dir=tmp_path / "recs")
    monkeypatch.setattr(ta, "config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(ta, "load_config", lambda _p: cfg)
    monkeypatch.setattr(ta, "appdata_dir", lambda: tmp_path / "appdata")

    app = ta.TrayApp()

    assert app.controller._notify_fn == app._notify_capture_warnings


def test_dead_mic_recording_produces_a_visible_toast(tmp_path, monkeypatch):
    """End-to-end over the seam: the 2026-07 dead-mic case must now be seen.

    The per-side unit tests can both stay green while the controller/tray call
    signatures drift apart, so this exercises the real chain.
    """
    import audiologger.tray_app as ta

    cfg = Config(output_dir=tmp_path / "recs")
    monkeypatch.setattr(ta, "config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(ta, "load_config", lambda _p: cfg)
    monkeypatch.setattr(ta, "appdata_dir", lambda: tmp_path / "appdata")

    class DeadMicCapture:
        """Mirrors AudioCaptureThread when the mic stream dies."""

        def __init__(self, session_dir, sample_rate, source, app_names, mic_only=False):
            self.session_dir = session_dir
            self.warnings: list[str] = []

        def start(self):
            (self.session_dir / "mic.wav").write_bytes(b"x" * 44)  # header only
            (self.session_dir / "system.wav").touch()

        def stop(self):
            self.warnings = ["Microphone recording aborted."]

    monkeypatch.setattr(ta, "AudioCaptureThread", DeadMicCapture)
    monkeypatch.setattr(ta, "mix_to_file", lambda *a: None)

    app = ta.TrayApp()
    app.notifier = RecordingNotifier()
    monkeypatch.setattr(app.queue, "enqueue", lambda _s: None)

    app.controller.toggle()  # start
    app.controller.toggle()  # stop

    assert len(app.notifier.calls) == 1, "the dead mic must raise a toast"
    call = app.notifier.calls[0]
    assert "Microphone recording aborted." in call["message"]
    assert call["launch"].startswith("file:///")
