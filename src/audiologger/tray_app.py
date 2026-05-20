"""Tray application — wires hotkey + controller + queue + notifications."""
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import pystray
from pystray import MenuItem, Menu

from audiologger.audio_capture import AudioCaptureThread
from audiologger.audio_mix import mix_to_file
from audiologger.config import Config, load_config, save_config
from audiologger.controller import RecordingController, RecordingState
from audiologger.hotkey import HotkeyManager
from audiologger.icons import idle_icon, recording_icon, transcribing_icon
from audiologger.job_queue import TranscriptionJobQueue
from audiologger.notifications import Action, Notifier
from audiologger.paths import appdata_dir, config_path
from audiologger.recovery import find_orphaned_sessions


log = logging.getLogger("tray_app")


class TrayApp:
    LONG_RECORDING_WARN_SECONDS = 3 * 60 * 60  # 3 hours

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
        self.dictation_hotkey = HotkeyManager()
        self.extend_hotkey = HotkeyManager()
        self.icon: pystray.Icon | None = None
        self._stop_event = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def run(self) -> None:
        self._handle_orphaned_sessions()
        if self.cfg.worker_prewarm:
            log.info("Pre-warming transcription worker")
            self.queue.prewarm()
        self._bind_hotkeys()
        self.icon = pystray.Icon(
            "AudioLogger",
            icon=idle_icon(),
            title="AudioLogger",
            menu=self._build_menu(),
        )
        threading.Thread(target=self._status_refresh_loop, daemon=True).start()
        self.icon.run()

    def _bind_hotkeys(self) -> None:
        ok = self.hotkey.bind(self.cfg.hotkey, self._on_meeting_hotkey)
        if not ok:
            self.notifier.notify(
                "Hotkey conflict",
                f"'{self.cfg.hotkey}' could not be bound. Change it from the tray menu.",
            )
        ok2 = self.dictation_hotkey.bind(self.cfg.dictation_hotkey, self._on_dictation_hotkey)
        if not ok2:
            self.notifier.notify(
                "Hotkey conflict",
                f"'{self.cfg.dictation_hotkey}' could not be bound. Change it from the tray menu.",
            )
        ok3 = self.extend_hotkey.bind(self.cfg.extend_hotkey, self._on_extend_hotkey)
        if not ok3:
            self.notifier.notify(
                "Hotkey conflict",
                f"'{self.cfg.extend_hotkey}' could not be bound. Change it from the tray menu.",
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

    # --- Hotkey + controller -------------------------------------------

    def _on_meeting_hotkey(self) -> None:
        prev_state = self.controller.state
        try:
            self.controller.toggle(mode="meeting")
        except Exception:
            log.exception("toggle failed")
            self.notifier.notify("Error", "Recording could not start/stop — see logs.")
            return
        new_state = self.controller.state
        if prev_state is RecordingState.IDLE and new_state is RecordingState.RECORDING:
            self.notifier.notify("Recording started", "Press hotkey again to stop.")
            self._set_icon(recording_icon())
        elif prev_state is RecordingState.RECORDING and new_state is RecordingState.IDLE:
            self.notifier.notify("Recording stopped", "Transcribing...")
            self._set_icon(transcribing_icon())

    def _on_dictation_hotkey(self) -> None:
        prev_state = self.controller.state
        try:
            self.controller.toggle(mode="dictation")
        except Exception:
            log.exception("toggle failed")
            self.notifier.notify("Error", "Dictation could not start/stop — see logs.")
            return
        new_state = self.controller.state
        if prev_state is RecordingState.IDLE and new_state is RecordingState.RECORDING:
            self.notifier.notify("Dictation started", "Press hotkey again to stop.")
            self._set_icon(recording_icon())
        elif prev_state is RecordingState.RECORDING and new_state is RecordingState.IDLE:
            self.notifier.notify("Dictation stopped", "Transcribing...")
            self._set_icon(transcribing_icon())

    def _on_extend_hotkey(self) -> None:
        prev_state = self.controller.state
        try:
            self.controller.toggle(mode="dictation_extend")
        except Exception:
            log.exception("toggle failed")
            self.notifier.notify("Error", "Note could not start/stop — see logs.")
            return
        new_state = self.controller.state
        if prev_state is RecordingState.IDLE and new_state is RecordingState.RECORDING:
            self.notifier.notify("Note recording started", "Press hotkey again to stop.")
            self._set_icon(recording_icon())
        elif prev_state is RecordingState.RECORDING and new_state is RecordingState.IDLE:
            self.notifier.notify("Note stopped", "Transcribing...")
            self._set_icon(transcribing_icon())

    def _on_recording_finished(self, session_dir: Path) -> None:
        """Called by controller after stop. Hand off to queue."""
        self.queue.enqueue(session_dir)

    # --- Settings helpers ----------------------------------------------

    def _set_model(self, field: str, value: str) -> None:
        """Mutate cfg.field, save, terminate worker so it reloads on next job."""
        setattr(self.cfg, field, value)
        save_config(config_path(), self.cfg)
        # Terminate running worker — it will be respawned with the new model on next job.
        worker = getattr(self.queue, "_worker", None)
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
                worker.wait(timeout=5)
            except Exception:
                log.exception("Failed to terminate worker after model change")
            self.queue._worker = None
        self.notifier.notify(
            "Model changed",
            f"Whisper model set to {value}. Worker will reload on next job.",
        )

    def _set_bool_setting(self, field: str, label: str, value: bool) -> None:
        """Mutate a boolean cfg field, save, and fire a toast."""
        setattr(self.cfg, field, value)
        save_config(config_path(), self.cfg)
        state = "Enabled" if value else "Disabled"
        self.notifier.notify(
            "Setting changed",
            f"{label} set to {state}. Restart may be required.",
        )

    def _set_device(self, value: str) -> None:
        self.cfg.device = value
        save_config(config_path(), self.cfg)
        self.notifier.notify(
            "Setting changed",
            f"Device set to {value}. Restart may be required.",
        )

    def _make_model_menu(
        self,
        *,
        current: Callable[[], str],
        setter: Callable[[str], None],
    ) -> Menu:
        options = ["tiny", "base", "small", "medium", "large-v3"]
        return Menu(*[
            MenuItem(
                opt,
                lambda _, value=opt: setter(value),
                radio=True,
                checked=lambda _, value=opt: current() == value,
            )
            for opt in options
        ])

    def _make_bool_menu(
        self,
        *,
        current: Callable[[], bool],
        setter: Callable[[bool], None],
    ) -> Menu:
        return Menu(
            MenuItem(
                "Enabled",
                lambda _: setter(True),
                radio=True,
                checked=lambda _: current() is True,
            ),
            MenuItem(
                "Disabled",
                lambda _: setter(False),
                radio=True,
                checked=lambda _: current() is False,
            ),
        )

    def _make_device_menu(self) -> Menu:
        return Menu(
            MenuItem(
                "CUDA (GPU)",
                lambda _: self._set_device("cuda"),
                radio=True,
                checked=lambda _: self.cfg.device == "cuda",
            ),
            MenuItem(
                "CPU",
                lambda _: self._set_device("cpu"),
                radio=True,
                checked=lambda _: self.cfg.device == "cpu",
            ),
        )

    # --- Tray menu -----------------------------------------------------

    def _build_menu(self) -> Menu:
        return Menu(
            MenuItem(lambda _: f"Status: {self._status_text()}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Start/Stop Meeting Recording", lambda _: self._on_meeting_hotkey()),
            MenuItem("Start/Stop Dictation", lambda _: self._on_dictation_hotkey()),
            MenuItem("Append Note (Extend)", lambda _: self._on_extend_hotkey()),
            Menu.SEPARATOR,
            MenuItem("Change Output Folder...", lambda _: self._pick_output_dir()),
            MenuItem(
                "Audio Source",
                Menu(
                    MenuItem(
                        "All (System Loopback)",
                        lambda _: self._set_audio_source("all"),
                        radio=True,
                        checked=lambda _: self.cfg.audio_source == "all",
                    ),
                    MenuItem(
                        "Selected Apps Only",
                        lambda _: self._set_audio_source("apps"),
                        radio=True,
                        checked=lambda _: self.cfg.audio_source == "apps",
                    ),
                ),
            ),
            MenuItem("Open Config File...", lambda _: self._open_config_file()),
            MenuItem(
                "Settings",
                Menu(
                    MenuItem(
                        "Whisper Model (Meetings)",
                        self._make_model_menu(
                            current=lambda: self.cfg.whisper_model,
                            setter=lambda v: self._set_model("whisper_model", v),
                        ),
                    ),
                    MenuItem(
                        "Whisper Model (Dictation)",
                        self._make_model_menu(
                            current=lambda: self.cfg.dictation_model,
                            setter=lambda v: self._set_model("dictation_model", v),
                        ),
                    ),
                    MenuItem(
                        "Device",
                        self._make_device_menu(),
                    ),
                    MenuItem(
                        "Diarization",
                        self._make_bool_menu(
                            current=lambda: self.cfg.diarization_enabled,
                            setter=lambda v: self._set_bool_setting("diarization_enabled", "Diarization", v),
                        ),
                    ),
                    MenuItem(
                        "Notifications",
                        self._make_bool_menu(
                            current=lambda: self.cfg.notification_enabled,
                            setter=lambda v: self._set_bool_setting("notification_enabled", "Notifications", v),
                        ),
                    ),
                    MenuItem(
                        "Worker Pre-warm",
                        self._make_bool_menu(
                            current=lambda: self.cfg.worker_prewarm,
                            setter=lambda v: self._set_bool_setting("worker_prewarm", "Worker Pre-warm", v),
                        ),
                    ),
                ),
            ),
            MenuItem("Re-transcribe Last Recording", lambda _: self._retry_last()),
            Menu.SEPARATOR,
            MenuItem("Quit", lambda _: self._quit()),
        )

    def _status_text(self) -> str:
        if self.controller.state is RecordingState.RECORDING:
            return "Recording"
        s = self.queue.status()
        if s.running:
            return f"Transcribing: {s.running}"
        if s.queued:
            return f"Queued: {len(s.queued)}"
        if s.last_failed and not s.running and not s.queued:
            return f"Last recording failed: {s.last_failed}"
        if s.warming:
            return "Worker warming up..."
        return "Ready"

    def _set_icon(self, image) -> None:
        if self.icon is not None:
            self.icon.icon = image

    def _pick_output_dir(self) -> None:
        """Show a native Windows folder picker. Thread-safe (unlike tkinter)."""
        try:
            chosen = _pick_folder_native(
                title="Select output folder",
                initial=str(self.cfg.output_dir.resolve()),
            )
        except Exception:
            log.exception("Folder picker failed")
            self.notifier.notify(
                "Folder picker failed",
                "See %APPDATA%/AudioLogger/tray.log for details.",
            )
            return
        if not chosen:
            return  # user cancelled
        try:
            self.cfg.output_dir = Path(chosen)
            save_config(config_path(), self.cfg)
            self.notifier.notify("Output folder changed", chosen)
        except Exception:
            log.exception("Failed to save new output_dir")
            self.notifier.notify(
                "Output folder NOT saved",
                "See %APPDATA%/AudioLogger/tray.log for details.",
            )

    def _set_audio_source(self, source: str) -> None:
        self.cfg.audio_source = source
        save_config(config_path(), self.cfg)
        self.notifier.notify("Audio source", f"New source: {source}")

    def _open_config_file(self) -> None:
        os.startfile(str(config_path()))

    def _retry_last(self) -> None:
        out = self.cfg.output_dir
        if not out.exists():
            self.notifier.notify("No recordings", f"{out} does not exist.")
            return
        sessions = sorted([d for d in out.iterdir() if d.is_dir()], key=lambda p: p.name)
        if not sessions:
            self.notifier.notify("No recordings", "Output folder is empty.")
            return
        last = sessions[-1]
        self.queue.enqueue(last)
        self.notifier.notify("Re-transcription started", last.name)

    def _quit(self) -> None:
        self.hotkey.unbind()
        self.dictation_hotkey.unbind()
        self.extend_hotkey.unbind()
        self._stop_event.set()
        worker = getattr(self.queue, "_worker", None)
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
                worker.wait(timeout=5)
            except Exception:
                log.exception("Failed to terminate worker cleanly")
        if self.icon is not None:
            self.icon.stop()

    # --- Background icon-refresh loop ----------------------------------

    @staticmethod
    def _finished_key(last_finished: dict | None) -> tuple | None:
        if last_finished is None:
            return None
        return (
            last_finished.get("session_id"),
            last_finished.get("success"),
            last_finished.get("was_extend"),
        )

    def _status_refresh_loop(self) -> None:
        from datetime import datetime
        long_warning_fired = False
        recording_started_at: datetime | None = None

        # Seed prev_finished_key from current state so we don't re-fire stale completions.
        try:
            prev_finished_key = self._finished_key(self.queue.status().last_finished)
        except Exception:
            prev_finished_key = None

        while not self._stop_event.is_set():
            if self.controller.state is RecordingState.RECORDING:
                if recording_started_at is None:
                    recording_started_at = datetime.now()
                    long_warning_fired = False
                elapsed = (datetime.now() - recording_started_at).total_seconds()
                if elapsed > self.LONG_RECORDING_WARN_SECONDS and not long_warning_fired:
                    self.notifier.notify(
                        "Long recording",
                        f"Recording has been running for {int(elapsed // 3600)}+ hours — still good?",
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

                # Detect new completion via last_finished change.
                current_key = self._finished_key(s.last_finished)
                if (
                    s.last_finished is not None
                    and prev_finished_key is not None
                    and current_key != prev_finished_key
                ):
                    self._handle_completion(s.last_finished)
                prev_finished_key = current_key
            time.sleep(1.0)

    def _handle_completion(self, payload: dict) -> None:
        mode = payload.get("mode", "meeting")
        success = bool(payload.get("success", False))
        was_extend = bool(payload.get("was_extend", False))
        session_id = payload.get("session_id", "")
        chunk_preview = payload.get("chunk_preview")

        if mode == "meeting":
            if success:
                self._notify_transcription_done(session_id)
            else:
                self._notify_transcription_failed(session_id)
        else:
            # dictation (regular or extend)
            if was_extend:
                if success:
                    self._notify_extend_done(session_id, chunk_preview)
                else:
                    self._notify_extend_failed(session_id)
            else:
                if success:
                    self._notify_dictation_done(session_id, chunk_preview)
                else:
                    self._notify_dictation_failed(session_id)

    def _notify_transcription_done(self, session_name: str) -> None:
        session_dir = (self.cfg.output_dir / session_name).resolve()
        transcript = session_dir / "transcript.md"
        launch = transcript.as_uri() if transcript.exists() else session_dir.as_uri()
        actions = [
            Action(label="Open transcript", launch=transcript.as_uri()),
            Action(label="Open folder", launch=session_dir.as_uri()),
        ] if transcript.exists() else [
            Action(label="Open folder", launch=session_dir.as_uri()),
        ]
        self.notifier.notify(
            "Transcription complete",
            session_name,
            launch=launch,
            actions=actions,
        )

    def _notify_transcription_failed(self, session_name: str) -> None:
        session_dir = (self.cfg.output_dir / session_name).resolve()
        job_log = session_dir / "job.log"
        actions = [
            Action(label="Open folder", launch=session_dir.as_uri()),
        ]
        if job_log.exists():
            actions.insert(0, Action(label="Open error log", launch=job_log.as_uri()))
        self.notifier.notify(
            "Transcription failed",
            session_name,
            launch=session_dir.as_uri(),
            actions=actions,
        )

    def _notify_dictation_done(self, session_name: str, chunk_preview: str | None = None) -> None:
        session_dir = (self.cfg.output_dir / session_name).resolve()
        transcript = session_dir / "transcript.txt"
        if chunk_preview:
            preview = chunk_preview
        elif transcript.exists():
            try:
                text = transcript.read_text(encoding="utf-8")
                preview = text[:140] + "..." if len(text) > 140 else text
            except OSError:
                preview = ""
        else:
            preview = ""
        launch = transcript.as_uri() if transcript.exists() else session_dir.as_uri()
        actions = [Action(label="Open file", launch=transcript.as_uri())]
        self.notifier.notify(
            "Dictation complete",
            preview or session_name,
            launch=launch,
            actions=actions,
        )

    def _notify_dictation_failed(self, session_name: str) -> None:
        session_dir = (self.cfg.output_dir / session_name).resolve()
        job_log = session_dir / "job.log"
        actions = [
            Action(label="Open folder", launch=session_dir.as_uri()),
        ]
        if job_log.exists():
            actions.insert(0, Action(label="Open error log", launch=job_log.as_uri()))
        self.notifier.notify(
            "Dictation failed",
            session_name,
            launch=session_dir.as_uri(),
            actions=actions,
        )

    def _notify_extend_done(self, session_name: str, chunk_preview: str | None = None) -> None:
        session_dir = (self.cfg.output_dir / session_name).resolve()
        transcript = session_dir / "transcript.txt"
        preview = chunk_preview or session_name
        launch = transcript.as_uri() if transcript.exists() else session_dir.as_uri()
        actions = [Action(label="Open file", launch=transcript.as_uri())]
        self.notifier.notify(
            "Note appended",
            preview,
            launch=launch,
            actions=actions,
        )

    def _notify_extend_failed(self, session_name: str) -> None:
        session_dir = (self.cfg.output_dir / session_name).resolve()
        job_log = session_dir / "job.log"
        actions = [
            Action(label="Open folder", launch=session_dir.as_uri()),
        ]
        if job_log.exists():
            actions.insert(0, Action(label="Open error log", launch=job_log.as_uri()))
        self.notifier.notify(
            "Note append failed",
            session_name,
            launch=session_dir.as_uri(),
            actions=actions,
        )


# ---------------------------------------------------------------------------
# Native Windows folder picker (thread-safe; tkinter is not).
# ---------------------------------------------------------------------------

def _pick_folder_native(*, title: str, initial: str) -> str | None:
    """Show Windows native folder browser. Returns chosen path or None on cancel.

    Uses SHBrowseForFolderW + SHGetPathFromIDListW from shell32.dll via ctypes.
    Safe to call from any thread (unlike tkinter, which insists on the main thread).
    """
    import ctypes
    from ctypes import wintypes

    BIF_RETURNONLYFSDIRS = 0x00000001
    BIF_NEWDIALOGSTYLE = 0x00000040
    BIF_EDITBOX = 0x00000010
    BFFM_INITIALIZED = 1
    BFFM_SETSELECTIONW = 0x400 + 103
    MAX_PATH = 260

    BFFCALLBACK = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HWND, wintypes.UINT, wintypes.LPARAM, wintypes.LPARAM
    )

    class BROWSEINFOW(ctypes.Structure):
        _fields_ = [
            ("hwndOwner", wintypes.HWND),
            ("pidlRoot", ctypes.c_void_p),
            ("pszDisplayName", wintypes.LPWSTR),
            ("lpszTitle", wintypes.LPCWSTR),
            ("ulFlags", wintypes.UINT),
            ("lpfn", BFFCALLBACK),
            ("lParam", wintypes.LPARAM),
            ("iImage", ctypes.c_int),
        ]

    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    user32 = ctypes.windll.user32  # noqa: F841

    shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
    shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wintypes.LPWSTR]
    shell32.SHGetPathFromIDListW.restype = wintypes.BOOL

    # Callback to preselect the initial path
    def _callback(hwnd, msg, lparam, data):
        if msg == BFFM_INITIALIZED and data:
            user32.SendMessageW(hwnd, BFFM_SETSELECTIONW, 1, data)
        return 0

    cb = BFFCALLBACK(_callback)
    initial_buf = ctypes.create_unicode_buffer(initial)

    display = ctypes.create_unicode_buffer(MAX_PATH)
    bi = BROWSEINFOW()
    bi.hwndOwner = None
    bi.pidlRoot = None
    bi.pszDisplayName = ctypes.cast(display, wintypes.LPWSTR)
    bi.lpszTitle = title
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX
    bi.lpfn = cb
    bi.lParam = ctypes.cast(initial_buf, ctypes.c_void_p).value or 0
    bi.iImage = 0

    # COM apartment init — required for SHBrowseForFolderW from non-main threads
    ole32.CoInitialize(None)
    try:
        pidl = shell32.SHBrowseForFolderW(ctypes.byref(bi))
        if not pidl:
            return None  # cancelled
        path_buf = ctypes.create_unicode_buffer(MAX_PATH)
        ok = shell32.SHGetPathFromIDListW(pidl, path_buf)
        ole32.CoTaskMemFree(pidl)
        if not ok:
            return None
        return path_buf.value or None
    finally:
        ole32.CoUninitialize()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _setup_tray_logging() -> Path:
    """Write tray logs to %APPDATA%/AudioLogger/tray.log and install excepthook."""
    log_path = appdata_dir() / "tray.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    # Log any uncaught exception before the process dies
    def _excepthook(exc_type, exc_value, tb):
        logging.getLogger("tray_app").critical(
            "UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, tb)
        )
    sys.excepthook = _excepthook
    return log_path


def main() -> None:
    log_path = _setup_tray_logging()
    log.info("AudioLogger tray starting — log: %s", log_path)
    try:
        TrayApp().run()
    except Exception:
        log.exception("Tray crashed")
        raise


if __name__ == "__main__":
    main()
