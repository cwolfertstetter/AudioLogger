"""Tray application — wires hotkey + controller + queue + notifications."""
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pystray
from pystray import MenuItem, Menu

from audiologger.audio_capture import AudioCaptureThread
from audiologger.audio_mix import mix_to_file
from audiologger.config import Config, load_config, save_config
from audiologger.controller import RecordingController, RecordingState
from audiologger.hotkey import HotkeyManager
from audiologger.icons import idle_icon, recording_icon, transcribing_icon
from audiologger.job_queue import TranscriptionJobQueue
from audiologger.notifications import Notifier
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
        self.icon: pystray.Icon | None = None
        self._stop_event = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def run(self) -> None:
        self._handle_orphaned_sessions()
        self._bind_hotkey()
        self.icon = pystray.Icon(
            "AudioLogger",
            icon=idle_icon(),
            title="AudioLogger",
            menu=self._build_menu(),
        )
        threading.Thread(target=self._status_refresh_loop, daemon=True).start()
        self.icon.run()

    def _bind_hotkey(self) -> None:
        ok = self.hotkey.bind(self.cfg.hotkey, self._on_hotkey)
        if not ok:
            self.notifier.notify(
                "Hotkey-Konflikt",
                f"'{self.cfg.hotkey}' konnte nicht gebunden werden. Bitte in Tray ändern.",
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

    def _on_hotkey(self) -> None:
        prev_state = self.controller.state
        try:
            self.controller.toggle()
        except Exception:
            log.exception("toggle failed")
            self.notifier.notify("Fehler", "Aufnahme konnte nicht (be)endet werden — siehe Logs.")
            return
        new_state = self.controller.state
        if prev_state is RecordingState.IDLE and new_state is RecordingState.RECORDING:
            self.notifier.notify("Aufnahme gestartet", "Hotkey erneut drücken zum Stoppen.")
            self._set_icon(recording_icon())
        elif prev_state is RecordingState.RECORDING and new_state is RecordingState.IDLE:
            self.notifier.notify("Aufnahme beendet", "Transkription läuft...")
            self._set_icon(transcribing_icon())

    def _on_recording_finished(self, session_dir: Path) -> None:
        """Called by controller after stop. Hand off to queue."""
        self.queue.enqueue(session_dir)

    # --- Tray menu -----------------------------------------------------

    def _build_menu(self) -> Menu:
        return Menu(
            MenuItem(lambda _: f"Status: {self._status_text()}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Aufnahme starten/stoppen", lambda _: self._on_hotkey()),
            Menu.SEPARATOR,
            MenuItem("Output-Ordner ändern...", lambda _: self._pick_output_dir()),
            MenuItem(
                "Audio-Quelle",
                Menu(
                    MenuItem(
                        "Alles (System-Loopback)",
                        lambda _: self._set_audio_source("all"),
                        radio=True,
                        checked=lambda _: self.cfg.audio_source == "all",
                    ),
                    MenuItem(
                        "Nur ausgewählte Apps",
                        lambda _: self._set_audio_source("apps"),
                        radio=True,
                        checked=lambda _: self.cfg.audio_source == "apps",
                    ),
                ),
            ),
            MenuItem("Config-Datei öffnen...", lambda _: self._open_config_file()),
            MenuItem("Letzte Aufnahme erneut transkribieren", lambda _: self._retry_last()),
            Menu.SEPARATOR,
            MenuItem("Beenden", lambda _: self._quit()),
        )

    def _status_text(self) -> str:
        if self.controller.state is RecordingState.RECORDING:
            return "Aufnahme läuft"
        s = self.queue.status()
        if s.running:
            return f"Transkribiert: {s.running}"
        if s.queued:
            return f"In Warteschlange: {len(s.queued)}"
        if s.last_failed and not s.running and not s.queued:
            return f"Letzte Aufnahme fehlgeschlagen: {s.last_failed}"
        return "Bereit"

    def _set_icon(self, image) -> None:
        if self.icon is not None:
            self.icon.icon = image

    def _pick_output_dir(self) -> None:
        # Use tkinter folder picker (built-in)
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(initialdir=str(self.cfg.output_dir))
        root.destroy()
        if chosen:
            self.cfg.output_dir = Path(chosen)
            save_config(config_path(), self.cfg)
            self.notifier.notify("Output-Ordner geändert", chosen)

    def _set_audio_source(self, source: str) -> None:
        self.cfg.audio_source = source
        save_config(config_path(), self.cfg)
        self.notifier.notify("Audio-Quelle", f"Neue Quelle: {source}")

    def _open_config_file(self) -> None:
        os.startfile(str(config_path()))

    def _retry_last(self) -> None:
        out = self.cfg.output_dir
        if not out.exists():
            self.notifier.notify("Keine Aufnahmen", f"{out} existiert nicht.")
            return
        sessions = sorted([d for d in out.iterdir() if d.is_dir()], key=lambda p: p.name)
        if not sessions:
            self.notifier.notify("Keine Aufnahmen", "Output-Ordner ist leer.")
            return
        last = sessions[-1]
        self.queue.enqueue(last)
        self.notifier.notify("Re-Transkription gestartet", last.name)

    def _quit(self) -> None:
        self.hotkey.unbind()
        self._stop_event.set()
        if self.icon is not None:
            self.icon.stop()

    # --- Background icon-refresh loop ----------------------------------

    def _status_refresh_loop(self) -> None:
        from datetime import datetime
        long_warning_fired = False
        recording_started_at: datetime | None = None
        while not self._stop_event.is_set():
            if self.controller.state is RecordingState.RECORDING:
                if recording_started_at is None:
                    recording_started_at = datetime.now()
                    long_warning_fired = False
                elapsed = (datetime.now() - recording_started_at).total_seconds()
                if elapsed > self.LONG_RECORDING_WARN_SECONDS and not long_warning_fired:
                    self.notifier.notify(
                        "Lange Aufnahme",
                        f"Aufnahme läuft seit {int(elapsed // 3600)}+ Stunden — alles ok?",
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
            time.sleep(1.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    TrayApp().run()


if __name__ == "__main__":
    main()
