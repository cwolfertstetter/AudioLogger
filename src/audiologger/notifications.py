"""Thin wrapper around winotify for tray toast notifications."""
from winotify import Notification


APP_NAME = "AudioLogger"


class Notifier:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def notify(self, title: str, message: str) -> None:
        if not self.enabled:
            return
        toast = Notification(app_id=APP_NAME, title=title, msg=message, duration="short")
        toast.show()
