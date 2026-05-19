"""Thin wrapper around winotify for tray toast notifications."""
from dataclasses import dataclass
from winotify import Notification


APP_NAME = "AudioLogger"


@dataclass(frozen=True)
class Action:
    """A clickable button on a toast. `launch` may be a file:// URL or http URL."""
    label: str
    launch: str


class Notifier:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def notify(
        self,
        title: str,
        message: str,
        *,
        launch: str = "",
        actions: list[Action] | None = None,
    ) -> None:
        """Show a Windows toast.

        Args:
            title, message: visible text.
            launch: URL/file opened when the toast body is clicked.
                    Use 'file:///C:/path/to/file.md' for local files.
            actions: extra clickable buttons under the body.
        """
        if not self.enabled:
            return
        toast = Notification(
            app_id=APP_NAME,
            title=title,
            msg=message,
            duration="short",
            launch=launch,
        )
        for a in actions or []:
            toast.add_actions(label=a.label, launch=a.launch)
        toast.show()
