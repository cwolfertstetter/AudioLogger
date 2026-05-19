"""Global hotkey registration with rebind support."""
import logging
from typing import Callable

import keyboard


log = logging.getLogger(__name__)


class HotkeyManager:
    def __init__(self):
        self._current_hotkey: str | None = None
        self._current_handle = None

    def bind(self, hotkey: str, callback: Callable[[], None]) -> bool:
        """Bind `hotkey` to `callback`. Returns True on success, False on conflict."""
        self.unbind()
        try:
            self._current_handle = keyboard.add_hotkey(hotkey, callback)
            self._current_hotkey = hotkey
            return True
        except Exception:
            log.exception("Failed to bind hotkey %s", hotkey)
            self._current_handle = None
            self._current_hotkey = None
            return False

    def unbind(self) -> None:
        if self._current_handle is not None:
            try:
                keyboard.remove_hotkey(self._current_handle)
            except Exception:
                log.exception("Failed to remove hotkey")
        self._current_handle = None
        self._current_hotkey = None

    @property
    def current(self) -> str | None:
        return self._current_hotkey
