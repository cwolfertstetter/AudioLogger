"""Per-app loopback ("apps" mode) failing must be visible, never silent."""
import sys
import threading
import types

import pytest

from audiologger import process_loopback as pl


class FakeStream:
    def __init__(self, reads_before_failure, stop_raises=False):
        self._left = reads_before_failure
        self._stop_raises = stop_raises
        self.closed = False

    def read(self, n, exception_on_overflow=False):
        if self._left == 0:
            raise OSError("device invalidated")
        self._left -= 1
        return b"\x01\x00" * n

    def stop_stream(self):
        if self._stop_raises:
            raise OSError("device invalidated")

    def close(self):
        self.closed = True


def fake_pyaudio(stream, *, with_process_loopback=True):
    mod = types.ModuleType("pyaudiowpatch")
    mod.paInt16 = 8

    class PyAudio:
        def get_loopback_device_info_generator(self):
            yield {"index": 3, "name": "Speakers [Loopback]"}

        def open(self, **kw):
            return stream

        def terminate(self):
            pass

    mod.PyAudio = PyAudio
    if with_process_loopback:
        mod.paWinWasapiProcessLoopback = 1
        mod.PaWasapiStreamInfo = lambda **kw: kw
    return mod


@pytest.fixture
def one_app(monkeypatch):
    monkeypatch.setattr(pl, "_HAS_PYAUDIOWPATCH", True)
    monkeypatch.setattr(pl, "_resolve_pids", lambda names: [4242])

    def install(mod):
        monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)

    return install


def test_a_read_error_mid_call_is_raised_not_swallowed(tmp_path, one_app):
    stream = FakeStream(reads_before_failure=2, stop_raises=True)
    one_app(fake_pyaudio(stream))

    with pytest.raises(pl.ProcessLoopbackAborted):
        pl.record_app_loopback(tmp_path / "system.wav", ["Teams.exe"], 10, threading.Event())

    assert stream.closed, "close() runs even though stop_stream() failed"


def test_without_process_loopback_support_the_filter_is_not_faked(tmp_path, one_app):
    """Opening the stream anyway recorded every app while claiming to filter."""
    one_app(fake_pyaudio(FakeStream(1), with_process_loopback=False))

    with pytest.raises(pl.ProcessLoopbackNotAvailable):
        pl.record_app_loopback(tmp_path / "system.wav", ["Teams.exe"], 10, threading.Event())


def test_the_capture_warns_when_the_app_stream_breaks(tmp_path, monkeypatch):
    from audiologger import audio_capture

    def aborted(*a, **kw):
        raise pl.ProcessLoopbackAborted("device invalidated")

    monkeypatch.setattr(audio_capture, "record_app_loopback", aborted)
    cap = audio_capture.AudioCaptureThread(tmp_path, 10, audio_source="apps",
                                           filtered_app_names=["Teams.exe"])

    cap._run_system(tmp_path / "system.wav")

    assert any("App audio recording stopped" in w for w in cap.warnings), cap.warnings
