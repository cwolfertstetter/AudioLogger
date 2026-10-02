"""Mic branch of AudioCaptureThread.

The original failure mode was silent: soundcard raised, the warning went into
a file nobody reads, and the pipeline happily transcribed an empty mic track.
A capture that yields no frames must say so.
"""
import threading
from pathlib import Path

from audiologger import audio_capture
from audiologger.audio_capture import AudioCaptureThread
from audiologger.mic_capture import MicCaptureResult, MicrophoneNotAvailable


def make_capture(tmp_path: Path) -> AudioCaptureThread:
    return AudioCaptureThread(tmp_path, 48000, mic_only=True)


def test_records_via_portaudio_and_reports_the_device_name(tmp_path, monkeypatch):
    seen = {}

    def fake_record(out_path, sample_rate, stop_event, **kw):
        seen["out_path"] = out_path
        seen["sample_rate"] = sample_rate
        return MicCaptureResult("Microphone (KLIM Mantis Audio 7.1)", 48000)

    monkeypatch.setattr(audio_capture, "record_microphone", fake_record)
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    assert seen["out_path"] == tmp_path / "mic.wav"
    assert seen["sample_rate"] == 48000
    assert cap.mic_device_name == "Microphone (KLIM Mantis Audio 7.1)"
    assert cap.warnings == []


def test_warns_when_the_microphone_captured_no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audio_capture, "record_microphone",
        lambda *a, **k: MicCaptureResult("Microphone (KLIM Mantis Audio 7.1)", 0),
    )
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    assert any("no audio" in w.lower() for w in cap.warnings), cap.warnings


def test_warns_with_the_reason_when_the_microphone_is_unavailable(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise MicrophoneNotAvailable("No default recording device")

    monkeypatch.setattr(audio_capture, "record_microphone", boom)
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    assert any("No default recording device" in w for w in cap.warnings), cap.warnings


def test_warns_when_capture_crashes_unexpectedly(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(audio_capture, "record_microphone", boom)
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    assert any("aborted" in w.lower() for w in cap.warnings), cap.warnings


# --- dropouts --------------------------------------------------------------------
# A mic that vanished and came back must still be visible to the user: the
# warning reaches capture_warnings.txt and the toast shown at stop.

def test_warns_when_the_microphone_dropped_out_and_came_back(tmp_path, monkeypatch):
    from audiologger.mic_capture import MicDropout

    monkeypatch.setattr(
        audio_capture, "record_microphone",
        lambda *a, **k: MicCaptureResult(
            "Microphone (KLIM Mantis Audio 7.1)", 48000,
            (MicDropout(at_s=752.0, gap_s=6.0, resumed_on="Microphone (KLIM Mantis Audio 7.1)"),),
        ),
    )
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    [w] = cap.warnings
    assert "dropped out" in w and "12:32" in w and "6 s" in w and "silence" in w


def test_names_the_device_recording_continued_on_when_it_changed(tmp_path, monkeypatch):
    from audiologger.mic_capture import MicDropout

    monkeypatch.setattr(
        audio_capture, "record_microphone",
        lambda *a, **k: MicCaptureResult(
            "Microphone (KLIM Mantis Audio 7.1)", 48000,
            (MicDropout(at_s=60.0, gap_s=3.0, resumed_on="Microphone Array (Realtek(R) Audio)"),),
        ),
    )
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    assert "Microphone Array (Realtek(R) Audio)" in cap.warnings[0]


def test_says_so_when_the_microphone_never_came_back(tmp_path, monkeypatch):
    from audiologger.mic_capture import MicDropout

    monkeypatch.setattr(
        audio_capture, "record_microphone",
        lambda *a, **k: MicCaptureResult(
            "Microphone (KLIM Mantis Audio 7.1)", 48000,
            (MicDropout(at_s=793.0, gap_s=14826.0, resumed_on=None),),
        ),
    )
    cap = make_capture(tmp_path)

    cap._run_mic(tmp_path / "mic.wav")

    assert "did not come back" in cap.warnings[0]
