"""Dropping segments that sit on silence.

Pinning the language stopped Whisper inventing Russian and Norwegian, and
normalizing stopped it looping on quiet speech, but neither makes it stay
quiet when there is nothing to transcribe.  On a 14-minute recording where the
user spoke 2.5% of the time, 10 of 13 "Me" entries were the word "Vielen
Dank." dropped onto silence.

So measure each segment against the actual audio underneath it and throw away
the ones sitting on nothing.
"""
import numpy as np
import pytest

from audiologger.segment import Segment
from audiologger.transcribe_worker import drop_silent_segments

SR = 16000


def tone(seconds: float, dbfs: float, sr: int = SR) -> np.ndarray:
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def seg(start, end, text="hallo"):
    return Segment(start=start, end=end, text=text, speaker="Me")


@pytest.fixture
def speech_then_silence():
    """0-2 s speech at -20 dBFS, 2-10 s digital silence."""
    return np.concatenate([tone(2.0, -20.0), silence(8.0)])


def test_keeps_a_segment_that_sits_on_speech(speech_then_silence):
    kept = drop_silent_segments([seg(0.0, 2.0)], speech_then_silence, SR)

    assert len(kept) == 1


def test_drops_a_segment_that_sits_on_silence(speech_then_silence):
    kept = drop_silent_segments([seg(5.0, 6.0, "Vielen Dank.")], speech_then_silence, SR)

    assert kept == []


def test_keeps_the_real_one_and_drops_the_invented_one(speech_then_silence):
    segments = [seg(0.0, 2.0, "echt"), seg(5.0, 6.0, "Vielen Dank.")]

    kept = drop_silent_segments(segments, speech_then_silence, SR)

    assert [s.text for s in kept] == ["echt"]


def test_keeps_a_quiet_but_real_utterance():
    """Speech 12 dB below the loud part must survive -- it is still speech."""
    audio = np.concatenate([tone(2.0, -20.0), silence(2.0), tone(2.0, -32.0), silence(2.0)])

    kept = drop_silent_segments([seg(4.0, 6.0, "leise")], audio, SR)

    assert [s.text for s in kept] == ["leise"]


def test_returns_segments_unchanged_when_there_is_no_audio():
    segments = [seg(0.0, 1.0)]

    assert drop_silent_segments(segments, np.zeros(0, dtype=np.float32), SR) == segments


def test_returns_segments_unchanged_when_the_track_is_pure_digital_silence():
    """Nothing to compare against -- do not silently throw the transcript away."""
    segments = [seg(0.0, 1.0)]

    assert drop_silent_segments(segments, silence(5.0), SR) == segments


def test_no_segments_yields_no_segments(speech_then_silence):
    assert drop_silent_segments([], speech_then_silence, SR) == []


def test_a_segment_running_past_the_end_of_the_audio_does_not_crash(speech_then_silence):
    kept = drop_silent_segments([seg(9.5, 30.0, "Tschuess.")], speech_then_silence, SR)

    assert kept == []


# --- wiring ------------------------------------------------------------------

class TwoSegmentModel:
    """Returns one segment on speech and one invented onto silence."""

    def transcribe(self, audio, **kw):
        return {
            "language": "de",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "echte Aussage"},
                {"start": 5.0, "end": 6.0, "text": "Vielen Dank."},
            ],
        }


def test_transcribe_drops_the_segment_that_landed_on_silence(tmp_path, monkeypatch):
    import whisperx
    from audiologger.transcribe_worker import WhisperXPipeline

    audio = np.concatenate([tone(2.0, -20.0), silence(8.0)])
    monkeypatch.setattr(whisperx, "load_audio", lambda p: audio)

    pipe = WhisperXPipeline(
        model_size="large-v3", device="cpu", compute_type="int8",
        diarization_enabled=False, hf_token=None, language="de",
    )
    pipe._models["large-v3"] = TwoSegmentModel()
    f = tmp_path / "mic.wav"
    f.touch()

    result = pipe.transcribe(f, diarize=False, align=False, language="de")

    assert [s.text for s in result.segments] == ["echte Aussage"]


# --- threshold calibration ---------------------------------------------------
# Measured on three real recordings: genuine speech reached -22.7 dB below the
# track's own speech level ("Beide Zeiten.", which follows coherently from what
# the other speaker just said), while invented segments started at -27.8 dB and
# ran down to -60 dB. The default sits in that gap.

def test_keeps_genuine_speech_that_is_22_db_below_the_speech_level():
    audio = np.concatenate([tone(2.0, -20.0), tone(2.0, -42.0), silence(2.0)])

    kept = drop_silent_segments([seg(2.0, 4.0, "Beide Zeiten.")], audio, SR)

    assert [s.text for s in kept] == ["Beide Zeiten."]


def test_drops_an_invented_segment_30_db_below_the_speech_level():
    audio = np.concatenate([tone(2.0, -20.0), tone(2.0, -50.0), silence(2.0)])

    kept = drop_silent_segments([seg(2.0, 4.0, "Vielen Dank.")], audio, SR)

    assert kept == []
