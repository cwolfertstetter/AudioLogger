"""Dropping Whisper's subtitle boilerplate.

The level filter catches text invented onto silence.  It cannot catch text
invented onto real audio -- breathing, background noise, the tail of a word.
On a 71-minute recording those sat at -23.3 to -2.4 dB below the speech level
while genuine content sat at -20.4 to +1.9 dB: the distributions overlap, so no
threshold separates them.

What does separate them is the text.  Whisper was trained on subtitle files and
falls back on their stock phrases -- "Vielen Dank.", "Untertitelung des ZDF,
2020", bare ellipses.  Match those, but only when they make up a whole segment:
"Vielen Dank für die Erklärung" is somebody talking.
"""
import numpy as np
import pytest

from audiologger.segment import Segment
from audiologger.transcribe_worker import drop_boilerplate_segments


def seg(text, start=0.0):
    return Segment(start=start, end=start + 1.0, text=text, speaker="Me")


def texts(segments):
    return [s.text for s in segments]


# --- the phrases actually seen in these recordings ---------------------------

@pytest.mark.parametrize("text", [
    "Vielen Dank.",
    "Untertitelung des ZDF, 2020",
    "Untertitel 2.97",
    "... ... ... ... ...",
    "Продолжение следует...",
    "Takk for at du så med.",
    "Teksting av Nicolai Winther",
])
def test_drops_boilerplate_that_makes_up_a_whole_segment(text):
    assert drop_boilerplate_segments([seg(text)]) == []


# --- genuine speech must survive ---------------------------------------------

@pytest.mark.parametrize("text", [
    "Vielen Dank für die Erklärung, das hilft mir weiter.",
    "Wir brauchen Untertitel für das Video.",
    "Also den Satz müssen wir umdrehen, ne?",
    "Ja, das stimmt.",
    "Beide Zeiten.",
])
def test_keeps_genuine_speech(text):
    assert texts(drop_boilerplate_segments([seg(text)])) == [text]


def test_keeps_the_real_line_and_drops_the_stock_phrase():
    segments = [seg("Vielen Dank.", 1.0), seg("Der hat aber hier gestimmt.", 2.0)]

    assert texts(drop_boilerplate_segments(segments)) == ["Der hat aber hier gestimmt."]


# --- matching details ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "vielen dank",
    "  Vielen   Dank  ",
    "VIELEN DANK!",
    "Vielen Dank…",
])
def test_matching_ignores_case_spacing_and_trailing_punctuation(text):
    assert drop_boilerplate_segments([seg(text)]) == []


@pytest.mark.parametrize("text", ["...", ". . .", "…", "-- --"])
def test_drops_segments_that_are_only_punctuation(text):
    assert drop_boilerplate_segments([seg(text)]) == []


def test_keeps_a_segment_that_is_only_punctuation_plus_a_word():
    assert texts(drop_boilerplate_segments([seg("... genau")])) == ["... genau"]


def test_no_segments_yields_no_segments():
    assert drop_boilerplate_segments([]) == []


# --- wiring ------------------------------------------------------------------

class BoilerplateModel:
    """One stock phrase and one real sentence, both on loud audio."""

    def transcribe(self, audio, **kw):
        return {
            "language": "de",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Vielen Dank."},
                {"start": 1.0, "end": 2.0, "text": "Der hat aber hier gestimmt."},
            ],
        }


def test_transcribe_drops_boilerplate_even_when_it_sits_on_loud_audio(tmp_path, monkeypatch):
    import whisperx
    from audiologger.transcribe_worker import WhisperXPipeline

    t = np.arange(2 * 16000) / 16000
    loud = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: loud)

    pipe = WhisperXPipeline(
        model_size="large-v3", device="cpu", compute_type="int8",
        diarization_enabled=False, hf_token=None, language="de",
    )
    pipe._models["large-v3"] = BoilerplateModel()
    f = tmp_path / "mic.wav"
    f.touch()

    result = pipe.transcribe(f, diarize=False, align=False, language="de")

    assert [s.text for s in result.segments] == ["Der hat aber hier gestimmt."]
