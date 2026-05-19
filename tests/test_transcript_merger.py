from audiologger.segment import Segment
from audiologger.transcript_merger import (
    merge_segments,
    format_timestamp,
    render_markdown,
)


def test_format_timestamp_short():
    assert format_timestamp(3.4) == "00:00:03"
    assert format_timestamp(125.0) == "00:02:05"
    assert format_timestamp(3725.0) == "01:02:05"


def test_merge_chronological_order():
    mic = [Segment(5.0, 6.0, "Hallo", "Me")]
    sys = [Segment(0.0, 4.0, "Was?", "Speaker 1")]
    merged = merge_segments(mic, sys)
    assert [s.start for s in merged] == [0.0, 5.0]
    assert [s.speaker for s in merged] == ["Speaker 1", "Me"]


def test_merge_overlap_orders_by_start():
    mic = [Segment(1.0, 5.0, "A", "Me")]
    sys = [Segment(2.0, 4.0, "B", "Speaker 1")]
    merged = merge_segments(mic, sys)
    assert [s.text for s in merged] == ["A", "B"]


def test_merge_stable_for_equal_start():
    mic = [Segment(1.0, 2.0, "M", "Me")]
    sys = [Segment(1.0, 2.0, "S", "Speaker 1")]
    merged = merge_segments(mic, sys)
    # mic first when ties — implementation choice, document it
    assert merged[0].speaker == "Me"


def test_render_markdown_basic():
    segments = [
        Segment(3.0, 4.0, "Hi zusammen", "Me"),
        Segment(6.0, 7.0, "Hallo", "Speaker 1"),
    ]
    md = render_markdown(
        segments,
        recorded_at="2026-05-18 14:32:15",
        duration_seconds=420,
        source_label="mic + system (loopback, all)",
        model_label="WhisperX large-v3 + pyannote/speaker-diarization-3.1",
        warnings=[],
    )
    assert "# Recording 2026-05-18 14:32:15" in md
    assert "**Duration:** 07:00" in md
    assert "**[00:00:03] Me:** Hi zusammen" in md
    assert "**[00:00:06] Speaker 1:** Hallo" in md


def test_render_markdown_includes_warnings():
    md = render_markdown(
        [],
        recorded_at="2026-05-18 14:32:15",
        duration_seconds=10,
        source_label="mic only",
        model_label="WhisperX large-v3",
        warnings=["System audio not available"],
    )
    assert "System audio not available" in md
