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


# --- screenshots ------------------------------------------------------------------

from audiologger.screenshot_watch import Screenshot

HEADER = dict(
    recorded_at="2026-10-02 10:00:00",
    duration_seconds=900,
    source_label="mic + system (loopback)",
    model_label="WhisperX large-v3",
    warnings=[],
)
SPEECH = [
    Segment(748.0, 752.0, "Hier seht ihr die neue Rezeptstruktur.", "Speaker 1"),
    Segment(755.0, 757.0, "Okay, und wo kommt der Teig rein?", "Me"),
]


def body(md: str) -> list[str]:
    return md.split("---\n\n", 1)[1].splitlines()


def test_without_screenshots_the_transcript_is_exactly_as_before():
    assert render_markdown(SPEECH, **HEADER) == (
        "# Recording 2026-10-02 10:00:00\n"
        "\n"
        "**Duration:** 15:00\n"
        "**Source:** mic + system (loopback)\n"
        "**Model:** WhisperX large-v3\n"
        "\n"
        "---\n"
        "\n"
        "**[00:12:28] Speaker 1:** Hier seht ihr die neue Rezeptstruktur.\n"
        "**[00:12:35] Me:** Okay, und wo kommt der Teig rein?\n"
    )


def test_a_screenshot_sits_between_the_speech_lines_around_it():
    md = render_markdown(SPEECH, **HEADER,
                         screenshots=[Screenshot(751, "screenshots/screenshot_00-12-31.png")])

    assert body(md) == [
        "**[00:12:28] Speaker 1:** Hier seht ihr die neue Rezeptstruktur.",
        "**[00:12:31] Screenshot 1:** ![Screenshot 1](screenshots/screenshot_00-12-31.png)",
        "**[00:12:35] Me:** Okay, und wo kommt der Teig rein?",
    ]


def test_speech_starting_in_the_same_second_comes_first():
    md = render_markdown([Segment(751.0, 753.0, "Genau hier.", "Me")], **HEADER,
                         screenshots=[Screenshot(751, "screenshots/screenshot_00-12-31.png")])

    assert body(md)[0].startswith("**[00:12:31] Me:**")
    assert body(md)[1].startswith("**[00:12:31] Screenshot 1:**")


def test_screenshots_before_the_first_and_after_the_last_word_are_kept_in_order():
    md = render_markdown(SPEECH, **HEADER, screenshots=[
        Screenshot(800, "screenshots/screenshot_00-13-20.png"),
        Screenshot(10, "screenshots/screenshot_00-00-10.png"),
    ])

    assert body(md)[0] == (
        "**[00:00:10] Screenshot 1:** ![Screenshot 1](screenshots/screenshot_00-00-10.png)")
    assert body(md)[-1] == (
        "**[00:13:20] Screenshot 2:** ![Screenshot 2](screenshots/screenshot_00-13-20.png)")


def test_the_header_counts_the_screenshots():
    md = render_markdown(SPEECH, **HEADER, screenshots=[
        Screenshot(10, "screenshots/screenshot_00-00-10.png"),
        Screenshot(800, "screenshots/screenshot_00-13-20.png"),
    ])

    assert "**Model:** WhisperX large-v3\n**Screenshots:** 2\n" in md


def test_speech_in_the_same_displayed_second_comes_first_even_mid_second():
    """WhisperX starts are fractional; the tie rule works on whole seconds,
    as the timestamps are displayed."""
    md = render_markdown([Segment(751.4, 753.0, "Genau hier.", "Me")], **HEADER,
                         screenshots=[Screenshot(751, "screenshots/screenshot_00-12-31.png")])

    assert body(md)[0].startswith("**[00:12:31] Me:**")
    assert body(md)[1].startswith("**[00:12:31] Screenshot 1:**")
