"""Merge mic + system segments into a Markdown transcript."""
from typing import Iterable, Sequence

from audiologger.screenshot_watch import Screenshot
from audiologger.segment import Segment


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def merge_segments(
    mic: Iterable[Segment], system: Iterable[Segment]
) -> list[Segment]:
    """Stable chronological merge. Mic wins ties (mic appears first)."""
    tagged = [(s.start, 0, s) for s in mic] + [(s.start, 1, s) for s in system]
    tagged.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in tagged]


def render_markdown(
    segments: list[Segment],
    *,
    recorded_at: str,
    duration_seconds: int,
    source_label: str,
    model_label: str,
    warnings: list[str],
    screenshots: Sequence[Screenshot] = (),
) -> str:
    """Render the final transcript Markdown matching the spec format.

    Screenshots are placed among the speech lines by time, compared in whole
    seconds as the timestamps are shown; a speech line in the same second comes
    first, and screenshots within one second keep the caller's order (the order
    find_screenshots returns). `segments` must be in start order, as
    merge_segments returns them. Without screenshots, no screenshot header or
    lines are added.
    """
    duration_str = format_timestamp(duration_seconds)
    # Drop the leading "00:" for short recordings — keep it consistent: spec
    # showed "47:21" for sub-hour. Strip leading "00:" only if hours == 0.
    if duration_str.startswith("00:"):
        duration_str = duration_str[3:]

    shots = sorted(screenshots, key=lambda s: s.at_s)
    lines = [
        f"# Recording {recorded_at}",
        "",
        f"**Duration:** {duration_str}",
        f"**Source:** {source_label}",
        f"**Model:** {model_label}",
    ]
    if shots:
        lines.append(f"**Screenshots:** {len(shots)}")
    for w in warnings:
        lines.append(f"**Warning:** {w}")
    lines.extend(["", "---", ""])
    next_shot = 0
    for seg in segments:
        while next_shot < len(shots) and shots[next_shot].at_s < int(seg.start):
            lines.append(_screenshot_line(next_shot + 1, shots[next_shot]))
            next_shot += 1
        ts = format_timestamp(seg.start)
        lines.append(f"**[{ts}] {seg.speaker}:** {seg.text}")
    for i in range(next_shot, len(shots)):
        lines.append(_screenshot_line(i + 1, shots[i]))
    return "\n".join(lines) + "\n"


def _screenshot_line(number: int, shot: Screenshot) -> str:
    return (f"**[{format_timestamp(shot.at_s)}] Screenshot {number}:** "
            f"![Screenshot {number}]({shot.path})")
