"""Merge mic + system segments into a Markdown transcript."""
from typing import Iterable

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
) -> str:
    """Render the final transcript Markdown matching the spec format."""
    duration_str = format_timestamp(duration_seconds)
    # Drop the leading "00:" for short recordings — keep it consistent: spec
    # showed "47:21" for sub-hour. Strip leading "00:" only if hours == 0.
    if duration_str.startswith("00:"):
        duration_str = duration_str[3:]

    lines = [
        f"# Recording {recorded_at}",
        "",
        f"**Duration:** {duration_str}",
        f"**Source:** {source_label}",
        f"**Model:** {model_label}",
    ]
    for w in warnings:
        lines.append(f"**Warning:** {w}")
    lines.extend(["", "---", ""])
    for seg in segments:
        ts = format_timestamp(seg.start)
        lines.append(f"**[{ts}] {seg.speaker}:** {seg.text}")
    return "\n".join(lines) + "\n"
