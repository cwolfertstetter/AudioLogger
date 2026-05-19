"""Transcript segment with speaker label and timing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: float       # seconds from recording start
    end: float
    text: str
    speaker: str       # "Ich" | "Sprecher 1" | "Sprecher 2" | "Andere" | ...
