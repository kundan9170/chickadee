"""How a line looks where it lands: a terminal gets colour; a pipe gets plain
text hard-wrapped at `PIPE_WIDTH`, so nothing a script reads depends on a layout."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = ["PIPE_WIDTH", "Style", "Tone"]

PIPE_WIDTH: Final = 80
"""Line-oriented rendering, hard-wrapped at 80; no layout dependence."""


class Tone(StrEnum):
    """What a line is, which a terminal shows by colour and a pipe not at all."""

    PLAIN = ""
    DIM = "2"
    PERSON = "1"
    ASK = "33"
    BAD = "31"
    GOOD = "32"


@dataclass(frozen=True)
class Style:
    tty: bool

    def lines(self, text: str, tone: Tone = Tone.PLAIN, *, indent: str = "") -> list[str]:
        """`text` as the lines this destination prints, each prefixed by `indent`."""
        out: list[str] = []
        for raw in text.splitlines() or [""]:
            if self.tty:
                out.append(self._paint(indent + raw, tone))
            else:
                width = max(PIPE_WIDTH - len(indent), 20)
                wrapped = textwrap.wrap(raw, width, break_on_hyphens=False) or [""]
                out.extend(indent + w for w in wrapped)
        return out

    def verbatim(self, line: str, tone: Tone = Tone.PLAIN) -> str:
        """One line as it is, never wrapped or stripped: a diff's line means its whitespace."""
        return self._paint(line, tone) if self.tty else line

    @staticmethod
    def _paint(line: str, tone: Tone) -> str:
        return line if tone is Tone.PLAIN else f"\x1b[{tone.value}m{line}\x1b[0m"
