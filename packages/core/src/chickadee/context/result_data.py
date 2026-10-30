"""What a tool result was made from: the rows a `grep` or `glob` showed, a
command's end, handed to the worker's Python blocks beside the text. Its own module, apart from
the failures and results of `results.py`, which only refer to it."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Completed", "Match", "Matches", "Paths", "ResultData", "Structured"]


@dataclass(frozen=True)
class Match:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class Matches:
    """`grep`'s hits, exactly the rows its text shows: after the deny list and the
    search's own bound, so it never holds a hit the text left out."""

    rows: tuple[Match, ...]


@dataclass(frozen=True)
class Paths:
    """`glob`'s paths, exactly the rows its text shows."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class Completed:
    """A `shell` command's end, named as `subprocess.CompletedProcess` names it, since that is
    what a model was trained on. `stdout` and `stderr` are None when the output was cut: the
    text shows the cut, and the data never holds more than it."""

    returncode: int
    stdout: str | None
    stderr: str | None


ResultData = Matches | Paths | Completed
"""The data a result was made from, handed to the worker's Python blocks beside its text.
A native call is shown the text alone, as ever."""


class Structured(str):
    """Text a producer hands with the data it was made from, where its port returns `str` (the
    search port): `tools.tool.as_text` moves the data onto the answer, typed, and nothing
    downstream reads this class. A string, so a caller that wants text has text."""

    data: ResultData

    def __new__(cls, text: str, data: ResultData) -> Structured:
        made = super().__new__(cls, text)
        made.data = data
        return made
