"""The files on the draft, one chip each, above the composer: `[1] mockup.png · 1440x900 ·
~1,730 tok`, how Zeus would get it when that is less than
the file, a secret, and the total. Drawn only while something is staged."""

from __future__ import annotations

from collections.abc import Sequence

from textual.widgets import Static

from chickadee_cli.attach import Staged, chip_lines

__all__ = ["Chips"]


class Chips(Static):
    DEFAULT_CSS = """
    Chips { height: auto; color: $text-muted; display: none; }
    Chips.-shown { display: block; }
    """

    def show(self, staged: Sequence[Staged], over: str | None) -> None:
        rows = chip_lines(staged, over)
        self.update("\n".join(rows))
        self.set_class(bool(rows), "-shown")
