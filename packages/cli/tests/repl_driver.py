"""Drives the REPL as a person would, line by line, against a hosted engine: each line waits
for what the person would wait for (an answer printed, a question shown), so a test reads the
same conversation every time."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from engines import Step
from hosting import SID, hosted

from chickadee_cli.render.style import Style
from chickadee_cli.repl import Repl
from chickadee_cli.session import transcript_path

WAIT_S = 20.0


class CtrlC:
    """A line that is a Ctrl-C."""


@dataclass
class Typed:
    text: str | CtrlC
    after: Callable[[list[str]], bool] = lambda _out: True
    """Wait for this to hold over the output so far before the line is typed."""


@dataclass
class Script:
    """An `InputSource` over `lines`; Ctrl-D after the last."""

    lines: Sequence[Typed]
    out: list[str] = field(default_factory=list[str])
    transcript: Path | None = None
    """Where the REPL wrote what it printed; set once the REPL has run."""
    defaults: list[str] = field(default_factory=list[str])
    """Each non-empty `default` a read was offered: words put back to edit."""
    _next: int = 0

    def read(self, prompt_text: str, default: str = "") -> str:
        if default:
            self.defaults.append(default)
        if self._next >= len(self.lines):
            raise EOFError
        line = self.lines[self._next]
        self._next += 1
        deadline = time.monotonic() + WAIT_S
        while not line.after(self.out):
            assert time.monotonic() < deadline, f"waited for {line}; output:\n" + "\n".join(
                self.out
            )
            time.sleep(0.02)
        if isinstance(line.text, CtrlC):
            raise KeyboardInterrupt
        return line.text


def printed(text: str) -> Callable[[list[str]], bool]:
    """The output holds a line containing `text`."""
    return lambda out: any(text in line for line in out)


@contextmanager
def repl(
    monkeypatch: pytest.MonkeyPatch, script: Sequence[Step], lines: Sequence[Typed]
) -> Iterator[tuple[Script, int]]:
    """The REPL on session `SID` of a hosted engine answering from `script`, fed `lines`,
    as a pipe sees it: no colour, no progress. Yields the source (its `out`) and the code."""
    source = Script(lines)
    with hosted(monkeypatch, script) as host, host.client() as client:
        session = Repl(
            client,
            SID,
            source=source,
            write=source.out.append,
            style=Style(tty=False),
            state_dir=host.state_dir,
        )
        code = session.run()
        source.transcript = transcript_path(host.state_dir, SID)
        yield source, code
