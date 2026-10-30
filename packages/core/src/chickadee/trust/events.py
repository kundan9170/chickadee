"""The one `untrusted_read` line: untrusted text crossed into a model's context.

A third-party project skill's body and a web page write it, one line per crossing,
so "what did this rung actually read, and from where" is answered from the session log alone.
`span` says which part (`whole`, or a fetch's `chars 0-40000`) and `rung` which model read it;
a line written before they took any other value reads the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import with_config

from chickadee.context.vocabulary import LINE, SessionKind

__all__ = ["UNTRUSTED_READ", "UntrustedRead"]

UNTRUSTED_READ: Final[SessionKind] = "untrusted_read"


@with_config(LINE)
@dataclass(frozen=True)
class UntrustedRead:
    """Untrusted text crossed the trust boundary. Its own line, not a field on the read
    that carried it: the two answer different questions."""

    KIND: ClassVar[SessionKind] = UNTRUSTED_READ
    path: str
    """The file, or the URL."""
    bytes: int
    span: str = "whole"
    rung: str = "zeus"
