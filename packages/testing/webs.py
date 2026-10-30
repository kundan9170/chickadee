"""Doubles for the web's tests, one home for the external package's and the agents':
a session to record into, the lines it holds by kind, a resolver that finds every host public,
and an engine that answers from a dict."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from chickadee.context.record_lines import EXTERNAL_CALL, ExternalCallLine
from chickadee.context.results import FetchFailed
from chickadee.context.session import Session
from chickadee.external.fetch import Fetched, Moved
from chickadee.external.urls import Admitted
from chickadee.schema.journal import EventKind, ExternalCallP
from chickadee.trust.events import UNTRUSTED_READ, UntrustedRead

PUBLIC = "93.184.216.34"


def public(host: str, port: int) -> Sequence[str]:
    del host, port
    return (PUBLIC,)


@contextmanager
def session(tmp_path: Path) -> Iterator[Session]:
    opened = Session.create(tmp_path / "sessions", "S-web")
    try:
        yield opened
    finally:
        opened.close()


def _of[T](session: Session, kind: str, model: type[T]) -> list[T]:
    adapter: TypeAdapter[Any] = TypeAdapter(model)
    return [
        adapter.validate_json(session.blob(e.payload_sha))
        for e, _ in session.state.lines()
        if e.kind == kind
    ]


def calls(session: Session) -> list[ExternalCallP]:
    """Each `external_call` line's payload, in order."""
    out: list[ExternalCallP] = []
    for line in _of(session, EXTERNAL_CALL, ExternalCallLine):
        assert line.event is EventKind.EXTERNAL_CALL and isinstance(line.payload, ExternalCallP)
        out.append(line.payload)
    return out


def reads(session: Session) -> list[UntrustedRead]:
    return _of(session, UNTRUSTED_READ, UntrustedRead)


@dataclass
class DictEngine:
    """An engine answering each URL from `pages`: text, a redirect's target, or a failure."""

    pages: Mapping[str, str | Moved | FetchFailed]
    asked: list[str] = field(default_factory=list[str])
    provider: str = "web.fetch.test"

    def get(self, target: Admitted) -> Fetched | Moved | FetchFailed:
        self.asked.append(target.url)
        page = self.pages[target.url]
        return Fetched(target.url, page, False) if isinstance(page, str) else page
