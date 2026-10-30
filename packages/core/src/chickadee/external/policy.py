"""The one checkpoint a call leaving the machine on a model's behalf passes.

`ExternalPolicy.run(call, perform, reply)` classifies the call, redacts its arguments for the
record, runs it, and records one `external_call` line with what came back as a blob, so what a
model was shown is recoverable and the citation check reads it back.
`ExternalPolicy.shown(...)` frames text from outside in the one untrusted envelope and records
the crossing, an `untrusted_read` line: a page fetched once and shown in parts is one call and
one crossing per part.

The web uses it now; the MCP client is its second user, so nothing here is the
web's. Every web call is `DISCOVERY`, approved by nobody because a read is self-securing.
`ACTION` is the shape `T-5.1` fills under D40's lifecycle; until it does, one reaching the
policy is a `HarnessBug`, never a call made without that lifecycle.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from chickadee.context.record_lines import ExternalCallLine
from chickadee.context.vocabulary import Line
from chickadee.errors import HarnessBug
from chickadee.executor.types import PolicyClass
from chickadee.schema.journal import EventKind, ExternalCallP
from chickadee.secrets.redact import redact_text
from chickadee.tools.catalogue import args_sha
from chickadee.trust.events import UntrustedRead
from chickadee.trust.regions import envelope

__all__ = ["CallReply", "ExternalCall", "ExternalPolicy", "ExternalRecorder"]


class ExternalRecorder(Protocol):
    """Where the caller journals: for an agent, its session (`context/session.Session`)."""

    def put_blob(self, data: bytes) -> str: ...

    def append(self, line: Line) -> object: ...


@dataclass(frozen=True)
class ExternalCall:
    """One call to make: the tool call it serves, who answers it and what is asked."""

    operation_id: str
    provider: str
    """`web.fetch.http`, `web.search.github`; later `mcp:<server>`."""
    tool: str
    arguments: Mapping[str, object] = field(default_factory=dict[str, object])
    policy: PolicyClass = PolicyClass.DISCOVERY


@dataclass(frozen=True)
class CallReply:
    """What the record says came back: the text, or the failure's name."""

    text: str | None
    """None when nothing came back."""
    truncated: bool = False
    error: str | None = None


class ExternalPolicy:
    """One caller's checkpoint: its recorder, and the rung whose model reads what comes back.
    Safe across threads when its recorder is (the session's is locked)."""

    def __init__(
        self,
        recorder: ExternalRecorder,
        *,
        rung: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recorder = recorder
        self._rung = rung
        self._clock = clock

    def run[T](
        self, call: ExternalCall, perform: Callable[[], T], reply: Callable[[T], CallReply]
    ) -> T:
        """`perform`'s answer, after one `external_call` line recording it as `reply` reads it.
        An exception out of `perform` is a bug in the code that made it, and is not recorded
        as a call that came back."""
        if call.policy is not PolicyClass.DISCOVERY:
            raise HarnessBug(
                f"{call.provider}: a {call.policy.value} call reached the external policy, "
                "which runs discovery calls only until T-5.1 gives actions their lifecycle"
            )
        asked = args_sha({k: _redacted(v) for k, v in call.arguments.items()})
        started = self._clock()
        answer = perform()
        wall_ms = max(0, round((self._clock() - started) * 1000))
        said = reply(answer)
        body = None if said.text is None else said.text.encode()
        payload = ExternalCallP(
            operation_id=call.operation_id,
            provider=call.provider,
            tool=call.tool,
            args_sha=asked,
            policy="discovery",
            ok=said.error is None,
            result_sha=None if body is None else self._recorder.put_blob(body),
            result_bytes=0 if body is None else len(body),
            wall_ms=wall_ms,
            truncated=said.truncated,
            error=said.error,
        )
        self._recorder.append(ExternalCallLine(EventKind.EXTERNAL_CALL, payload))
        return answer

    def shown(self, tag: str, body: str, *, source: str, span: str, **attrs: str) -> str:
        """`body` in the untrusted `<tag>` region, after one `untrusted_read` line naming
        `source`, the `span` shown and this policy's rung."""
        self._recorder.append(
            UntrustedRead(path=source, bytes=len(body.encode()), span=span, rung=self._rung)
        )
        return envelope(tag, body, **attrs)


def _redacted(value: object) -> object:
    """Every string in `value` with its credentials replaced (`secrets/redact`), so a record
    never holds what a call was sent with."""
    if isinstance(value, str):
        return redact_text(value)[0]
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {str(k): _redacted(v) for k, v in items}
    if isinstance(value, list | tuple):
        return [_redacted(v) for v in cast("Sequence[object]", value)]
    return value
