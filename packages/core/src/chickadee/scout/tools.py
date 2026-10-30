"""The scout's surface: the ladder's shared tools, and its own two.

The architecture puts rungs 1-4 of the targeting ladder inside the scout, so its
surface is those tools — `symbol`, `grep`, `glob`, `neighbors`, `impact`, `read`, `grep_in` —
reached through the **same** implementations Zeus's calls reach: one tool implementation,
not three forks. They arrive through the ports the caller supplies, because the scout owns none
of them: `read` and `grep_in` read the `files` port (`context.Reader`), the graph tools the
`navigator` port, Zeus's own navigator handed over as it is. Each is bounded once, at source;
nothing here cuts a shared tool's result a second time.

The scout's own is `request_exec` (`escapes.py`); `grep` and `glob`, text and path search
including files the graph does not index, are every role's over a `search` port this
module supplies. Never on the surface: `atlas` (the snapshot is the
scout's map), `recall`, `locate` (a scout never spawns a scout), `shell`, `coder`, `ask_user`,
and the record tools.

Every call goes through one dispatcher per `locate` call, over `ROLES`' `locate` policy:
its count, its caps, its dedup (an identical call answers with a marker instead of running
again) and its no-progress streak are that pipeline's rules, and each tool is the
registry's: the shared ones Zeus's own, `request_exec` the scout's, each over a port this module
supplies.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Literal

from chickadee.adapters.base import tool_descriptions
from chickadee.adapters.frontier import TEMPLATES
from chickadee.client.request import DocumentPart, ImagePart, ToolSchema
from chickadee.context.results import ToolResult
from chickadee.errors import HarnessBug
from chickadee.hooks.fire import RoleHooks
from chickadee.limits import ResolvedLimit, limit, resolved_value, tokens_for_chars
from chickadee.roles import role
from chickadee.schema import sha256_hex
from chickadee.schema.adapter import ToolCall
from chickadee.schema.journal import EventKind, ScoutToolP
from chickadee.scout.escapes import EscapeRequest, Escapes
from chickadee.scout.search import Search
from chickadee.scout.sink import ScoutSink
from chickadee.scout.snapshot import SNAPSHOT_TOKENS
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import NoProgress, answer_text, call_key, result_tokens
from chickadee.tools.context import FileView, Navigator, ToolContext
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.policy import RolePolicy
from chickadee.tools.rules import (
    Answered,
    BoundReached,
    CallOutcome,
    Level,
    Pass,
    Pending,
    ScopeState,
    Stop,
    Verdict,
)
from chickadee.tools.sink import Counters, no_hold
from chickadee.tools.spec import INLINE_BYTES_DEFAULT, RESULT_LINES_DEFAULT, ToolSpec
from chickadee.tools.tool import PORTS, Invocation, Port
from chickadee.tools.validate import InvalidArguments

__all__ = [
    "LocateCaps",
    "ScoutTools",
    "ToolAnswer",
]


# -- the bounds -------------------------------------------------------------------------


@dataclass(frozen=True)
class LocateCaps:
    """One `locate` call's bounds. The first two are the limits register's rows;
    the rest bound the scout's own `search`."""

    max_tool_calls: int
    max_result_tokens: int
    """`locate.max_cumulative_result_tokens`, resolved to tokens on the scout's rung."""
    search_hits: int = RESULT_LINES_DEFAULT
    search_bytes: int = INLINE_BYTES_DEFAULT
    no_progress: int = 3
    """This many consecutive dedup hits ends the loop early."""
    snapshot_tokens: int = SNAPSHOT_TOKENS
    """`locate.snapshot_tokens`: the budget `build_snapshot` meets. The register's
    default, or the resolved value when the caller holds the config."""

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1 or self.max_result_tokens < 1:
            raise HarnessBug("a locate call is bounded by positive caps (D48)")

    def limits(self) -> Mapping[str, int]:
        """The caps by the register's names, as the `locate` pipeline's rules read them."""
        return {
            "locate.max_tool_calls": self.max_tool_calls,
            "locate.max_cumulative_result_tokens": self.max_result_tokens,
            "locate.no_progress": self.no_progress,
        }

    @staticmethod
    def from_resolved(resolved: Sequence[ResolvedLimit]) -> LocateCaps:
        """The caps as the config resolved them — overrides applied, the fraction sized by
        the model the scout's rung runs."""
        return LocateCaps(
            resolved_value(resolved, "locate", "locate.max_tool_calls"),
            resolved_value(resolved, "locate", "locate.max_cumulative_result_tokens"),
            snapshot_tokens=resolved_value(resolved, "locate", "locate.snapshot_tokens"),
        )

    @staticmethod
    def for_window(window: int) -> LocateCaps:
        """The register's defaults on a window of `window` tokens, for a caller holding no
        config."""
        calls = int(limit("locate.max_tool_calls").value)
        fraction = float(limit("locate.max_cumulative_result_tokens").value)
        return LocateCaps(calls, max(1, int(fraction * window)))


# -- the surface ------------------------------------------------------------------------


Ended = Literal["cap", "no_progress"]
"""Why a search ended on a call: a cap (calls, result tokens, an escape's room), or no
progress."""


@dataclass(frozen=True)
class ToolAnswer:
    tool: str
    text: str
    """What the scout's model reads."""
    tokens: int
    """What the call's answer measured, withheld or not: what its `scout_tool` line records."""
    dedup_hit: bool = False
    searched: str | None = None
    """What this call looked for, for `UNRESOLVED`'s account of what was searched."""
    ended: Ended | None = None
    """Why the search ended on this call, when it did: a cap, or no progress."""
    image: ImagePart | DocumentPart | None = None
    """The pixels of an image the scout's reader returned at `depth="full"`, or a PDF
    on a model that reads PDFs: sent
    beside `text`, which names it, as Zeus's context sends one (`context/request.py`)."""


_UNCAPPED = LocateCaps(max_tool_calls=1 << 30, max_result_tokens=1 << 30)
"""The caps of a `ScoutTools` used outside the loop: its tools alone, as a test exercises
them. The loop always gives its own (`begin(caps)`)."""

_EXEC_PORTS: frozenset[Port] = frozenset({"exec_request"})


def _bound(what: str, used: int, limit: int) -> str:
    return f"BOUND_REACHED{{bound: {what}, used: {used}, limit: {limit}}} — not dispatched"


def _subject(name: str, args: Mapping[str, object]) -> str:
    for key in ("query", "name", "path", "handle", "target", "symbols", "pattern", "command"):
        if key in args:
            return f"{name} {args[key]!r}"
    return name


@dataclass
class _Now:
    """What the call being dispatched was given: where its records go and its stop. One
    `locate` call's tools serve one loop on one thread, so this is never shared."""

    sink: ScoutSink | None = None
    cancel: threading.Event | None = None


class _EscapePort:
    """The `exec_request` port over the scout's `Escapes` for one `locate` call: the request
    is granted or refused by their policy, and the escape's records go to this call's sink."""

    def __init__(self, escapes: Escapes, now: _Now) -> None:
        self._escapes = escapes
        self._now = now

    def request(self, command: str, reason: str) -> str:
        req = EscapeRequest.of(command, reason)
        if isinstance(req, InvalidArguments):
            return req.as_content()
        sink = self._now.sink
        if sink is None:
            raise HarnessBug("an escape ran with no sink to record it")
        return self._escapes.request(req, sink, self._now.cancel).render()


@dataclass(frozen=True)
class _EscapeRoom:
    """The scout's own admit rule, its pipeline's `escape_room` slot: a call whose run is an
    effect (`effects="self"`, `request_exec`) is refused before it runs when its capped output
    could not fit what the `locate` call may still read. Once run, its budget is spent and
    its run recorded, so it is never run and then withheld."""

    escapes: Escapes | None
    key: str = "locate.max_cumulative_result_tokens"
    stage: Literal["admit", "dispatch"] = "admit"

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        spec = pending.spec
        if spec is None or spec.effects != "self" or self.escapes is None:
            return Pass()
        need = self.escapes.room(pending.raw.get("command"))
        limit = state.limit(self.key)
        used = state.tokens("call")
        if need and used + need > limit:
            return Stop(self.key, BoundReached("result_tokens", used + need, limit, "call"))
        return Pass()


class _ToolLines:
    """The scout's sink: its bytes, its `scout_tool` line, and no replay —
    each `locate` call starts at zero. A call stopped before it reached a tool is answered
    and not recorded; every other call, withheld or not, records what it measured."""

    def __init__(self, now: _Now, max_calls: int, calls: Callable[[], int]) -> None:
        self._now = now
        self._max_calls = max_calls
        self._calls = calls
        self._ended = False

    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        return no_hold(spec)

    def replay(self, level: Level) -> Counters:
        return Counters()

    def render(self, outcome: CallOutcome) -> str:
        stop = outcome.ended_by
        if stop is None:
            return answer_text(outcome)
        if self._ended:
            return self.search_ended()
        failure = stop.failure
        if isinstance(failure, BoundReached):
            return _bound(stop.reason, failure.used, failure.limit)
        return answer_text(replace(outcome, stop=None))

    def search_ended(self) -> str:
        """What a call after the search ended is answered: it is never dispatched."""
        return _bound("the search ended", self._calls(), self._max_calls)

    def record(self, call: Invocation, outcome: CallOutcome) -> ToolAnswer:
        pending = outcome.pending
        decided = pending.decided
        text = self.render(outcome)
        stop = outcome.ended_by
        ended: Ended | None = None
        if stop is not None and not self._ended:
            ended = "cap" if isinstance(stop.failure, BoundReached) else "no_progress"
        if stop is not None:
            self._ended = True
        if isinstance(decided, Stop):
            # The loop never made this call: nothing reached a tool, nothing is recorded.
            return ToolAnswer(pending.sent, text, tokens_for_chars(len(text)), ended=ended)
        whole = replace(outcome, answer=outcome.withheld or outcome.answer, stop=None)
        tokens = result_tokens(whole)
        dedup = isinstance(decided, Answered)
        ran = pending.tool is not None and not dedup
        sink = self._now.sink
        if sink is None:
            raise HarnessBug("a scout tool call was recorded with no sink")
        sink.record(
            EventKind.SCOUT_TOOL,
            ScoutToolP(
                tool=pending.sent,
                args_sha=sha256_hex(call_key(pending)),
                result_tokens=tokens,
                dedup_hit=dedup,
            ),
        )
        shown = outcome.answer if stop is None else None
        return ToolAnswer(
            pending.sent,
            text,
            tokens,
            dedup,
            _subject(pending.sent, pending.raw) if ran else None,
            ended,
            shown.image if isinstance(shown, ToolResult) else None,
        )


class ScoutTools:
    """The tools of one `locate` call, through one dispatcher over `ROLES`' `locate` policy:
    its caps, its dedup, its count and its no-progress streak are that
    pipeline's rules, and each tool is the registry's, the shared ones Zeus's own.
    `escapes` is None when the policy offers none — trust, or a model
    without function calling — and `request_exec` is then never offered.

    `begin(caps)` opens a call: a fresh dispatcher, so a fresh count, dedup cache and streak,
    and a fresh escape budget; what the session knows (the escapes' memo) carries over. One
    `locate` call's tools serve one loop on one thread and are never shared."""

    def __init__(
        self,
        navigator: Navigator,
        reader: FileView,
        search: Search,
        *,
        escapes: Escapes | None = None,
        policy: RolePolicy | None = None,
        hooks: RoleHooks | None = None,
    ) -> None:
        """`policy` is the session's `[tools.locate]` as the config resolves it; the cast's
        default when none is given (a test's composition). `hooks`
        are the scout's, at its calls' points."""
        self.escapes = escapes
        self._now = _Now()
        self._ctx = ToolContext(
            role="locate",
            files=reader,
            navigator=navigator,
            search=search,
            exec_request=None if escapes is None else _EscapePort(escapes, self._now),
            hooks=hooks,
        )
        chosen = policy if policy is not None else role("locate").tool_defaults
        if chosen is None:
            raise HarnessBug("ROLES gives the locate role no tools")
        self._policy = replace(
            chosen,
            pipeline=chosen.pipeline.fill({"escape_room": _EscapeRoom(escapes)}),
        )
        self._no_progress = next(
            (r for r in self._policy.pipeline.result if isinstance(r, NoProgress)), None
        )
        self._ids = itertools.count(1)
        self._offer: bool | None = None
        self.begin()

    def begin(self, caps: LocateCaps | None = None) -> None:
        """Start one `locate` call."""
        caps = caps or _UNCAPPED
        self._lines = _ToolLines(self._now, caps.max_tool_calls, lambda: self.calls)
        self._dispatcher = Dispatcher[ToolAnswer](
            BUILTINS,
            self._policy,
            self._ctx,
            self._lines,
            limits=caps.limits(),
        )
        self._dispatcher.open_scope("call")
        self._offer = None
        if self.escapes is not None:
            self.escapes.begin()

    @property
    def calls(self) -> int:
        """The calls this `locate` call has made: every one the loop did not stop."""
        return self._dispatcher.state.calls("call")

    @property
    def dedup_streak(self) -> int:
        """Consecutive calls answered from the dedup cache: the `NoProgress` rule's."""
        if self._no_progress is None:
            return 0
        return self._no_progress.streak(self._dispatcher.state)

    def search_ended(self) -> str:
        """What a tool call after the search ended is answered: never dispatched."""
        return self._lines.search_ended()

    def names(self, *, offer_exec: bool) -> tuple[str, ...]:
        return tuple(s.name for s in self._offered(offer_exec))

    def registry_name(self, shown: str) -> str | None:
        """The tool a call names, as the dispatcher reads it."""
        return self._policy.registry_name(shown)

    def schemas(self, *, offer_exec: bool) -> tuple[ToolSchema, ...]:
        # The scout's own descriptions first (it has `search`, never a shell), then the shared.
        offered = self._offered(offer_exec)
        described = tool_descriptions(offered, TEMPLATES)
        return tuple(
            ToolSchema(name=s.name, description=described[s.description], schema=s.schema)
            for s in offered
        )

    def call(
        self,
        name: str,
        arguments: Mapping[str, object],
        sink: ScoutSink,
        *,
        offer_exec: bool,
        cancel: threading.Event | None = None,
    ) -> ToolAnswer:
        """One call through the pipeline, recorded `scout_tool`. A name the surface does not
        offer is answered, not raised: the model asked for a tool it was not given, and
        telling it so is the correction."""
        self._offered(offer_exec)
        self._now.sink, self._now.cancel = sink, cancel
        try:
            call = ToolCall(id=f"s{next(self._ids)}", name=name, arguments=dict(arguments))
            return self._dispatcher.call(call)
        finally:
            self._now.sink = self._now.cancel = None

    def _offered(self, offer_exec: bool) -> tuple[ToolSpec, ...]:
        """What this call's requests offer: `request_exec` only when escapes are offered.
        The loop offers the same set for a whole `locate` call (the provider's cached prefix);
        a change opens a new request scope and keeps the call's counts, and a search that has
        ended stays ended: a new offer is not a new search."""
        offer = offer_exec and self.escapes is not None
        if offer != self._offer:
            given = PORTS if offer else PORTS - _EXEC_PORTS
            ended = self._dispatcher.stopped
            self._dispatcher.open_scope("request", given)
            self._dispatcher.stopped = ended
            self._offer = offer
        return self._dispatcher.offered()
