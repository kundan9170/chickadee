"""The shared rule catalogue: each written once, parameterised by what
differs between roles, and placed in a role's `Pipeline` (`rules.py`).

| Rule | Stage | Runs for, or replaces |
|---|---|---|
| `CallCap` | admit | Zeus; the scout loop's cap; `Packer.query`'s |
| `ResultTokenCap` | admit, dispatch | Zeus's two read bounds; `Packer.query`'s |
| `Offered` | admit | Zeus's `UNKNOWN_TOOL`; the scout's `UNKNOWN_TOOL` text |
| `Valid` | admit | `validate` |
| `PerResponse` | admit | Zeus's second `coder` and `locate` fan-out |
| `Repetition` | admit | Zeus, with its identity and note (`zeus/calls.py`) |
| `Dedup` | admit | `ScoutTools._seen` |
| `AskWhen` | admit | the approval router, asked inside the shell's executor today |
| `Account` | result | Zeus's turn; the scout's result tokens; `_Session.result_tokens` |
| `ResultFits` | result | the scout loop's withheld result |
| `NoProgress` | result | the scout loop's dedup streak |
| `Clip` | result | the packer's head+tail cut of a worker result |
| `BoundText` | result | Zeus's navigation answers |
| `Redact`, `Untrusted` | result | `secrets/redact.py`; `20-injection.md` §3.1 |

A rule that reads one role's own state lives with that role and fills a `Slot` (Zeus's
`findings`). No call carries a purpose line.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from chickadee.client.request import DocumentPart
from chickadee.context.document import document_est
from chickadee.context.handle import Handle, HandleKind
from chickadee.context.outline import image_est
from chickadee.context.results import InvalidArgument, ToolResult, Truncation, UnresolvedSearch
from chickadee.context.spans import Span
from chickadee.edits.lines import split_lines
from chickadee.errors import HarnessBug
from chickadee.limits import CHARS_PER_TOKEN, tokens_est, tokens_for_chars
from chickadee.schema import sha256_hex
from chickadee.schema.authorization import Tier
from chickadee.secrets.redact import redact_text
from chickadee.shell_line import prints_only
from chickadee.suggest import nearest
from chickadee.tools.foreign import translate
from chickadee.tools.meta import FINDINGS, split_meta, with_meta
from chickadee.tools.rules import (
    RETIRED,
    Answered,
    AskPerson,
    BoundReached,
    CallOutcome,
    Level,
    Noted,
    NotOffered,
    OnCap,
    Pass,
    Pending,
    Refuse,
    ScopeState,
    Stop,
    Verdict,
    repeated,
)
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Answer, Defaulted, Said, TextAnswer
from chickadee.tools.validate import InvalidArguments, validate

if TYPE_CHECKING:
    from chickadee.tools.context import ToolContext

__all__ = [
    "TRUNCATION_MARK",
    "Account",
    "AskWhen",
    "BoundText",
    "CallCap",
    "Clip",
    "Dedup",
    "InvalidCall",
    "NoProgress",
    "Offered",
    "PerResponse",
    "PrintsOnly",
    "Redact",
    "Repetition",
    "ResultFits",
    "ResultTokenCap",
    "ShowExample",
    "Untrusted",
    "Valid",
    "answer_text",
    "args_digest",
    "args_sha",
    "call_key",
    "capped",
    "failed",
    "result_tokens",
    "truncate",
]


# -- admit and dispatch rules ---------------------------------------------------------------


def _refused(resource: str, used: int, limit: int, level: str) -> BoundReached:
    return BoundReached(resource, used, limit, level)


@dataclass(frozen=True)
class CallCap:
    """`limit` calls per `level` scope; from the (limit+1)th. A limit of 0 never
    refuses. Zeus refuses without counting; the scout stops; the worker's packer
    refuses as a counted denial."""

    level: Level
    key: str
    on_cap: OnCap = "refuse"
    counts: bool = False
    stage: Literal["admit", "dispatch"] = "admit"
    exempt: frozenset[str] = frozenset()
    """Families never refused by it, as `ResultTokenCap`'s: Athena's task tools, so a long
    investigation never strands an open task."""

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        limit = state.limit(self.key)
        used = state.calls(self.level)
        if not 0 < limit <= used or _exempt(pending, self.exempt):
            return Pass()
        failure = _refused("tool_calls", limit, limit, self.level)
        if self.on_cap == "stop":
            return Stop(self.key, failure)
        return Refuse(failure, counts=self.counts)


@dataclass(frozen=True)
class ResultTokenCap:
    """The tool-result tokens a `level` scope may take in. At admit and dispatch it
    refuses once the allowance is spent; at the result stage (`ResultFits`) it withholds a
    result that would pass it. A call of an `exempt` family is never refused by it: the
    worker's edits, whose answers are bounded one by one and must not starve because reads
    spent the budget."""

    level: Level
    key: str
    on_cap: OnCap = "refuse"
    counts: bool = False
    stage: Literal["admit", "dispatch"] = "admit"
    exempt: frozenset[str] = frozenset()

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        limit = state.limit(self.key)
        used = state.tokens(self.level)
        if not 0 < limit <= used or _exempt(pending, self.exempt):
            return Pass()
        failure = _refused("result_tokens", used, limit, self.level)
        if self.on_cap == "stop":
            return Stop(self.key, failure)
        return Refuse(failure, counts=self.counts)


@dataclass(frozen=True)
class Offered:
    """A name the scope does not offer is answered `UNKNOWN_TOOL`, never corrected,
    in the role's form (`show`): Zeus names the nearest offered name, the scout lists them.
    A name from another harness's table (`tools/foreign.py`) names the call to send."""

    counts: bool = True
    show: Literal["nearest", "offered"] = "nearest"
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        if pending.tool is not None:
            return Pass()
        offered = state.offered
        near = nearest(pending.sent, offered)
        instead = tuple(t for t in RETIRED.get(pending.sent, ()) if t in offered)
        failure = NotOffered(pending.sent, near, offered, self.show, instead)
        # Another harness's name is answered with the call to send in ours, never run.
        suggested = translate(pending.sent, pending.raw, state.specs, state.place)
        if suggested is not None:
            failure = replace(
                failure, send=suggested.call, takes=suggested.takes, note=suggested.note
            )
        return Refuse(failure, self.counts)


@dataclass(frozen=True)
class Valid:
    """The full schema (33 §5): `INVALID_ARGUMENTS{tool, violations, schema}`, or the read
    view the tool reads its arguments through. `meta` names the role's own arguments
    a call carries beside the tool's (Zeus's `findings`, `tools/meta.py`):
    set aside, never edited, so the tool's schema is checked against its own. One the tool
    names itself is its own too, and is kept. The refusal shows the schema as the model was
    shown it, meta included."""

    counts: bool = True
    meta: tuple[str, ...] = ()
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        if pending.spec is None:
            return Pass()
        named = pending.spec.arguments
        own = {k: v for k, v in pending.raw.items() if k not in self.meta or k in named}
        checked = validate(pending.spec, own)
        if isinstance(checked, InvalidArguments):
            if self.meta and pending.spec.grammar is None:
                schema = with_meta(checked.schema, findings=FINDINGS in self.meta)
                checked = replace(checked, schema=schema)
            return Refuse(InvalidCall(checked), self.counts)
        return Pass(checked)


@dataclass(frozen=True)
class InvalidCall:
    """`InvalidArguments` as a `Failure`: it renders as its content, and a role's sink may
    show `invalid` another way."""

    invalid: InvalidArguments
    failure: Literal["INVALID_ARGUMENTS"] = "INVALID_ARGUMENTS"

    def render(self) -> str:
        return self.invalid.as_content()


@dataclass(frozen=True)
class PerResponse:
    """33 §7, by the tool's `parallel`, within one response: a second `exclusive` call is
    refused, and `capped` calls past their fan-out. Neither counts.

    A `capped` tool's fan-out is the row its spec names (`ToolSpec.fanout`), else `key`; the
    calls of the tools sharing a row are counted together, apart from every other row's
    (`locate`'s fan-out and the agents' are two counts). A second call
    of a tool whose spec names `one_per` giving that argument a value already given is refused
    too (`follow_up` of one agent twice, decision 2)."""

    key: str
    stage: Literal["admit", "dispatch"] = "admit"

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def spec_keys(self, spec: ToolSpec) -> tuple[str, ...]:
        """The fan-out row a tool's calls are counted against, when it names its own."""
        return (spec.fanout,) if spec.parallel == "capped" and spec.fanout else ()

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        spec = pending.spec
        if spec is None:
            return Pass()
        seen = state.keep("response", "per_response", dict[str, int])
        if spec.one_per is not None:
            value = pending.raw.get(spec.one_per)
            once = f"{spec.name}:{spec.one_per}={value}"
            seen[once] = seen.get(once, 0) + 1
            if seen[once] > 1:
                why = f"{value} already has a {pending.sent} call in this response"
                do = "send it one call, with everything you want from it"
                return Refuse(InvalidArgument(spec.one_per, why, do=do), counts=False)
        if spec.parallel == "exclusive":
            seen["exclusive"] = seen.get("exclusive", 0) + 1
            if seen["exclusive"] > 1:
                return Refuse(BoundReached("coder", 1, 1, "response"), counts=False)
        if spec.parallel != "capped":
            return Pass()
        key = spec.fanout or self.key
        seen[key] = seen.get(key, 0) + 1
        fanout = state.limit(key)
        if 0 < fanout < seen[key]:
            failure = BoundReached("fan_out", seen[key], fanout, "response")
            return Refuse(failure, counts=False)
        return Pass()


def args_sha(raw: Mapping[str, object]) -> str:
    """A call's arguments as one sha: its `tool_call` line's `args_sha` and its effect's
    `approved_sha`, the one encoding both are written with, so a step is joined to its call
    by equality (`box/facts.py`)."""
    return sha256_hex(json.dumps(dict(raw), sort_keys=True, default=str))


def args_digest(pending: Pending) -> str:
    """D98's `args_digest`: of the call's own arguments, canonical. The meta arguments are
    left out, so a call cannot leave a repetition count by rewording its `findings`."""
    own, _ = split_meta(pending.raw)
    canonical = json.dumps(own, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_hex(canonical)[:16]


def _same_call(pending: Pending) -> Hashable:
    # A name this scope does not offer resolves to no tool, so the name as sent is what tells
    # two such calls apart; keyed by `tool` alone, every unknown name was one call repeated.
    name = pending.tool if pending.tool is not None else ("unoffered", pending.sent)
    return (name, args_digest(pending))


def _count_note(count: int, identity: Hashable) -> str:
    del identity
    return repeated(f"repetition{{count: {count}}}", count)


@dataclass
class _Streak:
    key: Hashable = None
    count: int = 0


@dataclass(frozen=True)
class Repetition:
    """Consecutive identical calls. A note from `n` (`key`), a refusal from `2n`; any
    other call resets the count. `identity` is what makes two calls identical and `note` is
    what the model is told; Zeus supplies both with its world's generation in them
    (`repetition{count, since_generation}`). `on_cap="stop"` ends the role's tool
    use at the refusal instead (the worker's)."""

    level: Level
    key: str
    identity: Callable[[Pending], Hashable] = field(default_factory=lambda: _same_call)
    note: Callable[[int, Hashable], str] = field(default_factory=lambda: _count_note)
    on_cap: OnCap = "refuse"
    stage: Literal["admit", "dispatch"] = "admit"
    by_span: bool = True
    """A `read` is keyed on the file, its bytes and the span it asks (`ScopeState.read_key`),
    not on its words: `5-208`, `6-207` and `depth: "full"` over lines
    already held are one span asked again. Zeus's identity carries his world's generation
    and his note reads it, so his reads keep his own key."""

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        streak = state.keep(self.level, "repetition", _Streak)
        identity = self._read_key(pending, state) or self.identity(pending)
        streak.count = streak.count + 1 if identity == streak.key else 1
        streak.key = identity
        n = state.limit(self.key)
        if n <= 0 or streak.count < n:
            return Pass()
        if streak.count >= 2 * n:
            # Not dispatched, so not counted: the streak stays where the refusal found it.
            streak.count -= 1
            failure = BoundReached("repetition", 2 * n, 2 * n, self.level)
            if self.on_cap == "stop":
                return Stop(self.key, failure)
            return Refuse(failure, counts=False)
        return Noted(self.note(streak.count, identity))

    def _read_key(self, pending: Pending, state: ScopeState) -> Hashable | None:
        if not self.by_span or pending.tool != "read" or state.read_key is None:
            return None
        raw = pending.raw
        target = raw.get("path", raw.get("handle"))
        if not isinstance(target, str):
            return None
        named = {k: v for k in ("lines", "at", "depth") if isinstance(v := raw.get(k), str)}
        key = state.read_key(target, **named)
        return None if key is None else ("read", key)


@dataclass
class _Printed:
    count: int = 0


@dataclass(frozen=True)
class PrintsOnly:
    """A `shell` call that only prints (`shell_line.prints_only`: `echo "task completed"`) is
    answered, never run: it changes and checks nothing, and a worker that ends its task this
    way loops on it instead of replying (the owner's REPL run). The answer says how a
    task ends; past `key`'s count, the role's tool use ends, as a repetition's does."""

    level: Level
    key: str
    stage: Literal["admit", "dispatch"] = "admit"

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        command = pending.raw.get("command")
        if pending.spec is None or pending.spec.name != "shell" or not isinstance(command, str):
            return Pass()
        if not prints_only(command):
            return Pass()
        printed = state.keep(self.level, "prints_only", _Printed)
        printed.count += 1
        n = state.limit(self.key)
        if n > 0 and printed.count > n:
            return Stop(self.key, BoundReached("print_only", printed.count, n, self.level))
        # Says the way to end that this role holds: `finish`, or a reply with no tool call.
        how = (
            "call `finish` with your report"
            if "finish" in state.offered
            else "reply without calling a tool: that reply ends it and is your report"
        )
        return Answered(
            f"not run: {command!r} only prints, which changes and checks nothing. If the task "
            f"is done, {how}."
        )


def call_key(pending: Pending) -> str:
    """One call's identity as sent: its name and its arguments, keys sorted. Two calls with
    one key are the same call to `Dedup`, and the scout's `scout_tool` line records its sha
    (`args_sha`)."""
    return f"{pending.sent}\0{json.dumps(dict(pending.raw), sort_keys=True, default=str)}"


@dataclass(frozen=True)
class Dedup:
    """An identical call in the `level` scope is answered with a marker, not run again.
    It counts."""

    level: Level = "call"
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        seen = state.keep(self.level, "dedup", dict[str, int])
        key = call_key(pending)
        first = seen.get(key)
        if first is not None:
            return Answered(
                f"[dedup: the same {pending.sent} call as call #{first}; nothing re-run]"
            )
        seen[key] = state.calls(self.level) + 1
        return Pass()


@dataclass(frozen=True)
class AskWhen:
    """Ask the person before a call `when` selects, at the tier the
    router computes for it: no tier above T0, no question. A role whose ceiling lacks the
    `user` port cannot hold one (decision 8)."""

    when: Callable[[ToolSpec], bool]
    tier: Callable[[Pending], Tier]
    stage: Literal["admit", "dispatch"] = "admit"
    asks: bool = True

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        spec = pending.spec
        if spec is None or not self.when(spec):
            return Pass()
        tier = self.tier(pending)
        if tier is Tier.T0:
            return Pass()
        return AskPerson(f"run {spec.name}?", tier)


# -- result rules ---------------------------------------------------------------------------


def answer_text(outcome: CallOutcome) -> str:
    """What the model reads of an outcome, before the sink frames it: the answer's rendering,
    or the failure or text that answered it instead, and the notes."""
    body = ""
    decided = outcome.pending.decided
    stop = outcome.ended_by
    if outcome.answer is not None:
        body = outcome.answer.render()
    elif isinstance(decided, Answered):
        body = decided.text
    elif isinstance(decided, Refuse):
        body = decided.failure.render()
    elif stop is not None and stop.failure is not None:
        body = stop.failure.render()
    return "\n".join((body, *outcome.pending.notes)) if outcome.pending.notes else body


def result_tokens(outcome: CallOutcome) -> int:
    """What an outcome takes from a result budget: its text, and the image it sends
    when it sends one, counted as the request will count it (`context.outline.image_est`). An
    image measured by its one-line header alone let a turn of screenshots pass a budget that
    the same text would not."""
    tokens = tokens_for_chars(len(answer_text(outcome)))
    answer = outcome.answer
    if isinstance(answer, ToolResult) and isinstance(answer.image, DocumentPart):
        tokens += document_est(answer.image.data, answer.handle.shape.est_tokens)
    elif isinstance(answer, ToolResult) and answer.image is not None:
        tokens += image_est(answer.image.data, answer.handle.shape.est_tokens)
    return tokens


@dataclass(frozen=True)
class Account:
    """What a call took in, added to each of `levels`: every call but one a
    non-counting verdict answered or a stop withheld, or (`served_only`) only those that
    ran and answered."""

    levels: tuple[Level, ...]
    measure: Callable[[CallOutcome], int] = result_tokens
    served_only: bool = False
    exempt: frozenset[str] = frozenset()
    """Families whose answers take nothing from the scope (the worker's edits)."""

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        if not outcome.pending.counts or outcome.stop is not None:
            return outcome
        if _exempt(outcome.pending, self.exempt):
            return outcome
        if self.served_only and (outcome.answer is None or failed(outcome.answer)):
            return outcome
        tokens = self.measure(outcome)
        for level in self.levels:
            state.took(level, tokens)
        return outcome


def _exempt(pending: Pending, families: frozenset[str]) -> bool:
    """Whether `pending` names a tool of one of `families`."""
    return pending.spec is not None and pending.spec.family in families


def failed(answer: Answer) -> bool:
    """Whether an answer is a failure rather than a result."""
    return not isinstance(answer, (ToolResult, TextAnswer, Said, Defaulted))


def refused(answer: Answer | None) -> bool:
    """Whether a call was refused: no answer, or a failure other than a search that ran and
    found nothing (`UNRESOLVED`), which is what was asked for, not a refusal. A box
    role's record names a refusal's text as its `denied_reason`."""
    return answer is None or (failed(answer) and not isinstance(answer, UnresolvedSearch))


@dataclass(frozen=True)
class ResultFits:
    """The scout's form of D121: what a call would show, when it would take the scope past
    its allowance, is withheld, and the loop stops, rather than shown and then counted over.
    Every counted call is judged, a refusal's or a dedup marker's text as much as a result
    (the scout's loop compared every answer). The withheld answer stays on the outcome
    (`withheld`), for the role's record."""

    level: Level
    key: str
    measure: Callable[[CallOutcome], int] = result_tokens

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        if not outcome.pending.counts or outcome.stop is not None:
            return outcome
        limit = state.limit(self.key)
        would = state.tokens(self.level) + self.measure(outcome)
        if would <= limit:
            return outcome
        failure = BoundReached("result_tokens", would, limit, self.level)
        return replace(
            outcome, answer=None, withheld=outcome.answer, stop=Stop(self.key, failure)
        )


TRUNCATION_MARK = "…[truncated]"
"""What `truncate` puts between the head and the tail it keeps."""


def truncate(text: str, max_tokens: int) -> str:
    """Head + tail with an explicit marker. Never silent, and never over `max_tokens`.

    The marker is inside the budget rather than added to it: at a small remaining budget
    the old form returned the marker alone and overshot the very cap that produced it.
    """
    if tokens_est(text) <= max_tokens:
        return text
    chars = max(int(max_tokens * CHARS_PER_TOKEN), 0)
    if chars <= len(TRUNCATION_MARK):
        return TRUNCATION_MARK[:chars]
    budget = chars - len(TRUNCATION_MARK)
    head = budget // 2
    tail = budget - head
    return text[:head] + TRUNCATION_MARK + (text[-tail:] if tail else "")


@dataclass(frozen=True)
class Clip:
    """The worker's form of D78 and D121: a text answer cut, head and tail, to the
    smaller of the per-result bound (`per_key`) and what the `level` scope has left of its
    allowance (`total_key`), so one result cannot spend more than the whole allowance. A
    failure and a `ToolResult` pass unchanged. A bound of 0 is no bound, as it is for
    `ResultTokenCap` beside it: read as a number, it would cut every answer to nothing."""

    level: Level
    per_key: str
    total_key: str
    exempt: frozenset[str] = frozenset()
    """Families cut by the per-result bound alone, never by what the scope has left (the
    worker's edits, which take nothing from it)."""

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.per_key, self.total_key)

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        answer = outcome.answer
        if not isinstance(answer, TextAnswer):
            return outcome
        bounds: list[int] = []
        per = state.limit(self.per_key)
        if per > 0:
            bounds.append(per)
        total = state.limit(self.total_key)
        if total > 0 and not _exempt(outcome.pending, self.exempt):
            bounds.append(max(0, total - state.tokens(self.level)))
        if not bounds:
            return outcome
        return replace(outcome, answer=replace(answer, text=truncate(answer.text, min(bounds))))


@dataclass(frozen=True)
class NoProgress:
    """`n` (`key`) consecutive calls answered from the dedup cache end the loop: no
    progress."""

    level: Level
    key: str

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def streak(self, state: ScopeState) -> int:
        """The consecutive calls answered from the dedup cache so far (the scout's
        `dedup_streak`)."""
        return state.keep(self.level, "no_progress", _Streak).count

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        streak = state.keep(self.level, "no_progress", _Streak)
        repeated = isinstance(outcome.pending.decided, Answered)
        streak.count = streak.count + 1 if repeated else 0
        n = state.limit(self.key)
        if 0 < n <= streak.count and outcome.stop is None:
            return replace(outcome, stop=Stop(f"{n} consecutive repeated calls: no progress"))
        return outcome


def capped(
    text: str, handle: Handle, *, limit: int, unit: Literal["bytes", "lines"]
) -> tuple[str, Truncation | None]:
    """`text` under its tool's bound, and the `Truncation` that says what was not shown.

    Under the bound it returns untouched, with no marker. Over it, the head that fits —
    whole lines, and at least one — and a `Truncation` naming the total and the handle that
    holds the rest. Bytes cut as `byte_cap`; lines as `result_cap`, since a line-bounded
    result is a list of hits. Lines are `split_lines`', as the handle's reader numbers them, so
    the counts and where `continue_with` picks up are lines of the text it holds."""
    lines = split_lines(text, keepends=True)
    if unit == "lines":
        if len(lines) <= limit:
            return text, None
        kept = max(limit, 1)
        truncation = Truncation(
            (Span(1, kept),), len(lines) - kept, len(lines), "result_cap", handle
        )
        return "".join(lines[:kept]), truncation
    if len(text.encode("utf-8")) <= limit:
        return text, None
    used = kept = 0
    for line in lines:
        size = len(line.encode("utf-8"))
        if used + size > limit:
            break
        used += size
        kept += 1
    if kept == 0:
        # One line alone over the cap is still shown, cut and marked, so a read progresses.
        head = lines[0].encode("utf-8")[:limit].decode("utf-8", "ignore")
        content = f"{head}\n[line 1 is cut at {limit:,} bytes]"
        kept = 1
    else:
        content = "".join(lines[:kept])
    truncation = Truncation((Span(1, kept),), len(lines) - kept, len(lines), "byte_cap", handle)
    return content, truncation


@dataclass(frozen=True)
class BoundText:
    """D78 for a `TextAnswer`: stored behind a handle (`handles` port) and shown under its
    unit's cap (`bytes_key`, `lines_key`). A `ToolResult` was bounded by its producer and
    passes unchanged: bounded exactly once."""

    bytes_key: str
    lines_key: str
    kind: HandleKind = "findings"

    @property
    def limit_keys(self) -> tuple[str, ...]:
        return (self.bytes_key, self.lines_key)

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        answer = outcome.answer
        if not isinstance(answer, TextAnswer):
            return outcome
        if ctx.handles is None:
            raise HarnessBug(f"{self!r} needs the handles port, and the role gave none")
        handle = ctx.handles.store(self.kind, answer.text.encode("utf-8"))
        key = self.lines_key if answer.unit == "lines" else self.bytes_key
        content, truncation = capped(
            answer.text, handle, limit=state.limit(key), unit=answer.unit
        )
        return replace(
            outcome, answer=ToolResult(handle, content=content, truncation=truncation)
        )


@dataclass(frozen=True)
class ShowExample:
    """An `INVALID_ARGUMENTS` a tool's own check gave a structured argument carries a minimal
    valid shape for it, from the tool's spec (`ToolSpec.examples`), so the next call is fixed
    from an example rather than guessed at. The longest declared path that the field
    lies under wins, list indexes ignored: `request.hunks.0.path` reads `request.hunks`."""

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        del state, ctx
        answer, spec = outcome.answer, outcome.pending.spec
        if not isinstance(answer, InvalidArgument) or answer.example or spec is None:
            return outcome
        parts = [p for p in answer.field.split(".") if not p.isdigit()]
        for n in range(len(parts), 0, -1):
            example = spec.examples.get(".".join(parts[:n]))
            if example is not None:
                return replace(outcome, answer=replace(answer, example=example))
        return outcome


@dataclass(frozen=True)
class Redact:
    """Secrets out of what the model sees (`secrets/redact.py`, the one home)."""

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        answer = outcome.answer
        if isinstance(answer, TextAnswer):
            text, _ = redact_text(answer.text)
            return replace(outcome, answer=replace(answer, text=text))
        if isinstance(answer, ToolResult) and answer.content is not None:
            content, _ = redact_text(answer.content)
            return replace(outcome, answer=replace(answer, content=content))
        return outcome


@dataclass(frozen=True)
class Untrusted:
    """External content framed as data.
    `wrap` is that framing's one home, `regions.wrap_untrusted` once it is built; nothing
    here invents an envelope."""

    wrap: Callable[[str, str], str]
    """`(content, source) -> framed`, `source` being the tool's name."""

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        answer = outcome.answer
        source = outcome.pending.tool or outcome.pending.sent
        if isinstance(answer, TextAnswer):
            return replace(outcome, answer=replace(answer, text=self.wrap(answer.text, source)))
        if isinstance(answer, ToolResult) and answer.content is not None:
            framed = self.wrap(answer.content, source)
            return replace(outcome, answer=replace(answer, content=framed))
        return outcome
