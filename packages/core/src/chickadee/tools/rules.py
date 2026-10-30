"""Call policy as small, shared, ordered steps:
the verdicts, a call's `Pending` and `CallOutcome`, the `ScopeState` rules count in, the stage
protocols and the `Pipeline`. The shared rules are `catalogue.py`'s; the around-stage seams
are `wrappers.py`'s.

Each rule is written once and parameterised by what differs between roles: the scope level it
counts over, the limit key it reads, whether its refusal counts, what happens at its cap. A
role's pipeline is the order of its rules, per stage; the dispatcher only runs the lists.

    admit may this call run at all `Rule`, stage "admit"
    dispatch may it run now `Rule`, stage "dispatch"
    around how it runs `Wrapper`
    result what the model sees, what it cost `ResultRule`
    observe watching, never changing `Observer`

**Counting is not a step.** Every role counts the same way: a call counts toward each of its
pipeline's `counts_at` levels unless the verdict that decided it says it does not (Zeus's
bound refusals). So the dispatcher counts once, after admit, and uncounts a call a
dispatch rule refuses without counting.

**Limits are read by key.** A rule names the limit it reads (`zeus.max_tool_calls`), and the
role's composition root hands the dispatcher the resolved numbers: a register row's value,
or a number the role's own config holds until it becomes one (the worker's
`RunConfig` allowance). A pipeline in `ROLES` therefore holds no number.

**State** is the dispatcher's `ScopeState`: one loop's, on one thread, never shared. A rule
keeps what it remembers there (`ScopeState.keep`), never on itself,
because one rule instance serves every dispatcher built from a role's default pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

from chickadee.context.results import instructed
from chickadee.errors import HarnessBug
from chickadee.paths import Place
from chickadee.schema.authorization import Tier
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Answer, Failure, Invocation
from chickadee.tools.validate import Arguments

if TYPE_CHECKING:
    from chickadee.hooks.fire import RoleHooks
    from chickadee.hooks.payloads import BeforeTool
    from chickadee.tools.context import ToolContext

__all__ = [
    "BOUND_REACHED",
    "RETIRED",
    "Answered",
    "Around",
    "AskPerson",
    "BoundReached",
    "CallOutcome",
    "Declined",
    "Level",
    "NotOffered",
    "Noted",
    "Observer",
    "OnCap",
    "Pass",
    "Pending",
    "Pipeline",
    "Refuse",
    "ResultRule",
    "Rule",
    "ScopeState",
    "Slot",
    "Stop",
    "Verdict",
    "Wrapper",
    "repeated",
]


Level = Literal["turn", "call", "request", "attempt", "packet", "response", "task"]
"""The register's scopes, and `response`: one model response's calls, which Zeus's
per-response rules count over and the dispatcher resets at each `admit`."""

OnCap = Literal["refuse", "stop"]
"""What a cap does: refuse the call (Zeus, the worker's denials), or end the role's loop
(the scout's caps: `truncated: true`)."""

BOUND_REACHED: Final = "BOUND_REACHED"

_WHY: Final[Mapping[str, str]] = {
    "tool_calls": "bounded for safety, not for spend",
    "repetition": "identical call, world unchanged",
    "print_only": "a command that only prints changes and checks nothing",
    "result_tokens": "this turn has read what one turn may; decide from what it holds",
    "coder": "one coder delegation at a time; read the report of the one that ran",
    "fan_out": "each locate is a model call; the first ones ran",
}

_DO: Final[Mapping[str, str]] = {
    "tool_calls": "stop calling tools: give your answer or your report now, from what you have",
    "repetition": (
        "this exact call was answered the same way each time you sent it, and every identical "
        "one is now refused: read that answer, then change the call or take another step"
    ),
    "print_only": "run a command that checks the change (its tests), not one that only prints",
    "result_tokens": "stop reading: decide from what you hold, and answer",
    "coder": "read the report of the delegation that ran, then decide; one at a time",
    "fan_out": "read the answers of the locate calls that ran before you ask more",
}
"""What a model does next at each bound, beside its reason: one home per resource."""
_DO_ANY: Final = "do not send this call again: go on from what you have"


def repeated(note: str, count: int) -> str:
    """The note on a call sent `count` times in a row, with what to do: the
    shared form of the worker's, Athena's and Zeus's notes."""
    return instructed(
        note,
        f"you sent this exact call {count} times in a row and got this answer each time; "
        "change the call or take another step, or the next identical ones are refused",
    )


# -- failures the dispatcher emits ----------------------------------------------------------


@dataclass(frozen=True)
class BoundReached:
    """`BOUND_REACHED{resource, used, limit, scope, reason}` (33 §5): emitted by the dispatcher,
    not a tool."""

    resource: str
    used: int
    limit: int
    scope: str
    failure: Literal["BOUND_REACHED"] = BOUND_REACHED

    @property
    def why(self) -> str:
        return _WHY.get(self.resource, "")

    @property
    def instruction(self) -> str:
        return _DO.get(self.resource, _DO_ANY)

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{resource: {self.resource}, used: {self.used}, "
            f"limit: {self.limit}, scope: {self.scope}, reason: {self.why!r}}}",
            self.instruction,
        )


@dataclass(frozen=True)
class NotOffered:
    """`UNKNOWN_TOOL`: a call named a tool the scope did not offer. `nearest` is a
    hint, never a substitution: the call is recorded as sent. Zeus shows `{name, nearest}`;
    the scout shows `{name, offered}`, its `Offered(show="offered")`. The form is the
    failure's, not the sink's: the result rules measure what the model reads, and they run
    before the sink.

    A name another harness uses (`tools/foreign.py`) is answered with the call to send
    instead, in our terms, and what that tool takes (`send`, `takes`), or with why nothing
    here is to be called (`note` alone). It is still refused and never run."""

    name: str
    nearest: str | None
    offered: tuple[str, ...]
    show: Literal["nearest", "offered"] = "nearest"
    replaced_by: tuple[str, ...] = ()
    """The offered tools that replaced a retired name (`RETIRED`), which it meant."""
    send: str | None = None
    """The call to send instead, as the model writes it: `read {"path": "config.py"}`."""
    takes: tuple[str, ...] = ()
    """The arguments of `send`'s tool."""
    note: str = ""
    """What else the refusal says: why nothing is to be called, or what has no equivalent."""
    failure: Literal["UNKNOWN_TOOL"] = "UNKNOWN_TOOL"

    def render(self) -> str:
        # An instruction, not a diagnosis: a small model acts on "do not / call instead", and
        # whose habit the name was is nothing it can act on.
        refused = f"{self.failure}: {self.name} is not a tool here. Do not call it again."
        if self.send is not None:
            tool = self.send.split(" ", 1)[0]
            note = f" {_sentence(self.note)}" if self.note else ""
            return (
                f"{refused} Call instead: {self.send}. "
                f"{tool} takes: {', '.join(self.takes)}.{note}"
            )
        if self.note:
            return f"{refused} {_sentence(self.note)}"
        instead = f", replaced_by: [{', '.join(self.replaced_by)}]" if self.replaced_by else ""
        if self.show == "offered":
            return instructed(
                f"{self.failure}{{name: {self.name!r}{instead}, "
                f"offered: [{', '.join(self.offered)}]}}",
                "do not call it again; call one of the offered tools above",
            )
        do = (
            f"do not call it again; call {self.nearest} if that is the tool you meant"
            if self.nearest is not None
            else "do not call it again; call one of the tools you are offered"
        )
        return instructed(
            f"{self.failure}{{name: {self.name!r}{instead}, nearest: {self.nearest!r}}}", do
        )


def _sentence(text: str) -> str:
    """`reason in your reply` as the sentence a refusal ends with: `Reason in your reply.`"""
    return text[:1].upper() + text[1:].rstrip(".") + "."


RETIRED: Final[Mapping[str, tuple[str, ...]]] = {
    "find": ("symbol", "grep"),
    "search": ("grep", "glob"),
}
"""A retired tool's name and the tools that replaced it: `find` searched symbol names,
now `symbol`'s, and models meant text by it, `grep`'s; `search` is `grep` and `glob`. Data, as
`ToolSpec.renamed` is for an argument."""


@dataclass(frozen=True)
class Declined:
    """The person said no to a `AskPerson`, or could not be asked: the call did not run."""

    tool: str
    tier: Tier
    failure: Literal["DECLINED"] = "DECLINED"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{tool: {self.tool}, tier: {self.tier.value}}}",
            "the person said no, or could not be asked: do not send it again; go on without "
            "it, or say in your answer what it would have needed",
        )


# -- verdicts -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pass:
    """Continue. `Valid` passes the call's read view with it."""

    arguments: Arguments | None = None


@dataclass(frozen=True)
class Refuse:
    """A typed failure instead of a run. `counts` says whether the call still uses a call."""

    failure: Failure
    counts: bool = True


@dataclass(frozen=True)
class Answered:
    """An answer without running (the scout's dedup marker)."""

    text: str
    counts: bool = True


@dataclass(frozen=True)
class Noted:
    """Run, and carry this note on the result (D98's `repetition`)."""

    note: str


@dataclass(frozen=True)
class Stop:
    """End the role's loop: this call and every later one is answered, never run (the
    scout's caps, `truncated: true`). `failure` is what this call is answered with."""

    reason: str
    failure: Failure | None = None


@dataclass(frozen=True)
class AskPerson:
    """Ask the person through the `user` port first: yes continues, no
    is a refusal carrying `Declined`."""

    question: str
    tier: Tier


Verdict = Pass | Refuse | Answered | Noted | Stop | AskPerson

_DECIDING = (Refuse, Answered, Stop)


# -- one call, before and after it runs ------------------------------------------------------


@dataclass(frozen=True)
class Pending:
    """One call between admit and its record."""

    call: Invocation
    sent: str
    """The tool name as the model sent it (invariant 19)."""
    raw: Mapping[str, object]
    """The arguments as sent, never edited."""
    tool: str | None
    """The registry name, or None when the name is not offered in this scope."""
    spec: ToolSpec | None
    arguments: Arguments | None = None
    """The validated read view; None until `Valid` passes."""
    verdicts: tuple[Verdict, ...] = ()
    counted: tuple[Level, ...] = ()
    """The levels this call was counted at, so a dispatch refusal can uncount it."""
    dispatched: bool = False
    """The dispatch rules have been asked of it (`Dispatcher.dispatch`); `execute` does not
    ask again."""

    @property
    def decided(self) -> Refuse | Answered | Stop | None:
        """The verdict that answered the call without running it, if one did."""
        for v in self.verdicts:
            if isinstance(v, _DECIDING):
                return v
        return None

    @property
    def counts(self) -> bool:
        """Whether the call uses a call: every one but a refusal or answer that says it does
        not, and a stopped one, which the loop never made (the scout's cap, the worker's
        allowance: neither is counted today, nor replayed on resume)."""
        decided = self.decided
        if isinstance(decided, Stop):
            return False
        return decided is None or decided.counts

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(v.note for v in self.verdicts if isinstance(v, Noted))

    def with_verdict(self, verdict: Verdict) -> Pending:
        out = replace(self, verdicts=(*self.verdicts, verdict))
        if isinstance(verdict, Pass) and verdict.arguments is not None:
            out = replace(out, arguments=verdict.arguments)
        return out


@dataclass(frozen=True)
class CallOutcome:
    """A call's end: what it answered (`None` when a verdict answered it), and whether it
    ends the role's loop."""

    pending: Pending
    answer: Answer | None
    stop: Stop | None = None
    withheld: Answer | None = None
    """An answer a result rule kept from the model (`ResultFits`): never shown, kept so the
    role's record can say what it would have cost (the scout's `scout_tool`)."""

    @property
    def ended_by(self) -> Stop | None:
        """What ends the role's loop with this call, or None: the run's stop, or the verdict
        that answered the call instead of running it."""
        decided = self.pending.decided
        return self.stop or (decided if isinstance(decided, Stop) else None)


# -- state ----------------------------------------------------------------------------------


class AskGate(Protocol):
    """What may refuse a call before its question reaches the person, never answer it: the
    role's hooks, known here by this shape
    alone, so the dispatcher imports no hook."""

    def ask_refused(self, pending: Pending, ask: AskPerson) -> Refuse | None: ...


class ScopeState:
    """The counters of the open scopes and what the rules remember, for one dispatcher.

    Not thread-safe, and not shared: a dispatcher and its state belong to one loop on one
    thread, as each scout `locate` gets its own today."""

    def __init__(self, limits: Mapping[str, int]) -> None:
        self._limits = dict(limits)
        self.offered: tuple[str, ...] = ()
        """The names the open scope offers, as the model sees them."""
        self.specs: Mapping[str, ToolSpec] = {}
        """The offered tools' specs by their registry name, each under the name the model sees
        it by: what a refusal says to send and what that tool takes."""
        self.place: Callable[[str], Place | None] | None = None
        """Where a path a model sent lands in the repository: the role's file view's
        (`FileView.place`), which the dispatcher hands it; None for a role with no files."""
        self.read_key: Callable[..., Hashable | None] | None = None
        """What a `read` asks as a repetition counts it: the role's file view's
        (`FileView.repeat_key`), which the dispatcher hands it."""
        self.hooks: RoleHooks | None = None
        """The role's hooks, which the dispatcher hands the tool points' rules
        (`hooks/rule.py`); None for a role given none."""
        self.admitted: dict[str, BeforeTool] = {}
        """What `before_tool` was told of each call, by call id, until its `after_tool` reads
        it back (`hooks/rule.py`): a call's paths resolved and its line read once."""
        self._calls: dict[str, int] = {}
        self._tokens: dict[str, int] = {}
        self._kept: dict[tuple[str, str], object] = {}

    def limit(self, key: str) -> int:
        try:
            return self._limits[key]
        except KeyError:
            raise HarnessBug(f"no limit {key!r} was given to this dispatcher") from None

    def has_limit(self, key: str) -> bool:
        return key in self._limits

    def calls(self, level: Level) -> int:
        return self._calls.get(level, 0)

    def tokens(self, level: Level) -> int:
        return self._tokens.get(level, 0)

    def count(self, level: Level, n: int = 1) -> None:
        self._calls[level] = self.calls(level) + n

    def took(self, level: Level, tokens: int) -> None:
        self._tokens[level] = self.tokens(level) + tokens

    def reset(self, level: Level, *, calls: int = 0, tokens: int = 0) -> None:
        """Open `level` afresh: its counters from `calls` and `tokens` (a sink's replay) and
        nothing remembered."""
        self._calls[level] = calls
        self._tokens[level] = tokens
        for key in [k for k in self._kept if k[0] == level]:
            del self._kept[key]

    def keep[T](self, level: Level, key: str, make: Callable[[], T]) -> T:
        """What a rule remembers for the life of one `level` scope."""
        slot = (level, key)
        if slot not in self._kept:
            self._kept[slot] = make()
        return cast("T", self._kept[slot])


# -- the stage protocols ---------------------------------------------------------------------


class Rule(Protocol):
    """Admit and dispatch stages: may this call run (at all, or now)?"""

    @property
    def stage(self) -> Literal["admit", "dispatch"]: ...

    def check(self, pending: Pending, state: ScopeState) -> Verdict: ...


class ResultRule(Protocol):
    """Result stage: may change what the model sees, and accounts what it cost."""

    def apply(
        self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext
    ) -> CallOutcome: ...


Around = Callable[[], Answer]


class Wrapper(Protocol):
    """Around stage: wraps the run itself. Composed in pipeline order, the first outermost."""

    def around(self, pending: Pending, run: Around, ctx: ToolContext) -> Answer: ...


class Observer(Protocol):
    """Observe stage: after the record, never changing it. An observer that raises is logged
    and dropped; it cannot fail the call."""

    def observe(self, outcome: CallOutcome) -> None: ...


@dataclass(frozen=True)
class Slot:
    """A role's own rule, by name, filled at its composition root (`Pipeline.fill`): Zeus's
    `findings` read Zeus's session, so they live in `zeus/` and `ROLES` names the place they
    run in without importing them."""

    name: str
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        raise HarnessBug(f"the pipeline's slot {self.name!r} was never filled")


@dataclass(frozen=True)
class Pipeline:
    """A role's call policy: its rules in order, per stage. Code, not
    configuration (decision 7)."""

    admit: tuple[Rule, ...] = ()
    dispatch: tuple[Rule, ...] = ()
    around: tuple[Wrapper, ...] = ()
    result: tuple[ResultRule, ...] = ()
    observe: tuple[Observer, ...] = ()
    counts_at: tuple[Level, ...] = ()
    """The levels every call counts toward unless its verdict says it does not."""

    def __post_init__(self) -> None:
        for rule in self.admit:
            if rule.stage != "admit":
                raise HarnessBug(f"{rule!r} is a {rule.stage} rule in the admit list")
        for rule in self.dispatch:
            if rule.stage != "dispatch":
                raise HarnessBug(f"{rule!r} is a {rule.stage} rule in the dispatch list")

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(r.name for r in (*self.admit, *self.dispatch) if isinstance(r, Slot))

    def fill(self, rules: Mapping[str, Rule]) -> Pipeline:
        """This pipeline with each named slot replaced by the role's rule."""
        unknown = set(rules) - set(self.slots)
        if unknown:
            raise HarnessBug(f"no slot named {sorted(unknown)} in this pipeline")

        def put(seq: tuple[Rule, ...]) -> tuple[Rule, ...]:
            return tuple(rules.get(r.name, r) if isinstance(r, Slot) else r for r in seq)

        return replace(self, admit=put(self.admit), dispatch=put(self.dispatch))

    def measured_by(self, measure: Callable[[CallOutcome], int]) -> Pipeline:
        """This pipeline with every result rule that measures an outcome (`Account`,
        `ResultFits`) measuring it by `measure`: the tokens of what the role's sink records,
        which only the role's composition root holds."""
        result: list[ResultRule] = []
        for rule in self.result:
            # Only a dataclass rule declaring a `measure` field is one (`is_dataclass` first:
            # `fields` raises on anything else, and a rule need not be a dataclass).
            if is_dataclass(rule) and "measure" in {f.name for f in fields(rule)}:
                rule = cast("ResultRule", replace(cast("Any", rule), measure=measure))
            result.append(rule)
        return replace(self, result=tuple(result))

    @property
    def confirms(self) -> bool:
        """Whether any rule can ask the person (D164 decision 8 refuses one for a role
        whose ceiling lacks the `user` port)."""
        return any(getattr(r, "asks", False) for r in (*self.admit, *self.dispatch))

    def keys(self, specs: Iterable[ToolSpec] = ()) -> frozenset[str]:
        """Every limit key a rule of this pipeline reads: its own (`limit_keys`), and those it
        reads for a tool of `specs`, the tools it runs (`spec_keys`)."""
        offered = tuple(specs)
        out: set[str] = set()
        for rule in (*self.admit, *self.dispatch, *self.result):
            out.update(getattr(rule, "limit_keys", ()))
            per_spec = getattr(rule, "spec_keys", None)
            if per_spec is not None:
                for spec in offered:
                    out.update(cast("Callable[[ToolSpec], tuple[str, ...]]", per_spec)(spec))
        return frozenset(out)
