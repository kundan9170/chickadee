"""The cast as a value: `Role`, `Sampling`, `ROLES`, and the rungs' rules:
`DEFAULT_RUNGS`, `FORMER_RUNGS`, `rung_name_error`, `assign`.

Every place that needs a per-role fact reads it from here — the config loader (which rung,
which cap), the adapter (which template, what sampling), the dispatcher (which tool
surface), the compaction predicate (which rung's window), and the cast table in
the architecture notes, which is a *rendering* of `ROLES` and fails a test on drift.

A role is a trust boundary, so it is added here, never by
configuration. Each role has a **ceiling**, fixed here, and a **default policy**: which
tools it is offered and the pipeline its calls pass (not a bare name). A config chooses within
the ceiling
(`[tools.<role>]`), never the ceiling. `tools` still names whose code builds the role's tools
today, the cast table's
column, until each loop moves onto the dispatcher.

A role's caps are not typed here: they are rows of the limits register,
read when `ROLES` is built, so a role's default is the register's default. A pipeline names
the limits it reads by key, and holds no number: a key that is a register row is resolved from
the config, and one that is not (Zeus's `zeus.repeat`, the worker's allowance) is the role's
own config's until it becomes a row, supplied by the role's composition root.

This module imports nothing from `zeus/`, `orchestrator/` or `packer/`: both sides read it and
neither may import the other. A rule that reads a role's own state is a named `Slot` here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from chickadee.errors import ConfigError
from chickadee.hooks.rule import HookAdmit, HookResult
from chickadee.limits import limit
from chickadee.schema.authorization import Tier
from chickadee.tools.catalogue import (
    Account,
    BoundText,
    CallCap,
    Clip,
    Dedup,
    NoProgress,
    Offered,
    PerResponse,
    PrintsOnly,
    Repetition,
    ResultFits,
    ResultTokenCap,
    ShowExample,
    Valid,
)
from chickadee.tools.meta import FINDINGS
from chickadee.tools.policy import Ceiling, FileRule, RolePolicy
from chickadee.tools.rules import Pipeline, Slot
from chickadee.tools.spec import SURFACE
from chickadee.tools.tool import PORTS, Port

__all__ = [
    "ATHENA_CEILING",
    "ATHENA_TASK",
    "CRITIC_CEILING",
    "DEFAULT_RUNGS",
    "FORMER_RUNGS",
    "LOCATE_CEILING",
    "NO_TOOLS",
    "READER_CEILING",
    "RESEARCHER_CEILING",
    "REVIEWER_CEILING",
    "ROLES",
    "TASK_WORKER",
    "UNWRITTEN_PROMPTS",
    "WORKER_CEILING",
    "ZEUS_CEILING",
    "ZEUS_META_ARGS",
    "Former",
    "Proposal",
    "Role",
    "Sampling",
    "ToolSurface",
    "assign",
    "current_rung",
    "former_rung_error",
    "render_cast",
    "role",
    "rung_name_error",
]

DEFAULT_RUNGS: Final = ("high-param", "mid-param", "low-param")
"""The tiers the cast runs on by default. A rung exists because the engine
config's `[rungs]` names it; these are only what each role's `default_rung` is, and a config
that points every role elsewhere need not declare them (decision 4)."""


@dataclass(frozen=True)
class Former:
    """A rung name a decision retired: what replaced it, if anything."""

    successor: str | None
    why: str


FORMER_RUNGS: Mapping[str, Former] = MappingProxyType(
    {
        "zeus": Former("high-param", "the conversation's rung, a default tier since D176"),
        "planner": Former("high-param", "Athena's rung, a default tier since D176"),
        "scout": Former("mid-param", "the scout's rung, a default tier since D176"),
        "critic": Former("mid-param", "the critic's rung, a default tier since D176"),
        "coder": Former("low-param", "the worker's rung, a default tier since D176"),
        "mid": Former(None, "retired by D131 with the elaborator, the only role on it"),
    }
)
"""The one table of retired rung names. A former name may not be declared again, so a line an
old session log holds (`spend`, `cap`) means one thing: its successor (`current_rung`)."""

_RUNG_NAME: Final = re.compile(r"[a-z][a-z0-9-]*")
_RUNG_NAME_MAX: Final = 32


def rung_name_error(name: str) -> str | None:
    """The one name rule: None when `name` may name a rung, else why not.
    The config's layers and model, and the session log's `spend` and `cap` lines, all ask it."""
    if _RUNG_NAME.fullmatch(name) and len(name) <= _RUNG_NAME_MAX:
        return None
    return (
        "a rung name is lowercase letters, digits and '-', starting with a letter, at most "
        f"{_RUNG_NAME_MAX} characters; got {name!r}"
    )


def former_rung_error(name: str) -> str | None:
    """None unless `name` is a former rung; else why it may not be used again, naming its
    successor. The config's load and `set_cap` both refuse with it."""
    former = FORMER_RUNGS.get(name)
    if former is None:
        return None
    instead = (
        f"its successor is {former.successor!r}" if former.successor else "it has no successor"
    )
    return f"{name!r} is a former rung ({former.why}); {instead}"


def current_rung(name: str) -> str:
    """What a rung name on an old line means now: a former name's successor, else the name.
    A former name with no successor (`mid`) stays itself, a rung nothing runs on."""
    former = FORMER_RUNGS.get(name)
    return former.successor if former is not None and former.successor else name


ToolSurface = Literal["zeus", "athena", "coder", "scout", "agent", "none"]

NO_TOOLS = Ceiling("nothing", Tier.T0, frozenset(), FileRule("packet", "hidden"))
"""The ceiling of a role with no tools, and the narrowest there is."""
Proposal = Literal["patch", "command", "check", "design"]


@dataclass(frozen=True)
class Sampling:
    temperature: float | None = None
    """None → `RungEndpoint.temperature`, itself None → the provider's default."""
    ladder: Literal["fixed", "d5"] = "fixed"
    """`d5` moves the temperature per attempt. Reasoning is NOT here —
    it is config of the model a rung runs on (`RungEndpoint.reasoning`)."""


@dataclass(frozen=True)
class Role:
    name: str
    """The config key, and the cast's name."""
    default_rung: str
    """One of `DEFAULT_RUNGS`: the rung the role runs on unless `[roles]` moves it. Read only
    by `assign` and the config; every other reader asks `EngineConfig.assignment`."""
    tools: ToolSurface
    """The tool surface, by owner: the tool `SURFACE` · `AthenaTools` · the adapter's
    `ToolRequest` Literal · the scout's table · nothing."""
    max_tool_calls: int
    """Per role; 0 for a role with no tools. The register's default
    (`<role>.max_tool_calls`), recalibrated from telemetry — the point is that the knob
    exists per role and is enforced."""
    max_cumulative_result_tokens: float
    """A window fraction — the tool-result tokens one request may hold — resolved to
    tokens per rung at config load; 0.0 for a role with no tools."""
    prompt: str
    """The template under `adapters/<family>/templates/`."""
    sampling: Sampling
    judgment: bool
    """The cast's named / unnamed column, as a bit: a named agent reasons and can be wrong
    in interesting ways, so its output is a claim something downstream must judge."""
    may_propose: frozenset[Proposal]
    """What its output can *be*. Empty for a role whose output is only text."""
    tool_ceiling: Ceiling = NO_TOOLS
    """The most this role's tools may ever do: a trust boundary, never
    config. The default is the narrowest: nothing, at T0, through no port."""
    tool_defaults: RolePolicy | None = None
    """Its default tools and pipeline, today's behaviour; None for a role with no tools."""
    distinct_from: tuple[str, ...] = ()
    """Roles this one must not share a catalog model with, checked at load:
    the critic does not check work on the model that did it."""

    @property
    def family(self) -> str:
        """The adapter family whose templates the role's prompt is under, and whose adapter
        renders and parses its calls: `prompt`'s first part."""
        return self.prompt.partition("/")[0]

    @property
    def template(self) -> str:
        """The role's own template, under its family's `templates/`."""
        return self.prompt.partition("/")[2]

    def __post_init__(self) -> None:
        if self.default_rung not in DEFAULT_RUNGS:
            raise ConfigError(
                f"role {self.name!r} defaults to rung {self.default_rung!r}, not one of "
                f"{DEFAULT_RUNGS}"
            )
        if self.max_tool_calls < 0:
            raise ConfigError(f"role {self.name!r} has a negative tool-call cap")
        if not 0.0 <= self.max_cumulative_result_tokens <= 1.0:
            raise ConfigError(
                f"role {self.name!r}: max_cumulative_result_tokens is a window fraction, got "
                f"{self.max_cumulative_result_tokens}"
            )
        if self.tools == "none" and (self.max_tool_calls or self.max_cumulative_result_tokens):
            raise ConfigError(f"role {self.name!r} has no tool surface but a cap")
        # A zero here is a register row gone wrong, found when `ROLES` is built — not later,
        # as a config load blaming the person for a cap they never set.
        if self.tools != "none" and not (
            self.max_tool_calls > 0 and self.max_cumulative_result_tokens > 0
        ):
            raise ConfigError(f"role {self.name!r} has a tool surface but a zero cap")
        if (self.tool_defaults is None) != (self.tools == "none"):
            raise ConfigError(f"role {self.name!r}: a tool surface has defaults, and only one")
        if self.tool_defaults is not None and self.tool_defaults.role != self.name:
            raise ConfigError(
                f"role {self.name!r} holds {self.tool_defaults.role!r}'s defaults"
            )


FIXED = Sampling()
NO_PROPOSAL: frozenset[Proposal] = frozenset()


def _caps(name: str, tools: ToolSurface) -> tuple[int, float]:
    """A role's call cap and result-token fraction, as the register states them —
    never typed here, so a role's defaults cannot drift from the register's. A role with no
    tool surface has nothing to bound, and the register has no row for it."""
    if tools == "none":
        return 0, 0.0
    calls = limit(f"{name}.max_tool_calls").value
    results = limit(f"{name}.max_cumulative_result_tokens").value
    return int(calls), float(results)


def _role(
    name: str,
    default_rung: str,
    tools: ToolSurface,
    prompt: str,
    *,
    ceiling: Ceiling,
    policy: RolePolicy | None,
    judgment: bool = False,
    may_propose: frozenset[Proposal] = NO_PROPOSAL,
    sampling: Sampling = FIXED,
    distinct_from: tuple[str, ...] = (),
) -> Role:
    calls, results = _caps(name, tools)
    return Role(
        name=name,
        default_rung=default_rung,
        tools=tools,
        max_tool_calls=calls,
        max_cumulative_result_tokens=results,
        prompt=prompt,
        sampling=sampling,
        judgment=judgment,
        may_propose=may_propose,
        tool_ceiling=ceiling,
        tool_defaults=policy,
        distinct_from=distinct_from,
    )


# -- ceilings and default policies ----------------------------------

_ALL_FILES = FileRule("repository", "visible")

ZEUS_META_ARGS: Final = (FINDINGS,)
"""The argument a Zeus call may carry beside its tool's own: set aside
before the tool's schema is checked, and shown with it (`tools/meta.py`). No purpose line.
"""
BOX_META_ARGS: Final[tuple[str, ...]] = ()
"""Athena's and the worker's: none, since only Zeus's conversation folds findings."""

_ZEUS_PIPELINE = Pipeline(
    # `loop.py`'s order: the two bounds, the name, Zeus's own two
    # checks, the schema, the per-response classes, repetition. Zeus fills the slots.
    admit=(
        CallCap("turn", "zeus.max_tool_calls"),
        ResultTokenCap("turn", "zeus.max_cumulative_result_tokens"),
        Offered(),
        Slot("findings"),
        Valid(meta=ZEUS_META_ARGS),
        # A second `coder` split into its own exchange, then the response's classes.
        Slot("excess_coder"),
        # `locate`'s fan-out, and each row a tool's spec names apart (the agents').
        PerResponse("zeus.locate_fanout"),
        # With Zeus's identity and note: both carry the workspace's generation.
        Slot("repetition"),
        # Last, so a hook sees only a call that would otherwise run.
        HookAdmit(),
    ),
    dispatch=(ResultTokenCap("turn", "zeus.max_cumulative_result_tokens", stage="dispatch"),),
    result=(
        ShowExample(),
        BoundText("zeus.inline_bytes", "zeus.result_lines"),
        HookResult(),
        Account(("turn",)),
    ),
    counts_at=("turn",),
)

_LOCATE_PIPELINE = Pipeline(
    # `scout/loop.py` asks its cap before a call reaches `ScoutTools.call`, which counts it,
    # answers an unknown name, dedups and validates; a result that would pass the allowance
    # is withheld and the search ends; repeated calls end it too.
    admit=(
        CallCap("call", "locate.max_tool_calls", on_cap="stop"),
        Slot("escape_room"),
        Offered(show="offered"),
        Dedup("call"),
        Valid(),
        HookAdmit(),
    ),
    result=(
        ResultFits("call", "locate.max_cumulative_result_tokens"),
        HookResult(),
        Account(("call",)),
        NoProgress("call", "locate.no_progress"),
    ),
    counts_at=("call",),
)

_TASK_WORKER_PIPELINE = Pipeline(
    # The task worker is an agentic coder working a whole task. Its calls
    # count per task, replayed from its `tool_call` lines on resume (`BoxSink.replay`), and
    # past `worker.max_tool_calls` its round ends (`limit`); the same call in a row is noted,
    # and repeated past the row ends the round (`looped`). Each result is cut at
    # `worker.result_max` alone: no cumulative read budget, since the conversation's fit
    # evicts instead. The step around each writing call is the round's own
    # `around` (`box/steps.StepRecord`), added where the task is known.
    admit=(
        Repetition("task", "worker.repeat", on_cap="stop"),
        # A command that only prints is answered, never run, and the next ends the round
        # (a worker echoed "task completed" in a loop instead of replying).
        PrintsOnly("task", "worker.print_only"),
        CallCap("task", "worker.max_tool_calls", on_cap="stop"),
        # A call to a tool not offered lists what is: "nearest: None" told a worker
        # with no shell nothing it could use.
        Offered(show="offered"),
        Valid(meta=BOX_META_ARGS),
        HookAdmit(),
    ),
    result=(
        Clip("call", "worker.result_max", "worker.result_max"),
        HookResult(),
        Account(("task",)),
    ),
    counts_at=("task",),
)

_TASK_WORKER_TOOLS: Final = (
    "read",
    "grep_in",
    "grep",
    "glob",
    "symbol",
    "neighbors",
    "impact",
    "tests_for",
    "edit",
    "create",
    "delete",
    "move",
    "shell",
    "finish",
)
"""Zeus's reads over the worktree, `tests_for`, the
edit tools (`edit` by default; a model's config may name another shape), and the box
shell; and `grep` and `glob`, the one way to search every file and list them without a shell;
and `finish`, which ends the task with its report."""

_READS = ("symbol", "grep", "glob", "neighbors", "impact", "read", "grep_in")
"""The ladder's shared reads, registered built-ins."""

ZEUS_CEILING = Ceiling("source", Tier.T2, PORTS, _ALL_FILES)
ATHENA_CEILING = Ceiling(
    # She edits the worktree while she holds a task, `source`, since the
    # edit tools say they mutate it and write through `edits`; her shell asks above T0, so T2;
    # and the `desk` is her plan, delegations and reviews.
    "source",
    Tier.T2,
    frozenset(
        {
            "files",
            "navigator",
            "graph",
            "handles",
            "journal",
            "design_notes",
            "exec_request",
            "mcp_request",
            "edits",
            "exec",
            "desk",
            "search",
            "docs",
        }
    ),
    _ALL_FILES,
)
WORKER_CEILING = Ceiling(
    # The task worker edits the worktree and its shell asks above T0.
    "source",
    Tier.T2,
    frozenset({"files", "navigator", "graph", "handles", "exec", "edits", "search", "finish"}),
    _ALL_FILES,
)
_JOBS_ARGUMENTS: Final = {"shell": ("background", "kind")}
"""The shell's job arguments, never shown to a box role: only Zeus's shell keeps jobs, and the
box's refuses them."""

TASK_WORKER = RolePolicy(
    role="worker",
    enabled=_TASK_WORKER_TOOLS,
    shown_as={},
    files=_ALL_FILES,
    graph_depth=None,
    pipeline=_TASK_WORKER_PIPELINE,
    # A syntax error an edit made is kept and said, for the model to fix.
    edit_check="report",
    withheld=_JOBS_ARGUMENTS,
)
"""The worker's tool policy: native calls only, within
`WORKER_CEILING`. The `worker` role's default since the switch."""

_TASK_FAMILY: Final = frozenset({"task"})
"""Athena's task tools (`plan`, `delegate`, `review`, `outgrown`, `reply`): never refused by
her caps."""

_ATHENA_TASK_PIPELINE = Pipeline(
    # Her request is the whole episode, so her caps
    # refuse past their bound (`BOUND_REACHED`) rather than end her tool use, which would
    # strand an open task; her task tools are exempt from both, so she can always delegate,
    # review or answer; and the request scope is reopened at each task tool's answer
    # (`box/athena.py`), so each stretch of her work between two of them has the allowance.
    # No `Dedup`: edits now land live under her, so a call identical to an
    # earlier one may answer differently, which was its premise.
    admit=(
        # The same call in a row is noted, and refused past its row (`BOUND_REACHED`), never
        # her tool use ended, which would strand an open task.
        Repetition("request", "athena.repeat"),
        CallCap("request", "athena.max_tool_calls", exempt=_TASK_FAMILY),
        ResultTokenCap("request", "athena.max_cumulative_result_tokens", exempt=_TASK_FAMILY),
        Offered(show="offered"),
        Valid(meta=BOX_META_ARGS),
        HookAdmit(),
    ),
    result=(HookResult(), Account(("request",))),
    counts_at=("request",),
)

_ATHENA_TASK_TOOLS: Final = (
    "atlas",
    "grep",
    "glob",
    "symbol",
    "neighbors",
    "impact",
    "read",
    "grep_in",
    "tests_for",
    "recall",
    "note",
    "plan",
    "delegate",
    "review",
    "outgrown",
    "reply",
    "doc",
    "edit",
    "create",
    "delete",
    "move",
    "shell",
)
"""The reads, the map, the journal and her note; her task tools,
`outgrown` among them; the worker's edit tools, always offered so her tool
list never changes inside her request, and refused by the desk while she
holds no task; her shell; `grep` and `glob`; `doc`, her plan's
documents."""

ATHENA_TASK = RolePolicy(
    role="athena",
    enabled=_ATHENA_TASK_TOOLS,
    shown_as={},
    files=_ALL_FILES,
    graph_depth=None,
    pipeline=_ATHENA_TASK_PIPELINE,
    # A syntax error an edit made is kept and said, for the model to fix.
    edit_check="report",
    withheld=_JOBS_ARGUMENTS,
)
"""Athena's policy: native calls only, within `ATHENA_CEILING`. The
`athena` role's default since the switch."""

LOCATE_CEILING = Ceiling(
    "nothing",
    Tier.T0,
    frozenset({"files", "navigator", "handles", "search", "exec_request", "mcp"}),
    _ALL_FILES,
)
CRITIC_CEILING = Ceiling(
    "nothing", Tier.T0, frozenset({"files", "navigator", "handles"}), _ALL_FILES
)

# The built-in agent profiles, roles because a profile is a trust
# boundary. Both read the whole repository through their own reader and Zeus's navigator
# and search; the reviewer also runs commands at T0, on the `agent` shell profile, which refuses
# anything above it and never asks (no person is in an agent's loop).
_AGENT_READS: frozenset[Port] = frozenset({"files", "navigator", "handles", "search"})
READER_CEILING = Ceiling("nothing", Tier.T0, _AGENT_READS, _ALL_FILES)
_REVIEWER_PORTS: frozenset[Port] = _AGENT_READS | frozenset[Port]({"exec"})
REVIEWER_CEILING = Ceiling("environment", Tier.T0, _REVIEWER_PORTS, _ALL_FILES)
# The researcher reads the web and nothing else. No file, no graph, no search
# of the repository and no handle of Zeus's, so what it sends out can only be what its task
# says; the narrowest file rule, since it reads none.
RESEARCHER_CEILING = Ceiling(
    "nothing", Tier.T0, frozenset[Port]({"web"}), FileRule("packet", "hidden")
)


def _agent_pipeline(name: str) -> Pipeline:
    """An agent's round: past its call cap or its result budget its tool use ends and
    its next turn is its report; a repeated call answers from the round's cache."""
    return Pipeline(
        admit=(
            CallCap("call", f"{name}.max_tool_calls", on_cap="stop"),
            Offered(show="offered"),
            Dedup("call"),
            Valid(),
        ),
        result=(
            ResultFits("call", f"{name}.max_cumulative_result_tokens"),
            Account(("call",)),
        ),
        counts_at=("call",),
    )


def _policy(
    name: str,
    enabled: tuple[str, ...],
    pipeline: Pipeline,
    *,
    files: FileRule = _ALL_FILES,
    shown_as: dict[str, str] | None = None,
    graph_depth: int | None = None,
) -> RolePolicy:
    return RolePolicy(
        role=name,
        enabled=enabled,
        shown_as=shown_as or {},
        files=files,
        graph_depth=graph_depth,
        pipeline=pipeline,
    )


ROLES: tuple[Role, ...] = (
    # The turn's call cap is a calibration; at the bound the turn finishes in prose
    # (BOUND_REACHED).
    _role(
        "zeus",
        "high-param",
        "zeus",
        "frontier/zeus.md",
        ceiling=ZEUS_CEILING,
        policy=_policy("zeus", tuple(s.name for s in SURFACE), _ZEUS_PIPELINE),
        judgment=True,
        may_propose=frozenset({"command", "design"}),
    ),
    # One request for the whole episode, which plans, delegates and reviews; her
    # instructions are a prompt tied to no model family.
    _role(
        "athena",
        "high-param",
        "athena",
        "shared/athena.md",
        ceiling=ATHENA_CEILING,
        policy=ATHENA_TASK,
        judgment=True,
        may_propose=frozenset({"design", "patch"}),
    ),
    # An agentic coder working one task, on any model.
    _role(
        "worker",
        "low-param",
        "coder",
        "shared/worker.md",
        ceiling=WORKER_CEILING,
        policy=TASK_WORKER,
        may_propose=frozenset({"patch"}),
    ),
    _role(
        "locate",
        "mid-param",
        "scout",
        "frontier/locate.md",
        ceiling=LOCATE_CEILING,
        policy=_policy("locate", (*_READS, "request_exec"), _LOCATE_PIPELINE),
    ),
    _role(
        "critic",
        "mid-param",
        "scout",
        "frontier/critic.md",
        # The critic's tools are not registered yet.
        ceiling=CRITIC_CEILING,
        policy=_policy(
            "critic",
            (),
            Pipeline(
                admit=(CallCap("request", "critic.max_tool_calls"), Offered(), Valid()),
                counts_at=("request",),
            ),
        ),
        may_propose=frozenset({"check"}),
        distinct_from=("worker",),
    ),
    # The built-in agent profiles, spawned by Zeus's `spawn_agent`.
    _role(
        "reader",
        "mid-param",
        "agent",
        "shared/reader.md",
        ceiling=READER_CEILING,
        policy=_policy("reader", _READS, _agent_pipeline("reader")),
    ),
    _role(
        "reviewer",
        "mid-param",
        "agent",
        "shared/reviewer.md",
        ceiling=REVIEWER_CEILING,
        policy=_policy(
            "reviewer", (*_READS, "tests_for", "shell"), _agent_pipeline("reviewer")
        ),
    ),
    # The web, and nothing else.
    _role(
        "researcher",
        "mid-param",
        "agent",
        "shared/researcher.md",
        ceiling=RESEARCHER_CEILING,
        policy=_policy(
            "researcher",
            ("web_search", "web_fetch"),
            _agent_pipeline("researcher"),
            files=RESEARCHER_CEILING.widest_files,
        ),
    ),
    # L3_MODEL and CHECKPOINT_MODEL are these two roles' rungs, kept apart
    # so an experiment moves one without the other. The residue recalls the
    # conversation on the model that held it (the owner's call); the checkpoint is typed
    # extraction against a fixed schema.
    _role(
        "residue", "high-param", "none", "frontier/residue.md", ceiling=NO_TOOLS, policy=None
    ),
    _role(
        "checkpoint",
        "mid-param",
        "none",
        "frontier/checkpoint.md",
        ceiling=NO_TOOLS,
        policy=None,
    ),
    # A session's name, from its first exchange; one short call, on the cheapest tier.
    _role("title", "low-param", "none", "shared/title.md", ceiling=NO_TOOLS, policy=None),
)
"""The roles, in order. The one list."""

_BY_NAME = {r.name: r for r in ROLES}

UNWRITTEN_PROMPTS: Final[Mapping[str, str]] = {
    "critic": "written with the critic's tools",
    "checkpoint": "checkpoints have no module yet",
}
"""Each role whose prompt file does not exist yet, and why: its row stays, since its
rung is configurable, and the config measures its template as 0 tokens until it is written.
`test_roles.py` fails on any other role naming a missing file, and on an entry here whose file
now exists, so this list only shrinks."""


def _distinct_from_roles() -> None:
    """Each `distinct_from` names a cast role: checked once, as `ROLES` is built."""
    for r in ROLES:
        unknown = sorted(set(r.distinct_from) - set(_BY_NAME))
        if unknown:
            raise ConfigError(f"role {r.name!r} is distinct from {unknown}: not cast roles")


_distinct_from_roles()


def role(name: str) -> Role:
    """A role by name. An unknown name is a config error, at load — never a `KeyError`
    escaping from wherever the name was typed."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ConfigError(
            f"no role named {name!r}; the cast is {[r.name for r in ROLES]} (34-roles.md §3)"
        ) from None


def assign(overrides: Mapping[str, str]) -> Mapping[str, str]:
    """Role → rung for every cast role: `overrides` (the config's `[roles]`) over each role's
    `default_rung`. The one computation of the assignment: the config holds
    its result as `EngineConfig.assignment`, and every reader takes that."""
    return MappingProxyType({r.name: overrides.get(r.name, r.default_rung) for r in ROLES})


CAST_HEADER = (
    "| Role | Default rung | Tools | Cap | Prompt | Sampling | Judgment | May propose |"
)


def render_cast(roles: tuple[Role, ...] = ROLES) -> str:
    """What the architecture notes' cast table contains — the same rows, rendered."""
    lines = [CAST_HEADER, "|---|---|---|---|---|---|---|---|"]
    for r in roles:
        sampling = "fixed" if r.sampling.ladder == "fixed" else f"`{r.sampling.ladder}` ladder"
        if r.sampling.temperature is not None:
            sampling += f", t={r.sampling.temperature}"
        proposes = ", ".join(sorted(r.may_propose)) or "—"
        lines.append(
            f"| `{r.name}` | `{r.default_rung}` | `{r.tools}` | {r.max_tool_calls} | "
            f"`{r.prompt}` | {sampling} | {'yes' if r.judgment else 'no'} | {proposes} |"
        )
    return "\n".join(lines)
