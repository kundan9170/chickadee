"""What a tool is: `ToolSpec`, `Bound`, and `SURFACE`.

`SURFACE` is Zeus's sixteen tools in five families, and Zeus's default enabled list.
`render_table` renders it as the family and parallel-class tables, and
`test_tool_surface.py` checks it, per tool, on the facts the tables carry.

A spec carries no behaviour: a `Tool` binds one to a run function (`tool.py`), and the
`Registry` holds the tools. The
dispatcher reads `parallel` from here and enforces it; `exclusive` is the row that matters:
a second `coder` call while one is in flight is refused with the in-flight handle, never
queued.

Schemas are the **full** JSON schemas; the stub a
dynamic tool is shown as is derived in `validate.py`. Descriptions are templates under
`adapters/frontier/templates/`, one per tool, whose first line is the stub (`ToolSpec.template`
says where).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast, get_args

from chickadee.limits import limit
from chickadee.schema.authorization import Tier
from chickadee.schema.isolation import WorkspaceMode
from chickadee.schema.journal import AGENT_ID

__all__ = [
    "ANCHOR_ARG",
    "BLOCK",
    "DYNAMIC_STUB_MAX",
    "FAMILIES",
    "INLINE_BYTES_DEFAULT",
    "INVALID_ARGUMENTS",
    "LOCATE_MAX_CALLS",
    "RESULT_LINES_DEFAULT",
    "SHELL_INLINE_BYTES_DEFAULT",
    "SURFACE",
    "Anchor",
    "Bound",
    "Family",
    "Parallel",
    "ToolSpec",
    "parallel_class_of",
    "render_table",
    "spec",
]

Family = Literal["navigate", "read", "act", "talk", "record", "edit", "task"]
FAMILIES: tuple[str, ...] = ("navigate", "read", "act", "talk", "record", "edit", "task")
"""The first five families, in order, then the worker's edit tools, the worker's edit tools:
offered one, a
model is incremental; then G, Athena's task tools (`plan`, `delegate`, `review`), which her caps
exempt so she can always hand out, review or answer a task."""

Anchor = Literal["whole", "text", "line", "symbol"]
"""How an edit tool says where: the whole file, copied text, a line, or a symbol's
name."""

BLOCK = "x-block"
"""The schema annotation on a string argument that holds a file's text (`old`, `new`,
`content`, `patch`): `prompt_simple` writes it as a `~~~<argument>` block. An
annotation the validator ignores, as JSON Schema lets any unknown keyword be."""

ANCHOR_ARG = "x-anchor"
"""The schema annotation on a line tool's anchor argument: `line` names a line,
`after` the line an insertion follows (`0`, the file's start). It stands in for the argument's
type, which is the policy's anchor style's (`RolePolicy.resolved`), so the request, the prompt
and the validation read the one schema the style gives."""

Parallel = Literal["any", "capped", "serial", "exclusive"]

DYNAMIC_STUB_MAX = 32
"""Config `dynamic_stub_max`: past this many dynamic tools the stubs leave the
prefix and `tools(query)` is how the model finds them. A starting value, to be measured."""

INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
"""The one failure on every tool."""

# The four numbers below are rows of the limits register and are
# read from it, so there is one place to change them. The names stay because the surface's
# callers and tests import them.

LOCATE_MAX_CALLS = int(limit("locate.max_tool_calls").value)
"""The cheap-rung calls one `locate` may make (`limits.locate.max_tool_calls`)."""

RESULT_LINES_DEFAULT = int(limit("zeus.result_lines").value)
"""`result_cap`: the lines a line-bounded result (`find`, `tools`, `grep_in`) carries inline
before the rest is a handle. **A starting value, to be measured** (`zeus.result_lines`).
`tools()` also applies it as its entry cap, because one brief is one line."""

INLINE_BYTES_DEFAULT = int(limit("zeus.inline_bytes").value)
"""The bytes a byte-bounded read (`atlas`, `symbol`, `neighbors`, `impact`, `recall`,
`read`) returns inline (`read_inline_cap` on `read`; the others carry no key yet). A
starting value, to be measured (`zeus.inline_bytes`)."""

FIND_SKILL_RESULTS = int(limit("skills.find_results").value)
"""`find_skill`'s rows: the register's `skills.find_results`."""

SHELL_INLINE_BYTES_DEFAULT = int(limit("zeus.shell_inline_bytes").value)
"""`shell_inline_cap`: a command's inline output before the rest is a handle. Twice
`INLINE_BYTES_DEFAULT` because a test run's tail is what the model reads next. A starting
value, to be measured (`zeus.shell_inline_bytes`)."""


@dataclass(frozen=True)
class Bound:
    """What this tool caps, and how — *bounded exactly once, by whoever produced it*
    needs the producer to say what its cap is."""

    kind: Literal["inline_bytes", "inline_lines", "calls", "one_line", "none"]
    value: int | None = None
    """None for `one_line` / `none`."""
    config_key: str | None = None
    """Where the number is tunable, if it is."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    family: Family
    parallel: Parallel
    """`any`: pure reads, dispatch as many as the turn emits. `capped`: each costs a model
    call, fan-out bounded. `serial`: one at a time. `exclusive`: never more than one in
    flight, refused not queued."""
    mutates: Literal["nothing", "environment", "source"]
    tier: Tier | Literal["computed"]
    """T0 for every read; `computed` for `shell` and `coder`."""
    model_calls: Literal["none", "cheap", "box"]
    schema: Mapping[str, object]
    """The FULL JSON schema; the stub is derived."""
    description: str
    """The description template's name: a file under `tools/` for `SURFACE`'s, or a role's own
    path (`scout/search.md`, `worker/packet_read.md`). `template` is where it is."""
    bound: Bound
    failures: frozenset[str]
    """A subset of the vocabulary; `INVALID_ARGUMENTS` on every tool."""
    effects: Literal["none", "dispatcher", "self"]
    """Who records the effect lifecycle: `none` for a tool that changes
    nothing; `self` for one that journals its own (`shell` in its executor, `coder` in the
    box's run journal); `dispatcher` for a mutating tool the `EffectLifecycle` wrapper runs
    inside `journal.effects.perform`."""
    idempotent: bool
    """Whether running it twice is safe: the recovery rule for an effect, and what
    `RetryIdempotent` requires."""
    anchor: Anchor | None = None
    """An edit tool's anchor; None on every tool that edits nothing."""
    grammar: str | None = None
    """The grammar an edit tool's call may be decoded under: its format's, beside its
    parser in `edits/formats/`. None unless a tool carries one."""
    examples: Mapping[str, str] = field(default_factory=dict[str, str])
    """A minimal valid shape per structured argument, by its path (`request.hunks`): what an
    `INVALID_ARGUMENTS` on that argument, or anything inside it, shows (`ShowExample`).
    Data, not code: the one rule reads every tool's the same way."""
    paths: tuple[str, ...] = ()
    """The arguments that name repository files, each a string or a list of strings: what a
    hook's `paths` matcher reads a call by. Data on the tool, so a new tool
    says it rather than a list elsewhere. A patch's files are known only once it is read
    against the run's targets, so `apply_patch` names none; its step's `changed` is matched."""
    exactly_one: tuple[tuple[str, ...], ...] = ()
    """Groups of arguments of which a call gives exactly one, whole (`read`'s `path` or
    `handle`; `edit`'s `old` and `new`, or `edits`). Checked by `validate`, never written
    as a top-level `oneOf`: Anthropic's `input_schema` refuses one, and the schema is sent to
    every provider as it is."""
    renamed: Mapping[str, str] = field(default_factory=dict[str, str])
    """An argument's old name and what replaced it: a call that still sends the old name
    is refused naming the new one, so the model's retry is right first time."""
    retyped: Mapping[str, str] = field(default_factory=dict[str, str])
    """An argument whose type changed, or that was retired, and what to say when a call sends
    the old one (`shell`'s `network`, once a host list, and its `effects`): as `renamed`,
    the refusal names the change rather than only the type, so the model's retry is right
    first time."""
    threads: bool = False
    """Whether its run takes a thread of its own beside the rest of its group
    (`Dispatcher.execute_group`): a call that waits on a model for
    most of its time, and whose run touches none of its loop's state. `spawn_agent` and
    `follow_up`."""
    one_per: str | None = None
    """An argument a response may give one value of once: a second call of this tool naming
    the same value is refused and not counted (`PerResponse`). `follow_up`'s `agent`: one
    agent is never continued twice at once."""
    fanout: str | None = None
    """The limits row a `capped` call's fan-out is counted against, with every other tool's
    that names it and apart from the rest (`PerResponse`); None counts it under its role's.
    The agents' `agents.max_parallel`, apart from `locate`'s. A
    dispatcher counting fan-out and offering the tool is given the row (`Pipeline.keys`)."""

    @property
    def template(self) -> str:
        """Where the description template is, under a family's `templates/`. `SURFACE`'s
        tools keep their words in `tools/`, one description however many roles offer the tool;
        a tool only one role has keeps them in that role's folder, and its `description` says
        so with a `/`. `test_tool_surface.py` holds `tools/` to exactly `SURFACE`."""
        return self.description if "/" in self.description else f"tools/{self.description}"

    @property
    def arguments(self) -> frozenset[str]:
        """The names the tool's own schema declares."""
        props = self.schema.get("properties")
        if not isinstance(props, Mapping):
            return frozenset()
        return frozenset(cast("Mapping[str, Mapping[str, object]]", props))

    def __post_init__(self) -> None:
        if INVALID_ARGUMENTS not in self.failures:
            raise ValueError(f"tool {self.name!r} must declare {INVALID_ARGUMENTS}")
        if self.schema.get("type") != "object":
            raise ValueError(f"tool {self.name!r}: a tool schema is an object schema")
        if (self.family == "edit") != (self.anchor is not None):
            raise ValueError(
                f"tool {self.name!r}: an edit tool, and only an edit tool, names its anchor "
            )
        if self.grammar is not None and self.family != "edit":
            raise ValueError(f"tool {self.name!r}: only an edit tool carries a grammar")
        if self.one_per is not None and self.one_per not in self.arguments:
            raise ValueError(f"tool {self.name!r}: one_per names no argument of its schema")
        if self.fanout is not None and self.parallel != "capped":
            raise ValueError(f"tool {self.name!r}: only a capped call has a fan-out")
        if self.threads and self.parallel not in ("any", "capped"):
            raise ValueError(
                f"tool {self.name!r}: only a call that runs in a group runs on a thread"
            )
        if self.mutates != "nothing" and self.effects == "none":
            raise ValueError(
                f"tool {self.name!r} mutates {self.mutates} and records no effect lifecycle "
            )


def _str(desc: str, **extra: object) -> dict[str, object]:
    return {"type": "string", "description": desc, **extra}


def _int(desc: str, **extra: object) -> dict[str, object]:
    return {"type": "integer", "description": desc, **extra}


def _strings(desc: str, **extra: object) -> dict[str, object]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "description": desc,
        **extra,
    }


def _object(
    properties: Mapping[str, object], required: tuple[str, ...] = (), **extra: object
) -> dict[str, object]:
    out: dict[str, object] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
        **extra,
    }
    if required:
        out["required"] = list(required)
    return out


_INLINE_BYTES = Bound("inline_bytes", INLINE_BYTES_DEFAULT, None)
_RESULT_LINES = Bound("inline_lines", RESULT_LINES_DEFAULT, "result_cap")
_T0 = Tier.T0
_INV = frozenset({INVALID_ARGUMENTS})


def _fails(*names: str) -> frozenset[str]:
    return _INV | frozenset(names)


SURFACE: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="atlas",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object({}),
        description="atlas.md",
        bound=_INLINE_BYTES,
        failures=_fails("INDEX_UNAVAILABLE"),
        effects="none",
        idempotent=True,
    ),
    # `grep` and `glob`, where `find` (symbol names, read by every model as Unix `find`)
    # and the scout's `search` were: two tools over the one `search` port, named as models are
    # trained to call them. A definition by name is `symbol`'s.
    ToolSpec(
        name="grep",
        paths=("paths",),
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "pattern": _str(
                    "a regular expression (case-sensitive) to find in file contents",
                    minLength=1,
                ),
                "paths": _strings("globs limiting which files are searched"),
            },
            ("pattern",),
        ),
        description="grep.md",
        bound=_RESULT_LINES,
        failures=_INV,
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="glob",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {"pattern": _str("a glob matched against file paths, e.g. `**/*.py`", minLength=1)},
            ("pattern",),
        ),
        description="glob.md",
        bound=_RESULT_LINES,
        failures=_INV,
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="symbol",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {"name": _str("the symbol's name or qualified name", minLength=1)}, ("name",)
        ),
        description="symbol.md",
        bound=_INLINE_BYTES,
        failures=_fails("INDEX_UNAVAILABLE", "UNRESOLVED"),
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="neighbors",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "target": _str("a symbol or a path", minLength=1),
                "depth": _int(
                    "how many hops out; 1 is the immediate neighbourhood",
                    minimum=1,
                    maximum=3,
                    default=1,
                ),
            },
            ("target",),
        ),
        description="neighbors.md",
        bound=_INLINE_BYTES,
        failures=_fails("INDEX_UNAVAILABLE"),
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="impact",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {"symbols": _strings("the symbols that would change", minItems=1)}, ("symbols",)
        ),
        description="impact.md",
        bound=_INLINE_BYTES,
        failures=_fails("INDEX_UNAVAILABLE"),
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="recall",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "query": _str(
                    "what to look for in prior designs, runs and outcomes", minLength=1
                )
            },
            ("query",),
        ),
        description="recall.md",
        bound=_INLINE_BYTES,
        failures=_fails("UNRESOLVED"),
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="locate",
        family="navigate",
        parallel="capped",
        mutates="nothing",
        tier=_T0,
        model_calls="cheap",
        schema=_object(
            {"question": _str("a description of the thing you cannot name", minLength=1)},
            ("question",),
        ),
        description="locate.md",
        bound=Bound("calls", LOCATE_MAX_CALLS, "limits.locate.max_tool_calls"),
        failures=_fails("UNRESOLVED", "UNAVAILABLE", "INDEX_UNAVAILABLE", "TIMED_OUT"),
        effects="none",
        idempotent=True,
    ),
    # An agent on a profile's role, for a job Zeus hands over whole. Its round
    # is bounded by its profile's rows (`reader.*`, `reviewer.*`); up to `agents.max_parallel`
    # in one response, each on a thread of its own (Amendment 1 decision 1).
    ToolSpec(
        name="spawn_agent",
        family="navigate",
        parallel="capped",
        # An agent's commands are T0, which change nothing the tier system tracks, and
        # no agent has a tool that writes: so the call mutates nothing.
        mutates="nothing",
        tier=_T0,
        model_calls="cheap",
        schema=_object(
            {
                "task": _str(
                    "the job: what to find out, what to return, where to look, and what is "
                    "out of scope",
                    minLength=1,
                ),
                "skill": _str("an agent skill to run, by name", minLength=1),
                "profile": _str(
                    "a profile to run with no skill: reader, reviewer or researcher",
                    minLength=1,
                ),
                "handles": _strings("handles of results to give the agent, whole"),
            },
            ("task",),
        ),
        description="spawn_agent.md",
        bound=Bound("none"),
        failures=_INV,
        effects="none",
        idempotent=False,
        threads=True,
        fanout="agents.max_parallel",
    ),
    # An agent continued, its conversation kept, a new report.
    ToolSpec(
        name="follow_up",
        family="navigate",
        parallel="capped",
        mutates="nothing",
        tier=_T0,
        model_calls="cheap",
        schema=_object(
            {
                "agent": _str(
                    "the agent to continue: the id its report opens with", pattern=AGENT_ID
                ),
                "message": _str("what you want from it now", minLength=1),
            },
            ("agent", "message"),
        ),
        description="follow_up.md",
        bound=Bound("none"),
        failures=_INV,
        effects="none",
        idempotent=False,
        threads=True,
        one_per="agent",
        fanout="agents.max_parallel",
    ),
    ToolSpec(
        name="tools",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "query": _str("text to match against tool names and descriptions", minLength=1),
                "name": _str("one tool's exact name, for its full schema", minLength=1),
            },
        ),
        description="tools.md",
        bound=_RESULT_LINES,
        failures=_INV,
        effects="none",
        idempotent=True,
        # Exactly one of the two: `{query}` or `{name}`, never both or neither.
        exactly_one=(("query",), ("name",)),
    ),
    ToolSpec(
        name="read",
        paths=("path",),
        family="read",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "path": _str("a file, from the repository root: opens it", minLength=1),
                "handle": _str("a handle a result returned: reuses it", minLength=1),
                "lines": _str("a span, `start-end`, one-based inclusive", pattern=r"^\d+-\d+$"),
                "at": _str("an outline entry to return the span of", minLength=1),
                "depth": {
                    "type": "string",
                    "enum": ["outline", "signatures", "full"],
                    "description": "how much of the file: outline, signatures, or the text",
                },
                "max_bytes": _int(
                    "lower the tool's inline cap for this call; never raises it", minimum=1
                ),
            },
        ),
        description="read.md",
        # The spec bounds `read` three ways — inline bytes, inline lines per language, and
        # `image_max_edge` for an image — and `Bound` holds one clause, so this
        # records the byte cap alone. Whether `Bound` admits several clauses or the row
        # splits is a spec-shape question, left open here.
        bound=Bound("inline_bytes", INLINE_BYTES_DEFAULT, "read_inline_cap"),
        failures=_fails(
            "HANDLE_EXPIRED",
            "STALE_GENERATION",
            "INDEX_UNAVAILABLE",
            "SECRET_PATH",
            "NOT_TEXT",
            "NOT_SUPPORTED",
        ),
        effects="none",
        idempotent=True,
        # One name per meaning; `target` was both a path and a handle.
        exactly_one=(("path",), ("handle",)),
        renamed={"target": "path, or handle"},
    ),
    ToolSpec(
        name="grep_in",
        family="read",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "handle": _str("the handle of an already-opened file", minLength=1),
                "pattern": _str("a regular expression", minLength=1),
                "context": _int("lines of context around each hit", minimum=0, default=2),
            },
            ("handle", "pattern"),
        ),
        description="grep_in.md",
        bound=_RESULT_LINES,
        failures=_fails("HANDLE_EXPIRED", "STALE_GENERATION"),
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="load_skill",
        family="read",
        # As every read: the pin budget is checked and filled under the session's lock.
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {"name": _str("a skill's name, as the skills index lists it", minLength=1)},
            ("name",),
        ),
        description="load_skill.md",
        # The body is whole or not at all: its own cap is `SKILL_BODY_MAX`, on the file.
        bound=Bound("none"),
        failures=_INV,
        effects="none",
        idempotent=True,
    ),
    # The way to the right skill, past the resident index.
    ToolSpec(
        name="find_skill",
        family="read",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {"query": _str("what the skill is for, in a few words", minLength=1)},
            ("query",),
        ),
        description="find_skill.md",
        bound=Bound("inline_lines", FIND_SKILL_RESULTS, "limits.skills.find_results"),
        failures=_INV,
        effects="none",
        idempotent=True,
    ),
    ToolSpec(
        name="shell",
        family="act",
        parallel="serial",
        mutates="environment",
        tier="computed",
        model_calls="none",
        schema=_object(
            {
                "command": _str("the command line to run", minLength=1),
                "cwd": _str("working directory, relative to the workspace"),
                "network": {
                    "type": "boolean",
                    "description": (
                        "this command needs the network; opened for it alone, on approval"
                    ),
                },
                "idempotent": {
                    "type": "boolean",
                    "description": "safe to run twice; gates whether recovery may retry it",
                },
                "sandbox": {
                    "type": "boolean",
                    "description": (
                        "false asks the user to run this command outside the sandbox: needed "
                        "for every server, and for a command that failed because of the "
                        "sandbox; approved, it runs as the user would"
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "true runs it as a job at once: the call answers and you are told "
                        "when it exits; a server is one unless sent false"
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": ["task", "server"],
                    "description": (
                        "the job it may become: a task tells you on every exit, a server "
                        "only on one nobody asked for; a dev server's row is a server"
                    ),
                },
            },
            ("command",),
        ),
        description="shell.md",
        bound=Bound("inline_bytes", SHELL_INLINE_BYTES_DEFAULT, "shell_inline_cap"),
        retyped={
            "network": "network is now true or false; hosts are not named",
            "effects": (
                "effects is retired: the shell writes the worktree, and each command is "
                "recorded so it can be undone"
            ),
        },
        failures=_fails(
            "TIMED_OUT",
            "REFUSED_ABOVE_T0",
            "NETWORK_DENIED",
            "WRITE_DENIED",
            "SECRET_PATH",
            "SANDBOX_REQUIRED",
            "JOBS_FULL",
        ),
        effects="self",
        idempotent=False,
    ),
    # The jobs a shell command became. T0: it can only end a job this
    # session started, and it starts nothing.
    ToolSpec(
        name="job",
        family="act",
        parallel="serial",
        mutates="environment",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "action": {
                    "type": "string",
                    "enum": ["list", "tail", "wait", "kill"],
                    "description": "list the session's jobs, or tail, wait on or kill one",
                },
                "id": _str("the job, as the shell named it: j1, j2"),
                "lines": _int("tail: how many lines from the end", minimum=1),
                "stream": {
                    "type": "string",
                    "enum": ["out", "err", "both"],
                    "description": "tail: stdout, stderr or both (the default)",
                },
                "timeout_s": _int("wait: the longest to wait, in seconds", minimum=1),
            },
            ("action",),
        ),
        description="job.md",
        bound=Bound("inline_bytes", SHELL_INLINE_BYTES_DEFAULT, "shell_inline_cap"),
        failures=_fails("UNRESOLVED"),
        effects="self",
        idempotent=False,
    ),
    ToolSpec(
        name="coder",
        family="act",
        parallel="exclusive",
        mutates="source",
        tier="computed",
        model_calls="box",
        schema=_object(
            {
                "work": _str(
                    "a design: the open work's folder name, its design/ written with doc",
                    minLength=1,
                ),
                "title": _str("a one-task plan's title, one line", minLength=1),
                "brief": _str(
                    "a one-task plan's brief, written to the brief template", minLength=1
                ),
                "workspace": {
                    "type": "string",
                    "enum": list(get_args(WorkspaceMode)),
                    "description": (
                        "where the work runs: the user's checkout by default, or the "
                        "session's worktree once it has one"
                    ),
                },
            },
            (),
        ),
        description="coder.md",
        bound=Bound("none"),
        failures=_fails("OUT_OF_SCOPE"),
        effects="self",
        retyped={
            "design": (
                "design is retired: write the design with doc into its work folder, "
                "then call coder(work=<its name>)"
            ),
        },
        examples={
            "work": "2026-10-18-add-a-retry",
            "brief": "# T1: <title>\n\n## Goal\n<what the task delivers>\n\n## Done when\n...",
        },
        idempotent=False,
    ),
    # Zeus's design and Athena's plan, the documents of the work folder.
    ToolSpec(
        name="doc",
        family="act",
        parallel="serial",
        mutates="source",
        tier=_T0,
        model_calls="none",
        schema={
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["write", "edit", "remove"],
                    "description": (
                        "write: create or replace the whole file; edit: replace exact text; "
                        "remove: delete it"
                    ),
                },
                "file": {
                    "type": "string",
                    "description": "the document, from your folder: README.md, tasks/T1-x.md",
                },
                "title": {
                    "type": "string",
                    "description": "the work's title, on the first call: it opens the folder",
                },
                "content": {
                    "type": "string",
                    "description": "write: the whole file",
                    BLOCK: True,
                },
                "old": {"type": "string", "description": "edit: the exact text to replace"},
                "new": {"type": "string", "description": "edit: what replaces it"},
                "replace_all": {
                    "type": "boolean",
                    "description": "edit: replace every occurrence, not exactly one",
                },
            },
            "required": ["op", "file"],
            "additionalProperties": False,
        },
        description="doc.md",
        bound=Bound("none"),
        failures=_fails("DENIED", "NO_MATCH", "AMBIGUOUS", "FILE_ABSENT"),
        effects="dispatcher",
        idempotent=False,
    ),
    # The session's run goes on in the background; Zeus pauses it, which suspends it.
    ToolSpec(
        name="pause",
        family="act",
        parallel="exclusive",
        mutates="nothing",
        tier=Tier.T0,
        model_calls="none",
        schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        description="pause.md",
        bound=Bound("none"),
        failures=_fails(),
        effects="none",
        idempotent=True,
    ),
    # Zeus's words to the team about the session's run, the person's turns their
    # authority; the run delivers them at its next boundary.
    ToolSpec(
        name="tell",
        family="act",
        parallel="exclusive",
        mutates="nothing",
        tier=Tier.T0,
        model_calls="none",
        schema=_object(
            {
                "text": _str("what the team should know, in your words", minLength=1),
                "from": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 1,
                    "description": "the person's turns it came from, by their [turn N]",
                },
            },
            ("text", "from"),
        ),
        description="tell.md",
        bound=Bound("none"),
        failures=_fails(),
        effects="none",
        idempotent=False,
    ),
    # The person asks, Zeus moves the work. Entering and leaving write the
    # session's `workspace` line; landing and discarding are the handoff, asked of the person.
    ToolSpec(
        name="worktree",
        family="act",
        parallel="exclusive",
        mutates="source",
        tier="computed",
        model_calls="none",
        schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["enter", "leave", "land", "discard"],
                    "description": (
                        "enter: work in a separate worktree; leave: back to the user's "
                        "checkout, the worktree kept; land: bring its work into the checkout; "
                        "discard: drop it"
                    ),
                },
                "how": {
                    "type": "string",
                    "enum": ["bring", "apply", "squash"],
                    "description": (
                        "land: bring writes its changes into the checkout uncommitted (the "
                        "default); apply puts its commits on the user's branch as they are; "
                        "squash makes them one commit"
                    ),
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        description="worktree.md",
        bound=Bound("none"),
        failures=_fails("DENIED"),
        effects="self",
        idempotent=False,
    ),
    ToolSpec(
        name="ask_user",
        family="talk",
        parallel="serial",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "question": _str(
                    "the question, with its default and the options if any", minLength=1
                )
            },
            ("question",),
        ),
        description="ask_user.md",
        bound=Bound("none"),
        failures=_INV,
        effects="none",
        idempotent=False,
    ),
    ToolSpec(
        name="note_concern",
        family="record",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "text": _str("the advisory, in your own words", minLength=1),
                "anchor": _strings(
                    "the paths it is about; an advisory naming no path can never surface",
                    minItems=1,
                ),
            },
            ("text", "anchor"),
        ),
        description="note_concern.md",
        bound=Bound("one_line"),
        failures=_INV,
        effects="none",
        idempotent=False,
    ),
    ToolSpec(
        name="note_rejected",
        family="record",
        parallel="any",
        mutates="nothing",
        tier=_T0,
        model_calls="none",
        schema=_object(
            {
                "text": _str("the alternative that was considered", minLength=1),
                "reason": _str(
                    "why it was declined; a rejection without one is a preference as history",
                    minLength=1,
                ),
            },
            ("text", "reason"),
        ),
        description="note_rejected.md",
        bound=Bound("one_line"),
        failures=_INV,
        effects="none",
        idempotent=False,
    ),
)
"""The tools, in order. The one list."""

_BY_NAME = {s.name: s for s in SURFACE}


def spec(name: str) -> ToolSpec | None:
    return _BY_NAME.get(name)


_FAMILY_LETTER = {
    "navigate": "A",
    "read": "B",
    "act": "C",
    "talk": "D",
    "record": "E",
    "edit": "F",
    "task": "G",
}


def parallel_class_of(cell: str) -> Parallel | None:
    """The `Parallel` value a prose cell of the parallel-class table's third column states, or
    `None` when the cell says nothing recognisable — the check that reads it then reports
    the tool rather than passing quietly. The vocabulary is the table's own: *yes* is
    `any`, *yes, but … cap* is `capped`, *never more than one* is `exclusive`, *one at a
    time* is `serial`."""
    text = cell.replace("*", "").strip().lower()
    if "never more than one" in text:
        return "exclusive"
    if "one at a time" in text:
        return "serial"
    if text.startswith("yes"):
        return "capped" if "cap" in text else "any"
    return None


def _bound_text(b: Bound) -> str:
    if b.kind in ("one_line", "none"):
        return b.kind
    text = b.kind if b.kind != "calls" else f"calls = {b.value}"
    return f"{text} (`{b.config_key}`)" if b.config_key else text


def render_table(surface: tuple[ToolSpec, ...], *, style: Literal["arch", "design"]) -> str:
    """The family and parallel-class tables — one row per
    family (`design`: family, tools, model calls) or one row per tool (`arch`: family,
    parallel class, what it mutates, tier, bound).

    Nothing compares these tables byte for byte. What is enforced is the facts:
    `test_tool_surface.py` asserts each tool's fields against `surface`. This function
    exists for a reader who wants the list as a table.
    """
    if style == "design":
        lines = ["| Family | Tools | Model calls |", "|---|---|---|"]
        for family in FAMILIES:
            members = [s for s in surface if s.family == family]
            if not members:
                continue  # a family `surface` holds none of has no row
            names = ", ".join(f"`{s.name}`" for s in members)
            calls = sorted({s.model_calls for s in members})
            lines.append(
                f"| {_FAMILY_LETTER[family]} — {family} | {names} | {' / '.join(calls)} |"
            )
        return "\n".join(lines)
    lines = [
        "| Tool | Family | Parallel | Mutates | Tier | Bound |",
        "|---|---|---|---|---|---|",
    ]
    for s in surface:
        tier = s.tier.value.upper() if isinstance(s.tier, Tier) else s.tier
        lines.append(
            f"| `{s.name}` | {_FAMILY_LETTER[s.family]} | {s.parallel} | {s.mutates} | "
            f"{tier} | {_bound_text(s.bound)} |"
        )
    return "\n".join(lines)
