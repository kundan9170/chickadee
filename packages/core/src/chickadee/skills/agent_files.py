"""Other tools' agent files as agent skills.

Claude Code keeps its subagents in `.claude/agents/*.md` and OpenCode in `.opencode/agent/` or
`.opencode/agents/`. Each is a prompt with front matter, the same thing an agent skill is, so it
is read as one: its body is the skill's body, its `description` is what `find_skill` matches,
and its tools narrow the profile it runs as. Nothing in it grants a tool.

**The two formats.**

| | Claude Code | OpenCode |
|---|---|---|
| name | `name:` | the file's stem |
| tools | `tools: Read, Grep`; none named is all | a map of `name: true/false`; the same |
| skipped | — | `mode: primary`, an agent a person talks to, never a subagent |

**Tools, ours or none.** `TOOL_NAMES` is the one map from another tool's names to ours. A name
with no counterpart is ignored, and the index says which. A file that names a tool that writes
is listed as not runnable: no agent here writes, and an author who gave it one
expected it to. A file that names no tools would have every tool in its own harness; here it
runs read-only as `reader`, and the index says so, as D213 says of `general-purpose`.

**One rule for every agent skill (`runs_as`).** A `SKILL.md` with `context: fork` and an
agent file are read by the same rule: the profile it names, mapped (`AGENT_TYPES`), or else the
one its tools make it (`reviewer` for tools that run commands, `reader` otherwise); its tools in
our names; and a note only for what the index row does not already say: a profile mapped from
another tool's type, no tools named, a tool with no counterpart here, or why it cannot run.

What is not read is listed as ignored, as for an external skill: `model` first, since
rung choice is config's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chickadee.errors import ConfigError
from chickadee.skills.frontmatter import (
    FrontMatter,
    Raw,
    read_block,
    scalar,
    string_map,
    tool_names,
    valid_name,
)
from chickadee.tools.builtin import BUILTINS

__all__ = [
    "AGENT_DIRS",
    "AGENT_TYPES",
    "TOOL_NAMES",
    "WRITING_TOOLS",
    "AgentFile",
    "Runs",
    "read_agent_file",
    "runs_as",
]

AGENT_DIRS: Final = (
    (".claude", "agents"),
    (".opencode", "agent"),
    (".opencode", "agents"),
)
"""Where another tool keeps its agent files, under the repository and under the user's home.
OpenCode's user root is `~/.config/opencode/`, which `discover` adds."""

AGENT_TYPES: Final[dict[str, str]] = {
    "Explore": "reader",
    "Plan": "reader",
    "general-purpose": "reader",
}
"""Claude Code's subagent types, to our profiles. `general-purpose` has every
tool there; a file never gets every tool here, so it runs as `reader` with a note."""

TOOL_NAMES: Final[dict[str, str]] = {
    "Read": "read",
    "Grep": "grep",
    "Glob": "glob",
    "LS": "glob",
    "Bash": "shell",
    "read": "read",
    "grep": "grep",
    "glob": "glob",
    "list": "glob",
    "bash": "shell",
}
"""Claude Code's and OpenCode's tool names, to ours. Our own names map to themselves in
`narrow`, so one of our skills may write ours."""

WRITING_TOOLS: Final = frozenset(
    {"Edit", "Write", "MultiEdit", "NotebookEdit", "edit", "write", "patch", "multiedit"}
)
"""Another tool's names for a tool that writes the repository."""

_READ_FIELDS = ("name", "description", "tools", "mode")


@dataclass(frozen=True)
class _Narrowed:
    """A tool list in our names. `tools` None narrows nothing."""

    tools: tuple[str, ...] | None
    unmapped: tuple[str, ...] = ()
    """Names with no counterpart here, ignored."""
    writes: tuple[str, ...] = ()
    """Names of a tool that writes: an agent asking for one is not runnable."""


def _narrow(names: tuple[str, ...] | None, ours: frozenset[str] | None) -> _Narrowed:
    """`names` in our tool names. A Claude Code pattern (`Bash(git:*)`) narrows by its tool:
    the pattern itself is a pre-approval, which a file never grants. `ours` is the set
    of our own names a file may write, or None to take any lower-case name as ours (one of our
    skills, whose names the agents layer checks against its profile when it spawns)."""
    if names is None:
        return _Narrowed(None)
    tools: list[str] = []
    unmapped: list[str] = []
    writes: list[str] = []
    for raw in names:
        name = raw.split("(", 1)[0]
        if name in WRITING_TOOLS:
            writes.append(name)
        elif name in TOOL_NAMES:
            tools.append(TOOL_NAMES[name])
        elif (ours is None and name.islower()) or (ours is not None and name in ours):
            tools.append(name)
        else:
            unmapped.append(name)
    return _Narrowed(tuple(dict.fromkeys(tools)), tuple(unmapped), tuple(writes))


@dataclass(frozen=True)
class Runs:
    """How an agent skill runs, whichever way it was written."""

    profile: str
    tools: tuple[str, ...] | None
    """What narrows its profile's tools, in our names; None narrows nothing."""
    note: str | None
    """What the index says of it beyond its profile, or None."""
    runnable: bool


def runs_as(
    names: tuple[str, ...] | None, agent: str | None, *, ours: frozenset[str] | None
) -> Runs:
    """How an agent skill whose tools are `names` (None: it named none) and whose profile is
    `agent` (None: it named none) runs. `ours` is as `_narrow` takes it. A skill asking for a
    tool that writes is not runnable, and that is its whole note: no agent here writes."""
    narrowed = _narrow(names, ours)
    notes: list[str] = []
    if agent is None:
        # By its tools: those that run commands (the `exec` port) are the reviewer's.
        profile = "reviewer" if "exec" in BUILTINS.needs(narrowed.tools or ()) else "reader"
        if names is None:
            notes.append(f"named no tools, so runs read-only as {profile}")
    else:
        profile = AGENT_TYPES.get(agent, agent)
        if agent == "general-purpose":
            notes.append(f"runs read-only as {profile}")
        elif agent in AGENT_TYPES:
            notes.append(f"{agent} runs as {profile}")
    if narrowed.unmapped:
        notes.append(f"no tool here for {', '.join(narrowed.unmapped)}")
    if narrowed.writes:
        writes = ", ".join(narrowed.writes)
        note = f"not runnable: it asks for {writes}, and no agent here writes"
        return Runs(profile, narrowed.tools, note, runnable=False)
    return Runs(profile, narrowed.tools, "; ".join(notes) or None, runnable=True)


@dataclass(frozen=True)
class AgentFile:
    """One agent file, read: `front` is an agent skill's (`context: fork`), naming the profile
    `runs` decided."""

    front: FrontMatter
    body: str
    runs: Runs
    ignored: tuple[str, ...]


def read_agent_file(path: Path, text: str, *, opencode: bool) -> AgentFile | None:
    """`path`'s agent, or None for one that is never a subagent (OpenCode's `mode: primary`). A
    file this cannot read is `ConfigError`, which `discover` turns into a skipped file."""
    raw, body = read_block(path, text)
    if scalar(path, raw, "mode").strip() == "primary":
        return None
    name = path.stem if opencode else scalar(path, raw, "name").strip()
    if not valid_name(name):
        raise ConfigError(
            f"{path}: agent name {name!r} is not a skill name: lower case letters, digits "
            "and single hyphens"
        )
    description = " ".join(scalar(path, raw, "description").split())
    if not description:
        raise ConfigError(f"{path}: an agent needs a `description`")
    names = _opencode_tools(path, raw) if opencode else _claude_tools(path, raw)
    runs = runs_as(names, None, ours=frozenset())
    front = FrontMatter(
        name, description, context="fork", agent=runs.profile, allowed_tools=runs.tools
    )
    ignored = tuple(k for k in raw if k not in _READ_FIELDS)
    return AgentFile(front, body, runs, ignored)


def _claude_tools(path: Path, raw: Raw) -> tuple[str, ...] | None:
    if "tools" not in raw:
        return None
    return tool_names(scalar(path, raw, "tools"))


def _opencode_tools(path: Path, raw: Raw) -> tuple[str, ...] | None:
    """OpenCode's map names what is switched on or off, every other tool on. So a map that
    switches off every writer is a reader's, and one that does not is a writer's."""
    if "tools" not in raw:
        return None
    pairs = string_map(raw["tools"][1])
    if pairs is None:
        raise ConfigError(f"{path}: `tools` must be a map of names to true or false")
    off = {k for k, v in pairs if v.strip().lower() == "false"}
    writers = sorted(w for w in WRITING_TOOLS if w.islower() and w not in off)
    on = [k for k, v in pairs if v.strip().lower() == "true"]
    shell = [] if "bash" in off else ["bash"]
    return tuple(dict.fromkeys([*writers, *on, "read", "grep", "glob", *shell]))
