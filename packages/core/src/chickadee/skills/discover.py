"""Discovery: which skills exist, in which scope, and which shadow which.

**Four roots, one precedence.** Ours first, then the compatibility roots other agents already
use, which are read and never written:

| Scope | Root | Layout |
|---|---|---|
| project | `<repo>/.chickadee/skills/` | `<name>.md`, or `<dir>/SKILL.md` and its resources |
| user | `$XDG_CONFIG_HOME/chickadee/skills/` | the same |
| external (project) | `<repo>/.claude/skills/`, `<repo>/.agents/skills/` | `**/SKILL.md` |
| external (user) | `~/.claude/skills/`, `~/.agents/skills/` | `**/SKILL.md` |
| agent files (project) | `<repo>/.claude/agents/`, `<repo>/.opencode/agent(s)/` | `**/*.md` |
| agent files (user) | `~/.claude/agents/`, `~/.config/opencode/agent(s)/` | `**/*.md` |

**Two kinds.** A skill with `context: fork` is an agent skill, one `spawn_agent` runs on
a profile; every other is knowledge, loaded whole. Another tool's agent file is an agent
skill too (`agent_files.py`), and both are read by its one rule (`agent_files.runs_as`).
`Skill.profile` is the profile it runs as, with Claude Code's agent types mapped;
`Skill.tools` is what narrows it, in our tool names; a skill that asks for a tool that writes
is listed but not runnable. Which profiles exist, and whether a profile's tools hold `tools`,
is the agents layer's to check when it spawns.

**What runs is the file as it is now (`reread`).** The catalog is built once a session; a
skill used later is read again, whole, so its body, profile, tools and whether it may run are
the file's now, never the index's.

The first skill of a name wins and is indexed; every later one is kept as shadowed, with its
winner, so the index can show both. A knowledge layer that silently replaces knowledge
is worse than one that has none.

**Built-in names are reserved.** `/<name>` shares a namespace with the CLI's slash
commands, so a skill named for one is refused here, at index time, naming the file and the
built-in. The set lives in core because core cannot import the CLI; `12-cli.md` the table is
its source, and a test reads the table. Another tool's agent file named for a built-in is
skipped and named instead: an agent called `plan` is common there, and is never `/plan` here.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chickadee.errors import ConfigError
from chickadee.skills.agent_files import AGENT_DIRS, AgentFile, read_agent_file, runs_as
from chickadee.skills.frontmatter import FrontMatter, parse
from chickadee.state.roots import Roots

__all__ = [
    "RESERVED_NAMES",
    "SKILL_FILE",
    "Origin",
    "Scope",
    "Shadow",
    "Skill",
    "SkillCatalog",
    "SkillRoots",
    "SkippedSkill",
    "discover",
    "reread",
]

Scope = Literal["project", "user", "external"]
Kind = Literal["knowledge", "agent"]
Form = Literal["skill", "claude-agent", "opencode-agent"]
"""Which format the file is in, so it can be read again (`reread`)."""
Origin = Literal["project", "user"]
"""Whose tree a skill came from: an external skill is under the repository or under `$HOME`."""

SKILL_FILE = "SKILL.md"
SKILLS_DIR = "skills"
EXTERNAL_DIRS = (".claude", ".agents")
"""The compatibility roots: another agent's skills, in the same open format."""

RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "ask",
        "attach",
        "budget",
        "clear",
        "config",
        "context",
        "diff",
        "drop",
        "exit",
        "gc",
        "handoff",
        "help",
        "investigate",
        "jobs",
        "journal",
        "notify",
        "pause",
        "plan",
        "redo",
        "rename",
        "replay",
        "resume",
        "run",
        "rung",
        "skills",
        "status",
        "undo",
        "unload",
        "verbose",
        "workspace",
    }
)
"""`12-cli.md` the built-in slash commands. Adding a built-in reserves its name
in the same edit; `tests/skills/test_reserved_names.py` reads the table, so neither drifts."""


@dataclass(frozen=True)
class Skill:
    name: str
    scope: Scope
    origin: Origin
    path: Path
    """The file `load_skill` reads: `<name>.md`, or a directory's `SKILL.md`."""
    front: FrontMatter
    ignored: tuple[str, ...] = ()
    """An external skill's fields outside the closed set, for the index's note."""
    profile: str | None = None
    """An agent skill's profile; None for knowledge."""
    tools: tuple[str, ...] | None = None
    """What narrows an agent skill's tools, in our names; None narrows nothing."""
    note: str | None = None
    """What the index says of how an agent skill runs: a mapped profile, an ignored tool, or why
    it cannot run."""
    runnable: bool = True
    form: Form = "skill"

    @property
    def kind(self) -> Kind:
        return "agent" if self.front.is_agent else "knowledge"


@dataclass(frozen=True)
class Shadow:
    """A skill another of the same name hides. Shown, never loaded."""

    skill: Skill
    by: Skill


@dataclass(frozen=True)
class SkippedSkill:
    """An external file that is not a skill we can read: passed over and named,
    never refused. It was written for another tool, and one broken file of theirs must not
    stop a session."""

    path: Path
    why: str


@dataclass(frozen=True)
class SkillCatalog:
    """A session's skills: the indexed ones in precedence order, the shadowed ones, and the
    external files passed over."""

    skills: tuple[Skill, ...] = ()
    shadowed: tuple[Shadow, ...] = ()
    skipped: tuple[SkippedSkill, ...] = ()

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.skills if s.name == name), None)

    def near(self, name: str) -> tuple[str, ...]:
        """Indexed names close to `name`, best first: what an unresolved load offers instead
        of guessing which file was meant."""
        names = [s.name for s in self.skills]
        return tuple(difflib.get_close_matches(name, names, n=3, cutoff=0.6))

    def shadows_of(self, skill: Skill) -> tuple[Skill, ...]:
        return tuple(s.skill for s in self.shadowed if s.by == skill)


@dataclass(frozen=True)
class SkillRoots:
    """Where a session's skills are looked for. `external = False` is `skills.external`."""

    project: Path
    """`<repo>/.chickadee`."""
    user: Path
    """`$XDG_CONFIG_HOME/chickadee`."""
    repo: Path
    """The worktree, whose `.claude/` and `.agents/` are external project roots."""
    home: Path
    """`$HOME`, whose `.claude/` and `.agents/` are external user roots."""
    external: bool = True

    @classmethod
    def of(cls, roots: Roots, *, home: Path | None = None, external: bool = True) -> SkillRoots:
        return cls(
            roots.project,
            roots.user,
            roots.project.parent,
            home if home is not None else Path.home(),
            external,
        )


def discover(roots: SkillRoots) -> SkillCatalog:
    """Every skill under `roots`, precedence first. One of ours that is malformed or declares a
    field outside the closed set, or a skill in any scope named for a built-in, is
    `ConfigError` naming the file: refused at index time, never loaded, never shadowing.
    An external file that is not a readable skill is skipped and named."""
    winners: dict[str, Skill] = {}
    shadowed: list[Shadow] = []
    skipped: list[SkippedSkill] = []
    for skill in _walk(roots):
        if isinstance(skill, SkippedSkill):
            skipped.append(skill)
            continue
        if skill.name in RESERVED_NAMES:
            raise ConfigError(
                f"{skill.path}: skill {skill.name!r} has the name of the built-in command "
                f"`/{skill.name}` (12-cli.md §6); rename the skill (SK-18)"
            )
        won = winners.get(skill.name)
        if won is None:
            winners[skill.name] = skill
        else:
            shadowed.append(Shadow(skill, won))
    return SkillCatalog(tuple(winners.values()), tuple(shadowed), tuple(skipped))


def _walk(roots: SkillRoots) -> Iterator[Skill | SkippedSkill]:
    yield from _ours(roots.project / SKILLS_DIR, "project")
    yield from _ours(roots.user / SKILLS_DIR, "user")
    if not roots.external:
        return
    bases: tuple[tuple[Path, Origin], ...] = ((roots.repo, "project"), (roots.home, "user"))
    for base, origin in bases:
        for tool in EXTERNAL_DIRS:
            root = base / tool / SKILLS_DIR
            if root.is_dir():
                for path in sorted(root.rglob(SKILL_FILE)):
                    try:
                        yield _read(path, "external", origin)
                    except ConfigError as exc:
                        yield SkippedSkill(path, str(exc).removeprefix(f"{path}: "))
    for base, origin in bases:
        for tool, folder in AGENT_DIRS:
            # OpenCode's user files are under `~/.config/opencode/`, not `~/.opencode/`.
            user_opencode = origin == "user" and tool == ".opencode"
            root = (
                base / ".config" / "opencode" / folder
                if user_opencode
                else base / tool / folder
            )
            yield from _agent_files(root, origin, opencode=tool == ".opencode")


def _agent_files(
    root: Path, origin: Origin, *, opencode: bool
) -> Iterator[Skill | SkippedSkill]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.md")):
        try:
            got = read_agent_file(path, _text(path), opencode=opencode)
        except ConfigError as exc:
            yield SkippedSkill(path, str(exc).removeprefix(f"{path}: "))
            continue
        if got is None:
            continue
        if got.front.name in RESERVED_NAMES:
            yield SkippedSkill(path, f"the name of the built-in command `/{got.front.name}`")
            continue
        yield _agent_skill(path, origin, got, opencode=opencode)


def _agent_skill(path: Path, origin: Origin, got: AgentFile, *, opencode: bool) -> Skill:
    return Skill(
        got.front.name,
        "external",
        origin,
        path,
        got.front,
        got.ignored,
        profile=got.runs.profile,
        tools=got.runs.tools,
        note=got.runs.note,
        runnable=got.runs.runnable,
        form="opencode-agent" if opencode else "claude-agent",
    )


def _ours(root: Path, scope: Origin) -> Iterator[Skill]:
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix == ".md":
            yield _read(entry, scope, scope)
        elif entry.is_dir() and (entry / SKILL_FILE).is_file():
            yield _read(entry / SKILL_FILE, scope, scope)


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: a skill must be readable UTF-8 text ({exc})") from None


def _read(path: Path, scope: Scope, origin: Origin) -> Skill:
    return _read_with_body(path, _text(path), scope, origin)[0]


def _read_with_body(path: Path, text: str, scope: Scope, origin: Origin) -> tuple[Skill, str]:
    external = scope == "external"
    parsed = parse(path, text, external=external)
    front = parsed.front
    if not front.is_agent:
        return Skill(front.name, scope, origin, path, front, parsed.ignored), parsed.body
    # Ours name our tools; another tool's names are mapped, and a name with no counterpart is
    # ignored. `None` for `ours` takes any lower-case name as one of ours.
    runs = runs_as(front.allowed_tools, front.agent, ours=None if not external else frozenset())
    skill = Skill(
        front.name,
        scope,
        origin,
        path,
        front,
        parsed.ignored,
        profile=runs.profile,
        tools=runs.tools,
        note=runs.note,
        runnable=runs.runnable,
    )
    return skill, parsed.body


def reread(skill: Skill) -> tuple[Skill, str]:
    """`skill` as its file says now, and its body: it may have changed since the catalog was
    built, and what is used is what is on disk. `ConfigError` or `OSError` when it no longer
    reads."""
    text = skill.path.read_text(encoding="utf-8")
    if skill.form == "skill":
        return _read_with_body(skill.path, text, skill.scope, skill.origin)
    opencode = skill.form == "opencode-agent"
    got = read_agent_file(skill.path, text, opencode=opencode)
    if got is None:
        raise ConfigError(f"{skill.path}: is now a primary agent, not a subagent")
    return _agent_skill(skill.path, skill.origin, got, opencode=opencode), got.body
