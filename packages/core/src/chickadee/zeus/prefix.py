"""Zeus's system prefix, composed.

Four parts, most stable first, because each invalidates only what follows it:

    1 template adapters/frontier/templates/zeus.md per release
    2 tools SURFACE's schemas, full per release
    3 skills the skills index per session
    4 env workspace, platform, sandbox, model, trust per machine

The template, the index and the environment block are the system message; the schemas are
the request's `tools` array, which is where a provider holds them. One sha covers all four,
and each part's own sha goes beside it to the session log (`record_prefix`), so a prefix that
changed says which part moved.

**Each part's text is a session blob under its own sha,** so the `prefix` line, which
names those shas, is also the full prompt: `system_text` puts the system message back
together from them, the way `assemble` made it, for a reader that wants to see what Zeus was
told. The line's bytes do not change for it.

**What is not here.** The date and the git head: either would be a cache miss on the largest
region for the cheapest fact, so they are the turn frame the loop writes before a user turn.
The served model: the block names the configured one, because it is written before the first
call and requested and served are never conflated. And no transform hook: instruction
enters as a pinned skill or an amendment to `zeus.md`, both diffable.

**A call may carry `findings`, and no purpose line.** Layer 1
collapses an exchange to a receipt only when it has a finding, written by Zeus inside its
ordinary turn. The resident schemas carry `findings` beside each tool's arguments
(`tools/meta.py`); the loop validates a call against its tool's own schema with it set aside,
and records the call as it was sent (invariant 19).
"""

from __future__ import annotations

import platform as _platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Literal

from pydantic import PlainSerializer, with_config

from chickadee.adapters.base import load_template, tool_descriptions
from chickadee.adapters.frontier import TEMPLATES as FRONTIER_TEMPLATES
from chickadee.client.request import Message, ToolSchema
from chickadee.context.lines import LINE, SessionKind, canonical_json
from chickadee.errors import HarnessBug
from chickadee.schema import sha256_hex
from chickadee.schema.execution import SandboxLevel
from chickadee.tools.meta import shown
from chickadee.tools.spec import ToolSpec
from chickadee.tools.validate import resident
from chickadee.work import with_work_dir

if TYPE_CHECKING:
    from chickadee.context.session import Session
    from chickadee.roles import Role

__all__ = [
    "PARTS",
    "TEMPLATES",
    "Env",
    "PartName",
    "Platform",
    "Prefix",
    "assemble",
    "current_platform",
    "record_prefix",
    "render_env",
    "store_texts",
    "system_text",
]

TEMPLATES: Final = FRONTIER_TEMPLATES
"""The frontier family's templates: `zeus.md`, and `tools/<name>.md` beside it (33 §6)."""

PartName = Literal["template", "tools", "skills", "env"]
PARTS: Final[tuple[PartName, ...]] = ("template", "tools", "skills", "env")
"""The order. The order is a cost decision, not a layout one."""

Platform = Literal["linux", "darwin", "wsl2"]


# -- the environment block ----------------------------------------------------------------


@dataclass(frozen=True)
class Env:
    """Part 4: what Zeus cannot infer and needs (D99's four facts, less the date)."""

    workspace: str
    """The resolved repository root."""
    platform: Platform
    sandbox: SandboxLevel
    reads_masked: bool
    """Whether the deny list is masked at the sandbox, not only refused by `read`."""
    model: str
    """The zeus rung's CONFIGURED `model_id`. A substitution is a notice, never an edit here."""
    repo_trust: Literal["first_party", "third_party"]


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def render_env(env: Env) -> str:
    return "\n".join(
        [
            "<env>",
            f"workspace: {env.workspace}",
            f"platform: {env.platform}",
            f"sandbox: {env.sandbox.value} · reads_masked: {_yes(env.reads_masked)}",
            f"model: {env.model}",
            f"repo_trust: {env.repo_trust}",
            "</env>",
        ]
    )


def current_platform() -> Platform:
    """The platform the shell runs on. WSL2 is Linux to Python and the Windows path to a
    person, and `sed -i` does not care which, but a path the user pastes does."""
    system = _platform.system()
    if system == "Darwin":
        return "darwin"
    if system == "Linux" and "microsoft" in _platform.release().lower():
        return "wsl2"
    return "linux"


# -- the prefix ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prefix:
    """What `assemble` composes: the system text, the resident tools, and the shas."""

    text: str
    """The system message: the template, the skills index, the environment block."""
    tools: tuple[ToolSchema, ...]
    """Part 2, as the request's `tools` array."""
    parts: Mapping[PartName, str]
    """Each part's sha, in the order."""
    sha: str
    """Over the four part shas, in order: any part that moves, moves it."""
    model_id: str
    """The configured model the environment block names."""
    texts: Mapping[PartName, str]
    """Each part's text, whose sha is `parts`': what `record_prefix` stores."""

    def system(self) -> Message:
        return Message("system", self.text)

    def line(self) -> PrefixLine:
        return PrefixLine(self.sha, self.parts, self.model_id)


@with_config(LINE)
@dataclass(frozen=True)
class PrefixLine:
    """The `prefix` line: the prefix's sha, its parts' shas, and the
    model the environment block names. Zeus's own kind, so its type is here."""

    KIND: ClassVar[SessionKind] = "prefix"
    sha: str
    parts: Annotated[Mapping[PartName, str], PlainSerializer(dict, return_type=dict[str, str])]
    model_id: str


def assemble(
    role: Role,
    surface: tuple[ToolSpec, ...],
    skills: str,
    env: Env,
    *,
    templates: Path = TEMPLATES,
    handoffs: Sequence[tuple[str, str]] = (),
    work_dir: str,
) -> Prefix:
    """The four parts in order, once per session — and again only when `/rung` moves
    Zeus's model, since the block must name the model now configured.

    `surface` is `SURFACE` less any tool a standing deny covers wholly. `skills` is
    the index `32-skills.md` §4 renders — name, scope and description per skill, never a
    body — and is empty when there are none."""
    if role.tools != "zeus":
        raise HarnessBug(f"role {role.name!r} does not hold Zeus's surface; this is its prefix")
    if role.family != "frontier":
        raise HarnessBug(f"Zeus's template is a frontier template, not {role.prompt!r}")
    # The design and brief templates in force follow the instructions that tell Zeus to write
    # to them: each as `## <title>` and its text, a repository's own where it has one.
    template = "\n\n".join(
        [
            # `[work] dir`, where his first `doc` opens a design's folder.
            with_work_dir(load_template(templates, role.template), work_dir).rstrip("\n"),
            *(f"## {title}\n\n{text.strip()}" for title, text in handoffs),
        ]
    )
    tools = shown(resident(surface, tool_descriptions(surface, templates)), findings=True)
    listed = [
        {"name": t.name, "description": t.description, "parameters": t.schema} for t in tools
    ]
    tools_text = canonical_json({"tools": listed})
    env_text = render_env(env)
    index = skills.strip("\n")
    shas: dict[PartName, str] = {
        "template": sha256_hex(template),
        "tools": sha256_hex(tools_text),
        "skills": sha256_hex(index),
        "env": sha256_hex(env_text),
    }
    parts: dict[PartName, str] = {p: shas[p] for p in PARTS}
    texts: dict[PartName, str] = {
        "template": template,
        "tools": tools_text.decode("utf-8"),
        "skills": index,
        "env": env_text,
    }
    sha = sha256_hex(canonical_json({"parts": [[p, parts[p]] for p in PARTS]}))
    return Prefix(
        text=system_text(texts),
        tools=tools,
        parts=parts,
        sha=sha,
        model_id=env.model,
        texts={p: texts[p] for p in PARTS},
    )


def system_text(texts: Mapping[PartName, str]) -> str:
    """The system message from its parts' texts: the template, the skills index and the
    environment block, each that is not empty, a blank line between. The one place this join
    is spelled, so a reader rebuilding it from the session's blobs gets the bytes Zeus got."""
    return "\n\n".join(t for t in (texts["template"], texts["skills"], texts["env"]) if t)


def store_texts(session: Session, prefix: Prefix) -> None:
    """Each part's text as a session blob. Idempotent, and cheap when they are there,
    so a session resumed from in older versions gains them at its next turn."""
    for part in PARTS:
        sha = session.put_blob(prefix.texts[part].encode("utf-8"))
        if sha != prefix.parts[part]:
            raise HarnessBug(f"the prefix's {part} text does not hash to its recorded sha")


def record_prefix(session: Session, prefix: Prefix) -> int:
    """The prefix's sha to the session log: at session start, and after a `/rung` switch. The
    parts' texts are stored first, since a line names nothing that is not yet durable."""
    store_texts(session, prefix)
    return session.append(prefix.line()).seq
