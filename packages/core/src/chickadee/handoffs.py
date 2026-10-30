"""The hand-off templates: the design Zeus
writes, the plan Athena writes, and the brief either writes for a task.

A template is prose a model follows, never a schema: nothing parses a design, a plan or a
brief, and nothing refuses one for its shape. So a template can be switched
without a code change, and the one that ran is recorded by its sha, so two can be compared
(the journal's `task_delegated.template_sha`; the eval manifest).

Where one comes from, first match wins, through the lookup conventions use
(`profiles/loader.resolve_document`): the repository's `.chickadee/templates/<kind>.md`,
then the file `[templates]` names, then the shipped one under
`adapters/shared/templates/handoffs/`. A repository file that cannot be read as UTF-8
text falls back to the shipped one with a warning, as an unreadable conventions file does.

Read by Athena's head (`boxctx/athena.athena_head`), and by the stops, which bind each approval
to its template's sha (`delegation.stop_at_dispatch`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, get_args

from chickadee.adapters.base import SHARED_TEMPLATES
from chickadee.errors import HarnessBug
from chickadee.profiles.loader import resolve_document
from chickadee.schema.ids import sha256_hex

__all__ = [
    "HANDOFFS",
    "REPO_TEMPLATES",
    "SHIPPED",
    "ZEUS_HANDOFFS",
    "Handoff",
    "HandoffTemplate",
    "resolve",
]

Handoff = Literal["design", "plan", "brief"]

HANDOFFS: Final[tuple[Handoff, ...]] = get_args(Handoff)

REPO_TEMPLATES: Final = Path(".chickadee") / "templates"
"""Where a repository's own templates live, beside `.chickadee/conventions.md`."""

SHIPPED: Final = SHARED_TEMPLATES / "handoffs"


@dataclass(frozen=True)
class HandoffTemplate:
    kind: Handoff
    text: str
    sha: str
    """`sha256_hex(text)`: what the journal and the eval manifest record."""
    source: Literal["repository", "config", "shipped"]
    warning: str | None = None
    """Why a file earlier in the order was not used (a named file missing, a repository file
    unreadable), for the caller to show; None when the first choice was used."""


ZEUS_HANDOFFS: Final[tuple[tuple[Handoff, str], ...]] = (
    ("design", "The design template"),
    ("brief", "The brief template (a one-task plan)"),
)
"""The templates Zeus writes to, each under the heading his prefix shows it with: the
same files Athena's head and the delegation read, so what he is shown is what is in force."""


def resolve(repo: Path, configured: Mapping[str, Path], kind: Handoff) -> HandoffTemplate:
    """The template in force for `kind`: the repository's, else the configured one (a relative
    path read from `repo`), else the shipped one."""
    own = repo / REPO_TEMPLATES / f"{kind}.md"
    named = configured.get(kind)
    if named is not None and not named.is_absolute():
        named = repo / named
    shipped = SHIPPED / f"{kind}.md"
    path, warning = resolve_document(own, named, shipped)
    if path is None:
        raise HarnessBug(f"the shipped {kind} template is missing: {shipped}")
    source: Literal["repository", "config", "shipped"] = (
        "repository" if path == own else "config" if path == named else "shipped"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        if path == shipped:
            raise HarnessBug(f"the shipped {kind} template is unreadable: {shipped}") from None
        text = _shipped(kind)
        warning = f"{kind} template {path} unreadable; using the shipped default"
        source = "shipped"
    return HandoffTemplate(kind, text, sha256_hex(text), source, warning)


def _shipped(kind: Handoff) -> str:
    try:
        return (SHIPPED / f"{kind}.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HarnessBug(f"the shipped {kind} template is unreadable") from exc
