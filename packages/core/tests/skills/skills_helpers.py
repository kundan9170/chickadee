"""Helpers for the skills suite: skill files on disk, a session, a load made the way
Zeus's loop makes one, and a cut through the real `SessionCompaction`."""

from __future__ import annotations

from pathlib import Path

from chickadee.config import PriceRow, RungEndpoint
from chickadee.context.compaction.cut import Compaction
from chickadee.context.compaction.session import SessionCompaction
from chickadee.context.request import assemble
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.schema.adapter import CacheMode, ToolCall
from chickadee.skills.discover import SkillRoots
from chickadee.skills.load import Skills
from chickadee.skills.pin import pinned_seqs

SESSION_ID = "S-skills"


def skill_text(name: str, description: str, body: str, *, front: str = "") -> str:
    return f"---\nname: {name}\ndescription: {description}\n{front}---\n{body}"


def write_skill(
    root: Path,
    name: str,
    description: str = "how this project does a thing",
    body: str = "# Doing it\n\nAlways do it this way.\n",
    *,
    front: str = "",
    bundle: bool = False,
) -> Path:
    """One skill under `root` (a `skills/` directory): `<name>.md`, or `<name>/SKILL.md` when
    `bundle`, as our two layouts are."""
    path = root / name / "SKILL.md" if bundle else root / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(skill_text(name, description, body, front=front), encoding="utf-8")
    return path


class Tree:
    """A repository, an XDG config home and a `$HOME`, each with its skills directories."""

    def __init__(self, base: Path) -> None:
        self.repo = base / "repo"
        self.home = base / "home"
        self.xdg = base / "xdg" / "chickadee"
        for path in (self.repo, self.home, self.xdg):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def project(self) -> Path:
        return self.repo / ".chickadee" / "skills"

    @property
    def user(self) -> Path:
        return self.xdg / "skills"

    def external(self, tool: str = ".claude", *, home: bool = False) -> Path:
        return (self.home if home else self.repo) / tool / "skills"

    def roots(self, *, external: bool = True) -> SkillRoots:
        return SkillRoots(self.repo / ".chickadee", self.xdg, self.repo, self.home, external)


def new_session(base: Path) -> Session:
    return Session.create(base / "sessions", SESSION_ID)


def load(session: Session, skills: Skills, name: str, cid: str) -> str:
    """One `load_skill` call as the loop makes it: the call, the port's answer, its line, and
    the next turn's finding from it (a result with no finding is never collapsed)."""
    session.record_assistant(
        "", [ToolCall(id=cid, name="load_skill", arguments={"name": name})]
    )
    got = skills.load(name, cid)
    text = got if isinstance(got, str) else got.render()
    session.record_tool(cid, text)
    session.record_assistant("noted", findings={cid: f"the {name} skill says how to work"})
    return text


ENDPOINT = RungEndpoint(
    base_url="http://x", model_id="m", context_window=4_000, max_output_tokens=200
)


def compaction() -> SessionCompaction:
    """Zeus's compaction as the engine composes it, with the pinned set from the log."""
    return SessionCompaction(
        ENDPOINT,
        PriceRow(price_in=3.0, price_out=15.0, price_cached_in=0.3, cache_ttl_s=300),
        CacheMode.NONE,
        resolver=lambda: HandleResolver(Path("."), lambda sha: b""),
        pinned=pinned_seqs,
    )


def must_fit(session: Session, driver: SessionCompaction | None = None) -> Compaction:
    """A cut on the provider's word that the request was too long: everything evictable is."""
    return (driver or compaction()).overflowed(session, 50)


def context_text(session: Session) -> str:
    """Every message the next request would carry, as one string."""
    return "\n".join(str(m.content) for m in assemble(session.state, session.blob).messages)


def filler(session: Session, turns: int = 6) -> None:
    """Prose and reads enough that a cut has a region to take."""
    for n in range(turns):
        session.record_user(f"step {n}: " + "explain the plan in detail please " * 20)
        session.record_assistant(f"answer {n}: " + "here is a long explanation " * 20)
