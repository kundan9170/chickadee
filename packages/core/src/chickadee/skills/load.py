"""`load_skill`: a skill's body, whole, pinned, or nothing.

**Its own loader, never the read path.** One read of the skill's file, no tier, no span, no
preview: a partial body presents as complete knowledge, and prose has no compiler to catch the
qualifier that was not read. So the bound is on the file. Over `SKILL_BODY_MAX` the load
refuses, naming the file and its size, and loads nothing. A bundled `scripts/` or reference
file is never read here: level 3 is the ordinary read path's, or the Executor's.

**The outcomes.** The framed body, recorded and pinned. Or a typed refusal, which records
nothing:
- an unknown name, offering the near ones;
- a body over the cap;
- a full pin budget, naming what is pinned;
- under `third_party`, a project skill the person has not accepted.

The model never makes room for itself. Whatever decides to load, the model selecting
or the person invoking (`invoke.py`), comes through `Skills.load`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from chickadee.context.results import InvalidArgument
from chickadee.context.session import Session
from chickadee.errors import ConfigError
from chickadee.limits import limit, tokens_est
from chickadee.schema import sha256_hex
from chickadee.skills.discover import Skill, SkillCatalog, reread
from chickadee.skills.frame import frame_body, is_untrusted
from chickadee.skills.index import HOW, skill_row
from chickadee.skills.pin import SKILL_PINNED_MAX, SkillLoaded, pinned
from chickadee.skills.search import rank
from chickadee.trust.events import UntrustedRead

__all__ = ["SKILL_BODY_MAX", "NoSkills", "Skills", "read_again"]

SKILL_BODY_MAX = 12_000
"""Tokens, on the file. Over it the load refuses; a document that size should be two."""


class Skills:
    """The `skills` port over one session's catalog. `accepted` is whether the person accepted
    this repository's executable config (`TrustDecision.accepted`), which under
    `repo_trust = third_party` is also what admits its skills' bodies."""

    def __init__(
        self,
        catalog: SkillCatalog,
        session: Session,
        *,
        repo_trust: Literal["first_party", "third_party"] = "first_party",
        accepted: bool = False,
        find_results: int | None = None,
        desc_chars: int | None = None,
        costs: Mapping[str, str] | None = None,
    ) -> None:
        """`find_results` and `desc_chars` are the config's `skills.find_results` and
        `skills.index_desc_chars`, None their defaults; `costs` says what one round of each
        profile may spend, for `find`'s cost hint."""
        self.catalog = catalog
        self._session = session
        self._untrusted_repo = repo_trust == "third_party"
        self._accepted = accepted
        self._find_results = (
            find_results
            if find_results is not None
            else int(limit("skills.find_results").value)
        )
        self._desc_chars = (
            desc_chars
            if desc_chars is not None
            else int(limit("skills.index_desc_chars").value)
        )
        self._costs = dict(costs or {})

    def find(self, query: str) -> str:
        """The best matches for `query`, in the index's rows
        (`index.skill_row`), an agent skill's with what one round of its profile may cost,
        under a line saying how each kind is used."""
        hits = rank(self.catalog, query, self._find_results)
        profiles = self._profiles(query, every=not hits)
        if not hits:
            none = f"no skill matches {query!r}; the index lists every skill this session has"
            return "\n".join((none, *profiles))
        rows = [
            skill_row(
                hit.skill,
                third_party=self._untrusted_repo,
                desc_chars=self._desc_chars,
                cost=self._costs.get(hit.skill.profile or ""),
            )
            for hit in hits
        ]
        return "\n".join((f"Best first. {HOW}", *rows, *profiles))

    def _profiles(self, query: str, *, every: bool) -> tuple[str, ...]:
        """The built-in and person's profiles, which are not skills and so never rank: every
        one when no skill matched, else those `query` names. Without them a model looking
        for the researcher here finds nothing and is told nothing exists. `costs` is keyed by
        the session's profiles; a port given none lists none."""
        shown = [
            f"- {name} · profile, no skill: spawn_agent(profile={name!r}, task=…) · {cost}"
            for name, cost in self._costs.items()
            if every or name in query.lower()
        ]
        if not shown:
            return ()
        return (
            "Profiles run an agent with no skill; send one as `profile`, never `skill`:",
            *shown,
        )

    def load(self, name: str, call_id: str) -> str | InvalidArgument:
        """`name`'s body, framed, after recording its load against `call_id`: the call whose
        tool line will carry it, and which a cut will lift while it is pinned."""
        skill = self.catalog.get(name)
        if skill is None:
            near = self.catalog.near(name)
            hint = (
                f"; did you mean {', '.join(near)}?" if near else "; the index lists them all"
            )
            return InvalidArgument("name", f"no skill is named {name!r}{hint}")
        untrusted = is_untrusted(skill, self._untrusted_repo)
        if untrusted and not self._accepted:
            return InvalidArgument(
                "name",
                f"{name} is this third-party repository's skill, and the person has not "
                "accepted the repository's executable config, which admits its skills (§8)",
            )
        with self._session.lock:
            pins = pinned(self._session)
            if any(p.name == name for p in pins):
                return f"{name} is already loaded and pinned for this session."
            if len(pins) >= SKILL_PINNED_MAX:
                held = ", ".join(p.name for p in pins)
                return InvalidArgument(
                    "name",
                    f"{SKILL_PINNED_MAX} skills are pinned ({held}), the most a session holds. "
                    "Say which one you believe is finished; only the person can /unload it",
                )
            again = read_again(skill)
            if isinstance(again, InvalidArgument):
                return again
            _, body = again
            size = tokens_est(body)
            if size > SKILL_BODY_MAX:
                return InvalidArgument(
                    "name",
                    f"{skill.path} is about {size} tokens, over the {SKILL_BODY_MAX} a skill "
                    "body may be; nothing was loaded. It should be split into two skills",
                )
            content_sha = sha256_hex(body.encode("utf-8"))
            self._session.append(
                SkillLoaded(
                    call_id=call_id,
                    name=name,
                    scope=skill.scope,
                    origin=skill.origin,
                    path_sha=sha256_hex(str(skill.path).encode("utf-8")),
                    content_sha=content_sha,
                    shadowed=bool(self.catalog.shadows_of(skill)),
                )
            )
            if untrusted:
                self._session.append(
                    UntrustedRead(path=str(skill.path), bytes=len(body.encode("utf-8")))
                )
        return frame_body(skill, body, content_sha, untrusted=untrusted)


class NoSkills:
    """The `skills` port of a session with none indexed: `load_skill` is still offered, as the
    prefix lists it, and says there is nothing to load."""

    def load(self, name: str, call_id: str) -> str | InvalidArgument:
        del call_id
        return InvalidArgument("name", f"no skill is named {name!r}; this session has none")

    def find(self, query: str) -> str:
        return f"no skill matches {query!r}: this session has none"


def read_again(skill: Skill) -> tuple[Skill, str] | InvalidArgument:
    """The skill as its file says now, and its body, read whole (`discover.reread`): what is
    pinned, or run, is what is on disk now. Refused when the file no longer reads or now names
    another skill."""
    try:
        now, body = reread(skill)
    except (OSError, UnicodeDecodeError) as exc:
        return InvalidArgument("name", f"{skill.path} cannot be read: {exc}")
    except ConfigError as exc:
        return InvalidArgument("name", str(exc))
    if now.name != skill.name:
        return InvalidArgument(
            "name", f"{skill.path} now names itself {now.name!r}; nothing was loaded"
        )
    return now, body
