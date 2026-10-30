"""`approval.toml`: two levels, one `Preferences`.

Standing preferences are **the user's judgment written down** — the only reason they may
lower a floor when nothing else can. Two levels: project (`<repo>/.chickadee/approval.toml`,
committed, team policy) and user (`$XDG_CONFIG_HOME/chickadee/approval.toml`, personal).

The levels do not merge; they **compose**: denials are unioned, and a class one level
asks about is asked about whatever another allows, since the highest floor wins.

**The repository's `allow` is executable config**. A committed `allow` lowers
a command's tier for everyone who clones it, so it is inert until this clone accepts it: it
shares the digest of the repository's `[[approval.grants]]` (`RepoApproval`), which `chickadee
trust` shows and records. Its `ask` and `deny` only tighten, and apply at once.

`always_ask`, `never_ask` and `[approval.facts]` set the work tier, which was removed:
every design and plan is approved at its stop. They are refused at load, so no rule
a file states is silently not kept.

Within a level, `deny > ask > allow`, and anything unrecognised escalates. **A preference
cannot disable a gate**: the loader refuses a file that tries, because approval and
verification are separate axes and a config that could conflate them would eventually be
used to.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from chickadee.approval.commands import (
    CLASS_TIERS,
    Classification,
    CommandClass,
    CommandMatch,
)
from chickadee.approval.tier import Level, Preferences, Tier, higher, highest
from chickadee.paths import glob_match
from chickadee.schema import sha256_hex
from chickadee.state.resolve import canonical
from chickadee.state.roots import Roots
from chickadee.state.trust import trusted_digests

__all__ = [
    "APPROVAL_FILE",
    "PreferenceError",
    "RepoApproval",
    "Verdict",
    "class_floor",
    "command_denied",
    "command_floor",
    "command_rule",
    "denied",
    "load_level",
    "load_preferences",
    "merge",
    "path_rule",
    "paths_for",
    "read_approval",
    "repo_approval",
]

APPROVAL_FILE = "approval.toml"


class PreferenceError(ValueError):
    """A preference file that cannot be used. Raised at load, never at compute."""


ASK = "ask"
ALLOW = "allow"
DENY = "deny"
VERDICTS = frozenset({ASK, ALLOW, DENY})

_TOP_KEYS = frozenset({"deny", "commands", "grants"})
"""What `[approval]` may contain. `grants` are not preferences: `approval/grants.py`
reads and writes them, and they are accepted here so a file carrying them still loads."""

_RETIRED_KEYS = frozenset({"always_ask", "never_ask", "facts"})
"""The work tier's three rules: refused with that reason, never as unknown."""

_GATE_WORDS = (
    "gate",
    "lockdown",
    "verif",
    "check",
    "skip",
    "disable",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
)
"""Spellings a person reaching for "turn a gate off" would use. A key containing one is
refused with the reason, not with a generic unknown-key message."""


@dataclass(frozen=True)
class Verdict:
    """One level's parsed file, before composition."""

    level: Level
    deny: tuple[str, ...] = ()
    commands: Mapping[str, str] = field(default_factory=dict[str, str])
    """Command class → `ask` / `allow`."""


def paths_for(roots: Roots) -> tuple[Path, Path]:
    """(project, user) — the two locations. `roots.project` is already the
    repository's `.chickadee/`, so the project file is `<repo>/.chickadee/approval.toml`."""
    return roots.project / APPROVAL_FILE, roots.user / APPROVAL_FILE


def load_preferences(roots: Roots) -> Preferences:
    """Both levels, composed. A missing file is an empty level, not an error. The repository's
    `allow` rules count only once this clone accepted them (`RepoApproval`)."""
    project, user = paths_for(roots)
    levels: list[Verdict] = []
    if project.is_file():
        data = read_approval(project)
        verdict = parse_level(data, "repo", source=str(project))
        if not repo_approval(data, str(project)).accepted(roots):
            verdict = Verdict(
                verdict.level,
                verdict.deny,
                {k: v for k, v in verdict.commands.items() if v != ALLOW},
            )
        levels.append(verdict)
    if user.is_file():
        levels.append(load_level(user, "user"))
    return merge(levels)


def read_approval(path: Path) -> Mapping[str, Any]:
    """`path` parsed, `{}` when there is none; `PreferenceError` when it cannot be read."""
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PreferenceError(f"{path}: malformed TOML: {exc}") from exc
    except OSError as exc:
        raise PreferenceError(f"{path}: {exc}") from exc


def load_level(path: Path, level: Level) -> Verdict:
    return parse_level(read_approval(path), level, source=str(path))


# -- what the repository's file lowers, inert until accepted ----------------


@dataclass(frozen=True)
class RepoApproval:
    """The parts of the repository's `approval.toml` that make something run unasked: its
    `[[approval.grants]]` and the classes its `[approval.commands]` sets to `allow`. One digest
    covers both, so one acceptance in `trusted.toml` honours both and a change to either asks
    again. A file with neither is accepted by construction: there is nothing to accept."""

    grants: tuple[Mapping[str, Any], ...]
    lowerings: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not self.grants and not self.lowerings

    @property
    def digest(self) -> str:
        # The grants alone digest as they always did, so an accepted grant list stays accepted
        # until an `allow` joins it.
        body: dict[str, Any] = {"approval.grants": list(self.grants)}
        if self.lowerings:
            body["approval.commands"] = {c: ALLOW for c in self.lowerings}
        return sha256_hex(canonical(body).encode("utf-8"))

    def accepted(self, roots: Roots) -> bool:
        if self.empty:
            return True
        try:
            return self.digest in trusted_digests(roots.repo_local)
        except (OSError, ValueError, TypeError) as exc:
            # ValueError covers a TOML error; guessing either way would decide, unsaid,
            # whether a teammate's `allow` runs commands unasked.
            raise PreferenceError(
                f"{roots.repo_local} trusted.toml could not be read to tell whether this "
                f"clone accepted the repository's approval file: {exc}"
            ) from exc


def repo_approval(data: Mapping[str, Any], source: str) -> RepoApproval:
    """The repository's file's executable parts, from its parsed `data`."""
    body: object = data.get("approval", {})
    table: Mapping[str, Any] = (
        cast("Mapping[str, Any]", body) if isinstance(body, Mapping) else {}
    )
    raw: object = table.get("grants", [])
    if not isinstance(raw, list):
        raise PreferenceError(f"{source}: approval.grants must be a list of tables")
    items = cast("list[object]", raw)
    if not all(isinstance(g, Mapping) for g in items):
        raise PreferenceError(f"{source}: approval.grants must be a list of tables")
    commands: object = table.get("commands", {})
    lowerings = (
        tuple(
            sorted(
                str(name)
                for name, word in cast("Mapping[str, Any]", commands).items()
                if isinstance(word, str) and word.strip().lower() == ALLOW
            )
        )
        if isinstance(commands, Mapping)
        else ()
    )
    return RepoApproval(tuple(cast("list[Mapping[str, Any]]", items)), lowerings)


def parse_level(data: Mapping[str, Any], level: Level, *, source: str = "<memory>") -> Verdict:
    """Strict: unknown keys refuse with their path, a gate-shaped key and a retired one
    refuse with the reason, a hard command class with `allow` refuses."""
    body_raw: object = data.get("approval", {})
    if not isinstance(body_raw, Mapping):
        raise PreferenceError(f"{source}: [approval] must be a table")
    body = cast("Mapping[str, Any]", body_raw)
    extra = sorted(set(data) - {"approval"})
    if extra:
        raise PreferenceError(f"{source}: unknown top-level key(s) {extra}; only [approval]")
    for key in body:
        if key in _RETIRED_KEYS:
            raise PreferenceError(
                f"{source}: approval.{key} set the work tier, which the rebuild removed (D196 "
                "§12.3): every design and plan is approved at its stop. Remove it; deny paths "
                "with approval.deny, or a command class with [approval.commands]"
            )
        if key not in _TOP_KEYS:
            _refuse_key(source, f"approval.{key}")

    deny = _globs(body.get("deny", []), source, "deny")
    commands = _verdicts(
        body.get("commands", {}),
        source,
        "commands",
        {c.value: c for c in CommandClass},
        _class_hard,
    )
    return Verdict(level, deny, commands)


def _refuse_key(source: str, key: str) -> None:
    lowered = key.lower()
    if any(word in lowered for word in _GATE_WORDS):
        raise PreferenceError(
            f"{source}: `{key}` looks like it disables a gate. Preferences change whether "
            "you are asked, never whether anything is recorded or checked (11-router.md §6): "
            "journaling and the stops are not preferences and cannot be turned off here"
        )
    raise PreferenceError(
        f"{source}: unknown key `{key}` (known: {sorted(_TOP_KEYS)}); an unrecognised rule "
        "escalates rather than silently doing nothing, so it is refused at load"
    )


def _globs(raw: object, source: str, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(g, str) and g for g in raw):  # pyright: ignore[reportUnknownVariableType]
        raise PreferenceError(f"{source}: approval.{key} must be a list of path globs")
    return tuple(cast("list[str]", raw))


def _class_hard(name: str) -> bool:
    floor, lowerable = CLASS_TIERS[CommandClass(name)]
    return floor is Tier.T2 and not lowerable


def _verdicts(
    raw: object,
    source: str,
    table: str,
    known: Mapping[str, Any],
    hard: Any,
) -> Mapping[str, str]:
    if not isinstance(raw, Mapping):
        raise PreferenceError(f"{source}: [approval.{table}] must be a table")
    out: dict[str, str] = {}
    for name, verdict in cast("Mapping[str, Any]", raw).items():
        if name not in known:
            _refuse_key(source, f"approval.{table}.{name}")
        if not isinstance(verdict, str):
            raise PreferenceError(f"{source}: approval.{table}.{name} must be a string verdict")
        word = verdict.strip().lower()
        if word not in VERDICTS:
            # Unrecognised escalates: it is read as `ask`, never as `allow`.
            word = ASK
        if word == ALLOW and hard(name):
            raise PreferenceError(
                f'{source}: approval.{table}.{name} = "allow" would lower a hard floor; '
                "the T2 command classes are about irreversibility and no preference reaches "
                "them (11-router.md §5, R-4)"
            )
        out[name] = word
    return out


def merge(levels: Iterable[Verdict]) -> Preferences:
    """Compose the levels into one `Preferences`.

    Command verdicts compose by `max()`: the highest floor any level gives a class is its
    override, so a level saying `ask` wins over one saying `allow`. `deny` globs are unioned,
    and so is a command class set to `deny`: a denial anywhere denies.
    """
    deny: list[str] = []
    commands: dict[str, Tier] = {}
    denied_commands: set[str] = set()
    source: dict[str, Level] = {}
    for level in levels:
        for glob in level.deny:
            deny.append(glob)
            source.setdefault(f"deny:{glob}", level.level)
        for name, verdict in level.commands.items():
            _compose(commands, name, verdict, source, f"command:{name}", level.level)
            if verdict == DENY:
                denied_commands.add(name)
    return Preferences(
        deny=tuple(dict.fromkeys(deny)),
        command_overrides=commands,
        denied_commands=frozenset(denied_commands),
        source=source,
    )


def _compose[K](
    table: dict[K, Tier], key: K, verdict: str, source: dict[str, Level], tag: str, level: Level
) -> None:
    # A command class set to `deny` is refused before the tier (`command_denied`); as a
    # floor it composes like `ask`, so the class still raises where a denial is not asked.
    floor = Tier.T0 if verdict == ALLOW else Tier.T1
    current = table.get(key)
    if current is None or higher(floor, current) is not current:
        table[key] = floor
        source[tag] = level
    elif floor is current:
        source.setdefault(tag, level)


def denied(prefs: Preferences, path: str) -> str | None:
    """The `deny` glob that removes `path` from the permitted set, or `None`. `deny` is path
    globs, read as `.gitignore` reads them (`paths.glob_match`); a command is
    denied by its class (`command_denied`)."""
    return glob_match(path, prefs.deny)


def command_denied(prefs: Preferences, command: Classification) -> CommandClass | None:
    """The class, set to `deny` in `[approval.commands]`, that removes `command` from the
    permitted set, or `None`. Any command the line runs counts, as a grant needs all of
    them (`binding_of`): `ls && git push` is a push."""
    return next((m.cls for m in command.matches if m.cls.value in prefs.denied_commands), None)


# -- read where it still asks ------------------------


def class_floor(prefs: Preferences, match: CommandMatch) -> Tier:
    """One command's floor through the person's rule for its class: `allow` lowers a lowerable
    class to T0, `ask` (and `deny`, as a floor) raises a lower one to T1, and a hard floor
    never moves. Moved from the work tier's `apply_preferences` when it retired."""
    override = prefs.command_overrides.get(match.cls.value)
    if override is None:
        return match.floor
    if override is Tier.T0:
        return Tier.T0 if match.lowerable else match.floor
    return higher(match.floor, override)


def command_floor(prefs: Preferences, line: Classification) -> Tier:
    """Each command's floor in `line` through the person's class rules, and the highest of
    them: `Shell._tier`'s table term. Every class in the line counts, so `ls && npm test`
    with `tests = "ask"` asks."""
    return highest([class_floor(prefs, m) for m in line.matches])


_LEVEL_FILE: dict[Level, str] = {
    "repo": "the repository's approval.toml",
    "user": "your approval.toml",
}


def command_rule(prefs: Preferences, cls: CommandClass) -> str | None:
    """The rule the person wrote for `cls`, as a prompt or a refusal names it, or None."""
    if cls.value in prefs.denied_commands:
        word = DENY
    else:
        override = prefs.command_overrides.get(cls.value)
        if override is None:
            return None
        word = ALLOW if override is Tier.T0 else ASK
    level = prefs.source.get(f"command:{cls.value}", "user")
    return f'[approval.commands] {cls.value} = "{word}" in {_LEVEL_FILE[level]}'


def path_rule(prefs: Preferences, glob: str) -> str:
    """A `deny` glob, as a refusal names it."""
    level = prefs.source.get(f"deny:{glob}", "user")
    return f'deny "{glob}" in {_LEVEL_FILE[level]}'
