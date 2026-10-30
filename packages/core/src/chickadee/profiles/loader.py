"""Builtin discovery, repo-override deep-merge, and freeze at run start.

Frozen once at run start: a mid-run profile edit does not take effect, so a run's
behavior is reproducible from its config snapshot. Without that, two runs from the same
journal could diverge for reasons the journal does not record — which would make replay
a claim rather than a property.
"""

from __future__ import annotations

import functools
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from chickadee.profiles.model import LanguageProfile, ProfileError, TestRunner
from chickadee.profiles.resolve import FileLanguage, resolve

# `profiles/` at the repo root holds the TOML data; this package holds the loader code.
# The plan's §3 layout lists both, and the names are easy to confuse.
BUILTIN_DIR = Path(__file__).resolve().parents[4].parent / "profiles"
OVERRIDE_SUBDIR = Path(".chickadee") / "profiles"
RUNNERS_SUBDIR = "runners"
"""Each test runner's output as data, beside the profiles; not a profile,
and never globbed as one (`builtin_paths` reads the top level only)."""

CONVENTIONS_OVERRIDE = Path(".chickadee") / "conventions.md"


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """PR-2 — an override changes `tests.runner` without restating the whole file.

    Tables merge; scalars and arrays replace. Merging arrays element-wise would make
    "replace the test globs" impossible to express, which is a thing repos need to do.
    """
    out = dict(base)
    for key, value in over.items():
        existing = out.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            out[key] = deep_merge(
                cast("dict[str, Any]", existing), cast("dict[str, Any]", value)
            )
        else:
            out[key] = value
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"{path}: malformed TOML: {exc}") from exc


def _validate(data: dict[str, Any], source: Path) -> LanguageProfile:
    """Strict: an unknown key rejects with its path, matching contract discipline. A
    profile that does not state `[syntax].mode` is refused, naming the file:
    no mode is assumed, since a default once judged every such language as Python."""
    try:
        profile = LanguageProfile.model_validate(data)
    except ValidationError as exc:
        raise ProfileError(f"{source}: {exc}") from exc
    except ProfileError as exc:
        raise ProfileError(f"{source}: {exc}") from exc
    if profile.syntax is None:
        raise ProfileError(
            f'{source}: [syntax].mode must be stated ("inproc", "cmd" or "parse"); no mode is '
            "assumed"
        )
    return profile


def builtin_paths(builtin_dir: Path | None = None) -> list[Path]:
    d = builtin_dir or BUILTIN_DIR
    return sorted(d.glob("*.toml")) if d.is_dir() else []


def load_profile(
    name: str, repo: Path | None = None, builtin_dir: Path | None = None
) -> LanguageProfile:
    """One profile: builtin, deep-merged with a repo override when present."""
    d = builtin_dir or BUILTIN_DIR
    builtin = d / f"{name}.toml"
    if not builtin.exists():
        raise ProfileError(f"no builtin profile named {name!r} in {d}")
    data = _read_toml(builtin)
    source = builtin

    if repo is not None:
        override = repo / OVERRIDE_SUBDIR / f"{name}.toml"
        if override.exists():
            data = deep_merge(data, _read_toml(override))
            source = override
    return _validate(data, source)


def executable_overrides(repo: Path, builtin_dir: Path | None = None) -> dict[str, Any]:
    """The executable subset a repository's profile overrides contribute.

    A project profile that sets `[toolchain].bootstrap` puts a command in front of the
    harness — one that runs at worktree creation with the network up — so it belongs under
    `31-state.md` the trust digest with `[[hooks]].command` and the rest: inert on a fresh
    clone until a person has read it and accepted the digest. The shipped profiles'
    bootstraps are the harness's own and are **not** here; only what the repo declared.

    Returned in the shape `state.trust.EXECUTABLE_PATHS` names (`toolchain.bootstrap`,
    keyed by profile name), so a caller merges it into the config mapping it hands to
    `executable_subset` and the digest moves when the override does. An override that
    sets `bootstrap = []` is a declaration too — "this project installs nothing" — and is
    recorded as such.
    """
    overrides = repo / OVERRIDE_SUBDIR
    if not overrides.is_dir():
        return {}
    builtin = {p.stem for p in builtin_paths(builtin_dir)}
    declared: dict[str, list[str]] = {}
    for path in sorted(overrides.glob("*.toml")):
        if path.stem not in builtin:
            continue  # not a profile `ProfileSet.load` would ever read
        data = _read_toml(path)
        toolchain = data.get("toolchain")
        if not isinstance(toolchain, dict) or "bootstrap" not in toolchain:
            continue
        # Validated through the same model as the merged profile, so a string here refuses
        # here too rather than surfacing as a digest over something that cannot load.
        merged = load_profile(path.stem, repo=repo, builtin_dir=builtin_dir)
        bootstrap = merged.toolchain.bootstrap
        declared[path.stem] = list(bootstrap.argv) if bootstrap is not None else []
    return {"toolchain": {"bootstrap": declared}} if declared else {}


class ProfileSet:
    """Every profile a run will use, loaded once and frozen. The one home for which
    language a file is and which grammar parses it: `language_of`."""

    def __init__(self, profiles: dict[str, LanguageProfile], repo: Path | None = None) -> None:
        self._by_name = dict(profiles)
        self._repo = repo
        """The repository whose overrides were merged, and whose manifests break a tie between
        two profiles claiming one extension; None for the shipped set alone."""
        self._by_ext: dict[str, list[LanguageProfile]] = {}
        self._answered: dict[str, FileLanguage | None] = {}
        """`language_of`'s answer per extension, which is all it depends on: a tie a manifest
        breaks checks the disk once per set, not once per file. The set is frozen at run start,
        so a manifest added mid-run moves nothing, as an edited profile does not.
        Filling it twice from two threads writes one answer twice."""
        for p in self._by_name.values():
            for ext in p.extensions:
                self._by_ext.setdefault(ext, []).append(p)

    @staticmethod
    def load(repo: Path | None = None, builtin_dir: Path | None = None) -> ProfileSet:
        names = [p.stem for p in builtin_paths(builtin_dir)]
        return ProfileSet(
            {n: load_profile(n, repo=repo, builtin_dir=builtin_dir) for n in names}, repo=repo
        )

    @staticmethod
    def of(profiles: Iterable[LanguageProfile], repo: Path | None = None) -> ProfileSet:
        """A set of already-loaded profiles, such as the ones an episode detected, so a holder
        of a few asks `language_of` as every other caller does. `repo` breaks a tie as the
        index's set does, so the two read one file alike."""
        return ProfileSet({p.name: p for p in profiles}, repo=repo)

    def language_of(self, path: Path | str) -> FileLanguage | None:
        """`path`'s language and the grammar that parses it, or None: no profile owns its
        extension, or two do and no manifest picks one. None is a gap, never
        a guess."""
        suffix = Path(path).suffix
        if suffix not in self._answered:
            self._answered[suffix] = resolve(path, self, self._repo).language
        return self._answered[suffix]

    def by_name(self, name: str) -> LanguageProfile | None:
        return self._by_name.get(name)

    def for_extension(self, ext: str) -> list[LanguageProfile]:
        return list(self._by_ext.get(ext, []))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def all(self) -> tuple[LanguageProfile, ...]:
        """Every installed profile, by name: what may judge a file the repository does not
        hold yet, where detection would find nothing."""
        return tuple(self._by_name[n] for n in self.names())


def load_runners(builtin_dir: Path | None = None) -> tuple[TestRunner, ...]:
    """Every shipped test runner, by file name; a bad one refuses at load, naming its file."""
    d = (builtin_dir or BUILTIN_DIR) / RUNNERS_SUBDIR
    out: list[TestRunner] = []
    for path in sorted(d.glob("*.toml")) if d.is_dir() else []:
        try:
            out.append(TestRunner.model_validate(_read_toml(path)))
        except (ValidationError, ProfileError) as exc:
            raise ProfileError(f"{path}: {exc}") from exc
    return tuple(out)


@functools.cache
def shipped_runners() -> tuple[TestRunner, ...]:
    """The harness's test runners, loaded once per process: what reads a test run."""
    return load_runners()


@functools.cache
def shipped() -> ProfileSet:
    """The harness's own profiles, without any repository's overrides, loaded once per
    process. For a reader that labels a file and is handed no run's set (`context/outline`):
    a label needs the languages the harness knows, not a repository's test runner."""
    return ProfileSet.load()


def resolve_document(
    repo_file: Path,
    named: Path | None,
    shipped: Path | None,
    *,
    missing: str | None = None,
) -> tuple[Path | None, str | None]:
    """A document a repository may override: its own file, else a named one, else the
    shipped one. Conventions and the hand-off templates (`handoffs.resolve`) share it.

    Returns (path, warning). The repository's file wins as soon as it exists, whatever it is:
    its reader decides what an unreadable one falls back to. A named file that does not exist
    falls through with a warning rather than failing: these documents shape prose, and
    losing one should not end a run that would otherwise succeed. `missing` is that warning's
    words, when the caller names the file in its own terms; a shipped path that does not exist
    is no document."""
    if repo_file.exists():
        return repo_file, None
    warning: str | None = None
    if named is not None:
        if named.exists():
            return named, None
        warning = missing or f"{named} does not exist; falling back to the shipped default"
    if shipped is not None and shipped.exists():
        return shipped, warning
    return None, warning


def resolve_conventions(
    repo: Path, profile: LanguageProfile | None, builtin_dir: Path | None = None
) -> tuple[Path | None, str | None]:
    """PR-8 order: `.chickadee/conventions.md` → profile's `conventions.file` → builtin, through
    `resolve_document`. The builtin default is no file: the renderer reads none as none.

    Returns (path, warning). A profile that names a missing file falls through with a warning
    rather than failing."""
    if profile is None or not profile.conventions.file:
        return resolve_document(repo / CONVENTIONS_OVERRIDE, None, None)
    d = builtin_dir or BUILTIN_DIR
    return resolve_document(
        repo / CONVENTIONS_OVERRIDE,
        d / profile.conventions.file,
        None,
        missing=(
            f"profile {profile.name!r} names conventions file {profile.conventions.file!r}, "
            "which does not exist; falling back to the builtin default"
        ),
    )
