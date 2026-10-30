"""Extension → profile and grammar, with the manifest tie-break and unprofiled-file handling.


The one home for a file's language: the graph, the init scan and the reader ask it, through
`ProfileSet.language_of`, and keep no extension table of their own (a guard fails one,
`scripts/guards/languages.py`).

An unprofiled file is a **gap, not an error**. The repository is allowed to contain
files we cannot parse; refusing to index them would make the graph incomplete in a way that
nothing records, which is worse than recording that we could not read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chickadee.errors import HarnessBug
from chickadee.profiles.model import LanguageProfile

if TYPE_CHECKING:
    # The loader imports this module for `ProfileSet.language_of`; the set is only a type here.
    from chickadee.profiles.loader import ProfileSet


@dataclass(frozen=True)
class FileLanguage:
    """A file's language: the profile that judges it and the grammar that parses it."""

    profile: LanguageProfile
    grammar: str

    @property
    def name(self) -> str:
        """The profile's name: what the graph records the file under, so a reference resolves
        only among files of its own language."""
        return self.profile.name


@dataclass(frozen=True)
class Resolution:
    """What the resolver concluded, including why it could not conclude more."""

    language: FileLanguage | None
    reason: str
    """`matched`, `tie_broken_by_manifest`, `ambiguous`, or `no_profile`. Carried so the
    index report can count gaps by cause rather than lumping them together."""

    @property
    def profile(self) -> LanguageProfile | None:
        return self.language.profile if self.language is not None else None

    @property
    def is_gap(self) -> bool:
        return self.language is None


def _language(path: Path | str, profile: LanguageProfile) -> FileLanguage:
    grammar = profile.grammar_for(str(path))
    if grammar is None:  # pragma: no cover - the model names a grammar for each extension
        raise HarnessBug(f"profile {profile.name!r} owns {path} and names no grammar for it")
    return FileLanguage(profile, grammar)


def resolve(path: Path | str, profiles: ProfileSet, repo: Path | None = None) -> Resolution:
    candidates = profiles.for_extension(Path(path).suffix)

    if not candidates:
        return Resolution(None, "no_profile")
    if len(candidates) == 1:
        return Resolution(_language(path, candidates[0]), "matched")

    # Tie-break by manifest presence in the repo root: a `.h` in a repo with a
    # `CMakeLists.txt` is C++, and the manifest is the evidence rather than a guess.
    if repo is not None:
        present = [c for c in candidates if any((repo / m).exists() for m in c.deps.manifests)]
        if len(present) == 1:
            return Resolution(_language(path, present[0]), "tie_broken_by_manifest")

    # Still ambiguous. Reporting the ambiguity beats picking one and being silently wrong
    # about every symbol in the file.
    return Resolution(None, "ambiguous")
