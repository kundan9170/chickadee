"""Git identity and language-profile evidence.

Git plumbing, never a literal `.git/` directory test: a linked worktree has a `.git`
*file* pointing at the common directory, and D39 makes worktrees the default for
autonomous runs — so the case this must handle is the case the harness creates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chickadee.process import git
from chickadee.profiles.loader import ProfileSet


class NotAGitRepository(Exception):
    """Carries the one-line fix, because "not a git repository" without it is a dead end."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} is not a git repository — run `git init` first")
        self.path = path


@dataclass(frozen=True)
class ProfileEvidence:
    """Why a profile is considered active. Evidence, not a guess."""

    name: str
    file_count: int
    manifests: tuple[str, ...]
    evidence: str


@dataclass(frozen=True)
class RepoIdentity:
    root: Path
    git_common_dir: Path
    repository_id: str
    """Stable across linked worktrees: the lease is repository-wide, so two
    worktrees of one repository must resolve to one id or they would not exclude each
    other."""
    is_linked_worktree: bool


def _git(root: Path, *args: str) -> str:
    # A failing `rev-parse` here means "not a repository", whatever git said — the
    # one-line fix in NotAGitRepository is more useful than git's stderr.
    return git(root, *args, on_error=lambda _detail: NotAGitRepository(root))


def resolve(root: Path) -> RepoIdentity:
    """Worktree root, common Git directory, and a repository identity that survives
    linked worktrees."""
    if not root.exists():
        raise NotAGitRepository(root)
    toplevel = Path(_git(root, "rev-parse", "--show-toplevel"))
    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    git_dir = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-dir"))
    # The common dir is shared by every linked worktree and the main one, so hashing its
    # resolved path gives every worktree of a repository the same identity — which is what
    # makes one lease exclude them all.
    resolved = common.resolve()
    from chickadee.schema import sha256_hex

    return RepoIdentity(
        root=toplevel,
        git_common_dir=resolved,
        repository_id=sha256_hex(str(resolved).encode())[:16],
        is_linked_worktree=git_dir.resolve() != resolved,
    )


def repository_key(root: Path) -> str:
    """`resolve(root).repository_id`, the same in every linked worktree of one repository; for
    a directory that is not one, a key of its own resolved path. What a cache kept per
    repository is named by."""
    try:
        return resolve(root).repository_id
    except NotAGitRepository:
        from chickadee.schema import sha256_hex

        return sha256_hex(str(root.resolve()).encode())[:16]


def exclude_path(root: Path) -> Path:
    """`git rev-parse --git-path info/exclude`.

    Never `.git/info/exclude` by construction: in a linked worktree that path does not
    exist, and writing it would create a file git ignores while the real exclude went
    unedited.
    """
    return Path(_git(root, "rev-parse", "--path-format=absolute", "--git-path", "info/exclude"))


SKIP_DIRS = {
    ".git",
    ".chickadee",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
}


def detect_profiles(
    root: Path, profiles: ProfileSet | None = None, builtin_dir: Path | None = None
) -> tuple[tuple[ProfileEvidence, ...], tuple[str, ...]]:
    """Every supported profile with evidence, plus the files no profile claims.

    Per file, and several profiles may activate. There is no repo-wide winner
    and no majority language: a Python/TypeScript monorepo is two profiles, and forcing one
    would send half the repository through gates that cannot read it.

    Unsupported files are returned as gaps rather than assigned to a default, because a
    file the harness cannot parse must be visible as a hole — a graph that answers "no
    callers" for an unparsed file answers the same as one with no callers at all.
    """
    ps = (
        profiles
        if profiles is not None
        else ProfileSet.load(repo=root, builtin_dir=builtin_dir)
    )
    counts: dict[str, int] = {n: 0 for n in ps.names()}
    manifests: dict[str, set[str]] = {n: set() for n in ps.names()}
    gaps: list[str] = []

    # A3(b)'s manifest list, straight off the typed profile — a getattr chain here reads
    # as `Never` and silently disables the checks below.
    wanted: dict[str, set[str]] = {}
    for name in ps.names():
        profile = ps.by_name(name)
        wanted[name] = set(profile.deps.manifests) if profile is not None else set()
    all_manifest_names = {m for names in wanted.values() for m in names}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        # The profile that will read this file, the index's own answer: a file two profiles
        # claim with no manifest to pick one is a gap here as it is there.
        language = ps.language_of(rel)
        if language is not None:
            counts[language.name] += 1
        for name, names in wanted.items():
            if rel.name in names:
                manifests[name].add(rel.as_posix())
        if language is None and rel.name not in all_manifest_names:
            gaps.append(rel.as_posix())

    evidence = tuple(
        ProfileEvidence(
            name=name,
            file_count=counts[name],
            manifests=tuple(sorted(manifests[name])),
            evidence=_why(counts[name], sorted(manifests[name])),
        )
        # Deterministic order: a report whose rows move between runs cannot be diffed.
        for name in sorted(counts)
        if counts[name] or manifests[name]
    )
    return evidence, tuple(gaps)


def _why(file_count: int, manifests: list[str]) -> str:
    if manifests and file_count:
        return f"{file_count} source file(s) and manifest(s) {', '.join(manifests)}"
    if manifests:
        return f"manifest(s) {', '.join(manifests)} with no source files yet"
    return f"{file_count} source file(s)"
