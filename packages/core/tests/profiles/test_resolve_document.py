"""`resolve_conventions` through the shared lookup: its answers pinned before
the lookup moved out to `resolve_document`, and the lookup on its own.

The pins are every answer `resolve_conventions` gave: the repository's file whenever it exists
(a directory included: the renderer, not the lookup, falls back from an unreadable one), else
the profile's named file, else a warning naming the missing one, else nothing."""

from __future__ import annotations

import pathlib

from chickadee.profiles import load_profile, resolve_conventions, resolve_document
from chickadee.profiles.loader import CONVENTIONS_OVERRIDE

MISSING = (
    "profile 'python' names conventions file 'conventions/python.md', which does not exist; "
    "falling back to the builtin default"
)


def _override(repo: pathlib.Path, *, directory: bool = False) -> pathlib.Path:
    path = repo / CONVENTIONS_OVERRIDE
    path.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        path.mkdir()
    else:
        path.write_text("# repo conventions\n")
    return path


def test_the_profiles_named_file_when_the_repository_has_none(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    p = load_profile("python", builtin_dir=builtin)
    assert resolve_conventions(repo, p, builtin_dir=builtin) == (
        builtin / "conventions" / "python.md",
        None,
    )


def test_the_repositorys_file_first(builtin: pathlib.Path, repo: pathlib.Path) -> None:
    p = load_profile("python", builtin_dir=builtin)
    path = _override(repo)
    assert resolve_conventions(repo, p, builtin_dir=builtin) == (path, None)
    assert resolve_conventions(repo, None, builtin_dir=builtin) == (path, None)


def test_the_repositorys_path_wins_even_as_a_directory(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    p = load_profile("python", builtin_dir=builtin)
    path = _override(repo, directory=True)
    assert resolve_conventions(repo, p, builtin_dir=builtin) == (path, None)


def test_a_missing_named_file_warns_and_resolves_to_nothing(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    (builtin / "conventions" / "python.md").unlink()
    p = load_profile("python", builtin_dir=builtin)
    assert resolve_conventions(repo, p, builtin_dir=builtin) == (None, MISSING)


def test_no_profile_or_no_named_file_is_nothing(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    assert resolve_conventions(repo, None, builtin_dir=builtin) == (None, None)
    (builtin / "bare.toml").write_text(
        'name = "bare"\nextensions = [".bare"]\ntree_sitter_grammar = "g"\n'
        '[syntax]\nmode = "parse"\n'
    )
    bare = load_profile("bare", builtin_dir=builtin)
    assert resolve_conventions(repo, bare, builtin_dir=builtin) == (None, None)


# -- the lookup on its own ------------------------------------------------------------------


def _files(tmp: pathlib.Path, *names: str) -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    for name in ("repo.md", "named.md", "shipped.md"):
        out[name] = tmp / name
        if name in names:
            out[name].write_text(name)
    return out


def test_the_lookup_takes_the_first_that_exists(tmp_path: pathlib.Path) -> None:
    f = _files(tmp_path, "repo.md", "named.md", "shipped.md")
    assert resolve_document(f["repo.md"], f["named.md"], f["shipped.md"]) == (
        f["repo.md"],
        None,
    )
    f["repo.md"].unlink()
    assert resolve_document(f["repo.md"], f["named.md"], f["shipped.md"]) == (
        f["named.md"],
        None,
    )
    assert resolve_document(f["repo.md"], None, f["shipped.md"]) == (f["shipped.md"], None)


def test_a_missing_named_file_falls_through_to_the_shipped_with_a_warning(
    tmp_path: pathlib.Path,
) -> None:
    f = _files(tmp_path, "shipped.md")
    path, warning = resolve_document(f["repo.md"], f["named.md"], f["shipped.md"])
    assert path == f["shipped.md"]
    assert warning == f"{f['named.md']} does not exist; falling back to the shipped default"
    assert resolve_document(f["repo.md"], f["named.md"], None, missing="gone") == (None, "gone")


def test_a_shipped_file_that_does_not_exist_is_no_document(tmp_path: pathlib.Path) -> None:
    f = _files(tmp_path)
    assert resolve_document(f["repo.md"], None, f["shipped.md"]) == (None, None)
