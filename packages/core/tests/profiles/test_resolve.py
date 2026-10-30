"""PR-4 — extension match, manifest tie-break, unprofiled files are gaps not errors."""

from __future__ import annotations

import pathlib

from chickadee.profiles import ProfileSet, resolve


def _two_way_tie(builtin: pathlib.Path) -> None:
    for name, manifest in (("cpp", "CMakeLists.txt"), ("c", "Makefile")):
        (builtin / f"{name}.toml").write_text(
            f'name = "{name}"\nextensions = [".h"]\ntree_sitter_grammar = "{name}"\n'
            '[syntax]\nmode = "parse"\n'
            f'[deps]\nmanifests = ["{manifest}"]\n'
        )


def test_single_match(builtin: pathlib.Path) -> None:
    s = ProfileSet.load(builtin_dir=builtin)
    r = resolve("src/a.py", s)
    assert r.profile is not None
    assert r.profile.name == "python"
    assert r.reason == "matched"
    assert r.is_gap is False


def test_unprofiled_extension_is_a_gap_not_a_crash(builtin: pathlib.Path) -> None:
    """GR-4: the repo is allowed to contain files we cannot parse. Refusing them would
    make the graph incomplete in a way nothing records."""
    s = ProfileSet.load(builtin_dir=builtin)
    r = resolve("README.md", s)
    assert r.is_gap
    assert r.reason == "no_profile"


def test_manifest_breaks_the_tie(builtin: pathlib.Path, repo: pathlib.Path) -> None:
    _two_way_tie(builtin)
    (repo / "CMakeLists.txt").write_text("")
    s = ProfileSet.load(builtin_dir=builtin)
    r = resolve("src/a.h", s, repo=repo)
    assert r.profile is not None
    assert r.profile.name == "cpp"
    assert r.reason == "tie_broken_by_manifest"


def test_unbroken_tie_reports_ambiguity(builtin: pathlib.Path, repo: pathlib.Path) -> None:
    """Reporting ambiguity beats picking one and being silently wrong about every symbol
    in the file."""
    _two_way_tie(builtin)
    s = ProfileSet.load(builtin_dir=builtin)
    r = resolve("src/a.h", s, repo=repo)
    assert r.is_gap
    assert r.reason == "ambiguous"


def test_both_manifests_present_stays_ambiguous(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    _two_way_tie(builtin)
    (repo / "CMakeLists.txt").write_text("")
    (repo / "Makefile").write_text("")
    s = ProfileSet.load(builtin_dir=builtin)
    assert resolve("src/a.h", s, repo=repo).reason == "ambiguous"


def test_no_repo_means_no_tie_break(builtin: pathlib.Path) -> None:
    _two_way_tie(builtin)
    s = ProfileSet.load(builtin_dir=builtin)
    assert resolve("src/a.h", s).reason == "ambiguous"


def test_shipped_profiles_resolve_real_extensions() -> None:
    s = ProfileSet.load()
    for path, expected in [
        ("a.py", "python"),
        ("a.pyi", "python"),
        ("a.js", "javascript"),
        ("a.mjs", "javascript"),
        ("a.jsx", "javascript"),
    ]:
        r = resolve(path, s)
        assert r.profile is not None and r.profile.name == expected, path
