"""PR-9 / D36 — profiles declare mechanics; authority provenance is run data.

A profile can say *how* to run a suite. It cannot say whether that suite proves anything,
because whether the checks predate the run is a fact about the run, not about the language.
"""

from __future__ import annotations

import pathlib

from chickadee.profiles import ProfileSet, load_profile


def test_profile_declares_a_runner_without_claiming_authority(
    builtin: pathlib.Path,
) -> None:
    p = load_profile("python", builtin_dir=builtin)
    assert p.tests.runner is not None
    fields = set(type(p.tests).model_fields)
    assert not fields & {"independent", "authoritative", "provenance"}, (
        "a profile that could declare authority would let a language config decide what "
        "counts as proof, which is D36's job and run data"
    )


def test_no_profile_field_asserts_verification_depth() -> None:
    """Invariant 7: verification depth is never overstated. Nothing here may encode it."""
    s = ProfileSet.load()
    for name in s.names():
        p = s.by_name(name)
        assert p is not None
        blob = p.model_dump_json()
        for forbidden in ("verify_tier", "authoritative", "behavior_verified"):
            assert forbidden not in blob, f"{name} profile encodes {forbidden}"


def test_examples_are_prose_not_commands(builtin: pathlib.Path) -> None:
    """Examples may specify intent; only independent checks prove behavior. The
    snippet is never split or executed here."""
    s = ProfileSet.load()
    py = s.by_name("python")
    assert py is not None
    assert isinstance(py.examples.snippet, str)
    assert not hasattr(py.examples.snippet, "argv")


def test_greenfield_absence_of_checks_is_valid_configuration(
    builtin: pathlib.Path,
) -> None:
    """ "no authoritative greenfield checks" is not an error. Static gates continue and
    the handoff says behavior unverified — a profile with no runner must load."""
    (builtin / "bare.toml").write_text(
        'name = "bare"\nextensions = [".bare"]\ntree_sitter_grammar = "g"\n'
        '[syntax]\nmode = "parse"\n'
    )
    p = load_profile("bare", builtin_dir=builtin)
    assert p.tests.runner is None
    assert p.tests.globs == ()


def test_conventions_resolution_order(builtin: pathlib.Path, repo: pathlib.Path) -> None:
    """Repo-authored first, then the profile's file, then the builtin default."""
    from chickadee.profiles import resolve_conventions

    p = load_profile("python", builtin_dir=builtin)

    found, warn = resolve_conventions(repo, p, builtin_dir=builtin)
    assert found is not None and found.name == "python.md"
    assert warn is None

    (repo / ".chickadee").mkdir(parents=True, exist_ok=True)
    (repo / ".chickadee" / "conventions.md").write_text("# repo conventions\n")
    found, warn = resolve_conventions(repo, p, builtin_dir=builtin)
    assert found is not None and found.parent.name == ".chickadee"


def test_missing_conventions_file_warns_rather_than_fails(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    """R-5 idiom: conventions shape prose, and losing them should not end a run that would
    otherwise succeed."""
    from chickadee.profiles import resolve_conventions

    (builtin / "conventions" / "python.md").unlink()
    p = load_profile("python", builtin_dir=builtin)
    found, warn = resolve_conventions(repo, p, builtin_dir=builtin)
    assert found is None
    assert warn is not None and "does not exist" in warn
