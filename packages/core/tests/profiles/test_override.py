"""PR-2 — repo overrides deep-merge over builtins."""

from __future__ import annotations

import pathlib

from chickadee.profiles import deep_merge, load_profile


def _override(repo: pathlib.Path, body: str) -> None:
    d = repo / ".chickadee" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    (d / "python.toml").write_text(body)


def test_override_of_one_key_preserves_the_rest(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    """The point of deep-merge: change `tests.runner` without restating the whole file."""
    _override(repo, '[tests]\nrunner = "pytest -xvs {selector}"\n')
    p = load_profile("python", repo=repo, builtin_dir=builtin)
    assert p.tests.runner is not None
    assert p.tests.runner.argv[:2] == ("pytest", "-xvs")
    assert p.tests.timeout_s == 120, "sibling keys survive"
    assert p.tests.globs == ("tests/**", "test_*.py"), "sibling keys survive"
    assert p.extensions == (".py", ".pyi"), "top-level keys survive"


def test_an_override_naming_the_retired_gates_is_refused(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    """Moved: a repository's `[gates]` added a gate; the gates retired with the
    packet loop, so it is refused with the reason rather than loaded and ignored."""
    import pytest

    from chickadee.profiles import ProfileError

    _override(repo, '[gates]\na5_cmd = "mypy {file}"\n')
    with pytest.raises(ProfileError, match=r"\[gates\] is no longer read.*D196"):
        load_profile("python", repo=repo, builtin_dir=builtin)


def test_arrays_replace_rather_than_merge(builtin: pathlib.Path, repo: pathlib.Path) -> None:
    """Element-wise array merging would make "replace the test globs" inexpressible, and
    repos genuinely need to do that."""
    _override(repo, '[tests]\nglobs = ["spec/**"]\n')
    p = load_profile("python", repo=repo, builtin_dir=builtin)
    assert p.tests.globs == ("spec/**",)


def test_override_is_still_strictly_validated(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    import pytest

    from chickadee.profiles import ProfileError

    _override(repo, "[tests]\nunknown_key = 1\n")
    with pytest.raises(ProfileError, match="unknown_key"):
        load_profile("python", repo=repo, builtin_dir=builtin)


def test_no_override_uses_the_builtin(builtin: pathlib.Path, repo: pathlib.Path) -> None:
    p = load_profile("python", repo=repo, builtin_dir=builtin)
    assert p.tests.runner is not None
    assert p.tests.runner.argv[0] == "pytest"


def test_deep_merge_semantics() -> None:
    base = {"a": 1, "t": {"x": 1, "y": 2}, "arr": [1, 2]}
    over = {"t": {"y": 9, "z": 3}, "arr": [7]}
    assert deep_merge(base, over) == {"a": 1, "t": {"x": 1, "y": 9, "z": 3}, "arr": [7]}
