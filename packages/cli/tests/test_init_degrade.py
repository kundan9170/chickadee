"""Absent tools degrade; an absent grammar is fatal.

The asymmetry is the point. A missing linter means gate A4 records an honest skip at run
time — weaker verification, still a working harness. A missing grammar means the
file cannot be parsed at all, so it is not a weaker gate but no gate, and the index would
omit the code silently.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from chickadee.init import InitError, init, toolchain
from chickadee.profiles.loader import ProfileSet


def test_absent_tools_are_recorded_not_fatal(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Telling the user at init that a check will always skip is strictly better than letting
    them discover it in a handoff, after paying for the run. Moved: the gates A4
    and A5 retired, so the probe names the checks the task loop runs."""
    monkeypatch.setattr(shutil, "which", lambda _b, **_k: None)
    report = init(repo)
    assert report.degraded
    # A check skips; a language server's verdicts are unavailable.
    assert all("will skip" in d or "will be unavailable" in d for d in report.degraded)
    assert any("basedpyright-langserver not found" in d for d in report.degraded)
    assert any("the test runner" in d for d in report.degraded)


def test_a_degraded_init_still_publishes(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee.init import publish

    monkeypatch.setattr(shutil, "which", lambda _b, **_k: None)
    init(repo)
    assert publish.is_initialised(repo)


def test_present_tools_record_a_version(repo: pathlib.Path) -> None:
    report = init(repo)
    versions = [v for tools in report.toolchains.values() for v in tools.values()]
    assert any(v is not None for v in versions), "no toolchain was detected at all"


def test_a_missing_grammar_is_fatal(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(toolchain, "missing_grammars", lambda _n, _p: ("python",))
    with pytest.raises(InitError, match="tree-sitter grammar"):
        init(repo)


def test_a_missing_grammar_publishes_nothing(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee.init import publish

    monkeypatch.setattr(toolchain, "missing_grammars", lambda _n, _p: ("python",))
    with pytest.raises(InitError):
        init(repo)
    assert publish.read_state(repo) is None


def test_every_shipped_profile_declares_a_grammar() -> None:
    ps = ProfileSet.load()
    assert toolchain.missing_grammars(ps.names(), ps) == ()


def test_an_unparseable_file_is_a_gap_and_init_succeeds(repo: pathlib.Path) -> None:
    """Init records the hole rather than refusing: one bad file must not make a repository
    uninitialisable."""
    (repo / "broken.py").write_text("def (((\n")
    report = init(repo)
    assert report.root
    assert any(e.name == "python" for e in report.profiles)


def test_version_of_an_absent_binary_is_none() -> None:
    assert toolchain.version_of("definitely-not-a-real-binary-xyz") is None
