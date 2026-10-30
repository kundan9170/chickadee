"""An unusable work folder is offered a folder for this repository.

`doc` refused, with an exit only the person could take by editing their engine config, when git
ignores the work folder or a file stands at its path. Now the harness offers the first free
fallback folder, asking the person once; a yes writes the design there and keeps the choice for
this clone (`<repo-local>/work.toml`), a no or an unattended run keeps the refusal.

The real box and workspace, the model scripted and no sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo
from repos import git
from zeus_turn_helpers import WAIT_S, answer, call, calls, result, state

from chickadee.engine import PacketEngine, ScriptedInteraction
from chickadee.schema.interaction import QuestionKind
from chickadee.state.layers import load_layers
from chickadee.state.local import LOCAL_FILE
from chickadee.state.roots import roots
from chickadee.work import MARKER, works

TITLE = "Add a retry"
DESIGN = "# Add a retry\n\n## Goal\nRetry the call.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def _ignoring(repo: Path, *patterns: str) -> None:
    (repo / ".gitignore").write_text("".join(f"{p}\n" for p in patterns), encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")


def _doc_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    person: Any,
) -> PacketEngine:
    raw["sandbox"] = {"require_sandbox": False}
    write = call("doc", "d1", op="write", file="README.md", content=DESIGN, title=TITLE)
    engine, _ = build(monkeypatch, repo, raw, [calls(write), answer("ok")], interaction=person)
    engines.append(engine)
    engine.turn(SID, "design a retry").wait(timeout=WAIT_S)
    return engine


def _recorded(repo: Path) -> Path:
    return roots(repo).repo_local / LOCAL_FILE


@pytest.mark.writes_worktree(".gitignore")
def test_an_ignored_work_folder_is_offered_a_free_one_and_a_yes_writes_there(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    _ignoring(repo, "docs/")
    person = ScriptedInteraction(answers=["yes"])

    engine = _doc_turn(monkeypatch, repo, raw, engines, person)

    (asked,) = person.asked
    assert asked.kind is QuestionKind.WORK_FOLDER and asked.default == "no"
    assert "docs/chickadee" in asked.text and "notes/chickadee" in asked.text
    assert not result(engine, "d1", SID).startswith("DENIED")
    (work,) = works(state(engine, SID))
    assert work.dir == "notes/chickadee"
    readme = repo / "notes/chickadee" / work.name / "design" / "README.md"
    assert readme.read_text().startswith(MARKER) or DESIGN in readme.read_text()
    assert _recorded(repo).read_text() == 'dir = "notes/chickadee"\n'
    assert engine.config.work.dir == "notes/chickadee", "the rest of the session uses it"
    (local,) = [
        layer for layer in load_layers(roots(repo), environ={}) if layer.name == "local"
    ]
    assert local.body == {"work": {"dir": "notes/chickadee"}}, "and so does the next load"


@pytest.mark.writes_worktree("docs", "docs/chickadee")
def test_a_file_in_the_way_is_offered_a_free_folder_too(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    (repo / "docs").mkdir()
    (repo / "docs/chickadee").write_text("not a folder\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a file where the folder would be")
    person = ScriptedInteraction(answers=["yes"])

    engine = _doc_turn(monkeypatch, repo, raw, engines, person)

    assert len(person.asked) == 1
    assert not result(engine, "d1", SID).startswith("DENIED")
    assert works(state(engine, SID))[0].dir == "notes/chickadee"


@pytest.mark.parametrize("person", ["no", "unattended"])
@pytest.mark.writes_worktree(".gitignore")
def test_a_no_or_an_unattended_run_keeps_the_refusal_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    person: str,
) -> None:
    _ignoring(repo, "docs/")
    port = ScriptedInteraction(answers=["no"] if person == "no" else [])

    engine = _doc_turn(monkeypatch, repo, raw, engines, port)

    assert result(engine, "d1", SID).startswith("DENIED")
    assert "ignored by git" in result(engine, "d1", SID)
    assert not _recorded(repo).exists()
    assert engine.config.work.dir == "docs/chickadee"
    assert works(state(engine, SID)) == ()


@pytest.mark.writes_worktree(".gitignore")
def test_with_no_free_folder_nothing_is_asked_and_the_refusal_stands(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    _ignoring(repo, "docs/", "notes/", "design/", "chickadee-work")
    person = ScriptedInteraction(answers=["yes"])

    engine = _doc_turn(monkeypatch, repo, raw, engines, person)

    assert person.asked == []
    assert result(engine, "d1", SID).startswith("DENIED")
    assert not _recorded(repo).exists()
