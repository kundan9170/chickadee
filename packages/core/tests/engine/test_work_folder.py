"""Zeus's `doc` and the work folder (Amendment 2 B6).

B6 first: the WRITE lease is taken and given back around Zeus's first `doc` call as around a
`coder` call, the worktree that call opens is the one the next `coder` call reattaches, and
Zeus's shell writes the worktree from then on. Then the rest: no work open needs a `title`, a
path that leaves `design/` is refused, each call is one recorded write `/undo` takes back, a
dirty checkout is refused as `coder` refuses it, a folder of the person's is never written
over, and a name is never reused.

Engine-level, through the real view, workspace and session log, the model scripted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo, zeus_delegates
from repos import git
from test_session_worktree import lease, owned, worktrees
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

from chickadee.engine import PacketEngine
from chickadee.schema import sha256_hex
from chickadee.work import MARKER, works

TITLE = "Add a retry"
DESIGN = "# Add a retry\n\n## Goal\nRetry the call.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
) -> PacketEngine:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "go").wait(timeout=WAIT_S)
    return engine


def text(engine: PacketEngine, cid: str) -> str:
    line = state(engine, SID).result(cid)
    assert line is not None, f"call {cid} has no result"
    return line.text


def doc(cid: str, file: str = "README.md", content: str = DESIGN, **extra: object) -> Any:
    return call("doc", cid, op="write", file=file, content=content, **extra)


def tree(engine: PacketEngine) -> Path:
    (ref, *_) = owned(engine)
    assert ref.worktree_path is not None
    return Path(ref.worktree_path)


def folder(engine: PacketEngine) -> str:
    found = works(state(engine, SID))
    assert found, "no work was opened"
    return f"{found[-1].dir}/{found[-1].name}"


# -- B6: the worktree's timing -------------------------------------------------------------


@pytest.fixture
def raw(raw: dict[str, Any]) -> dict[str, Any]:
    """The worktree session D204 decision 4b describes. Since D230 decision 3 the default is
    in place, so these tests ask for the worktree they test; an in-place test sets its own."""
    raw["workspace"] = {"mode": "worktree"}
    return raw


def test_the_first_doc_opens_the_worktree_and_gives_the_lease_back(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [calls(doc("d1", title=TITLE)), answer("written")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert lease(repo) == "FREE", "the WRITE lease is per call, as a coder call's (B6)"
    assert worktrees(repo) == [SID]
    work = folder(engine)
    assert (tree(engine) / work / "design" / "README.md").read_text() == DESIGN
    assert not (repo / work).exists(), "written where the work runs, not the checkout"
    readme = (tree(engine) / work / "README.md").read_text()
    assert readme.startswith(MARKER) and TITLE in readme
    assert "view moved to the session's worktree" in text(engine, "d1")


def test_the_next_coder_call_reattaches_the_worktree_the_doc_opened(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [calls(doc("d1", title=TITLE)), *zeus_delegates(), answer("done")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert worktrees(repo) == [SID]
    assert len(owned(engine)) == 1, "reattached, nothing new recorded"
    assert "return 2" in (tree(engine) / "app.py").read_text()
    assert (tree(engine) / folder(engine) / "design" / "README.md").is_file()
    assert lease(repo) == "FREE"


def test_the_shell_runs_in_the_worktree_after_the_doc(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("shell", "s1", command="pwd")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert str(tree(engine).resolve()) in text(engine, "s1")


# -- the open work -------------------------------------------------------------------------


def test_a_first_doc_without_a_title_is_refused_and_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [calls(doc("d1")), answer("no")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert "DENIED" in text(engine, "d1") and "send the call again with title" in text(
        engine, "d1"
    )
    assert worktrees(repo) == [] and works(state(engine, SID)) == ()


def test_later_calls_write_the_same_work_and_edit_and_remove(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(doc("d1", title=TITLE)),
        calls(doc("d2", file="api.md", content="# API\n")),
        calls(
            call("doc", "d3", op="edit", file="README.md", old="Retry the call.", new="Retry.")
        ),
        calls(call("doc", "d4", op="remove", file="api.md")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    design = tree(engine) / folder(engine) / "design"
    assert "Retry.\n" in (design / "README.md").read_text()
    assert not (design / "api.md").exists()
    assert len(works(state(engine, SID))) == 1


def test_a_file_named_from_the_work_folder_and_an_edit_that_changes_nothing_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`design/api.md` from Zeus would land in `design/design/`, where nothing reads it; an
    edit whose old and new are one text changes nothing. Both are refused, saying what to
    send."""
    script = [
        calls(doc("d1", title=TITLE)),
        calls(doc("d2", file="design/api.md", content="# API\n")),
        calls(call("doc", "d3", op="edit", file="README.md", old="Retry", new="Retry")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    nested = text(engine, "d2")
    assert nested.startswith("DENIED") and "send file 'api.md'" in nested
    design = tree(engine) / folder(engine) / "design"
    assert not (design / "design").exists() and not (design / "api.md").exists()
    same = text(engine, "d3")
    assert same.startswith("BAD_FORMAT") and "changes nothing" in same


def test_a_design_file_that_is_not_markdown_is_refused_and_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Zeus's `doc` writes Markdown only, and a refused first call opens no
    work; `design/pose.md` is written where `design/pose.py` was refused."""
    script = [
        calls(doc("d1", file="pose.py", content="x = 1\n", title=TITLE)),
        calls(doc("d2", title=TITLE)),
        calls(doc("d3", file="pose.md", content="# Pose\n")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert text(engine, "d1").startswith("DENIED")
    assert "design files are Markdown" in text(engine, "d1")
    assert len(works(state(engine, SID))) == 1
    design = tree(engine) / folder(engine) / "design"
    assert (design / "pose.md").read_text() == "# Pose\n"
    assert not (design / "pose.py").exists()


@pytest.mark.parametrize("file", ["../../../app.py", "/etc/passwd", "../plan/README.md"])
def test_a_file_that_leaves_design_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    file: str,
) -> None:
    script = [calls(doc("d1", title=TITLE)), calls(doc("d2", file=file)), answer("no")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert text(engine, "d2").startswith("DENIED")
    assert "return 1" in (tree(engine) / "app.py").read_text()


def test_a_link_out_of_design_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(doc("d1", title=TITLE)),
        answer("opened"),
        calls(doc("d2", file="out/app.md", content="# x\n")),
        answer("no"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "open it").wait(timeout=WAIT_S)
    # Made after the folder opened, as a person or a command could have made it.
    (tree(engine) / folder(engine) / "design" / "out").symlink_to(tree(engine))
    engine.turn(SID, "write through it").wait(timeout=WAIT_S)

    assert "leads out of your folder" in text(engine, "d2")
    assert "return 1" in (tree(engine) / "app.py").read_text()


def test_each_doc_call_is_one_write_undo_takes_back(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [calls(doc("d1", title=TITLE)), calls(doc("d2", file="api.md")), answer("done")]
    engine = turn(monkeypatch, repo, raw, engines, script)
    design = tree(engine) / folder(engine) / "design"
    assert git(tree(engine), "status", "--porcelain") == "", "each committed on its own"
    assert git(tree(engine), "log", "--format=%s", "-1") == f"chickadee: doc {SID}:d2"

    done = engine.undo_operation(SID, f"{SID}:d2")

    assert done.restored == (f"{folder(engine)}/design/api.md",)
    assert not (design / "api.md").exists()
    assert (design / "README.md").is_file()


@pytest.mark.writes_worktree("app.py")
def test_a_dirty_checkout_seeds_the_worktree_as_coder_seeds_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Was: refused, as a `coder` call was. `T-3.212` moves
    both: the worktree starts with the checkout's changes, and the design is written there."""
    (repo / "app.py").write_text("def go() -> int:\n    return 9\n")
    script = [calls(doc("d1", title=TITLE)), answer("written")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert not text(engine, "d1").startswith("DENIED")
    (ref,) = owned(engine)
    assert "return 9" in (Path(str(ref.worktree_path)) / "app.py").read_text()
    assert worktrees(repo) == [SID] and works(state(engine, SID)) != ()


def test_in_place_the_folder_is_the_checkouts(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    raw["workspace"] = {"mode": "in_place"}
    script = [calls(doc("d1", title=TITLE)), answer("written")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert worktrees(repo) == []
    assert (repo / folder(engine) / "design" / "README.md").read_text() == DESIGN


@pytest.mark.writes_worktree("docs", "docs/chickadee", "docs/chickadee/notes.md")
def test_the_persons_own_file_in_the_work_dir_is_left_alone_and_blocks_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D230 Amendment 1. Was (D204 4c): refused, naming the file, the person to edit
    `[work] dir`. The design opens in a folder of its own beside the person's file."""
    (repo / "docs" / "chickadee").mkdir(parents=True)
    (repo / "docs" / "chickadee" / "notes.md").write_text("mine\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the person's notes")
    script = [calls(doc("d1", title=TITLE)), answer("written")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert not text(engine, "d1").startswith("DENIED")
    assert len(works(state(engine, SID))) == 1
    assert (tree(engine) / "docs" / "chickadee" / "notes.md").read_text() == "mine\n"
    assert (tree(engine) / folder(engine) / "design" / "README.md").read_text() == DESIGN


def test_a_second_session_never_reuses_a_work_name(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    first = turn(monkeypatch, repo, raw, engines, [calls(doc("d1", title=TITLE)), answer("x")])
    name = works(state(first, SID))[-1].name
    second, _ = build(monkeypatch, repo, raw, [calls(doc("e1", title=TITLE)), answer("y")])
    engines.append(second)
    second.create_session("S-other")
    second.turn("S-other", "go").wait(timeout=WAIT_S)
    other = works(state(second, "S-other"))[-1].name
    assert other == f"{name}-2"


def test_an_opening_call_the_draft_refuses_opens_no_work(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The call is all or none (decision 9): an `edit` of a file the new folder lacks writes
    nothing, the README included, so no work is open and the next call opens it."""
    script = [
        calls(call("doc", "d1", op="edit", file="README.md", old="a", new="b", title=TITLE)),
        calls(doc("d2", title=TITLE)),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert text(engine, "d1").startswith(("FILE_ABSENT", "DENIED"))
    (work,) = works(state(engine, SID))
    assert (tree(engine) / folder(engine) / "README.md").is_file()
    assert work.name.endswith("add-a-retry")


# -- calls by name, versions pinned (part 3) ---------------------------------------------------


def _design_run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> tuple[PacketEngine, Any]:
    from coder_helpers import athena_accepts, athena_plans_folder, worker_edit, zeus_designs

    script = [
        *zeus_designs(),
        *athena_plans_folder(),
        *worker_edit("return 1", "return 2"),
        *athena_accepts(),
        answer("done"),
    ]
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    engine, model = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
    return engine, model


def test_a_design_by_work_runs_from_its_pinned_folder(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = _design_run(monkeypatch, repo, raw, engines)
    work = tree(engine) / folder(engine)
    readme = (work / "design" / "README.md").read_text()
    assert "## Request\nmake go return 2" in readme, "the harness's sections, in the file"
    (episode,) = state(engine, SID).episodes()
    assert episode.design_sha == sha256_hex(readme), "the stop approved the README as pinned"
    assert "COMPLETED" in text(engine, "c1")
    assert "return 2" in (tree(engine) / "app.py").read_text()
    # Decision 6: the approved design and plan are commits of their own, before the task, and
    # the work's records (decision 5) the last.
    subjects = git(tree(engine), "log", "--format=%s", "-4").splitlines()
    assert subjects == [
        f"chickadee: report {folder(engine).rsplit('/', 1)[-1]}",
        "chickadee: T1 Return 2",
        "chickadee: plan approved",
        "chickadee: design approved",
    ]
    plan_files = git(tree(engine), "show", "--name-only", "--format=", "HEAD~2").split()
    assert f"{folder(engine)}/plan/README.md" in plan_files
    assert git(tree(engine), "log", "-1", "--format=%b", "HEAD~3").startswith("Tree: ")
    records = git(tree(engine), "show", "--name-only", "--format=", "HEAD").split()
    assert records == [f"{folder(engine)}/README.md", f"{folder(engine)}/report.md"]
    status = (work / "README.md").read_text()
    assert "· COMPLETED ·" in status and "- T1: Return 2 · committed · " in status
    assert (work / "report.md").read_text().startswith("<!-- written by chickadee")
    pins = git(tree(engine), "for-each-ref", "--format=%(refname)", "refs/chickadee/pins")
    assert f"refs/chickadee/pins/{SID}/design-" in pins and "/plan-v1" in pins
    # Athena works from the pinned README (15a), the worker from T1's pinned file (15, A7).
    athena = next(r for r in model.requests if "THE DESIGN" in _first_user(r))
    assert "## Request" in _first_user(athena)
    worker = next(r for r in model.requests if "## The work's documents" in _first_user(r))
    assert "# T1: Return 2" in _first_user(worker)
    assert f"`{folder(engine)}/plan/tasks/T1-task.md`" in _first_user(worker)


def _first_user(request: Any) -> str:
    users = [m for m in request.messages if m.role == "user"]
    return str(users[0].content) if users else ""


def test_coder_work_naming_another_folder_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("coder", "c1", work="2026-01-01-an-old-one")),
        answer("no"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    assert text(engine, "c1").startswith("INVALID_ARGUMENTS{field: work")
    assert "is not the open work" in text(engine, "c1")
    assert state(engine, SID).episodes() == ()


def test_a_design_out_of_shape_is_not_asked_and_names_what_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    from coder_helpers import work_name

    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("coder", "c1", work=work_name(TITLE))),
        answer("no"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    said = text(engine, "c1")
    assert "not asked, not run: the design is not in shape" in said
    assert "the What changes section is missing" in said


def test_the_design_is_frozen_while_its_episode_runs(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """B5: Zeus's `doc` on `design/` is refused while an episode builds it."""
    engine, _ = build(monkeypatch, repo, raw, [])
    engines.append(engine)
    conversation = engine._conversations.open(SID)  # pyright: ignore[reportPrivateUsage]
    del conversation
    from chickadee.zeus.docs import DesignDocs

    docs = DesignDocs(
        state_session(engine),
        lambda: repo,
        repo,
        work_dir="docs/chickadee",
        in_place=lambda: True,
        open_worktree=lambda _: None,
        edits=lambda _: None,  # type: ignore[arg-type, return-value]
        writes=lambda _: None,  # type: ignore[arg-type, return-value]
        running=lambda: "run R-1",
    )
    got = docs.resolve("README.md", TITLE)
    assert got.reason.startswith("design/ is frozen while run R-1 builds it")  # type: ignore[union-attr]


def state_session(engine: PacketEngine) -> Any:
    from test_session_worktree import session_of

    engine.open_session(SID)
    return session_of(engine)


def test_athenas_head_is_the_same_bytes_on_every_request(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D204 decision 15a: built from the pinned README when the episode opens, so the
    provider's cache holds it; the README edited after does not move it."""
    _, model = _design_run(monkeypatch, repo, raw, engines)
    heads = {
        (str(r.messages[0].content), _first_user(r))
        for r in model.requests
        if "THE DESIGN" in _first_user(r)
    }
    assert len(heads) == 1


def test_the_work_notice_is_said_once_per_repository(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """B7: the first design stop of a repository says what is committed, and only it."""
    from chickadee.state.roots import roots
    from chickadee.work import notice_once

    state = roots(repo).repo_local
    first = notice_once(state, "docs/chickadee", True)
    assert first is not None and "work.commit = false" in first
    assert notice_once(state, "docs/chickadee", True) is None
    assert notice_once(state, "docs/chickadee", False) is None


def _work_lines(engine: PacketEngine) -> list[str]:
    return [w.state for w in works(state(engine, SID))]


@pytest.mark.parametrize("choice", ["apply", "squash"])
def test_a_work_that_lands_gets_its_index_row_on_the_persons_branch(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    choice: str,
) -> None:
    """B1: after the landing, one harness commit appends the row; the session's branch never
    touched the index. The work is closed as landed (decision 4a)."""
    engine, _ = _design_run(monkeypatch, repo, raw, engines)
    name = folder(engine).rsplit("/", 1)[-1]
    assert not (tree(engine) / "docs/chickadee/README.md").exists(), "never on the branch"
    engine.finalize(SID, choice)
    assert git(repo, "log", "-1", "--format=%s") == f"chickadee: record {name}"
    landed = git(repo, "rev-parse", "--short=7", "HEAD~1")
    index = (repo / "docs/chickadee/README.md").read_text()
    assert index.startswith(MARKER)
    assert f"· [{name}]({name}/) · " in index and index.endswith(f"· landed {landed}\n")
    assert git(repo, "status", "--porcelain") == ""
    assert _work_lines(engine)[-1] == "landed"


def test_a_discarded_work_leaves_no_record_on_the_persons_branch(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _design_run(monkeypatch, repo, raw, engines)
    head = git(repo, "rev-parse", "HEAD")
    engine.finalize(SID, "discard")
    assert git(repo, "rev-parse", "HEAD") == head
    assert not (repo / "docs/chickadee").exists()
    assert _work_lines(engine)[-1] == "dropped"


def test_a_kept_work_merged_by_the_person_gets_no_commit_at_the_next_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """B1: `keep` appends nothing. Was B3: merged by the person, the next session's start
    committed its row on their branch. D230 decision 5: nothing commits as a session starts."""
    engine, _ = _design_run(monkeypatch, repo, raw, engines)
    branch = git(tree(engine), "rev-parse", "--abbrev-ref", "HEAD")
    engine.finalize(SID, "keep")
    assert not (repo / "docs/chickadee/README.md").exists()
    assert _work_lines(engine)[-1] == "kept"
    git(repo, "merge", "-q", "--ff-only", branch)
    head = git(repo, "rev-parse", "HEAD")
    engine.create_session("next")
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""


# -- D204 decision 18: after OUTGROWN, an unattended run carries on --------------------------


def test_an_unattended_run_carries_on_from_outgrown_into_a_design(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The one-task plan ends `OUTGROWN` with its accepted work committed; Zeus writes a design
    and calls `coder` again, and its stops are approved as every unattended stop is."""
    from coder_helpers import (
        athena_accepts,
        athena_plans_folder,
        box_call,
        worker_edit,
        zeus_designs,
    )

    one_task = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}
    script = [
        calls(call("coder", "c1", **one_task, workspace="worktree")),
        *worker_edit("return 1", "return 2"),
        calls(box_call("review", "a1", task="T1", answer="accept")),
        calls(box_call("outgrown", "o1")),
        answer("Missing: whether go() takes an argument. Done: T1. Proposed: a design."),
        *zeus_designs("Return 3", "`go()` returns 3", cid="c2"),
        *athena_plans_folder("Return 3", "`go()` returns 3", n=2),
        *worker_edit("return 2", "return 3", n=2),
        *athena_accepts(n=2),
        answer("done"),
    ]
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "make go return 3").wait(timeout=WAIT_S)

    first = text(engine, "c1")
    assert first.startswith("OUTGROWN") and "Do not call the work done" in first
    assert text(engine, "c2").startswith("COMPLETED")
    assert "return 3" in (tree(engine) / "app.py").read_text()
    subjects = git(tree(engine), "log", "--format=%s").splitlines()
    assert "chickadee: T1 Return 2" in subjects, "the outgrown plan's accepted work stays"


# -- the review of #607 ----------------------------------------------------------------------


def test_with_work_commit_off_a_doc_write_is_recorded_never_committed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """B7: `work.commit = false` keeps the folder out of the session's branch, Zeus's `doc`
    writes included; the write is still one recorded, undoable operation."""
    raw["work"] = {"commit": False}
    head = git(repo, "rev-parse", "HEAD")
    engine = turn(
        monkeypatch, repo, raw, engines, [calls(doc("d1", title=TITLE)), answer("ok")]
    )
    assert git(tree(engine), "rev-parse", "HEAD") == head, "nothing committed"
    assert (tree(engine) / folder(engine) / "design" / "README.md").read_text() == DESIGN
    assert "committed as" not in text(engine, "d1")


def test_a_stop_refused_before_it_is_shown_keeps_the_work_notice(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """B7: the notice is said once, at a stop the person sees; a design refused out of shape
    shows none, so the next stop still says it."""
    from coder_helpers import work_name

    from chickadee.state.roots import roots
    from chickadee.work import NOTICE_SHOWN

    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("coder", "c1", work=work_name(TITLE))),
        answer("no"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    assert "not asked, not run" in text(engine, "c1")
    assert not (roots(repo).repo_local / NOTICE_SHOWN).exists()


@pytest.mark.writes_worktree(".gitignore")
def test_a_work_dir_the_persons_ignore_rules_cover_is_refused_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """No snapshot holds a file git ignores: a work written there could be neither recorded
    nor undone, so the first `doc` call is refused and opens nothing."""
    (repo / ".gitignore").write_text("docs/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-qm", "ignore docs")
    engine = turn(
        monkeypatch, repo, raw, engines, [calls(doc("d1", title=TITLE)), answer("ok")]
    )
    said = text(engine, "d1")
    assert "DENIED" in said and "work.dir docs/chickadee is ignored by git" in said
    assert works(state(engine, SID)) == ()


def test_an_edit_after_the_approval_is_not_what_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Decision 12: the pin is the tree the person approved, never the disk re-read after."""
    from chickadee.approval.stops import ask_stop as asked
    from chickadee.workspace import pinned_text

    edited: list[Path] = []

    def ask_then_edit(*args: Any, **kwargs: Any) -> Any:
        said = asked(*args, **kwargs)
        readme = Path(args[0].folder) / "README.md"
        readme.write_text(readme.read_text() + "\nedited after the answer\n")
        edited.append(readme)
        return said

    monkeypatch.setattr("chickadee.delegation.ask_stop", ask_then_edit)
    engine, _ = _design_run(monkeypatch, repo, raw, engines)
    assert edited and "COMPLETED" in text(engine, "c1")
    (episode,) = state(engine, SID).episodes()
    ref = next(
        r
        for r in git(tree(engine), "for-each-ref", "--format=%(refname)").splitlines()
        if "/design-" in r
    )
    pinned = pinned_text(
        tree(engine), git(tree(engine), "rev-parse", f"{ref}^{{tree}}"), "README.md"
    )
    assert pinned is not None and "edited after the answer" not in pinned
    assert episode.design_sha == sha256_hex(pinned)


def test_a_failed_index_row_still_closes_the_landed_work(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The handoff is done when its row fails: the work is closed, the heal writes the row."""
    import chickadee.session_handoff as session_handoff
    from chickadee.errors import HandoffError

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise HandoffError("the index could not be committed")

    engine, _ = _design_run(monkeypatch, repo, raw, engines)
    monkeypatch.setattr(session_handoff, "record_works", refuse)
    engine.finalize(SID, "apply")
    assert _work_lines(engine)[-1] == "landed"
    assert not (repo / "docs/chickadee/README.md").exists()


@pytest.mark.writes_worktree("docs/chickadee/*/report.md")
def test_in_place_a_new_title_after_the_run_closes_the_work_and_opens_the_next(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """4a: in place there is no handoff; once the open work reported, a title opens the next.
    Before it reported, a title writes into the same work."""
    raw["workspace"] = {"mode": "in_place"}
    script = [
        calls(doc("d1", title=TITLE)),
        calls(doc("d2", title="Something else")),
        answer("written"),
        calls(doc("d3", title="Something else")),
        answer("written"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    first = folder(engine)
    assert [w.state for w in works(state(engine, SID))] == ["open"]
    (repo / first / "report.md").write_text("<!-- written by chickadee -->\n")
    engine.turn(SID, "again").wait(timeout=WAIT_S)
    found = works(state(engine, SID))
    assert [w.state for w in found][-2:] == ["landed", "open"]
    assert f"{found[-1].dir}/{found[-1].name}" != first
    assert (repo / folder(engine) / "design" / "README.md").read_text() == DESIGN


def test_a_one_task_plan_gives_athena_no_folder_of_the_open_work(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Her `plan/` is the approved design's work's; a one-task plan has none, though a design
    of Zeus's is open in the session."""
    from coder_helpers import athena_accepts, box_call, worker_edit

    one_task = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}
    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("coder", "c1", **one_task, workspace="worktree")),
        *worker_edit("return 1", "return 2"),
        calls(box_call("doc", "pd1", op="write", file="README.md", content="# Plan\n")),
        *athena_accepts(),
        answer("done"),
    ]
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    engine, model = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "go").wait(timeout=WAIT_S)
    said = [
        str(m.content)
        for r in model.requests
        for m in r.messages
        if m.role == "tool" and m.tool_call_id == "pd1"
    ]
    assert said and "no work folder is open" in said[-1]
    assert not (tree(engine) / folder(engine) / "plan").exists()


# -- the second review of #607 ---------------------------------------------------------------


def test_a_work_in_the_worktree_never_runs_in_place(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """4b: the run's root is where the folder is; the checkout has no `design/` to run."""
    from coder_helpers import work_name

    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("coder", "c1", work=work_name(TITLE), workspace="in_place")),
        answer("no"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    said = text(engine, "c1")
    assert "BLOCKED" in said and "is in the session's worktree" in said
    assert 'workspace="worktree"' in said


def test_a_held_repository_refuses_the_doc_call_never_raises(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    from chickadee.init.detect import resolve
    from chickadee.init.lease import take_write_lease

    raw["workspace"] = {"mode": "in_place"}
    lease = take_write_lease(resolve(repo))
    try:
        engine = turn(
            monkeypatch, repo, raw, engines, [calls(doc("d1", title=TITLE)), answer("ok")]
        )
    finally:
        lease.release()
    said = text(engine, "d1")
    assert "DENIED" in said and "call doc again once the run ends" in said
    assert works(state(engine, SID)) == ()


@pytest.mark.writes_worktree("docs/chickadee/*/report.md")
def test_in_place_a_refused_call_with_a_new_title_closes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    raw["workspace"] = {"mode": "in_place"}
    script = [
        calls(doc("d1", title=TITLE)),
        answer("written"),
        calls(doc("d2", file="../app.py", title="Something else")),
        calls(doc("d3", file="notes.md")),
        answer("written"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    first = folder(engine)
    (repo / first / "report.md").write_text("<!-- written by chickadee -->\n")
    engine.turn(SID, "again").wait(timeout=WAIT_S)
    assert "DENIED" in text(engine, "d2")
    assert [w.state for w in works(state(engine, SID))] == ["open"]
    assert (repo / first / "design" / "notes.md").read_text() == DESIGN, "the same work"


@pytest.mark.writes_worktree("docs", "docs/*", "big.bin")
def test_a_folders_tree_is_its_subtree_of_the_whole(repo: Path) -> None:
    """Hashed alone, the folder has the id it has in the repository's tree."""
    from chickadee.workspace import current_tree, folder_tree

    (repo / "docs" / "w" / "design").mkdir(parents=True)
    (repo / "docs" / "w" / "design" / "README.md").write_text("# Design\n")
    (repo / "big.bin").write_bytes(b"0" * 1000)
    whole, _ = current_tree(repo, untracked_max_bytes=None)
    assert folder_tree(repo, "docs/w") == git(repo, "rev-parse", f"{whole}:docs/w").strip()
    assert folder_tree(repo, "docs/absent") is None


# -- T-3.175: the configured work.dir in the prompts -----------------------------------------


def test_both_prompts_name_the_configured_work_dir(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Zeus and the worker are told the folder the harness fences, not the default's name."""
    raw["work"] = {"dir": "notes/designs"}
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    engine, model = build(monkeypatch, repo, raw, [*zeus_delegates(), answer("done")])
    engines.append(engine)
    engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
    assert text(engine, "c1").startswith("COMPLETED")
    systems = [str(r.messages[0].content) for r in model.requests]
    assert any("opens its folder under `notes/designs/`" in s for s in systems), "Zeus"
    assert any("The work folder (`notes/designs/`)" in s for s in systems), "the worker"
    assert not any("docs/chickadee" in s for s in systems)


def test_a_design_left_in_the_worktree_still_runs_there(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Final review of 2026-10-25: after `worktree leave` the folder's root was read from the
    view, the checkout, where the design is not, so the call answered that its README did not
    exist and Zeus would write it again there."""
    from coder_helpers import work_name

    script = [
        calls(doc("d1", title=TITLE)),
        calls(call("worktree", "w1", action="leave")),
        calls(call("coder", "c1", work=work_name(TITLE), workspace="in_place")),
        answer("no"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert text(engine, "w1").startswith("back in the user's checkout")
    said = text(engine, "c1")
    assert "is in the session's worktree" in said and "does not exist" not in said
