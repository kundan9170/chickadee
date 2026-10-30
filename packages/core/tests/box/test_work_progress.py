"""Where a work stands: `WorkProgress`, read
from each run's request and journal and from git. What these pin: the design's and the plan's
approval against the folder now; each plan task's state across runs (open while its run may go
on, pending again once it ended, committed only while its commit is on the branch, never
checkable in place); the latest run's state; a run of another work is not counted; and the
status text Zeus reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from repos import git, init_repo

import chickadee.work_progress as work_progress
from chickadee.box.carry import Carried
from chickadee.box.request import ApprovedDesign, CoderRequest, Continuation, WorkPin
from chickadee.context.lines import TellLine
from chickadee.ids import task_commit_id
from chickadee.journal.effects import EffectKind, Intent, on_journal, perform
from chickadee.journal.projection import RunProjection
from chickadee.journal.writer import Journal
from chickadee.schema.ids import sha256_hex
from chickadee.schema.journal import (
    EventKind,
    PlanWrittenP,
    RunEndP,
    RunStartP,
    TaskDelegatedP,
    TaskFactsP,
    TaskReportedP,
    TaskReviewedP,
    TellDeliveredP,
    TellReplyP,
)
from chickadee.state.roots import roots as resolve_roots
from chickadee.work_progress import Progresses, WorkKey, complete, progress, status_text
from chickadee.workspace import folder_tree, pin_tree

DIR = "docs/chickadee"
NAME = "2026-10-27-add-login"
FOLDER = f"{DIR}/{NAME}"
PLAN_README = (
    "# Plan: login\n\n## Tasks\n"
    "1. [T1: Page](tasks/T1-page.md): the page. Depends on: none\n"
    "2. [T2: Check](tasks/T2-check.md): the check. Depends on: T1\n\n"
    "## Final check\nThe suite.\n"
)
KEY = WorkKey(dir=DIR, name=NAME)
R1 = "2026-10-27-0000000a"
R2 = "2026-10-27-0000000b"


@dataclass
class ScriptedRuns:
    """`RunReader` over journals and requests the test made."""

    requests: dict[str, CoderRequest] = field(default_factory=lambda: {})
    journals: dict[str, Journal] = field(default_factory=lambda: {})

    def request(self, run_id: str) -> CoderRequest | None:
        return self.requests.get(run_id)

    def projection(self, run_id: str) -> RunProjection | None:
        journal = self.journals.get(run_id)
        return journal.projection() if journal is not None else None

    def candidates(self) -> tuple[str, ...]:
        return tuple(self.journals)

    def stamp(self, run_id: str) -> object:
        journal = self.journals.get(run_id)
        return journal.projection().last_seq if journal is not None else None


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {
            "app.py": "x = 1\n",
            f"{FOLDER}/design/README.md": "# Add login\n",
            f"{FOLDER}/plan/README.md": PLAN_README,
            f"{FOLDER}/plan/tasks/T1-page.md": "# T1: Page\n",
            f"{FOLDER}/plan/tasks/T2-check.md": "# T2: Check\n",
        },
    )


def _request(
    root: Path, *, continues: Continuation | None = None, in_place: bool = False
) -> CoderRequest:
    tree = folder_tree(root, f"{FOLDER}/design")
    assert tree is not None
    text = "# Add login\n"
    return CoderRequest(
        design=ApprovedDesign(
            text=text,
            sha=sha256_hex(text),
            template_sha="t" * 64,
            work=WorkPin(name=NAME, dir=DIR, tree=tree),
        ),
        continues=continues,
        workspace="in_place" if in_place else "worktree",
    )


def _run(
    tmp: Path, reader: ScriptedRuns, root: Path, run_id: str, request: CoderRequest
) -> Journal:
    state = tmp / f"state-{run_id}"
    (state / ".git" / "info").mkdir(parents=True)
    journal = Journal.create(resolve_roots(state), run_id, {})
    journal.append(
        EventKind.RUN_START,
        RunStartP(
            goal="Add login",
            work_sha=request.work_sha,
            config_sha="c" * 64,
            repo_head="h" * 40,
            engine_version="0",
            schema_version=2,
            isolation=request.workspace,
            hard_cost_cap_usd=5.0,
        ),
    )
    reader.requests[run_id] = request
    reader.journals[run_id] = journal
    return journal


def _approve_plan(journal: Journal, root: Path, run_id: str) -> None:
    tree = folder_tree(root, f"{FOLDER}/plan")
    assert tree is not None
    pin_tree(root, tree, f"{FOLDER}/plan", run_id, "plan-v1")
    journal.append(
        EventKind.PLAN_WRITTEN,
        PlanWrittenP(call_id="a-plan", sha="p" * 64, by="athena", version=1, asked=True),
    )


def _delegate(journal: Journal, task: str) -> None:
    journal.append(
        EventKind.TASK_DELEGATED,
        TaskDelegatedP(
            task_id=task,
            call_id=f"d-{task}",
            title=f"title of {task}",
            brief_sha=journal.put_blob(b"brief"),
            by="athena",
            start_tree="a" * 40,
            plan_task=task.split("@")[0],
        ),
        task_id=task,
    )


def _commit(journal: Journal, task: str, sha: str) -> None:
    journal.append(
        EventKind.TASK_REVIEWED,
        TaskReviewedP(task_id=task, call_id=f"r-{task}", answer="accept"),
        task_id=task,
    )
    intent = Intent(task_commit_id(task), EffectKind.TASK_COMMIT, "s" * 64, (), False, ("a",))
    with perform(on_journal(journal, task_id=task), intent) as fx:
        fx.complete("succeeded", post_state=sha)


def _end(journal: Journal, outcome: str) -> None:
    journal.append(
        EventKind.RUN_END,
        RunEndP(
            outcome=outcome,  # type: ignore[arg-type]
            why=None if outcome == "COMPLETED" else "stopped",
            tasks=2,
            committed=1,
            tokens=0,
            cost_usd=0.0,
            wall_ms=0,
        ),
    )


def _head_commit(root: Path, path: str) -> str:
    (root / path).write_text(f"changed at {git(root, 'rev-parse', 'HEAD').strip()}\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", path)
    return git(root, "rev-parse", "HEAD").strip()


def _progress(root: Path, reader: ScriptedRuns, runs: list[str], **kwargs: object):  # type: ignore[no-untyped-def]
    live = kwargs.pop("live", lambda _r: False)
    tells = kwargs.pop("tells", ())
    return progress(KEY, runs, reader, root, live=live, tells=tells)  # type: ignore[arg-type]


def _tell(n: int, run_id: str) -> TellLine:
    return TellLine(id=f"tell-{n}", run_id=run_id, text_sha="t" * 64, sources=(3,), seq=n)


@pytest.mark.writes_worktree("app.py")
def test_a_suspended_run_with_a_task_committed_and_one_open(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(journal, root, R1)
    _delegate(journal, "T1")
    _commit(journal, "T1", _head_commit(root, "app.py"))
    _delegate(journal, "T2")
    p = _progress(root, reader, [R1])
    assert (p.design.version, p.design.changed) == (1, False)
    assert (p.plan.version, p.plan.changed) == (1, False)
    assert p.plan.bound_to == reader.requests[R1].design.work.tree  # type: ignore[union-attr]
    assert [(t.id, t.state) for t in p.tasks] == [("T1", "committed"), ("T2", "open")]
    assert p.run is not None and (p.run.state, p.run.at_task) == ("suspended", "T2")
    assert not complete(p)
    text = status_text(p)
    assert "design  v1 approved · unchanged" in text
    assert "T1 committed" in text and "T2 open" in text
    assert f"run     {R1} suspended at T2" in text
    assert 'coder(work="2026-10-27-add-login") continues from where it stopped' in text


def test_a_live_run_is_the_teams(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    _run(tmp_path, reader, root, R1, _request(root))
    p = _progress(root, reader, [R1], live=lambda r: r == R1)
    assert p.run is not None and p.run.state == "live"
    assert "the team is working on it" in status_text(p)


@pytest.mark.writes_worktree("app.py")
def test_an_ended_runs_open_task_is_pending_again(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(journal, root, R1)
    _delegate(journal, "T1")
    _commit(journal, "T1", _head_commit(root, "app.py"))
    _delegate(journal, "T2")
    _end(journal, "ABORTED")
    p = _progress(root, reader, [R1])
    assert [(t.id, t.state) for t in p.tasks] == [("T1", "committed"), ("T2", "pending")]
    assert p.run is not None and (p.run.state, p.run.outcome) == ("ended", "ABORTED")


@pytest.mark.writes_worktree("app.py", f"{FOLDER}/design/README.md")
def test_a_design_edited_since_its_approval_is_changed(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    _run(tmp_path, reader, root, R1, _request(root))
    (root / FOLDER / "design" / "README.md").write_text("# Add login\n\n## Amendment 1\n")
    p = _progress(root, reader, [R1])
    assert p.design.changed
    assert "design  v1 approved · changed on disk since" in status_text(p)
    assert "asks the person to approve the design's changes" in status_text(p)


@pytest.mark.writes_worktree("app.py")
def test_a_commit_that_left_the_branch_is_lost(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(journal, root, R1)
    _delegate(journal, "T1")
    sha = _head_commit(root, "app.py")
    _commit(journal, "T1", sha)
    git(root, "reset", "-q", "--hard", "HEAD~1")
    p = _progress(root, reader, [R1])
    assert p.tasks[0].state == "lost"
    assert "T1 lost (its commit left the branch)" in status_text(p)


@pytest.mark.writes_worktree("app.py")
def test_in_place_a_done_task_is_never_lost(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root, in_place=True))
    _approve_plan(journal, root, R1)
    _delegate(journal, "T1")
    _commit(journal, "T1", "f" * 40)
    p = _progress(root, reader, [R1])
    assert p.tasks[0].state == "unchecked"


@pytest.mark.writes_worktree("app.py")
def test_a_completed_run_with_every_task_done_is_complete(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(journal, root, R1)
    for task in ("T1", "T2"):
        _delegate(journal, task)
        _commit(journal, task, _head_commit(root, "app.py"))
    _end(journal, "COMPLETED")
    p = _progress(root, reader, [R1])
    assert complete(p)
    assert "complete: change the design to add to it" in status_text(p)


@pytest.mark.writes_worktree("app.py")
def test_tasks_and_the_inherited_plan_read_across_runs(tmp_path: Path, root: Path) -> None:
    """A run that continues the work inherits its plan, bound to the design tree it was
    approved under, and the first run's committed task stays committed."""
    reader = ScriptedRuns()
    first = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(first, root, R1)
    _delegate(first, "T1")
    _commit(first, "T1", _head_commit(root, "app.py"))
    _end(first, "ABORTED")
    bound = reader.requests[R1].design.work.tree  # type: ignore[union-attr]
    second = _run(
        tmp_path,
        reader,
        root,
        R2,
        _request(
            root,
            continues=Continuation(
                mode="CONTINUE",
                prior_runs=(R1,),
                plan_run=R1,
                plan_version=1,
                plan_sha="p" * 64,
                plan_bound_to=bound,
            ),
        ),
    )
    tree = folder_tree(root, f"{FOLDER}/plan")
    assert tree is not None
    pin_tree(root, tree, f"{FOLDER}/plan", R2, "plan-v1")
    second.append(
        EventKind.PLAN_WRITTEN,
        PlanWrittenP(sha="p" * 64, by="athena", version=1, asked=False, inherited_from=R1),
    )
    _delegate(second, "T2")
    p = _progress(root, reader, [R1, R2])
    assert p.runs == (R1, R2)
    assert (p.plan.version, p.plan.bound_to, p.plan_run) == (1, bound, R2)
    assert [(t.id, t.state, t.run) for t in p.tasks] == [
        ("T1", "committed", R1),
        ("T2", "open", R2),
    ]


@pytest.mark.writes_worktree("app.py")
def test_a_lost_task_run_again_is_where_its_next_attempt_stands(
    tmp_path: Path, root: Path
) -> None:
    """A task whose commit left the branch runs again in a later run: that attempt is what
    the work has of it, in the progress and in what the next run carries."""
    reader = ScriptedRuns()
    first = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(first, root, R1)
    _delegate(first, "T1")
    _commit(first, "T1", _head_commit(root, "app.py"))
    _end(first, "COMPLETED")
    git(root, "reset", "-q", "--hard", "HEAD~1")
    continues = Continuation(mode="CONTINUE", prior_runs=(R1,), plan_run=R1, plan_version=1)
    second = _run(tmp_path, reader, root, R2, _request(root, continues=continues))
    _delegate(second, "T1@2")
    p = _progress(root, reader, [R1, R2])
    assert (p.tasks[0].id, p.tasks[0].key, p.tasks[0].state, p.tasks[0].run) == (
        "T1",
        "T1@2",
        "open",
        R2,
    ), "not held as lost"

    _end(second, "PARTIAL")
    later = Continuation(mode="CONTINUE", prior_runs=(R1, R2), plan_run=R1, plan_version=1)
    carried = Carried.of(later, lambda run_id: reader.journals[run_id].projection(), root)
    prior = carried.task("T1")
    assert prior is not None
    assert (prior.run_id, prior.entry.id, prior.attempts, prior.unfinished) == (
        R2,
        "T1@2",
        2,
        True,
    )
    assert carried.committed() == (), "the lost commit is not counted done"


@pytest.mark.writes_worktree("app.py")
def test_a_lost_commit_is_lost_to_the_desk_and_to_the_status_alike(
    tmp_path: Path, root: Path
) -> None:
    """One fold, `work_tasks`: a task whose commit left the branch is to do again for the run
    that continues the work, as Zeus's status says, never counted done by one and lost by the
    other."""
    reader = ScriptedRuns()
    first = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(first, root, R1)
    _delegate(first, "T1")
    _commit(first, "T1", _head_commit(root, "app.py"))
    _end(first, "PARTIAL")
    git(root, "reset", "-q", "--hard", "HEAD~1")

    assert _progress(root, reader, [R1]).tasks[0].state == "lost"
    continues = Continuation(mode="CONTINUE", prior_runs=(R1,), plan_run=R1, plan_version=1)
    carried = Carried.of(continues, lambda run_id: reader.journals[run_id].projection(), root)
    prior = carried.task("T1")
    assert prior is not None and prior.lost and not prior.done and not prior.unfinished
    assert carried.committed() == ()


def test_a_run_of_another_work_is_not_counted(tmp_path: Path, root: Path) -> None:
    reader = ScriptedRuns()
    _run(tmp_path, reader, root, R1, _request(root))
    other = WorkKey(dir=DIR, name="2026-10-26-something-else")
    p = progress(other, [R1], reader, root, live=lambda _r: False)
    assert (p.runs, p.run, p.design.version) == ((), None, 0)


@pytest.mark.writes_worktree("app.py", f"{FOLDER}/design/README.md")
def test_progress_is_read_again_only_when_something_it_reads_moved(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame asks at every turn; the git reads are made only after a journal, a run's
    liveness, a part's folder or the branch moved."""
    reads: list[str] = []
    real = work_progress.progress

    def counted(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        reads.append("read")
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(work_progress, "progress", counted)
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(journal, root, R1)
    running = {R1: False}
    memo = Progresses(reader)

    def of():  # type: ignore[no-untyped-def]
        return memo.of(KEY, root, live=lambda r: running.get(r, False))

    first = of()
    assert of() is first and len(reads) == 1, "nothing moved: not read again"

    _delegate(journal, "T1")
    assert of().tasks[0].state == "open" and len(reads) == 2, "the journal moved"

    running[R1] = True
    assert of().run.state == "live" and len(reads) == 3, "the run started running"  # type: ignore[union-attr]

    (root / FOLDER / "design" / "README.md").write_text("# Add login, and logout\n")
    assert of().design.changed and len(reads) == 4, "the design was edited"

    _head_commit(root, "app.py")
    of()
    assert len(reads) == 5, "the branch moved"
    of()
    assert len(reads) == 5


def test_a_held_task_and_the_works_tells_are_in_its_status(tmp_path: Path, root: Path) -> None:
    """D240 §2.1, §2.2: a task held for a message, and each tell to the work's runs, sent,
    delivered, replied, or unanswered once the run it was delivered in ended."""
    reader = ScriptedRuns()
    journal = _run(tmp_path, reader, root, R1, _request(root))
    _approve_plan(journal, root, R1)
    _delegate(journal, "T1")
    journal.append(
        EventKind.TASK_REPORTED,
        TaskReportedP(
            task_id="T1",
            round=1,
            stopped="held",
            facts=TaskFactsP(start_tree="a" * 40, end_tree="a" * 40),
        ),
        task_id="T1",
        round=1,
    )
    for tell in ("tell-1", "tell-2"):
        journal.append(
            EventKind.TELL_DELIVERED, TellDeliveredP(tell=tell, task_id="T1", round=1)
        )
    journal.append(
        EventKind.TELL_REPLY,
        TellReplyP(tell="tell-1", outcome="applied", text_sha="r" * 64, call_id="a-y1"),
    )
    tells = (_tell(1, R1), _tell(2, R1), _tell(3, R1), _tell(4, "2026-10-27-0000000f"))

    p = _progress(root, reader, [R1], live=lambda r: r == R1, tells=tells)
    assert [(t.id, t.state) for t in p.tasks] == [("T1", "held"), ("T2", "pending")]
    assert [(t.id, t.state, t.outcome) for t in p.tells] == [
        ("tell-1", "replied", "applied"),
        ("tell-2", "delivered", None),
        ("tell-3", "sent", None),
    ], "another run's tell is not this work's"
    text = status_text(p)
    assert "T1 held for a message" in text
    assert "tells   tell-1 replied: applied · tell-2 delivered · tell-3 sent" in text

    _end(journal, "PARTIAL")
    p = _progress(root, reader, [R1], tells=tells)
    assert (p.tasks[0].id, p.tasks[0].state) == ("T1", "pending")
    assert [t.state for t in p.tells] == ["replied", "unanswered", "sent"]
