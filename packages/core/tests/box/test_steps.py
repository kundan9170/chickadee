"""Steps: `StepRecord` around a box role's calls.

What these pin: one `worker_step` per writing call and none per read; each step's "before" is
the last step's "after"; a shell command's write is seen as an edit's is; a networked command's
hosts are in its intent; skipped files are named; every step's tree is kept on its task's chain;
a path the person's `deny` globs name that a command changed is put back and said.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from box_helpers import FakeRunJournal
from repos import git, init_repo

from chickadee.approval.prefs import PreferenceError, merge, parse_level
from chickadee.approval.tier import Preferences
from chickadee.box.edits import WorktreeEdits, wrote
from chickadee.box.steps import NETWORK, StepRecord, StepTrees, networked
from chickadee.box.syntax import SyntaxChecker
from chickadee.journal.effects import on_journal
from chickadee.journal.port import JournalPort
from chickadee.schema.journal import EffectCompletedP, EffectIntendedP, EventKind
from chickadee.secrets.denylist import Denylist
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import ToolContext
from chickadee.tools.rules import Pending
from chickadee.tools.tool import Answer, Invocation, TextAnswer
from chickadee.tools.validate import Arguments, validate
from chickadee.workspace import current_tree, step_ref

RUN = "2026-10-15-0a1b2c3d"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})


class _Spy:
    def __init__(self) -> None:
        self.reindexed: list[list[Path]] = []

    def reindex(self, changed: Sequence[Path] | None = None) -> object:
        self.reindexed.append(list(changed or ()))
        return None


def _record(
    root: Path, journal: FakeRunJournal, cap: int = 1024, deny: tuple[str, ...] = ()
) -> tuple[StepRecord, _Spy]:
    start, _ = current_tree(root, untracked_max_bytes=None)
    trees = StepTrees(start, cap, run_id=RUN, task="T1")
    spy = _Spy()
    log = on_journal(cast(JournalPort, journal), task_id="T1", round=1)
    prefs = merge([parse_level({"approval": {"deny": list(deny)}}, "user")])
    return StepRecord(root, log, trees, spy, prefs=lambda: prefs), spy


def _pending(tool: str, **raw: object) -> Pending:
    found = BUILTINS.get(tool)
    assert found is not None
    return Pending(Invocation(f"c-{tool}", f"op-{tool}", 0), tool, raw, tool, found.spec)


def _call(record: StepRecord, tool: str, act: Callable[[], object], **raw: object) -> Answer:
    return _call_in(record, ToolContext(role="worker"), tool, act, **raw)


def _call_in(
    record: StepRecord, ctx: ToolContext, tool: str, act: Callable[[], object], **raw: object
) -> Answer:
    def run() -> Answer:
        act()
        return TextAnswer("ok", "bytes")

    return record.around(_pending(tool, **raw), run, ctx)


def _intents(journal: FakeRunJournal) -> list[EffectIntendedP]:
    return [p for k, p, *_ in journal.lines if k is EventKind.EFFECT_INTENDED]


def _completions(journal: FakeRunJournal) -> list[EffectCompletedP]:
    return [p for k, p, *_ in journal.lines if k is EventKind.EFFECT_COMPLETED]


@pytest.mark.writes_worktree("*")
def test_a_writing_call_is_one_step_and_a_read_is_none(root: Path) -> None:
    journal = FakeRunJournal()
    record, spy = _record(root, journal)
    _call(record, "read", lambda: None, path="a.py")
    assert journal.lines == []
    _call(record, "create", lambda: (root / "b.py").write_text("b = 1\n"), path="b.py")
    [intent] = _intents(journal)
    assert intent.effect_kind == "worker_step" and intent.declared_surface == ("create",)
    [done] = _completions(journal)
    assert done.outcome == "succeeded" and done.observed_surface == ("b.py",)
    assert spy.reindexed == [[root / "b.py"]]


@pytest.mark.writes_worktree("*")
def test_each_step_starts_from_the_last_steps_tree(root: Path) -> None:
    journal = FakeRunJournal()
    record, _ = _record(root, journal)
    _call(record, "create", lambda: (root / "b.py").write_text("b = 1\n"), path="b.py")
    _call(record, "shell", lambda: (root / "a.py").write_text("a = 2\n"), command="sed")
    first, second = _completions(journal)
    _, later = _intents(journal)
    assert later.preimages_sha == first.post_state_sha
    assert second.observed_surface == ("a.py",), "a shell command's write is seen"
    chain = git(root, "rev-list", step_ref(RUN, "T1"))
    assert len(chain.splitlines()) == 3, "both steps on the chain, above the head"


@pytest.mark.writes_worktree("*")
def test_a_networked_command_says_it_opened_the_network(root: Path) -> None:
    journal = FakeRunJournal()
    record, _ = _record(root, journal)
    _call(record, "shell", lambda: None, command="pip download x", network=True)
    _call(record, "shell", lambda: None, command="pytest -q")
    networked_intent, sealed = _intents(journal)
    assert NETWORK in networked_intent.preconditions
    assert networked(networked_intent.preconditions)
    assert not networked(sealed.preconditions)


@pytest.mark.writes_worktree("*")
def test_an_unsealed_command_is_networked_whatever_it_declared(root: Path) -> None:
    """Review of #592: unsealed, a command has the network and the machine, so its step says so
    and a crash in it is never re-run on resume; an edit beside it is not a command."""
    from chickadee.box.shell import BoxShell
    from chickadee.executor.executor import Executor
    from chickadee.isolation.none import Passthrough

    journal = FakeRunJournal()
    record, _ = _record(root, journal)
    unsealed = BoxShell(
        Executor(Passthrough()),
        root,
        cast(JournalPort, journal),
        None,
        asker="worker",
        task=lambda: "T1",
    )
    ctx = ToolContext(role="worker", exec=unsealed)
    record.around(
        _pending("shell", command="pytest -q"), lambda: TextAnswer("ok", "bytes"), ctx
    )
    _call_in(record, ctx, "create", lambda: (root / "b.py").write_text("b = 1\n"), path="b.py")
    shell_step, edit_step = _intents(journal)
    assert networked(shell_step.preconditions)
    assert not networked(edit_step.preconditions)


@pytest.mark.writes_worktree("*")
def test_a_sealed_command_whose_class_needs_the_network_is_networked(root: Path) -> None:
    """Review of #592: an approved install gets the network from its class with no `network`
    argument, so its step says so and resume never runs it again."""
    from sandboxes import Enforcing, Scripted

    from chickadee.box.shell import BoxShell
    from chickadee.executor.executor import Executor

    journal = FakeRunJournal()
    record, _ = _record(root, journal)
    sealed = BoxShell(
        Executor(Enforcing(), supervisor=Scripted(b"")),
        root,
        cast(JournalPort, journal),
        None,
        asker="worker",
        task=lambda: "T1",
    )
    ctx = ToolContext(role="worker", exec=sealed)
    _call_in(record, ctx, "shell", lambda: None, command="npm ci")
    _call_in(record, ctx, "shell", lambda: None, command="pytest -q")
    install, tests = _intents(journal)
    assert networked(install.preconditions)
    assert not networked(tests.preconditions)


def test_a_step_intended_by_an_older_version_still_reads_as_networked() -> None:
    """A step cut off before the change named each host; resume never re-runs it."""
    assert networked(("before=abc", "host=pypi.org"))


@pytest.mark.writes_worktree("*")
def test_a_file_over_the_cap_is_named_and_not_stored(root: Path) -> None:
    journal = FakeRunJournal()
    record, _ = _record(root, journal, cap=8)
    _call(record, "shell", lambda: (root / "big.bin").write_text("x" * 64), command="make")
    assert record.last is not None and record.last.skipped == ("big.bin",)


@pytest.mark.writes_worktree("*")
def test_a_call_that_raises_is_put_back_and_its_step_failed(root: Path) -> None:
    """All or none: what a call wrote before it raised is undone, as resume undoes a step a
    crash cut off, and the next step starts from the tree before it."""
    journal = FakeRunJournal()
    record, _ = _record(root, journal)

    def half() -> None:
        (root / "a.py").write_text("a = 2\n")
        (root / "half.py").write_text("h = 1\n")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _call(record, "create", half, path="half.py")
    assert (root / "a.py").read_text() == "a = 1\n"
    assert not (root / "half.py").exists()
    [intent] = _intents(journal)
    [done] = _completions(journal)
    assert (done.outcome, done.observed_surface) == ("failed", ("a.py", "half.py"))
    assert done.post_state_sha == intent.preimages_sha


@pytest.mark.writes_worktree("*")
def test_a_denied_path_a_command_changed_is_put_back_and_said(root: Path) -> None:
    (root / "infra").mkdir()
    (root / "infra" / "main.tf").write_text("keep\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "infra")
    journal = FakeRunJournal()
    record, _ = _record(root, journal, deny=("infra/**",))

    def command() -> None:
        (root / "infra" / "main.tf").write_text("changed\n", encoding="utf-8")
        (root / "infra" / "new.tf").write_text("new\n", encoding="utf-8")
        (root / "a.py").write_text("a = 2\n", encoding="utf-8")

    answer = _call(record, "shell", command, command="terraform fmt")
    assert (root / "infra" / "main.tf").read_text(encoding="utf-8") == "keep\n"
    assert not (root / "infra" / "new.tf").exists()
    assert (root / "a.py").read_text(encoding="utf-8") == "a = 2\n"
    said = answer.render()
    assert '[put back infra/main.tf: deny "infra/**" in your approval.toml denies it]' in said
    assert "[put back infra/new.tf:" in said
    [done] = _completions(journal)
    assert done.observed_surface == ("a.py",)


@pytest.mark.writes_worktree("*")
def test_a_preference_file_that_cannot_be_read_after_a_command_still_puts_the_step_back(
    root: Path,
) -> None:
    """T-3.157 review: the deny read runs inside the step's guard, so an `approval.toml` the
    person broke while a command ran leaves no half-recorded writes: they are put back and
    the step is `failed`, as for a call that raised."""
    journal = FakeRunJournal()
    start, _ = current_tree(root, untracked_max_bytes=None)
    trees = StepTrees(start, 1024, run_id=RUN, task="T1")
    log = on_journal(cast(JournalPort, journal), task_id="T1", round=1)

    def broken() -> Preferences:
        raise PreferenceError("approval.toml: malformed TOML")

    record = StepRecord(root, log, trees, _Spy(), prefs=broken)

    def command() -> None:
        (root / "a.py").write_text("a = 2\n", encoding="utf-8")

    with pytest.raises(PreferenceError):
        _call(record, "shell", command, command="sed -i s/1/2/ a.py")
    assert (root / "a.py").read_text(encoding="utf-8") == "a = 1\n"
    [done] = _completions(journal)
    assert (done.outcome, done.observed_surface) == ("failed", ("a.py",))


# -- beside a running job ---------------------------------------------


class _BesideRecord:
    """The session's job table as a step sees it: one job running, and what each step said."""

    def __init__(self, running: tuple[str, ...] = ("j1",), started: int | None = None) -> None:
        self.jobs = running
        self.started = len(running) if started is None else started
        self.seen: list[tuple[tuple[str, ...], str]] = []

    def running_in(self, root: Path) -> tuple[str, ...]:
        return self.jobs

    def started_in(self, root: Path) -> int:
        return self.started

    def stepped(self, root: Path, step: str, paths: tuple[str, ...], tree: str) -> None:
        self.seen.append((paths, tree))


def _edited(root: Path, path: str, text: str) -> Callable[[], None]:
    """A call's write as the edit port makes it: on disk, and noted as the call's own."""

    def act() -> None:
        (root / path).write_text(text, encoding="utf-8")
        wrote((path,))

    return act


def _beside_record(root: Path, journal: FakeRunJournal, beside: _BesideRecord) -> StepRecord:
    start, _ = current_tree(root, untracked_max_bytes=None)
    trees = StepTrees(start, 1024, run_id=RUN, task="T1")
    log = on_journal(cast(JournalPort, journal), task_id="T1", round=1)
    return StepRecord(root, log, trees, _Spy(), beside=beside)


@pytest.mark.writes_worktree("*")
def test_beside_a_job_a_steps_before_is_the_disk_and_its_record_says_so(root: Path) -> None:
    journal = FakeRunJournal()
    beside = _BesideRecord()
    record = _beside_record(root, journal, beside)
    (root / "job.log").write_text("the job wrote this\n", encoding="utf-8")  # between steps
    answer = _call(record, "create", _edited(root, "b.py", "b = 1\n"), path="b.py")
    [intent] = _intents(journal)
    [done] = _completions(journal)
    assert "beside=j1" in intent.preconditions
    assert done.observed_surface == ("b.py",), "what the job wrote before the step is not its"
    assert "caught from" not in answer.render() and "may include" not in answer.render()
    assert beside.seen == [(("b.py",), done.post_state_sha)]


@pytest.mark.writes_worktree("*")
def test_an_edit_step_names_what_changed_during_it_that_it_did_not_write(root: Path) -> None:
    journal = FakeRunJournal()
    beside = _BesideRecord()
    record = _beside_record(root, journal, beside)

    def edit_while_the_job_writes() -> None:
        _edited(root, "b.py", "b = 1\n")()
        (root / "job.log").write_text("the job, during the call\n", encoding="utf-8")

    answer = _call(record, "create", edit_while_the_job_writes, path="b.py")
    assert "[caught from j1: job.log changed during this call; not its own]" in answer.render()
    [done] = _completions(journal)
    assert beside.seen == [(("b.py",), done.post_state_sha)], "job.log is left to the job"


@pytest.mark.writes_worktree("*")
def test_a_write_through_the_edit_port_is_the_call_s_own_beside_a_job(root: Path) -> None:
    journal = FakeRunJournal()
    beside = _BesideRecord()
    record = _beside_record(root, journal, beside)
    port = WorktreeEdits(root, Denylist(), SyntaxChecker(root, ()), untracked_max_bytes=1024)
    write = BUILTINS.get("write")
    assert write is not None
    args = validate(write.spec, {"path": "b.py", "content": "b = 2\n"})
    assert isinstance(args, Arguments)

    def run() -> Answer:
        answer = write.run(Invocation("c-write", "op-write", 0), args, ctx)
        (root / "job.log").write_text("the job, during the call\n", encoding="utf-8")
        return answer

    ctx = ToolContext(role="worker", edits=port)
    answer = record.around(_pending("write", path="b.py"), run, ctx)
    assert "[caught from j1: job.log changed during this call; not its own]" in answer.render()
    assert [paths for paths, _ in beside.seen] == [("b.py",)]


@pytest.mark.writes_worktree("*")
def test_a_command_step_beside_a_job_says_its_record_may_include_the_job_s(root: Path) -> None:
    journal = FakeRunJournal()
    record = _beside_record(root, journal, _BesideRecord())
    answer = _call(record, "shell", lambda: (root / "b.py").write_text("b = 1\n"), command="x")
    assert "j1 was running: this record may include j1's writes" in answer.render()
    assert "caught from" not in answer.render()


@pytest.mark.writes_worktree("*")
def test_a_job_that_ran_and_ended_between_two_steps_wrote_nothing_of_the_next_s(
    root: Path,
) -> None:
    """No job runs now, but one started since the last step: what it wrote is on disk, not in
    the last step's "after", and it is never the next step's write."""
    journal = FakeRunJournal()
    record = _beside_record(root, journal, _BesideRecord(running=(), started=1))
    (root / "job.log").write_text("the job wrote this, then ended\n", encoding="utf-8")
    _call(record, "create", _edited(root, "b.py", "b = 1\n"), path="b.py")
    [done] = _completions(journal)
    assert done.observed_surface == ("b.py",)


@pytest.mark.writes_worktree("*")
def test_a_whole_file_write_over_a_jobs_change_says_it_replaced_it(root: Path) -> None:
    journal = FakeRunJournal()
    record = _beside_record(root, journal, _BesideRecord())
    (root / "a.py").write_text("a = 'the job'\n", encoding="utf-8")
    answer = _call(record, "write", _edited(root, "a.py", "a = 3\n"), path="a.py")
    assert "this replaced j1's change to a.py" in answer.render()


@pytest.mark.writes_worktree("*")
def test_with_no_job_running_a_step_is_todays(root: Path) -> None:
    journal = FakeRunJournal()
    beside = _BesideRecord(running=())
    record = _beside_record(root, journal, beside)
    answer = _call(record, "create", lambda: (root / "b.py").write_text("b = 1\n"), path="b.py")
    [intent] = _intents(journal)
    assert not any(p.startswith("beside=") for p in intent.preconditions)
    assert "was running" not in answer.render()
