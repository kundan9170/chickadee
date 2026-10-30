"""Zeus's job table: a command that runs past its yield becomes a job, a
`JOB` effect on the session log; its waiter records the exit and, when nobody asked for it,
a `JOB_EXITED` notice.

Over a real repository with no sandbox, the person having lifted the unsandboxed floor,
as `test_shell_declared_write.py` runs its shell: the job, not the wall, is under test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from tool_args import checked

from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import JobsFull, ToolResult
from chickadee.context.rewind import safe_point_refusal
from chickadee.context.session import Session, Turn
from chickadee.context.standing import NoticeTrigger
from chickadee.context.terms import Vision
from chickadee.errors import SessionError
from chickadee.executor.executor import Executor
from chickadee.git_guard import dump, held
from chickadee.hooks.fire import BoundHook, Hooks, HookSet, RoleHooks
from chickadee.hooks.payloads import HookEvent, JobExited
from chickadee.hooks.sinks import SessionHookSink
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import HookVerdict, Note
from chickadee.ids import job_operation, job_precondition, operation_component
from chickadee.init.detect import resolve
from chickadee.init.lease import take_write_lease
from chickadee.isolation.none import Passthrough
from chickadee.journal.effects import EffectKind, Intent, begin
from chickadee.process.supervisor import GRACE_S, process_started
from chickadee.profiles.loader import ProfileSet
from chickadee.schema import ToolCall
from chickadee.shell import RefusedAboveT0, Shell, ShellClaim
from chickadee.undo import undo_operation
from chickadee.workspace import current_tree, restore_tree
from chickadee.zeus import jobs as jobs_module
from chickadee.zeus.jobs import JobLimits, JobReport, JobTable
from chickadee.zeus.writes import SessionEffects, ShellWrites, shell_effects

SID = "S-jobs"


class _SaysYes:
    """The person, saying yes: for a command that is asked about, as one touching git's
    state is outside the sandbox."""

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        return True


class JobTableRig:
    def __init__(
        self,
        tmp_path: Path,
        root: Path,
        limits: JobLimits,
        *,
        approves: bool = False,
        hooks: HookSet | None = None,
    ) -> None:
        self.root = root
        self.session = Session.create(tmp_path / "sessions", SID)
        reader = Reader(
            SessionHome(self.session),
            HandleResolver(workspace=root, session_blob=self.session.blob),
            vision=Vision("z"),
        )
        self.wakes: list[int] = []
        self.live: dict[str, Any] = {}
        self.table = JobTable(
            self.session,
            limits=limits,
            wake=lambda: self.wakes.append(1),
            hooks=None
            if hooks is None
            else RoleHooks(Hooks(hooks, SessionHookSink(self.session)), "zeus"),
        )
        self.shell = Shell(
            Executor(Passthrough()),
            root,
            lambda data: reader.store("output", data),
            authority=_SaysYes() if approves else None,
            allow_unsandboxed_shell=True,
            writes=ShellWrites(
                self.session,
                root,
                commits=True,
                profiles=ProfileSet.load(repo=root).all(),
                beside=self.table,
            ),
            jobs=self.table,
        )

    def run(self, command: str, op: str = "S:c1", **args: object) -> object:
        answer = self.shell.run(checked("shell", {"command": command, **args}), operation_id=op)
        # Held here, since the table lets go of an entry as its job ends.
        for job_id in self.table.running():
            entry = self.table.entry(job_id)
            if entry is not None:
                self.live[job_id] = entry
        return answer

    def join(self, job_id: str) -> None:
        """Until its waiter has written all it writes (`recorded`), and the fold says it
        ended, whether or not it already had."""
        entry = self.live.get(job_id) or self.table.entry(job_id)
        if entry is not None:
            assert entry.recorded.wait(15)
        deadline = time.monotonic() + 15
        while job_id in self.table.running() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job_id not in self.table.running()

    def jobs(self) -> list[object]:
        return [
            e
            for e in shell_effects(
                self.session.state, lambda e: self.session.blob(e.payload_sha)
            )
            if e.is_job
        ]

    def report(self, job_id: str) -> JobReport:
        (view,) = [v for v in self.table.views() if v.job_id == job_id]
        assert view.report is not None
        return view.report

    def notices(self) -> list[str]:
        return [
            t.text
            for t in self.session.state.turns()
            if t.notice is not None and t.notice.trigger is NoticeTrigger.JOB_EXITED
        ]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": "x = 1\n", "notes.txt": "a\n"})


def built(tmp_path: Path, repo: Path, **limits: int) -> JobTableRig:
    return JobTableRig(tmp_path, repo, JobLimits(**{"yield_s": 1, **limits}))


def test_a_command_that_ends_inside_its_window_is_todays_call(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    answer = b.run("echo hi")
    assert isinstance(answer, ToolResult) and "hi" in answer.render()
    assert b.jobs() == [] and b.session.state.jobs.ids() == ()
    assert not any((b.session.root / SID / "jobs").glob("*")), "its files are removed"


def test_a_command_past_its_window_becomes_a_job_and_its_exit_is_told(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    answer = b.run("echo started; sleep 2; echo done; exit 3")
    assert isinstance(answer, ToolResult)
    text = answer.render()
    assert "running as j1 (task)" in text and "started" in text and "do not poll" in text
    assert b.session.state.jobs.ids() == ("j1",)
    b.join("j1")
    assert b.session.state.jobs.ids() == ()
    report = b.report("j1")
    assert report.exit_code == 3 and report.ended == "exited" and report.notify
    (told,) = b.notices()
    assert "job j1 · task" in told and "exited 3" in told and "done" in told
    assert b.wakes == [1]


def test_a_jobs_exit_is_told_to_zeus_s_hooks_and_their_note_follows_it(
    tmp_path: Path, repo: Path
) -> None:
    """D227 `job_exited`: told each exit the table tells Zeus of, after the wake it opens (a
    hook's script never holds the turn back), its note a notice after."""
    told: list[JobExited] = []
    woken: list[int] = []

    def watch(event: HookEvent) -> HookVerdict:
        assert isinstance(event.payload, JobExited)
        told.append(event.payload)
        woken.append(len(b.wakes))
        return Note(f"{event.payload.job} failed: look at its tail")

    spec = HookSpec.model_validate({"event": "job_exited", "name": "w", "use": "w"})
    b = JobTableRig(
        tmp_path, repo, JobLimits(yield_s=1), hooks=HookSet((BoundHook(spec, watch, "w"),))
    )
    b.run("sleep 2; exit 3")
    b.join("j1")
    assert [(t.job, t.exit_code) for t in told] == [("j1", 3)]
    assert woken == [1], "the wake went first"
    hooked = [
        t.text
        for t in b.session.state.turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.HOOK
    ]
    assert hooked == ["j1 failed: look at its tail"]


def test_background_yields_at_once_and_a_dev_server_row_is_a_server(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo, yield_s=30)
    begun = time.monotonic()
    answer = b.run("sleep 30", background=True)
    assert time.monotonic() - begun < 5
    assert isinstance(answer, ToolResult) and "running as j1 (task)" in answer.render()
    b.table.close()
    # A server is asked, so the server half runs under a rig that approves.
    s = JobTableRig(tmp_path / "s", repo, JobLimits(yield_s=30), approves=True)
    served = s.run("npm run dev", op="S:c2", background=True)
    assert isinstance(served, ToolResult) and "(server)" in served.render()
    s.table.close()


def test_a_server_is_a_job_at_once_and_says_to_send_sandbox_false(
    tmp_path: Path, repo: Path
) -> None:
    """A server never ends, so it does not wait out the yield window. Asked by its row,
    it runs; the note tells the model to send the flag, for the server no row names."""
    b = JobTableRig(tmp_path, repo, JobLimits(yield_s=30), approves=True)
    begun = time.monotonic()
    served = b.run("python3 -m http.server 0 --bind 127.0.0.1")
    assert time.monotonic() - begun < 5, "a server is a job at once"
    assert isinstance(served, ToolResult)
    text = served.render()
    assert "running as j1 (server)" in text and "Send sandbox: false with every server" in text
    sent = b.run("python3 -m http.server 0 --bind 127.0.0.1", op="S:c2", sandbox=False)
    assert isinstance(sent, ToolResult) and "running as j2 (server)" in sent.render()
    assert "Send sandbox: false" not in sent.render(), "sent with the flag, nothing to say"
    script = b.run("sleep 30", op="S:c3", sandbox=False, kind="server")
    assert isinstance(script, ToolResult) and "running as j3 (server)" in script.render()
    b.table.close()


def test_a_call_that_says_server_is_asked_to_run_outside(tmp_path: Path, repo: Path) -> None:
    """`kind: "server"` on a command no row knows (a script) says it must be
    reached, so it is asked as `sandbox: false` is, not run sealed where nothing reaches it."""
    nobody = built(tmp_path / "n", repo, yield_s=30)
    refused = nobody.run("sleep 30", kind="server")
    assert (
        isinstance(refused, RefusedAboveT0) and refused.classification == "outside the sandbox"
    )
    nobody.table.close()
    b = JobTableRig(tmp_path / "y", repo, JobLimits(yield_s=30), approves=True)
    served = b.run("sleep 30", kind="server")
    assert isinstance(served, ToolResult)
    assert "running as j1 (server)" in served.render() and "sandbox none" in served.render()
    b.table.close()


def test_a_server_sent_background_false_waits_its_window(tmp_path: Path, repo: Path) -> None:
    b = JobTableRig(tmp_path, repo, JobLimits(yield_s=1), approves=True)
    begun = time.monotonic()
    served = b.run("python3 -m http.server 0 --bind 127.0.0.1", background=False)
    assert time.monotonic() - begun >= 1, "it waited out its window"
    assert isinstance(served, ToolResult) and "running as j1 (server)" in served.render()
    b.table.close()


def test_a_server_with_nobody_to_ask_is_refused_and_never_a_job(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo, yield_s=30)
    refused = b.run("python3 -m http.server 8000", background=True)
    assert isinstance(refused, RefusedAboveT0) and refused.classification == "dev_server"
    assert b.jobs() == [] and b.session.state.jobs.ids() == ()
    b.table.close()


def test_shell_yield_s_zero_blocks_as_before(tmp_path: Path, repo: Path) -> None:
    b = built(tmp_path, repo, yield_s=0)
    answer = b.run("sleep 1.5; echo late")
    assert isinstance(answer, ToolResult) and "late" in answer.render()
    assert b.jobs() == []


def test_with_max_jobs_running_background_is_jobs_full_and_a_command_runs_to_its_end(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo, max_jobs=1)
    b.run("sleep 30", background=True)
    refused = b.run("sleep 30", op="S:c2", background=True)
    assert isinstance(refused, JobsFull) and refused.running == ("j1",)
    assert "job(action='kill'" in refused.render()
    ran = b.run("sleep 1.5; echo end", op="S:c3")
    assert isinstance(ran, ToolResult)
    assert "end" in ran.render() and "could not become a job" in ran.render()
    b.table.close()


def test_a_command_that_ends_inside_its_window_says_nothing_of_the_jobs_running(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo, max_jobs=1)
    b.run("sleep 30", background=True)
    quick = b.run("echo quick", op="S:c2")
    assert isinstance(quick, ToolResult) and "quick" in quick.render()
    assert "could not become a job" not in quick.render(), "it never reached its yield"
    b.table.close()


def test_a_slot_filled_during_the_window_is_named_by_the_command_it_held(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo, max_jobs=1, yield_s=2)
    answers: list[object] = []
    ran = threading.Thread(target=lambda: answers.append(b.run("sleep 3; echo end")))
    ran.start()
    time.sleep(0.5)
    # Another job yields while this command is inside its window: the table decides at the
    # yield, and the note is its decision, not a count read before the command ran.
    begin(
        SessionEffects(b.session),
        Intent(
            job_operation("S:other"),
            EffectKind.JOB,
            approved_sha="a" * 64,
            preconditions=(job_precondition("j9"), "kind=task", "pid=0"),
            idempotent=False,
            surface=(),
        ),
    )
    ran.join(30)
    (answer,) = answers
    assert isinstance(answer, ToolResult) and "end" in answer.render()
    assert "(j9)" in answer.render() and "could not become a job" in answer.render()


def test_an_ended_job_s_output_is_its_record_and_its_files_are_gone(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("echo out; echo err >&2; sleep 2", op="S:c1")
    b.join("j1")
    assert not any((b.session.root / SID / "jobs").glob("*")), "one copy: the record"
    read = b.table.output("j1")
    assert read.kept and read.out == "out\n[stderr]\nerr\n" and read.skipped == 0
    report = b.report("j1")
    assert report.kept is not None and report.dropped == 0 and report.out_bytes == 4


def test_a_long_output_keeps_its_first_and_last_lines_and_says_what_it_dropped(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs_module, "KEPT_END_BYTES", 64)
    b = built(tmp_path, repo)
    # Still running at the yield, however loaded the machine: one ended by then is the call's.
    b.run("echo FIRST; seq 1 2000; sleep 1; echo LAST", background=True)
    b.join("j1")
    report = b.report("j1")
    assert report.dropped > 8000 and report.out_bytes > report.dropped
    kept = b.table.output("j1").out
    assert kept.startswith("FIRST\n") and kept.endswith("LAST\n")
    assert f"[... {report.dropped} bytes not kept ...]" in kept
    (told,) = b.notices()
    assert f"{report.dropped} bytes from its middle not kept" in told and "LAST" in told


def test_a_kill_records_the_exit_and_tells_nobody(tmp_path: Path, repo: Path) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 30", background=True)
    assert b.table.kill("j1", "zeus")
    b.join("j1")
    report = b.report("j1")
    assert report.ended == "killed" and report.by == "zeus" and not report.notify
    assert b.notices() == [] and b.wakes == []


def test_an_exit_a_wait_returned_is_seen_and_not_told(tmp_path: Path, repo: Path) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 2", background=True)
    exit_ = b.table.wait("j1", 10, None)
    assert exit_ is not None and exit_.seen
    b.join("j1")
    assert not b.report("j1").notify and b.notices() == []


def test_a_server_that_crashes_is_told(tmp_path: Path, repo: Path) -> None:
    # `kind: "server"` is asked since, so the rig approves.
    b = JobTableRig(tmp_path, repo, JobLimits(yield_s=1), approves=True)
    b.run("sleep 1; exit 1", background=True, kind="server")
    b.join("j1")
    assert b.report("j1").notify and len(b.notices()) == 1


def test_the_step_ends_at_the_yield_and_the_lease_is_free_while_the_job_runs(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("echo written > made.txt; sleep 30")
    steps = [
        e
        for e in shell_effects(b.session.state, lambda e: b.session.blob(e.payload_sha))
        if not e.is_job
    ]
    assert steps and steps[-1].outcome == "succeeded" and "made.txt" in steps[-1].written
    lease = take_write_lease(resolve(repo))
    lease.release()
    b.table.close()


def test_what_a_job_wrote_that_no_step_did_is_listed_and_after_a_step_is_overwrote(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 2; echo job > a.txt; echo job >> notes.txt", background=True)
    # A step beside the job writes notes.txt first: the job's later write is "overwrote".
    beside = b.run("echo step > notes.txt", op="S:c2")
    assert isinstance(beside, ToolResult) and "j1 was running" in beside.render()
    b.join("j1")
    report = b.report("j1")
    assert "a.txt" in report.wrote and "notes.txt" not in report.wrote
    assert report.overwrote == ("notes.txt",)
    (told,) = b.notices()
    assert "a.txt" in told and "notes.txt after a step" in told


def test_a_running_job_is_not_a_safe_point_and_a_job_is_never_undone(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 30", background=True)
    assert safe_point_refusal(b.session.state) == (
        "job j1 is still running; wait for it or kill it"
    )
    b.table.close()
    assert safe_point_refusal(b.session.state) is None
    with pytest.raises(SessionError, match="is a job"):
        undo_operation(b.session, None, repo, job_operation("S:c1"))  # type: ignore[arg-type]


def test_close_kills_every_job_completes_its_effect_and_wakes_nobody(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 30", background=True)
    b.run("sleep 30", op="S:c2", background=True)
    b.table.close()
    assert b.session.state.jobs.ids() == ()
    assert {b.report(j).by for j in ("j1", "j2")} == {"close"}
    assert b.notices() == [] and b.wakes == []


def test_resume_reaps_a_live_group_whose_start_time_matches_and_records_it_lost(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    # A job the harness died beside: its intent is on the log, its process still running.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    stranger = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        for n, (proc, started) in enumerate(
            [(child, process_started(child.pid)), (stranger, "not-its-start-time")], 1
        ):
            begin(
                SessionEffects(b.session),
                Intent(
                    job_operation(f"S:c{n}"),
                    EffectKind.JOB,
                    approved_sha="a" * 64,
                    preconditions=(
                        job_precondition(f"j{n}"),
                        "kind=task",
                        f"pid={proc.pid}",
                        f"pgid={proc.pid}",
                        f"started={started}",
                    ),
                    idempotent=False,
                    surface=(),
                ),
            )
        assert b.session.state.jobs.ids() == ("j1", "j2")
        b.table.reconcile()
        assert b.session.state.jobs.ids() == ()
        assert child.wait(10) is not None
        assert stranger.poll() is None, "a pid whose start time differs is never signalled"
        assert {b.report(j).ended for j in ("j1", "j2")} == {"lost"}
        assert len(b.notices()) == 2 and b.wakes == []
        assert not any(
            isinstance(t, Turn) and t.role == "user" for t in b.session.state.turns()
        )
    finally:
        for proc in (child, stranger):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def test_a_yield_header_never_claims_a_wall_the_run_did_not_have(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    answer = b.run("sleep 30", background=True)
    assert isinstance(answer, ToolResult)
    head = answer.render().splitlines()[0]
    assert "sandbox none" in head and "repo not protected" in head, head
    assert "worktree writable" not in head
    b.table.close()


def test_a_source_file_its_step_put_back_is_not_the_job_s_write(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    answer = b.run("echo 'x = 2' > app.py; sleep 2")
    assert isinstance(answer, ToolResult) and "put back app.py" in answer.render()
    b.join("j1")
    report = b.report("j1")
    assert report.wrote == () and not report.moved, "the step's put-back is where it began"


def test_two_jobs_in_one_workspace_say_a_change_may_be_either_s(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 2; echo a > a.txt; sleep 1", background=True)
    b.run("sleep 1; echo b > b.txt", op="S:c2", background=True)
    b.join("j2")
    b.join("j1")
    report = b.report("j1")
    assert report.alongside == ("j2",) and report.wrote == ("a.txt", "b.txt")
    (told, _) = sorted(b.notices(), key=lambda t: not t.startswith("job j1"))
    assert "j1 or j2, run beside it, wrote 2 file(s)" in told


def test_a_step_in_another_workspace_claims_nothing_of_a_job_s(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 2; echo job > a.txt", background=True)
    other = init_repo(tmp_path / "other", {"a.txt": "a\n"})
    assert b.table.running_in(other) == () and b.table.running_in(repo) == ("j1",)
    b.table.stepped(other, "S:x", ("a.txt",), "0" * 40)
    b.join("j1")
    assert b.report("j1").wrote == ("a.txt",)


def test_wait_answers_a_seen_exit_however_long_its_record_takes(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = built(tmp_path, repo)
    slow = current_tree

    def measured(root: Path, **kw: Any) -> Any:
        time.sleep(2)
        return slow(root, **kw)

    monkeypatch.setattr(jobs_module, "GRACE_S", 0.1)
    monkeypatch.setattr(jobs_module, "current_tree", measured)
    b.run("sleep 1; exit 4", background=True)
    assert b.table.await_exit("j1", 30, None) == "j1 exited 4 after 1s"
    assert b.notices() == [], "seen, so never told"


def test_an_exit_whose_recording_fails_is_closed_ambiguous_and_never_runs_on(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = built(tmp_path, repo)

    def unreadable(root: Path, **kw: Any) -> Any:
        raise OSError("the tree could not be read")

    b.run("sleep 1", background=True)
    monkeypatch.setattr(jobs_module, "current_tree", unreadable)
    b.join("j1")
    (effect,) = b.jobs()
    assert getattr(effect, "outcome", None) == "ambiguous"
    assert b.table.running() == () and safe_point_refusal(b.session.state) is None


def test_resume_keeps_a_lost_job_s_output_from_its_files_and_clears_the_folder(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    jobs_dir = b.session.root / SID / "jobs"
    jobs_dir.mkdir(parents=True)
    # What a crash leaves: an open job's two files, and a file nothing records.
    (jobs_dir / f"{operation_component(f'{SID}:c1')}.out").write_text("collected 3\nFAILED\n")
    (jobs_dir / "stray.out").write_text("cut off inside its window\n")
    begin(
        SessionEffects(b.session),
        Intent(
            job_operation(f"{SID}:c1"),
            EffectKind.JOB,
            approved_sha="a" * 64,
            preconditions=(job_precondition("j1"), "kind=task", "pid=0", f"step={SID}:c1"),
            idempotent=False,
            surface=(),
        ),
    )
    b.table.reconcile()
    assert not any(jobs_dir.iterdir()), "the record holds what is kept; the files go"
    report = b.report("j1")
    assert report.ended == "lost" and report.kept is not None
    assert b.table.output("j1").out == "collected 3\nFAILED\n"
    (told,) = b.notices()
    assert "FAILED" in told


def test_close_waits_for_an_exit_still_being_recorded(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that ended just before `close` is no longer running, but its waiter is still
    writing its exit: `close` waits for it, or the session's writer closes under it."""
    b = built(tmp_path, repo)
    slow = current_tree

    def measured(root: Path, **kw: Any) -> Any:
        time.sleep(2)
        return slow(root, **kw)

    monkeypatch.setattr(jobs_module, "current_tree", measured)
    b.run("sleep 1; exit 2", background=True)
    deadline = time.monotonic() + 10
    while b.table.entry("j1") is not None and time.monotonic() < deadline:
        time.sleep(0.05)  # ended: out of the running, its exit being recorded
    b.table.close()
    assert b.report("j1").exit_code == 2
    (told,) = b.notices()
    assert "exited 2" in told


def test_resume_puts_back_what_a_lost_job_no_wall_held_changed_of_git_s(
    tmp_path: Path, repo: Path
) -> None:
    """The step held git's settings before its command ran, and named the
    blob in its intent; the job ran with no wall, and the harness died before its exit. Resume
    puts back from that blob, says so, and leaves a walled job's alone."""
    b = built(tmp_path, repo)
    config = repo / ".git" / "config"
    original = config.read_bytes()
    before = b.session.put_blob(dump(held(repo)))
    for n, walled in ((1, "false"), (2, "true")):
        begin(
            SessionEffects(b.session),
            Intent(
                f"{SID}:c{n}",
                EffectKind.SHELL_WRITE,
                approved_sha="a" * 64,
                preconditions=("profile=zeus", "checkout", f"root={repo}", f"guarded={before}"),
                idempotent=False,
                surface=(),
            ),
        ).complete("succeeded")
        begin(
            SessionEffects(b.session),
            Intent(
                job_operation(f"{SID}:c{n}"),
                EffectKind.JOB,
                approved_sha="a" * 64,
                preconditions=(
                    job_precondition(f"j{n}"),
                    "kind=task",
                    "pid=0",
                    f"step={SID}:c{n}",
                    f"walled={walled}",
                ),
                idempotent=False,
                surface=(),
            ),
        )
    config.write_bytes(original + b"[core]\n\tsshCommand = touch pwned\n")
    b.table.reconcile()
    assert config.read_bytes() == original
    assert b.report("j1").git_restored == (str(config.resolve()),)
    assert b.report("j2").git_restored == (), "a walled job changed nothing of git's"
    (told, _) = sorted(b.notices(), key=lambda t: not t.startswith("job j1"))
    assert "changed git's own settings" in told


def test_a_job_that_ends_before_its_step_reports_never_puts_back_what_the_step_removed(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command changed git's config before its yield; its step's guard puts that back,
    slowly. The job ends meanwhile, and its exit waits for its step, then puts back from the
    step's own hold: the change stays gone. From the yield's snapshot, it would come back."""
    b = JobTableRig(tmp_path, repo, JobLimits(yield_s=1), approves=True)

    def restoring(*args: Any, **kw: Any) -> Any:
        time.sleep(1.5)
        return restore_tree(*args, **kw)

    monkeypatch.setattr("chickadee.zeus.writes.restore_tree", restoring)
    config = repo / ".git" / "config"
    original = config.read_bytes()
    # `sshCommand`, which no git of the harness's runs: the race under test is the exit's.
    answer = b.run("git config core.sshCommand 'touch pwned'; sleep 1.2")
    assert isinstance(answer, ToolResult) and "running as j1" in answer.render()
    b.join("j1")
    assert config.read_bytes() == original
    assert b.report("j1").git_restored == ()


def test_a_yield_whose_record_fails_leaves_no_job_running(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = built(tmp_path, repo)

    def broken(*args: Any, **kw: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(jobs_module, "snapshot_operation", broken)
    with pytest.raises(OSError):
        b.run("sleep 30", background=True)
    assert b.table.running_in(repo) == () and b.table.running() == ()
    assert b.table.entry("j1") is None


def test_wait_on_a_job_whose_exit_is_being_recorded_answers_its_exit(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = built(tmp_path, repo)
    slow = current_tree

    def measured(root: Path, **kw: Any) -> Any:
        time.sleep(2)
        return slow(root, **kw)

    monkeypatch.setattr(jobs_module, "current_tree", measured)
    b.run("sleep 1; exit 4", background=True)
    deadline = time.monotonic() + 10
    while b.table.entry("j1") is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert b.table.await_exit("j1", 30, None) == "j1 exited 4 after 1s"


@pytest.mark.writes_worktree("design.md")
def test_a_doc_write_beside_a_job_is_never_the_job_s(tmp_path: Path, repo: Path) -> None:
    b = built(tmp_path, repo)
    b.run("sleep 2", background=True)
    writes = ShellWrites(b.session, repo, commits=True, beside=b.table)
    with writes.recorded(f"{SID}:doc1", "a" * 64, "doc") as _notes:
        (repo / "design.md").write_text("# the design\n")
    b.join("j1")
    assert b.report("j1").wrote == (), "Zeus's own recorded write"


def test_output_past_the_ceiling_inside_the_window_says_the_harness_killed_it(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo, yield_s=5, max_log_bytes=100_000)
    answer = b.run("yes | head -c 2000000; sleep 10")
    assert isinstance(answer, ToolResult)
    assert "passed zeus.job_max_log_bytes" in answer.render()
    assert b.jobs() == []


def test_close_signals_every_job_at_once(tmp_path: Path, repo: Path) -> None:
    b = built(tmp_path, repo)
    stubborn = "trap '' TERM; sleep 60"
    for n in range(3):
        b.run(stubborn, op=f"S:c{n}", background=True)
    begun = time.monotonic()
    b.table.close()
    took = time.monotonic() - begun
    assert took < 2 * GRACE_S + 2, f"one grace for all, not one each ({took:.1f}s)"
    assert b.table.running() == ()


def test_a_job_settled_after_a_crash_names_its_command_from_its_call(
    tmp_path: Path, repo: Path
) -> None:
    b = built(tmp_path, repo)
    b.session.record_user("run it")
    b.session.record_assistant(
        "", [ToolCall(id="c1", name="shell", arguments={"command": "pytest -q"})]
    )
    begin(
        SessionEffects(b.session),
        Intent(
            job_operation(f"{SID}:c1"),
            EffectKind.JOB,
            approved_sha="a" * 64,
            preconditions=(job_precondition("j1"), "kind=task", "pid=0", f"step={SID}:c1"),
            idempotent=False,
            surface=(),
        ),
    )
    b.table.reconcile()
    assert b.report("j1").line == "pytest -q"
    (told,) = b.notices()
    assert "job j1 · task · pytest -q · lost" in told
    # Listed as what happened to it, never as an exit code nobody saw.
    assert "j1 · task · lost" in b.table.listing() and "None" not in b.table.listing()
