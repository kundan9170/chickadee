"""`journal/effects.py`: the one recorder of every effect's lifecycle, and the table of
what each effect kind is."""

from __future__ import annotations

import pytest

from chickadee.errors import HarnessBug, InfraError
from chickadee.journal import Journal, project
from chickadee.journal.effects import (
    EFFECT_EVENTS,
    KINDS,
    EffectKind,
    Intent,
    Reconcile,
    Restore,
    ambiguous,
    begin,
    failed_if,
    on_journal,
    perform,
    reopen,
)
from chickadee.journal.writer import FSYNC_KINDS
from chickadee.schema.journal import (
    EffectCompletedP,
    EffectIntendedP,
    EffectStartedP,
    EventKind,
    ExecP,
    P,
)
from chickadee.state.roots import Roots

RUN = "2026-10-05-0000e153"


class _Log:
    def __init__(self) -> None:
        self.records: list[tuple[EventKind, P]] = []

    def record(self, kind: EventKind, payload: P) -> int:
        self.records.append((kind, payload))
        return len(self.records)

    def kinds(self) -> list[EventKind]:
        return [k for k, _ in self.records]

    def completed(self) -> EffectCompletedP:
        (done,) = [p for _, p in self.records if isinstance(p, EffectCompletedP)]
        return done


def _intent(kind: EffectKind = EffectKind.WORKER_STEP, op: str = "exec:a1:1:1") -> Intent:
    return Intent(
        op,
        kind,
        approved_sha="a" * 64,
        preconditions=("world_generation=1",),
        idempotent=False,
        surface=("gen/**",),
        preimages="b" * 40,
    )


WHOLE = [EventKind.EFFECT_INTENDED, EventKind.EFFECT_STARTED, EventKind.EFFECT_COMPLETED]


# -- begin -----------------------------------------------------------------------------------


def test_begin_writes_the_intent_and_the_start_and_leaves_the_effect_open() -> None:
    log = _Log()
    effect = begin(log, _intent(EffectKind.JOB, "job:j1"))
    assert log.kinds() == WHOLE[:2]
    assert not effect.completed
    intended = log.records[0][1]
    assert isinstance(intended, EffectIntendedP)
    assert intended.effect_kind == EffectKind.JOB.value


def test_an_effect_begun_on_one_thread_completes_on_another() -> None:
    # A job is intended on the conversation's worker and completed on its waiter.
    import threading

    log = _Log()
    effect = begin(log, _intent(EffectKind.JOB, "job:j1"))
    done = threading.Thread(target=lambda: effect.complete("succeeded", observed=("a.txt",)))
    done.start()
    done.join()
    assert log.kinds() == WHOLE
    assert log.completed().observed_surface == ("a.txt",)


def test_perform_writes_what_begin_writes() -> None:
    begun, performed = _Log(), _Log()
    begin(begun, _intent()).complete("succeeded")
    with perform(performed, _intent()) as fx:
        fx.complete("succeeded")
    assert begun.records == performed.records


# -- perform ---------------------------------------------------------------------------------


def test_perform_writes_the_intent_and_the_start_before_the_effect_runs() -> None:
    log = _Log()
    with perform(log, _intent()) as fx:
        assert log.kinds() == WHOLE[:2], "durable before the effect is performed"
        fx.complete("succeeded", observed=("gen/a.py",), post_state="c" * 40, result="d" * 64)
    assert log.kinds() == WHOLE
    intended = log.records[0][1]
    assert intended == EffectIntendedP(
        operation_id="exec:a1:1:1",
        effect_kind="worker_step",
        approved_sha="a" * 64,
        preconditions=("world_generation=1",),
        idempotent=False,
        declared_surface=("gen/**",),
        preimages_sha="b" * 40,
    )
    assert log.records[1][1] == EffectStartedP(operation_id="exec:a1:1:1")
    assert log.completed() == EffectCompletedP(
        operation_id="exec:a1:1:1",
        outcome="succeeded",
        observed_surface=("gen/a.py",),
        post_state_sha="c" * 40,
        result_sha="d" * 64,
    )


def test_an_exception_closes_the_effect_ambiguous_by_default_and_is_reraised() -> None:
    log = _Log()
    error = RuntimeError("broke")
    with pytest.raises(RuntimeError) as info, perform(log, _intent()):
        raise error
    assert info.value is error
    assert log.kinds() == WHOLE
    done = log.completed()
    assert done.outcome == "ambiguous" and done.observed_surface == ()


@pytest.mark.parametrize(
    ("error", "outcome"), [(InfraError("refused"), "failed"), (OSError("io"), "ambiguous")]
)
def test_the_callers_rule_decides_what_an_exception_proves(
    error: Exception, outcome: str
) -> None:
    log = _Log()
    with pytest.raises(type(error)), perform(log, _intent(), on_raise=failed_if(InfraError)):
        raise error
    assert log.completed().outcome == outcome


def test_an_exception_after_the_completion_writes_nothing_more() -> None:
    log = _Log()
    with pytest.raises(InfraError), perform(log, _intent()) as fx:
        fx.complete("failed")
        raise InfraError("refused, and said so")
    assert log.kinds() == WHOLE
    assert log.completed().outcome == "failed"


def test_a_kill_writes_nothing_after_the_start() -> None:
    log = _Log()
    with pytest.raises(KeyboardInterrupt), perform(log, _intent()):
        raise KeyboardInterrupt
    assert log.kinds() == WHOLE[:2], "recovery reads the operation as started"


def test_an_effect_that_states_no_outcome_is_a_harness_bug() -> None:
    log = _Log()
    with pytest.raises(HarnessBug, match="left open"), perform(log, _intent()):
        pass


def test_an_effect_completes_once() -> None:
    log = _Log()
    with pytest.raises(HarnessBug, match="completed twice"), perform(log, _intent()) as fx:
        fx.complete("succeeded")
        fx.complete("succeeded")


def test_a_line_under_the_effect_goes_to_the_same_log_before_the_completion() -> None:
    log = _Log()
    exec_line = ExecP(
        operation_id="exec:a1:1:1",
        policy="action",
        label="make gen",
        argv_sha="e" * 64,
        cwd=".",
        exit_code=0,
        wall_ms=1,
        timed_out=False,
        sandbox="none",
        network=False,
        stdout_sha="f" * 64,
        stderr_sha="f" * 64,
        violations=(),
    )
    with perform(log, _intent()) as fx:
        assert fx.record(EventKind.EXEC, exec_line) == 3
        fx.complete("succeeded")
    assert log.kinds() == [*WHOLE[:2], EventKind.EXEC, EventKind.EFFECT_COMPLETED]


@pytest.mark.parametrize("phase", sorted(EFFECT_EVENTS))
def test_a_lifecycle_line_is_never_recorded_beside_the_recorder(phase: EventKind) -> None:
    log = _Log()
    with pytest.raises(HarnessBug), perform(log, _intent()) as fx:
        fx.record(phase, EffectStartedP(operation_id="exec:a1:1:1"))
    assert log.kinds() == WHOLE, "and the raise closed the effect"


# -- reopen ----------------------------------------------------------------------------------


@pytest.fixture
def journal(repo: Roots) -> Journal:
    return Journal.create(repo, RUN, {})


def _intended() -> EffectIntendedP:
    return EffectIntendedP(
        operation_id="exec:a1:1:1",
        effect_kind="worker_step",
        approved_sha="a" * 64,
        preconditions=(),
        idempotent=False,
        declared_surface=(),
    )


def test_reopen_starts_an_operation_cut_off_at_its_intent(journal: Journal) -> None:
    with journal:
        journal.append(EventKind.EFFECT_INTENDED, _intended(), task_id="T1", round=1)
        op = project(journal.events()).operations["exec:a1:1:1"]
        effect = reopen(on_journal(journal, task_id="T1", round=1), op)
        effect.complete("ambiguous")
        events = list(journal.events())
        state = project(events)
    assert [e.kind for e in events] == WHOLE
    assert all(e.task_id == "T1" and e.round == 1 for e in events)
    assert state.operations["exec:a1:1:1"].outcome == "ambiguous"
    assert not state.defects


def test_reopen_does_not_start_an_operation_twice(journal: Journal) -> None:
    with journal:
        journal.append(EventKind.EFFECT_INTENDED, _intended(), task_id="T1", round=1)
        journal.append(EventKind.EFFECT_STARTED, EffectStartedP(operation_id="exec:a1:1:1"))
        op = project(journal.events()).operations["exec:a1:1:1"]
        reopen(on_journal(journal, task_id="T1", round=1), op).complete("failed")
        kinds = [e.kind for e in journal.events()]
    assert kinds == WHOLE


def test_a_completed_operation_is_not_reopened(journal: Journal) -> None:
    with journal:
        log = on_journal(journal, task_id="T1", round=1)
        with perform(log, _intent()) as fx:
            fx.complete("succeeded")
        op = project(journal.events()).operations["exec:a1:1:1"]
        with pytest.raises(HarnessBug, match="already completed"):
            reopen(log, op)


# -- on_journal and the phase list -----------------------------------------------------------


def test_on_journal_binds_every_line_to_its_task_and_round(journal: Journal) -> None:
    with journal:
        with perform(on_journal(journal, task_id="T1", round=2), _intent()) as fx:
            fx.complete("succeeded")
        events = list(journal.events())
    assert [(e.task_id, e.round) for e in events] == [("T1", 2)] * 3


def test_on_journal_without_a_task_writes_at_run_scope(journal: Journal) -> None:
    with journal:
        with perform(on_journal(journal), _intent()) as fx:
            fx.complete("succeeded")
        events = list(journal.events())
    assert [(e.task_id, e.round) for e in events] == [(None, None)] * 3


def test_every_effect_phase_is_fsynced_by_the_journal() -> None:
    assert EFFECT_EVENTS <= FSYNC_KINDS


def test_ambiguous_is_the_safe_claim() -> None:
    assert ambiguous(InfraError("anything")) == "ambiguous"


# -- the table -------------------------------------------------------------------------------


def test_every_kind_has_a_rule() -> None:
    assert set(KINDS) == set(EffectKind)


def test_the_kinds_rules_are_the_table() -> None:
    table = {k.value: (r.recovery, r.restore, r.worktree_only) for k, r in KINDS.items()}
    # The packet loop's kinds (patch_apply, patch_rollback, packet_commit, packet_known_good,
    # action_exec, worker_shell, stage, unstage) retired with it; the rest are unchanged.
    assert table == {
        "bootstrap": (Reconcile.NONE, Restore.NONE, True),
        "scout_escape": (Reconcile.SEALED, Restore.NONE, False),
        # A declared Zeus write, on the session log, taken back by its snapshot ref.
        "shell_write": (Reconcile.NONE, Restore.TREE, False),
        "worker_step": (Reconcile.STEP, Restore.TREE, False),
        "task_commit": (Reconcile.TRAILER, Restore.NONE, True),
        "task_known_good": (Reconcile.HEAD, Restore.NONE, False),
        # The work folder's approval and report commits.
        "approval_commit": (Reconcile.TRAILER, Restore.NONE, True),
        # A background job, on the session log, never taken back.
        "shell_job": (Reconcile.JOB, Restore.NONE, False),
    }
