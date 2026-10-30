"""The session log is the only place a session lives, and replaying it reproduces the session
(invariant 3).

Every line kind folds back into what it produced live. A torn tail, the one damage a crash
produces, is skipped by a reader and cut by the writer; any other damage refuses to open and
names the seq, because a session that silently skipped a constraint is worse than one that
will not open. A fork reads its parent through and never writes it.
"""

from __future__ import annotations

import contextlib
import errno
import os
import threading
from collections.abc import Buffer, Callable, Mapping
from pathlib import Path
from typing import Any, Final, cast, get_args

import pytest
from context_helpers import (
    SESSION_ID,
    Caller,
    FakeRecords,
    cut,
    exchange,
    new_session,
    reader_for,
    reopen,
    run_start,
    work_unit,
    write_files,
)
from session_lines import raw
from spend_reads import family_spent, lines_of

import chickadee.durable.blobs as durable_blobs
import chickadee.durable.files as durable_files
import chickadee.durable.log as durable_log
from chickadee.approval.grants import ActionGrantedLine, CommandGrantLine
from chickadee.context.concerns import accrue, note_concern, promote
from chickadee.context.ledger import note_constraint, note_rejected, reconsider, rejected
from chickadee.context.lines import (
    PARSED,
    CapLine,
    PayloadError,
    RedoLine,
    RewoundCode,
    SessionKind,
    SpendLine,
    UndoLine,
    canonical_json,
)
from chickadee.context.questions import (
    QuestionAnsweredLine,
    QuestionAskedLine,
    record_answered,
    record_asked,
)
from chickadee.context.record_lines import ExternalCallLine
from chickadee.context.request import assemble
from chickadee.context.rewind import CodeSnapshotLine, ForkRefused, record_rewind
from chickadee.context.session import OtherLine, Session, SessionState
from chickadee.context.store import (
    BLOBS_DIR,
    LOG_NAME,
    SessionCorrupt,
    SessionEvent,
    SessionLog,
    SessionStoreError,
    SessionWriteError,
    session_dir,
)
from chickadee.context.work_line import WorkLine
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.errors import HarnessBug
from chickadee.hooks.sinks import HookLine
from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.projection import RunProjection
from chickadee.schema import ToolCall, sha256_hex
from chickadee.schema.adapter import Usage
from chickadee.schema.interaction import Option, Question, QuestionKind
from chickadee.schema.journal import EventKind, ExternalCallP, HookP
from chickadee.session_handoff import HandoffCompleted, HandoffIntended
from chickadee.sessions import SessionTitle, record_title
from chickadee.skills.pin import SkillLoaded, SkillUnloaded
from chickadee.trust.events import UntrustedRead
from chickadee.wiring import ModelSwitch, record_pinned, record_switch
from chickadee.work import record_work, works
from chickadee.zeus.snapshots import record as record_snapshot


def fold(root: Path, session_id: str) -> SessionState:
    """The session as a reader rebuilds it from disk, without its lock."""
    log = SessionLog.read(root, session_id)
    return SessionState.replay(session_id, root, log.events(), log.payload)


def seen(state: SessionState) -> dict[str, object]:
    """Everything the fold holds that anything reads."""
    return {
        "lines": list(state.lines()),
        "constraints": state.ledger.constraints(),
        "rejected": rejected(state),
        "concerns": state.concerns.all(),
        "handles": sorted(h.id for h in state.handles),
        "views": state.views(),
        "cuts": state.cuts(),
        "episodes": state.episodes(),
        "workspace": state.workspace_at(state.last_seq),
        "works": works(state),
        "pending": state.pending_calls(),
        "noticed": sorted(state.noticed_stale),
    }


def files(directory: Path) -> dict[str, bytes]:
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def forge(
    root: Path, session_id: str, seq: int, kind: SessionKind, payload: Mapping[str, object]
) -> None:
    """Append a well-formed line the writer would have refused, the way a hand edit would."""
    directory = session_dir(root, session_id)
    data = canonical_json(payload)
    sha = sha256_hex(data)
    (directory / BLOBS_DIR / sha).write_bytes(data)
    event = SessionEvent(seq, "2026-09-26T12:00:00Z", kind, sha, None, session_id)
    with (directory / LOG_NAME).open("a", encoding="utf-8") as fh:
        fh.write(event.line() + "\n")


# -- every kind replays ------------------------------------------------------------------------


def write_every_kind(session: Session, ws: Path) -> None:
    """Write a line of every kind this package folds and every kind another package owns,
    each in a context its fold accepts."""
    write_files(ws, {"a.py": "def a():\n    return 1\n"})
    call = Caller(session, reader_for(session, ws))
    said = session.record_user("keep a() pure")
    view = call("read", path="a.py", lines="1-2")
    session.record_assistant("read it", findings={view.call_id: "a returns 1"})
    assert note_constraint(session, "a() stays pure", [said.seq], "session") == "C-1"
    assert note_rejected(session, text="cache a()", reason="pure", by="zeus", at=1) == "R-1"
    # A skill's load, the person's unload, and a third-party body's crossing.
    session.append(
        SkillLoaded(view.call_id, "style", "project", "project", "p" * 64, "c" * 64, False)
    )
    session.append(UntrustedRead(path=".chickadee/skills/style.md", bytes=12))
    session.append(SkillUnloaded("style"))
    # The person's *always for this session*.
    session.append(
        CommandGrantLine("tests", ("pytest",), None, False, (), (), "shell-0123456789ab")
    )
    # An action that grant approved, one line and never a question.
    session.append(
        ActionGrantedLine(
            "shell-0123456789ac", "S:c2", "pytest -q", "session", "tests", ("pytest",)
        )
    )
    # A hook's verdict on Zeus's turn.
    session.append(
        HookLine(HookP(event="before_stop", name="again", decision="continue", wall_ms=1))
    )
    # A call the external-call policy made, its writer's kind.
    session.append(
        ExternalCallLine(
            EventKind.EXTERNAL_CALL,
            ExternalCallP(
                operation_id="c8",
                provider="web.fetch.http",
                tool="web_fetch",
                args_sha="a" * 64,
                policy="discovery",
                ok=True,
                wall_ms=3,
            ),
        )
    )
    assert isinstance(reconsider(session, "R-1", "it got slow", by="user"), int)
    assert note_concern(session, "a() may grow a side effect", ["a.py"], "critic") == "K-1"
    session.record_workspace(
        WorkspaceRef(
            mode="worktree",
            lease_id="L-1",
            source_ref="main",
            source_head="0" * 40,
            branch="chickadee/run",
            worktree_path="/tmp/wt",
            handoff="pending",
        )
    )
    record_work(session, WorkLine(name="2026-10-18-retry", dir="docs/chickadee", state="open"))
    said = next(t.seq for t in session.state.turns() if t.role == "user")
    session.record_tell("2026-10-18-a1b2c3d4", "use argon2 instead", (said,))
    alpha = work_unit(session, "alpha")
    assert accrue(session, ["a.py"], run_id=alpha)
    assert isinstance(promote(session, "K-1", "tests/test_a.py::test_pure"), int)
    work_unit(session, "beta")
    for kind in (
        "packet_record",
        "design_record",
        "summary",
        "note",
        "plan",
        "limits",
        "box_compaction",
    ):
        session.append(raw(kind, {"id": f"{kind}-1", "text": f"a {kind}"}), run_id=alpha)
    # Zeus's loop: the prefix's sha, a turn's bound, the turn frame.
    session.append(raw("prefix", {"sha": "p" * 64, "parts": {}, "model_id": "zeus"}))
    session.append(raw("bound", {"resource": "tool_calls", "outcome": "refused"}))
    session.append(raw("frame", {"date": "2026-09-27", "text": "2026-09-27 · head 0000000"}))
    # A Zeus `locate`'s scout records: the frozen journal payload, as the sink wrote it.
    session.append(raw("scout", {"event": "scout_start", "payload": {"rung": "m"}}))
    # A declared Zeus write's D40 record, in the scout line's shape.
    session.append(
        raw("shell_effect", {"event": "effect_started", "payload": {"operation_id": "S:c1"}})
    )
    # A spawned agent's round start, in the same shape.
    session.append(
        raw(
            "agent",
            {
                "event": "agent_start",
                "payload": {
                    "agent": "A1",
                    "round": 0,
                    "source": "built-in",
                    "profile": "reader",
                    "rung": "mid-param",
                    "model": "m",
                    "task_sha": "t" * 64,
                    "call_id": "c9",
                },
            },
        )
    )
    # One of Athena's calls before her run has a journal, in the same shape.
    session.append(
        raw(
            "box_call",
            {
                "event": "model_call",
                "payload": {
                    "role": "athena",
                    "request": "decompose",
                    "tokens_in": 10,
                    "tokens_out": 2,
                    "cost_usd": 0.0,
                    "stop_reason": "end_turn",
                },
            },
        )
    )
    # One Zeus model call's charge: the session's spend outlives the process.
    session.record_spend(
        "high-param-model", Usage(prompt_tokens=120, completion_tokens=8), 0.0042, "high-param"
    )
    # The person's cap, then lifted for the family: both survive a replay.
    session.record_cap(12.5)
    session.record_cap(None)
    # Per rung: a scout's charge, and a coder cap set and then lifted.
    session.record_spend(
        "mid-param-model", Usage(prompt_tokens=40, completion_tokens=2), 0.001, "scout"
    )
    session.record_cap(2.0, rung="coder")
    session.record_cap(None, rung="coder")
    # The person moved a rung to another catalog model, then the session was
    # pinned, after which a switch is refused: both survive a replay.
    record_switch(
        session, ModelSwitch("high-param", "high-param-model", "mid-param-model", "m", "s" * 64)
    )
    record_pinned(session)
    # The session's name: the title role's, then the person's.
    record_title(session, SessionTitle("Retry the fetch", "model"))
    record_title(session, SessionTitle("Fetcher retries", "person"))
    # A handoff, written ahead: its intent, then its completion.
    with session.lock:
        session.append(HandoffIntended("handoff:S:1", "keep", "a" * 40, "b" * 40))
        session.append(HandoffCompleted("handoff:S:1", "done", "kept", False))
    # A turn that wrote: its before and after of one workspace.
    wrote = session.record_user("now change it")
    for phase in ("before", "after"):
        record_snapshot(
            session,
            CodeSnapshotLine(
                turn=wrote.seq,
                phase=phase,
                key="in-place",
                mode="in_place",
                root=str(ws),
                tree="b" * 40,
                ref=f"refs/chickadee/turns/S/{wrote.seq}/in-place/{phase}",
                head="c" * 40,
                skipped=("big.bin",),
            ),
        )
    session.record_assistant("changed")
    # A rewind: the code taken back and given back; the conversation stays.
    kept = RewoundCode(
        key="in-place",
        mode="in_place",
        root=str(ws),
        target="e" * 40,
        redo_tree="d" * 40,
        redo_ref="r",
        branch_head=None,
        restored=("a.py",),
        conflicts_kept=(),
        committed=None,
    )
    undone = record_rewind(session, UndoLine(wrote.seq, True, False, (kept,)), (), ())
    session.append(RedoLine(undone, True, False))
    asked = Question(
        id="q-every-kind",
        kind=QuestionKind.CLARIFY_GOAL,
        text="Keep a() pure?",
        options=(Option(key="yes", label="Yes"), Option(key="no", label="No")),
        default="yes",
    )
    record_asked(session, QuestionAskedLine(question=asked))
    record_answered(
        session,
        QuestionAnsweredLine(id="q-every-kind", choice="yes", source="human", elapsed_ms=5),
    )
    # A design gate's decision, retired with the packet loop: an older log's
    # line, kept unread, as the parsed line it was wrote it.
    session.append(
        raw(
            "plan_decision",
            {
                "id": "design-every-kind",
                "decision": "approve",
                "design_sha": "e" * 64,
                "source": "human",
                "approved": {"goal": "keep a() pure"},
            },
        )
    )
    session.record_user("thanks")
    session.record_assistant("welcome")
    assert cut(session, ws).cut is not None


def test_every_line_kind_replays_to_what_it_was_live(session: Session, ws: Path) -> None:
    write_every_kind(session, ws)
    at = session.state.last_seq

    live, sent = seen(session.state), assemble(session.state, session.blob).messages
    assert seen(fold(session.root, SESSION_ID)) == live
    with reopen(session) as again:
        assert seen(again.state) == live
        assert assemble(again.state, again.blob).messages == sent
        with Session.fork(
            again.root, SESSION_ID, at, "try it another way", "S-child", code="keep"
        ) as child:
            child.record_user("in the child")
            child_seen = seen(child.state)
            kinds = {e.kind for e, _ in again.state.lines()} | {
                e.kind for e, _ in child.state.lines()
            }
    assert kinds == set(get_args(SessionKind)), "every kind was written"
    assert seen(fold(session.root, "S-child")) == child_seen


# -- spend ---------------------------------------------------------------------------


def test_spend_lines_fold_into_the_sessions_spend_and_replay_to_it(session: Session) -> None:
    cached = Usage(prompt_tokens=100, completion_tokens=10, cached_prompt_tokens=40)
    session.record_spend("high-param-model", cached, 0.25, "high-param")
    session.record_spend(
        "high-param-model",
        Usage(prompt_tokens=50, completion_tokens=5, estimated=True),
        0.5,
        "high-param",
    )

    spent = family_spent(session.root, session.session_id, session.state)
    assert (spent.tokens_in, spent.tokens_out, spent.cached_in) == (150, 15, 40)
    assert spent.cost_usd == pytest.approx(0.75)
    assert spent.estimated_calls == 1
    with reopen(session) as again:
        assert family_spent(again.root, again.session_id, again.state) == spent
        assert lines_of(again.state, SpendLine) == lines_of(session.state, SpendLine)


@pytest.mark.parametrize(
    "bad",
    [
        {"model": "", "cost_usd": 0.1},
        {"model": "m", "cost_usd": -0.1},
        {"model": "m", "cost_usd": float("nan")},
        {"model": "m", "cost_usd": "0.1"},
        {"model": "m", "cost_usd": 0.1, "tokens_in": -1},
        {"model": "m", "cost_usd": 0.1, "estimated": "no"},
    ],
)
def test_a_malformed_charge_is_refused_before_it_is_written(
    session: Session, bad: dict[str, object]
) -> None:
    payload: dict[str, object] = {
        "tokens_in": 1,
        "tokens_out": 1,
        "cached_in": 0,
        "cache_writes": 0,
        "estimated": False,
    }
    payload.update(bad)
    before = session.state.last_seq
    with pytest.raises(PayloadError):
        session.append(raw("spend", payload))
    assert session.state.last_seq == before


# -- cap ----------------------------------------------------------------------


def test_cap_lines_fold_in_order_and_replay_to_them(session: Session) -> None:
    session.record_cap(12.5)
    session.record_cap(None)

    caps = lines_of(session.state, CapLine)
    assert [c.hard_cost_usd for c in caps] == [12.5, None]
    with reopen(session) as again:
        assert lines_of(again.state, CapLine) == caps


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"hard_cost_usd": 0},
        {"hard_cost_usd": -1.0},
        {"hard_cost_usd": float("inf")},
        {"hard_cost_usd": float("nan")},
        {"hard_cost_usd": "5"},
        {"hard_cost_usd": True},
    ],
)
def test_a_malformed_cap_is_refused_before_it_is_written(
    session: Session, bad: dict[str, object]
) -> None:
    before = session.state.last_seq
    with pytest.raises(PayloadError):
        session.append(raw("cap", bad))
    assert session.state.last_seq == before


def test_a_cap_line_is_fsynced(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap the log loses is a raise or a lift the next open undoes."""
    synced: list[int] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    # A `note` line flushes and is not fsynced; each writes one new blob, which is. So a
    # `cap` line costs one fsync more than a `note` — its log's.
    session.append(raw("note", {"id": "note-1", "text": "flushed only"}))
    flushed = len(synced)
    session.record_cap(3.0)
    assert len(synced) - flushed == flushed + 1


# -- rung ---------------------------------------------------------------------------


def test_spend_and_cap_lines_carry_their_rung_through_a_replay(session: Session) -> None:
    session.record_spend(
        "high-param-model", Usage(prompt_tokens=1, completion_tokens=1), 0.1, "high-param"
    )
    session.record_spend(
        "mid-param-model", Usage(prompt_tokens=1, completion_tokens=1), 0.2, "scout"
    )
    session.record_cap(3.0)
    session.record_cap(1.0, rung="scout")
    session.record_cap(None, rung="scout")

    assert [c.rung for c in lines_of(session.state, SpendLine)] == ["high-param", "scout"]
    assert [(c.rung, c.hard_cost_usd) for c in lines_of(session.state, CapLine)] == [
        (None, 3.0),
        ("scout", 1.0),
        ("scout", None),
    ]
    with reopen(session) as again:
        assert lines_of(again.state, SpendLine) == lines_of(session.state, SpendLine)
        assert lines_of(again.state, CapLine) == lines_of(session.state, CapLine)


def test_a_line_written_before_rungs_existed_reads_as_zeus_and_as_the_total(
    session: Session,
) -> None:
    payload = {"model": "m", "tokens_in": 1, "tokens_out": 1, "cached_in": 0}
    payload |= {"cache_writes": 0, "cost_usd": 0.1, "estimated": False}
    session.append(raw("spend", payload))
    session.append(raw("cap", {"hard_cost_usd": 4.0}))

    (charge,) = lines_of(session.state, SpendLine)
    (cap,) = lines_of(session.state, CapLine)
    assert (charge.rung, cap.rung) == ("zeus", None)


@pytest.mark.parametrize("rung", ["Zeus", "-x", "high param", "r" * 33])
def test_a_rung_name_the_rule_refuses_is_refused_before_it_is_written(
    session: Session, rung: str
) -> None:
    """The log cannot know the config, so it checks the name's shape by the
    one rule; whether the rung is declared is its writer's check (`mid` and `nope` are
    well-formed, and a line may name them)."""
    before = session.state.last_seq
    with pytest.raises(PayloadError):
        session.record_spend("m", Usage(prompt_tokens=1, completion_tokens=1), 0.1, rung)
    with pytest.raises(PayloadError):
        session.record_cap(1.0, rung=rung)
    assert session.state.last_seq == before


@pytest.mark.parametrize("rung", ["", 3, None])
def test_a_rung_that_is_not_a_name_is_refused_in_any_line(
    session: Session, rung: object
) -> None:
    before = session.state.last_seq
    spend = {"model": "m", "tokens_in": 1, "tokens_out": 1, "cached_in": 0, "rung": rung}
    spend |= {"cache_writes": 0, "cost_usd": 0.1, "estimated": False}
    with pytest.raises(PayloadError):
        session.append(raw("spend", spend))
    with pytest.raises(PayloadError):
        session.append(raw("cap", {"hard_cost_usd": 1.0, "rung": rung}))
    assert session.state.last_seq == before


def test_a_former_rung_on_an_old_log_still_replays(session: Session) -> None:
    """D145, the review of #261, D176: the config's rungs are never checked when a line is
    read back. A rung retired after the log named it (`scout`) leaves the log readable,
    and its spend still counts toward the total."""
    session.record_spend("m", Usage(prompt_tokens=1, completion_tokens=1), 0.25, "scout")
    session.record_cap(1.0, rung="scout")
    session.close()

    with Session.open(session.root, SESSION_ID) as again:
        assert [c.rung for c in lines_of(again.state, SpendLine)] == ["scout"]
        assert [c.rung for c in lines_of(again.state, CapLine)] == ["scout"]
        assert family_spent(
            again.root, again.session_id, again.state
        ).cost_usd == pytest.approx(0.25)


# -- damage --------------------------------------------------------------------------------


def closed_with_lines(session: Session, n: int) -> Path:
    for i in range(1, n + 1):
        session.record_user(f"line {i}")
    session.close()
    return session_dir(session.root, SESSION_ID) / LOG_NAME


def test_a_torn_tail_is_skipped_by_a_reader_and_cut_by_the_writer(session: Session) -> None:
    log = closed_with_lines(session, 2)
    whole = log.read_bytes()
    log.write_bytes(whole + b'{"seq":3,"at":"2026-09-26T')
    assert SessionLog.read(session.root, SESSION_ID).last_seq == 2
    assert log.read_bytes().endswith(b'"at":"2026-09-26T'), "a reader repairs nothing"
    with Session.open(session.root, SESSION_ID) as again:
        assert log.read_bytes() == whole
        again.record_user("line 3")
        assert again.state.last_seq == 3


def test_a_torn_blob_is_rewritten_never_trusted(tmp_path: Path, session: Session) -> None:
    """A blob cut short under its final name — damage from a crash mid-write — is not a
    dedupe hit: the next write of the same content replaces it."""
    session.record_user("retry me")
    (event,) = SessionLog.read(session.root, SESSION_ID).events()
    raw = (session_dir(session.root, SESSION_ID) / BLOBS_DIR / event.payload_sha).read_bytes()
    with new_session(tmp_path, "S-torn") as torn:
        (session_dir(torn.root, "S-torn") / BLOBS_DIR / event.payload_sha).write_bytes(raw[:7])
        torn.record_user("retry me")
    with Session.open(session.root, "S-torn") as again:
        assert [t.text for t in again.state.turns()] == ["retry me"]


def test_a_blob_is_durable_before_the_line_that_names_it(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename that gives a blob its name is a directory entry: the blobs directory is
    fsynced after it, before the log line naming the blob is written."""
    order: list[str] = []
    real_fsync = os.fsync
    blobs = session_dir(session.root, SESSION_ID) / BLOBS_DIR
    log = session_dir(session.root, SESSION_ID) / LOG_NAME

    def fsync(fd: int) -> None:
        info = os.fstat(fd)
        if os.path.samestat(info, os.stat(blobs)):
            order.append("blobs/")
        elif os.path.samestat(info, os.stat(log)):
            order.append("log")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    session.record_user("durable")
    assert order == ["blobs/", "log"]


def test_a_failed_blob_directory_sync_is_not_bypassed_by_a_retry(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blob whose directory entry never synced is not a dedupe hit: the next write of the
    same content syncs the directory before its line is written."""
    blobs = session_dir(session.root, SESSION_ID) / BLOBS_DIR
    real_fsync = os.fsync
    synced: list[bool] = []
    fail = [True]

    def fsync(fd: int) -> None:
        if os.path.samestat(os.fstat(fd), os.stat(blobs)):
            if fail:
                fail.pop()
                synced.append(False)
                raise OSError(errno.EIO, "Input/output error")
            synced.append(True)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    store = session._store  # pyright: ignore[reportPrivateUsage]
    payload = canonical_json({"role": "user", "text": "twice"})
    with pytest.raises(SessionWriteError, match="blob directory sync failed"):
        store.put_blob(payload)
    store.put_blob(payload)
    assert synced == [False, True]


def test_a_seq_gap_refuses_to_open(session: Session) -> None:
    log = closed_with_lines(session, 3)
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text(lines[0] + lines[2], encoding="utf-8")
    with pytest.raises(SessionCorrupt) as refused:
        Session.open(session.root, SESSION_ID)
    assert (refused.value.seq, refused.value.detail) == (2, "seq gap: expected 2, got 3")


def test_a_payload_that_does_not_match_its_sha_refuses_to_open(session: Session) -> None:
    log = closed_with_lines(session, 2)
    second = SessionLog.read(session.root, SESSION_ID).events()[1]
    (log.parent / BLOBS_DIR / second.payload_sha).write_bytes(
        b'{"role":"user","text":"edited"}'
    )
    with pytest.raises(SessionCorrupt) as refused:
        Session.open(session.root, SESSION_ID)
    assert (refused.value.seq, refused.value.detail) == (2, "payload sha mismatch")


def test_a_missing_payload_refuses_to_open(session: Session) -> None:
    log = closed_with_lines(session, 2)
    second = SessionLog.read(session.root, SESSION_ID).events()[1]
    (log.parent / BLOBS_DIR / second.payload_sha).unlink()
    with pytest.raises(SessionCorrupt) as refused:
        Session.open(session.root, SESSION_ID)
    assert refused.value.seq == 2
    assert refused.value.detail == f"blob {second.payload_sha} is missing"


def test_a_line_the_fold_refuses_is_corruption_at_its_seq(session: Session) -> None:
    """Well-formed, correctly hashed, and still refused: a constraint citing a line the
    user never said. Replay is the same fold the writer ran, so it refuses the same way."""
    session.record_user("keep it small")
    session.record_assistant("sure")
    session.close()
    forged: dict[str, object] = {
        "id": "C-1",
        "text": "the API is frozen",
        "sources": [2],
        "author": "zeus",
        "scope": "session",
        "scope_id": None,
        "valid_from": 3,
        "valid_until": None,
        "superseded_by": None,
        "supersedes": None,
    }
    forge(session.root, SESSION_ID, 3, "constraint", forged)
    with pytest.raises(SessionCorrupt) as refused:
        Session.open(session.root, SESSION_ID)
    assert (refused.value.seq, refused.value.detail) == (
        3,
        "constraint C-1 cites seq 2, not a user turn",
    )


# -- a write that fails ------------------------------------------------------------------------


def full_disk(monkeypatch: pytest.MonkeyPatch, *, keep: int = 0, after: int = 0) -> None:
    """Every session-log line written after the next `after` puts its first `keep` bytes on
    disk and then fails, the way a full disk does. Blob and lock writes are not touched."""
    writing = os.write
    lines = 0

    def write(fd: int, data: Buffer) -> int:
        nonlocal lines
        raw = bytes(data)
        if not raw.startswith(b'{"seq":'):
            return writing(fd, data)
        lines += 1
        if lines <= after:
            return writing(fd, data)
        writing(fd, raw[:keep])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "write", write)


def failing_fsync(monkeypatch: pytest.MonkeyPatch, log: Path) -> None:
    """Every fsync of `log` fails, as on a disk that reported a write-back error."""
    syncing = os.fsync
    target = log.stat()

    def fsync(fd: int) -> None:
        if os.path.samestat(os.fstat(fd), target):
            raise OSError(errno.EIO, "Input/output error")
        syncing(fd)

    monkeypatch.setattr(os, "fsync", fsync)


def test_a_failed_append_is_cut_back_and_the_store_writes_nothing_more(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part of the line reached the disk before it filled. The log goes back to the last line
    folded and no seq is handed out. Nothing more goes through this store, because a retry
    would build on a log it can no longer vouch for; reopening writes that seq again."""
    session.record_user("one")
    log = session_dir(session.root, SESSION_ID) / LOG_NAME
    before = log.read_bytes()
    full_disk(monkeypatch, keep=12)
    with pytest.raises(SessionWriteError, match=r"append failed at seq=2: .*cut back to seq 1"):
        session.record_user("two")
    monkeypatch.undo()
    assert log.read_bytes() == before
    with pytest.raises(SessionWriteError, match="Reopen the session to write again"):
        session.record_user("two, again")
    assert (log.read_bytes(), [t.text for t in session.state.turns()]) == (before, ["one"])
    session.close()
    with Session.open(session.root, SESSION_ID) as again:
        assert again.record_user("two").seq == 2
    assert [t.text for t in fold(session.root, SESSION_ID).turns()] == ["one", "two"]


def test_a_turn_whose_fsync_fails_is_cut_back_too(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All of the line reached the file and its fsync failed, so whether it would outlive a
    crash is unknown. The caller is told it failed, so the log does not keep it."""
    session.record_user("one")
    log = session_dir(session.root, SESSION_ID) / LOG_NAME
    before = log.read_bytes()
    failing_fsync(monkeypatch, log)
    with pytest.raises(SessionWriteError, match=r"append failed at seq=2: .*cut back to seq 1"):
        session.record_user("two")
    assert log.read_bytes() == before
    with pytest.raises(SessionWriteError, match="Reopen the session to write again"):
        session.record_user("two, again")
    monkeypatch.undo()
    session.close()
    assert [t.text for t in fold(session.root, SESSION_ID).turns()] == ["one"]


def test_a_failed_sync_stops_the_store_and_cuts_nothing(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every line a sync covers was folded and handed out, so a failed one removes none of
    them. It only stops the store, since what is durable is no longer known."""
    session.record_user("one")
    session.record_assistant("two")
    log = session_dir(session.root, SESSION_ID) / LOG_NAME
    before = log.read_bytes()
    failing_fsync(monkeypatch, log)
    with pytest.raises(SessionWriteError, match="fsync failed after seq=2"):
        session.sync()
    with pytest.raises(SessionWriteError, match="Reopen the session to write again"):
        session.record_user("three")
    monkeypatch.undo()
    assert log.read_bytes() == before
    session.close()
    with Session.open(session.root, SESSION_ID) as again:
        assert again.record_user("three").seq == 3


def test_a_fork_whose_first_line_fails_leaves_nothing_and_can_be_made_again(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session.record_user("one")
    full_disk(monkeypatch)
    with pytest.raises(SessionWriteError, match="append failed at seq=2"):
        Session.fork(session.root, SESSION_ID, 1, "branch at one", "S-child", code="keep")
    monkeypatch.undo()
    assert not session_dir(session.root, "S-child").exists()
    with Session.fork(
        session.root, SESSION_ID, 1, "branch at one", "S-child", code="keep"
    ) as child:
        assert child.state.forked_from == (SESSION_ID, 1)


def test_a_fork_whose_arrival_notice_fails_leaves_nothing(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notice of what a fork does not undo is a child's second line. A
    child with its `fork` line and without that notice would reopen never told, and arrival
    does not happen twice, so it is removed like one that failed sooner."""
    session.record_user("migrate the webhooks")
    migrate = {
        "command": "curl -X POST https://api.example/migrate",
        "network": True,
    }
    exchange(session, "s1", "shell", migrate, "migrated\n")
    at = session.record_assistant("let us try another way").seq
    full_disk(monkeypatch, after=1)
    with pytest.raises(SessionWriteError, match=f"append failed at seq={at + 2}"):
        Session.fork(session.root, SESSION_ID, at, "try another way", "S-child", code="keep")
    monkeypatch.undo()
    assert not session_dir(session.root, "S-child").exists()


# -- forks read through ------------------------------------------------------------------------


def test_a_fork_reads_its_parent_through_and_never_writes_it(session: Session) -> None:
    for text in ("one", "two", "three"):
        session.record_user(text)
    parent = session_dir(session.root, SESSION_ID)
    before = files(parent)
    with Session.fork(
        session.root, SESSION_ID, 2, "branch at two", "S-child", code="keep"
    ) as child:
        child.record_user("the child's own")
        assert [(e.session_id, e.seq, e.kind) for e, _ in child.state.lines()] == [
            (SESSION_ID, 1, "turn"),
            (SESSION_ID, 2, "turn"),
            ("S-child", 3, "fork"),
            ("S-child", 4, "turn"),
        ]
        assert child.state.forked_from == (SESSION_ID, 2)
        live = seen(child.state)
    assert files(parent) == before
    session.record_user("the parent carries on")
    assert seen(fold(session.root, "S-child")) == live
    with Session.open(session.root, "S-child") as again:
        assert [t.text for t in again.state.turns()] == ["one", "two", "the child's own"]


def test_a_fork_of_a_fork_reads_the_whole_chain(session: Session) -> None:
    session.record_user("root line")
    with Session.fork(session.root, SESSION_ID, 1, "first", "S-child", code="keep") as child:
        child.record_user("child line")
    with Session.fork(session.root, "S-child", 3, "second", "S-grand", code="keep") as grand:
        grand.record_user("grand line")
        assert [(e.session_id, e.seq) for e, _ in grand.state.lines()] == [
            (SESSION_ID, 1),
            ("S-child", 2),
            ("S-child", 3),
            ("S-grand", 4),
            ("S-grand", 5),
        ]
        live = seen(grand.state)
    assert seen(fold(session.root, "S-grand")) == live


class Unreadable:
    """`RunRecords` whose journal exists and will not read."""

    def projection(self, run_id: str) -> RunProjection:
        raise JournalCorrupt(3, f"runs/{run_id}/journal.jsonl", "torn line")

    def blob(self, run_id: str, sha: str) -> bytes:
        raise FileNotFoundError(sha)


def test_a_fork_is_refused_where_a_runs_journal_will_not_read(session: Session) -> None:
    """A journal that exists and will not read says nothing about whether its run ended, so
    the fork is refused — typed, where the journal's own error would escape as itself."""
    run = work_unit(session, "alpha")
    with pytest.raises(ForkRefused) as refused:
        Session.fork(
            session.root,
            SESSION_ID,
            session.state.last_seq,
            "branch",
            "S-child",
            records=Unreadable(),
            code="keep",
        )
    assert f"run {run}'s journal did not read (JournalCorrupt)" in str(refused.value)
    assert not session_dir(session.root, "S-child").exists(), "the child was never made"


def test_a_fork_of_a_fork_at_an_earlier_seq_reads_only_up_to_it(session: Session) -> None:
    """Forking a fork at or before its own fork point reads through to the grandparent and
    stops at the seq asked for: nothing of the middle session comes with it."""
    for n in range(1, 5):
        session.record_user(f"root line {n}")
    with Session.fork(session.root, SESSION_ID, 4, "first", "S-child", code="keep") as child:
        child.record_user("child line")
    with Session.fork(session.root, "S-child", 2, "earlier", "S-grand", code="keep") as grand:
        assert [(e.session_id, e.seq) for e, _ in grand.state.lines()] == [
            (SESSION_ID, 1),
            (SESSION_ID, 2),
            ("S-grand", 3),
        ]
        assert grand.state.forked_from == ("S-child", 2)
    with Session.open(session.root, "S-grand") as again:
        assert again.state.last_seq == 3


def test_a_fork_of_a_fork_at_an_earlier_seq_names_its_own_parent(session: Session) -> None:
    """Read through to the grandparent, the store still names the parent its fork line
    names, as the fold does."""
    for n in range(1, 5):
        session.record_user(f"root line {n}")
    with Session.fork(session.root, SESSION_ID, 4, "first", "S-child", code="keep") as child:
        child.record_user("child line")
    with Session.fork(session.root, "S-child", 2, "earlier", "S-grand", code="keep") as grand:
        store = grand._store  # pyright: ignore[reportPrivateUsage]
        assert store.forked_from == ("S-child", 2)
        assert store.forked_from == grand.state.forked_from


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(b'{"forked_from": {"session_id": "X", "at_seq": 1}}', id="parses-as-json"),
        pytest.param(b"\xff\xfe not json", id="not-utf8"),
    ],
)
def test_a_damaged_fork_payload_is_corruption_not_a_decode_error(
    session: Session, damage: bytes
) -> None:
    """A fork line's payload is checked like every other line's: by its sha, then its JSON.
    Damage is `SessionCorrupt` at that seq, even damage that still parses."""
    session.record_user("root line")
    with Session.fork(session.root, SESSION_ID, 1, "branch", "S-child", code="keep") as child:
        child.record_user("child line")
    fork = SessionLog.read(session.root, "S-child").events()[1]
    assert fork.kind == "fork"
    (session_dir(session.root, "S-child") / BLOBS_DIR / fork.payload_sha).write_bytes(damage)
    for opening in (
        lambda: SessionLog.read(session.root, "S-child"),
        lambda: Session.open(session.root, "S-child"),
    ):
        with pytest.raises(SessionCorrupt) as refused:
            opening()
        assert refused.value.seq == fork.seq


def test_a_fork_is_refused_where_it_would_split_something(session: Session) -> None:
    session.record_user("go")
    waiting = session.record_assistant(
        "", [ToolCall(id="c1", name="shell", arguments={"command": "ls"})]
    )
    with pytest.raises(ForkRefused, match="call c1 is still waiting for its result"):
        Session.fork(session.root, SESSION_ID, waiting.seq, "now", "S-child", code="keep")
    session.record_tool("c1", "a.py\n")
    alpha = work_unit(session, "alpha")
    in_flight = FakeRecords({alpha: (run_start(alpha, "alpha"),)})
    with pytest.raises(ForkRefused, match=f"run {alpha} is still in flight"):
        Session.fork(
            session.root,
            SESSION_ID,
            session.state.last_seq,
            "now",
            "S-child",
            records=in_flight,
            code="keep",
        )
    with pytest.raises(SessionCorrupt):
        Session.fork(session.root, SESSION_ID, 99, "past the end", "S-child", code="keep")
    assert not session_dir(session.root, "S-child").exists()


# -- crash safety as a property: a fault at every file operation ------------------------

_FILE_OPS: Final = ("open", "write", "fsync", "replace", "ftruncate")


class _Killed(BaseException):
    """The process dying mid-operation. A BaseException, so no `except OSError` runs."""


def _use_os(monkeypatch: pytest.MonkeyPatch, shim: object) -> None:
    """Give the store's file operations `shim` for `os`. Since `T-3.48` to `T-3.50` they run
    in `durable` (`files`' `write_all` and `sync_dir`, `blobs`' blob writes, `log`'s
    appends), so the one counting shim goes to each, and the `at`-th operation is the `at`-th
    the store performs, wherever it runs."""
    monkeypatch.setattr(durable_files, "os", shim)
    monkeypatch.setattr(durable_blobs, "os", shim)
    monkeypatch.setattr(durable_log, "os", shim)


class _FaultyOs:
    """`os` as the session store sees it: the `at`-th file operation fails. As an I/O error,
    or as a kill — a write torn in half first — after which nothing runs."""

    def __init__(self, at: int, kill: bool) -> None:
        self.at, self.kill, self.count = at, kill, 0

    def __getattr__(self, name: str) -> object:
        real = getattr(os, name)
        if name not in _FILE_OPS:
            return real

        def op(*args: Any, **kwargs: Any) -> object:
            k, self.count = self.count, self.count + 1
            if k != self.at:
                return real(*args, **kwargs)
            if not self.kill:
                raise OSError(errno.EIO, "Input/output error")
            if name == "write":
                data = bytes(args[1])
                os.write(args[0], data[: len(data) // 2])
            raise _Killed

        return op


def _crash_script() -> list[Callable[[Session], object]]:
    """A session's ordinary writes, one of them two lines: a constraint and its notice."""
    call = ToolCall(id="c1", name="shell", arguments={"command": "ls"})
    return [
        lambda s: s.record_user("the first ask"),
        lambda s: s.record_assistant("", [call]),
        lambda s: s.record_tool("c1", "a\nb\n"),
        lambda s: s.record_assistant("noted", findings={"c1": "two files"}),
        lambda s: note_constraint(s, "keep the public api", [1], "session"),
        lambda s: s.record_user("the second ask"),
        lambda s: s.record_assistant("done"),
    ]


def _lines(root: Path) -> list[tuple[str, str]]:
    return [(e.kind, e.payload_sha) for e in SessionLog.read(root, SESSION_ID).events()]


def _clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Any], list[int], int]:
    """The lines the script writes with nothing failing, the count after each step, and how
    many file operations it took."""
    session = new_session(tmp_path / "clean")
    faulty = _FaultyOs(-1, kill=False)
    _use_os(monkeypatch, faulty)
    counts: list[int] = []
    for step in _crash_script():
        step(session)
        counts.append(len(SessionLog.read(session.root, SESSION_ID).events()))
    _use_os(monkeypatch, os)
    session.close()
    return _lines(session.root), counts, faulty.count


def _die(session: Session) -> None:
    """What a killed process leaves: its descriptors closed, nothing flushed or released."""
    store = session._store  # pyright: ignore[reportPrivateUsage]
    fd = store._log._fd  # pyright: ignore[reportPrivateUsage]
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)
    # Closing the lock's descriptor is what the kernel does at death; `release` closes it.
    lock = store._lock  # pyright: ignore[reportPrivateUsage]
    if lock is not None:
        lock.release()
    store._lock = None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("mode", ["error", "retry", "kill"])
def test_no_single_fault_leaves_a_session_that_will_not_open_or_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """For every file operation the script's writes perform, fail that one: as an I/O error
    the caller sees and stops at (`error`), as one it carries on past with the same session
    (`retry`), or as the process dying mid-write (`kill`). Then reopen. Every time:

    - the session opens — a torn tail is repaired, nothing else is damage;
    - its lines are a prefix of what the script writes with nothing failing: no line lost
      from a step that returned, none duplicated, none out of order;
    - no step lands only part-way unless it is the step that failed;
    - every failure the caller saw was typed, never a raw OS error;
    - and it takes the next line.

    Most of the store's review findings — a retried append reusing its seq, a torn blob
    trusted as a dedupe hit, writes carrying on past a failed payload — broke one of these."""
    expected, counts, ops = _clean_run(tmp_path, monkeypatch)
    problems: list[str] = []
    for at in range(ops):
        root_dir = tmp_path / f"{mode}-{at}"
        session = new_session(root_dir)
        root = session.root
        _use_os(monkeypatch, _FaultyOs(at, kill=mode == "kill"))
        failed: int | None = None
        script = _crash_script()
        for i, step in enumerate(script):
            try:
                step(session)
            except (_Killed, SessionStoreError):
                failed = i
                break
            except Exception as exc:
                problems.append(f"op {at}: step {i} raised raw {type(exc).__name__}: {exc}")
                failed = i
                break
        if mode == "retry" and failed is not None:
            for step in script[failed:]:
                with contextlib.suppress(SessionStoreError, PayloadError):
                    step(session)
        _use_os(monkeypatch, os)
        if mode == "kill" and failed is not None:
            _die(session)
        else:
            with contextlib.suppress(SessionStoreError):
                session.close()
        try:
            again = Session.open(root, SESSION_ID)
        except SessionStoreError as exc:
            problems.append(f"op {at}: step {failed}: will not reopen: {exc}")
            continue
        got = _lines(root)
        if got != expected[: len(got)]:
            problems.append(f"op {at}: step {failed}: lines are not a prefix of the clean run")
        elif failed is not None and mode != "retry":
            before = counts[failed - 1] if failed else 0
            if not before <= len(got) <= counts[failed]:
                problems.append(f"op {at}: step {failed}: {len(got)} lines of {before}..")
        pending = again.state.pending_calls()
        if pending:
            again.record_tool(pending[0], "answered after the fault")
        else:
            again.record_user("after the fault")
        again.close()
        if len(_lines(root)) != len(got) + 1:
            problems.append(f"op {at}: step {failed}: the reopened session took no line")
    assert ops > 40, "the script exercises the store's writes"
    assert problems == []


FOLDS_INTO: Final[Mapping[str, frozenset[str]]] = {
    "turn": frozenset({"Turn"}),
    # Zeus's shell's D40 records, folded for the jobs they open and close.
    "shell_effect": frozenset({"EffectRecord"}),
    # A spawned agent's lines, kept in order for `agents/`.
    "agent": frozenset({"AgentRecord"}),
    "tool": frozenset({"ToolLine"}),
    "episode": frozenset({"EpisodeLine"}),
    "workspace": frozenset({"WorkspaceLine"}),
    "work": frozenset({"WorkLine"}),
    # Zeus's messages to the team, kept in order for the run that delivers them.
    "tell": frozenset({"TellLine"}),
    "compaction": frozenset({"CutLine"}),
    "fork": frozenset({"ForkLine"}),
    "constraint": frozenset({"ConstraintLine"}),
    "rejected": frozenset({"RejectedAlt", "Reconsideration"}),
    "concern": frozenset({"Concern", "Touched", "Promoted"}),
    "frame": frozenset({"FrameLine"}),
    "spend": frozenset({"SpendLine"}),
    "cap": frozenset({"CapLine"}),
    "model_switched": frozenset({"OtherLine"}),
    "pinned": frozenset({"OtherLine"}),
    "title": frozenset({"OtherLine"}),
    "handoff_intended": frozenset({"OtherLine"}),
    "handoff_completed": frozenset({"OtherLine"}),
    "undo": frozenset({"UndoLine"}),
    "redo": frozenset({"RedoLine"}),
    **{
        kind: frozenset({"OtherLine"})
        for kind in (
            "packet_record",
            "design_record",
            "summary",
            "note",
            "plan",
            "limits",
            "box_compaction",
            "prefix",
            "bound",
            "scout",
            "box_call",
            "code_snapshot",
            "question_asked",
            "question_answered",
            "plan_decision",
            "skill_loaded",
            "skill_unloaded",
            "untrusted_read",
            "command_grant",
            "action_granted",
            "hook",
            "external_call",
        )
    },
}
"""What each kind folds into: a parsed kind into its own record, a kind another package owns
into `OtherLine`, kept unread. Pinned before `check` and `apply` read one table."""


def test_each_kind_folds_into_its_record_and_a_foreign_kind_is_kept_unread(
    session: Session, ws: Path
) -> None:
    write_every_kind(session, ws)
    at = session.state.last_seq
    with Session.fork(
        session.root, SESSION_ID, at, "another way", "S-child", code="keep"
    ) as child:
        child.record_user("in the child")
    folded: dict[str, set[str]] = {}
    for sid in (SESSION_ID, "S-child"):
        for event, record in fold(session.root, sid).lines():
            folded.setdefault(event.kind, set()).add(type(record).__name__)
            if isinstance(record, OtherLine):
                assert record.kind == event.kind and record.seq == event.seq
    assert folded == {kind: set(names) for kind, names in FOLDS_INTO.items()}
    assert set(FOLDS_INTO) == set(get_args(SessionKind))


def test_the_kind_table_folds_every_parsed_kind_and_only_those() -> None:
    """Totality: `check` and `apply` read one table, so a kind `lines.py` parses and
    the table lacks would fail on its first line. Every other kind is foreign, kept unread."""
    table = SessionState._FOLDS  # pyright: ignore[reportPrivateUsage]
    assert set(table) == set(PARSED)
    assert {k for k, names in FOLDS_INTO.items() if names != {"OtherLine"}} == set(PARSED)


def a_workspace() -> WorkspaceRef:
    return WorkspaceRef(
        mode="worktree",
        lease_id="L-1",
        source_ref="main",
        source_head="0" * 40,
        branch="chickadee/run",
        worktree_path="/tmp/wt",
        handoff="pending",
    )


@pytest.mark.parametrize(
    ("kind", "line"), [("turn", "workspace"), ("note", "workspace"), ("workspace", None)]
)
def test_a_line_of_another_kinds_type_is_a_harness_bug(
    kind: SessionKind, line: str | None
) -> None:
    """Only a caller bypassing `append` and replay could pair a kind with another's payload,
    since both hand `check` the payload `parse` built for that kind."""
    payload = a_workspace() if line else None
    with pytest.raises(HarnessBug, match=f"a {kind} line holds a"):
        SessionState("S-x").check(kind, payload, seq=1, run_id=None)


def test_a_kind_no_package_writes_has_no_fold() -> None:
    with pytest.raises(PayloadError, match="no fold for line kind 'gossip'"):
        SessionState("S-x").check(cast(SessionKind, "gossip"), None, seq=1, run_id=None)


def test_one_blob_written_from_many_threads_at_once_lands_whole(tmp_path: Path) -> None:
    """Parallel agents store the same bytes at once: two of them run on one
    skill open on one prompt. The blob store has one writer, so `put_blob` takes the session's
    lock: without it the threads share one temp file and the loser's rename fails."""
    session = Session.create(tmp_path / "sessions", "S-blobs")
    failed: list[BaseException] = []

    def write() -> None:
        try:
            for n in range(100):
                session.put_blob(b"the same prompt %d" % (n % 5))
        except BaseException as exc:
            failed.append(exc)

    try:
        threads = [threading.Thread(target=write) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert failed == []
        assert session.blob(session.put_blob(b"the same prompt 0")) == b"the same prompt 0"
    finally:
        session.close()
