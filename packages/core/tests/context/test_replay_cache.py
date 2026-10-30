"""`ReplayCache` advances a grown log to what a full replay folds.

Each test lays a recorded log down one prefix at a time, every byte length of it, so every
torn tail is seen, and asks one long-lived cache and a full replay the same question. A fork's
log is laid down over its parent's, so a fork chain is advanced too. A refusal is compared as
well as a state: the same exception, at the same seq.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from context_helpers import exchange, new_session, sessions_dir, work_unit

from chickadee.context.ledger import note_constraint
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session, SessionState, parsed
from chickadee.context.store import LOG_NAME, SessionLog, SessionStoreError
from chickadee.schema.adapter import Usage


def write_parent(tmp_path: Path) -> int:
    """A root session with lines of many kinds; returns the seq a child forks at."""
    with new_session(tmp_path, "S-root") as root:
        said = root.record_user("keep the parser pure")
        exchange(root, "q1", "shell", {"command": "ls"}, "a.py\nb.py\n")
        assert note_constraint(root, "the parser is pure", [said.seq], "session") == "C-1"
        root.record_spend(
            "high-param-model", Usage(prompt_tokens=10, completion_tokens=2), 0.5, "high-param"
        )
        at = root.state.last_seq
        work_unit(root, "alpha")
        root.record_cap(4.0)
        return at


def write_child(tmp_path: Path, at: int) -> None:
    with Session.fork(
        sessions_dir(tmp_path), "S-root", at, "another way", "S-child", code="keep"
    ) as child:
        child.record_user("and in the child")
        exchange(child, "q2", "shell", {"command": "pwd"}, "/ws\n")
        work_unit(child, "beta")


Outcome = tuple[str, object]


def folded(read: Callable[[], tuple[SessionLog, SessionState]]) -> Outcome:
    """What a read shows: every line and its record, and the last seq; or its refusal."""
    try:
        log, state = read()
    except SessionStoreError as exc:
        return type(exc).__name__, str(exc)
    return "state", (log.events(), tuple(state.lines()), state.last_seq)


def full(root: Path, session_id: str) -> tuple[SessionLog, SessionState]:
    log = SessionLog.read(root, session_id)
    return log, SessionState.replay(session_id, root, log.events(), log.payload)


def lay(path: Path, data: bytes) -> None:
    """Write `data` into the same file, as appends would have left it."""
    with path.open("r+b") as fh:
        fh.write(data)
        fh.truncate()


def every_prefix(root: Path, session_id: str, data: bytes) -> None:
    cache = ReplayCache(root)
    log = root / session_id / LOG_NAME
    for n in range(len(data) + 1):
        lay(log, data[:n])
        assert folded(lambda: cache.get(session_id)) == folded(
            lambda: full(root, session_id)
        ), n


def test_every_prefix_of_a_log_advances_to_a_full_replay(tmp_path: Path) -> None:
    write_parent(tmp_path)
    root = sessions_dir(tmp_path)
    every_prefix(root, "S-root", (root / "S-root" / LOG_NAME).read_bytes())


def test_every_prefix_of_a_forks_log_advances_to_a_full_replay(tmp_path: Path) -> None:
    write_child(tmp_path, write_parent(tmp_path))
    root = sessions_dir(tmp_path)
    every_prefix(root, "S-child", (root / "S-child" / LOG_NAME).read_bytes())


@pytest.mark.parametrize(
    "appended",
    [
        b"not json\n",
        b'{"seq": 99, "at": "x", "kind": "turn", "payload_sha": "%s", "run_id": null}\n'
        % (b"0" * 64),
        b'{"seq": %d, "at": "2026-10-05T00:00:00Z", "kind": "turn", "payload_sha": "%s", '
        b'"run_id": null}\n',
    ],
    ids=["unparseable", "seq-gap", "missing-blob"],
)
def test_a_damaged_appended_line_is_refused_at_a_full_replays_seq(
    tmp_path: Path, appended: bytes
) -> None:
    write_parent(tmp_path)
    root = sessions_dir(tmp_path)
    log = root / "S-root" / LOG_NAME
    cache = ReplayCache(root)
    _, before = cache.get("S-root")
    if b"%d" in appended:
        appended = appended % (before.last_seq + 1, b"f" * 64)
    with log.open("ab") as fh:
        fh.write(appended)
    refused = folded(lambda: cache.get("S-root"))
    assert refused[0] == "SessionCorrupt"
    assert refused == folded(lambda: full(root, "S-root"))


def test_a_line_the_fold_refuses_is_refused_at_a_full_replays_seq(tmp_path: Path) -> None:
    """A second result for a call already answered: it parses, and only the fold refuses it."""
    write_parent(tmp_path)
    root = sessions_dir(tmp_path)
    log = root / "S-root" / LOG_NAME
    cache = ReplayCache(root)
    _, before = cache.get("S-root")
    lines = log.read_bytes().splitlines(keepends=True)
    tool = next(line for line in lines if b'"kind":"tool"' in line.replace(b" ", b""))
    again = json.loads(tool)
    again["seq"] = before.last_seq + 1
    with log.open("ab") as fh:
        fh.write(json.dumps(again).encode() + b"\n")
    refused = folded(lambda: cache.get("S-root"))
    assert refused[0] == "SessionCorrupt"
    assert "is waiting" in str(refused[1])
    assert refused == folded(lambda: full(root, "S-root"))


def test_a_state_handed_out_never_changes_after_a_later_advance(tmp_path: Path) -> None:
    write_parent(tmp_path)
    root = sessions_dir(tmp_path)
    cache = ReplayCache(root)
    log, state = cache.get("S-root")
    held = (log.events(), tuple(state.lines()), state.last_seq)
    with Session.open(root, "S-root") as session:
        session.record_user("later")
    later_log, later = cache.get("S-root")
    assert later is not state and later_log is not log
    assert (log.events(), tuple(state.lines()), state.last_seq) == held
    assert later.last_seq == state.last_seq + 1


def test_lines_parsed_once_fold_alike_twice(tmp_path: Path) -> None:
    """A parsed line is reused, never changed, so one parse serves every refold."""
    write_parent(tmp_path)
    root = sessions_dir(tmp_path)
    log = SessionLog.read(root, "S-root")
    lines = [parsed(root, event, log.payload) for event in log.events()]
    first = SessionState.fold("S-root", root, lines)
    second = SessionState.fold("S-root", root, lines)
    assert first is not second
    assert tuple(first.lines()) == tuple(second.lines())
    assert first.last_seq == second.last_seq


def test_a_replaced_or_shorter_log_is_read_whole(tmp_path: Path) -> None:
    write_parent(tmp_path)
    root = sessions_dir(tmp_path)
    log = root / "S-root" / LOG_NAME
    data = log.read_bytes()
    cache = ReplayCache(root)
    cache.get("S-root")
    first = b"".join(data.splitlines(keepends=True)[:3])
    lay(log, first)
    assert folded(lambda: cache.get("S-root")) == folded(lambda: full(root, "S-root"))
    assert cache.get("S-root")[1] is cache.get("S-root")[1], "the shorter log is cached again"
    replacement = log.with_name("next")
    replacement.write_bytes(data)
    replacement.replace(log)
    assert folded(lambda: cache.get("S-root")) == folded(lambda: full(root, "S-root"))
