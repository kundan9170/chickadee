"""The replay: a client following a replay ends holding what one reading the recording holds;
the replay waits at a question as live; an answer never touches the recording; mutating routes
refuse."""

from __future__ import annotations

import time
from pathlib import Path

from demo_session import FIXTURE, QUESTION_ID, SESSION_ID, roots
from serve_helpers import recorded_lines, replay

from chickadee.context.session import SessionState
from chickadee.context.store import SessionLog, sessions_root


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_replay_waits_at_a_question_and_goes_on_when_answered(tmp_path: Path) -> None:
    with replay(tmp_path) as h:
        time.sleep(0.3)
        assert h.facade.horizon == 4, "nothing after the question is revealed before an answer"
        assert h.served.pending.waiting(SESSION_ID) == (QUESTION_ID,)
        r = h.client.post(
            f"/session/{SESSION_ID}/answer/{QUESTION_ID}",
            json={"choice": "yes"},
            headers=h.headers(),
        )
        assert r.status_code == 200
        deadline = time.monotonic() + 5
        while not h.facade.finished:
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_an_answer_goes_to_scratch_never_the_recording(tmp_path: Path) -> None:
    before = _files(FIXTURE)
    with replay(tmp_path) as h:
        while QUESTION_ID not in h.served.pending.waiting(SESSION_ID):
            time.sleep(0.01)
        h.client.post(
            f"/session/{SESSION_ID}/answer/{QUESTION_ID}",
            json={"choice": "no"},
            headers=h.headers(),
        )
        while not h.facade.finished:
            time.sleep(0.01)
        assert '"choice": "no"' in (h.scratch / "answers.jsonl").read_text()
    assert _files(FIXTURE) == before


def test_the_state_a_client_reaches_is_the_recordings(tmp_path: Path) -> None:
    """A client renders the session from its records and asks what is hidden: it reaches the
    state a direct reader of the recording reaches."""
    log = SessionLog.read(sessions_root(roots(FIXTURE)), SESSION_ID)
    direct = SessionState.replay(
        SESSION_ID, sessions_root(roots(FIXTURE)), log.events(), log.payload
    )
    with replay(tmp_path) as h:
        while QUESTION_ID not in h.served.pending.waiting(SESSION_ID):
            time.sleep(0.01)
        h.client.post(
            f"/session/{SESSION_ID}/answer/{QUESTION_ID}",
            json={"choice": "yes"},
            headers=h.headers(),
        )
        while not h.facade.finished:
            time.sleep(0.01)
        seqs = [r.seq for r in h.journal(0)]
        hidden = h.client.get(f"/session/{SESSION_ID}/hidden", headers=h.headers()).json()[
            "seqs"
        ]
    assert seqs == [d["seq"] for d in recorded_lines()]
    assert hidden == sorted(direct.hidden) and hidden, "the demo's undo hides its last exchange"


def test_mutating_routes_refuse_and_reading_ones_answer(tmp_path: Path) -> None:
    with replay(tmp_path) as h:
        for method, path, body in [
            ("POST", f"/session/{SESSION_ID}/turn", {"text": "hi"}),
            ("POST", f"/session/{SESSION_ID}/interrupt", {}),
            ("POST", f"/session/{SESSION_ID}/coder", {"request": {}}),
            ("POST", f"/session/{SESSION_ID}/finalize", {"choice": "keep"}),
            ("POST", f"/session/{SESSION_ID}/cap", {"hard_cost_usd": 5}),
            ("POST", "/session", {}),
        ]:
            r = h.client.request(method, path, json=body, headers=h.headers())
            assert r.status_code == 405, (path, r.text)
            assert r.json()["code"] == "REPLAY_READ_ONLY"
        assert (
            h.client.get("/session", headers=h.headers()).json()[0]["session_id"] == SESSION_ID
        )
        hello = h.client.get("/").json()
        assert hello["replay"] is True and "replay" in hello["capabilities"]
