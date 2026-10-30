"""The app over a live engine, a scripted model behind its zeus rung: a turn posted over the
socket runs, its records are read back by seq, a question is put and answered over the wire, and
the reads stage B added answer from the engine."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from engines import Step, answer, call, calls, config_for, raw_config, start, with_sessions
from repos import init_repo
from serve_helpers import Live, served_engine

from chickadee_serve.auth import TokenClass

SID = "S-live"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


@contextmanager
def live(
    monkeypatch: pytest.MonkeyPatch, repo: Path, script: Sequence[Step] = ()
) -> Iterator[Live]:
    engine, _ = start(monkeypatch, repo, config_for(raw_config()), script)
    try:
        with served_engine(engine) as h:
            yield h
    finally:
        engine.close()


def _texts(h: Live, sid: str = SID) -> list[str]:
    return [str(f.payload.get("text")) for f in h.journal(sid) if f.kind == "turn"]


def test_a_turn_over_the_socket_runs_and_reads_back_by_seq(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    with live(monkeypatch, repo, [answer("hello back")]) as h:
        r = h.client.post(f"/session/{SID}/turn", json={"text": "hello"}, headers=h.headers())
        assert r.status_code == 202, r.text
        h.until(SID, "turn", count=2)
        assert _texts(h) == ["hello", "hello back"]
        frames = h.journal(SID)
        seqs = [f.seq for f in frames]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        later = h.journal(SID, after=seqs[0])
        assert [f.seq for f in later] == seqs[1:], "a cursor resumes after itself"


def test_a_question_is_answered_over_the_wire_and_journaled_as_human(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    script: list[Step] = [calls(call("ask_user", "q1", question="which branch?")), answer()]
    with live(monkeypatch, repo, script) as h:
        h.client.post(f"/session/{SID}/turn", json={"text": "go"}, headers=h.headers())
        asked = h.until(SID, "question_asked")
        question = asked.payload["question"]
        assert isinstance(question, dict)
        qid = str(question["id"])
        liveness = h.client.get(f"/session/{SID}/liveness", headers=h.headers()).json()
        assert (liveness["kind"], liveness["question_id"]) == ("awaiting", qid)
        r = h.client.post(
            f"/session/{SID}/answer/{qid}",
            json={"choice": question["default"], "note": "main"},
            headers=h.headers(TokenClass.INTERACTIVE),
        )
        assert r.status_code == 200, r.text
        answered = h.until(SID, "question_answered")
    assert (answered.payload["id"], answered.payload["source"]) == (qid, "human")
    assert answered.payload["note"] == "main"


def test_the_session_reads_answer_from_the_engine(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    with live(monkeypatch, repo, [answer("hi")]) as h:
        h.client.post(
            f"/session/{SID}/turn", json={"text": "fix the fetcher"}, headers=h.headers()
        )
        h.until(SID, "turn", count=2)
        listed = h.client.get("/session", headers=h.headers()).json()
        assert [(s["session_id"], s["title"]) for s in listed] == [(SID, "fix the fetcher")]
        info = h.client.get(f"/session/{SID}", headers=h.headers()).json()
        assert info["title"] == "fix the fetcher"
        hidden = h.client.get(f"/session/{SID}/hidden", headers=h.headers()).json()
        assert hidden == {"seqs": []}
        idle = h.client.get(f"/session/{SID}/liveness", headers=h.headers()).json()
        assert idle["kind"] == "idle"
        missing = h.client.get("/session/S-nobody", headers=h.headers())
        assert missing.status_code == 404, missing.text
