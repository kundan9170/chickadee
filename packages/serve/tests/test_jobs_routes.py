"""The person's view of jobs over the wire: `GET /session/{sid}/jobs` lists
them from their effects, `POST …/jobs/{job_id}/kill` ends one and writes nothing itself, and a
job's exit reaches a following client as the harness turn it is, with no new record kind.

A live engine, its zeus rung scripted, no sandbox, the person having lifted the unsandboxed
floor: the job is what is under test.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from engines import Step, answer, call, calls, config_for, raw_config, start, with_sessions
from repos import init_repo
from serve_helpers import Live, served_engine

from chickadee import engine as engine_module
from chickadee.isolation.none import Passthrough
from chickadee_serve.frames import WIRE_VERSION

SID = "S-jobs"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


@contextmanager
def live(monkeypatch: pytest.MonkeyPatch, repo: Path, script: list[Step]) -> Iterator[Live]:
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    raw = raw_config()
    raw["sandbox"] = {"allow_unsandboxed_shell": {"zeus": True}, "require_sandbox": False}
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    try:
        with served_engine(engine) as h:
            yield h
    finally:
        engine.close()


def test_the_jobs_route_lists_a_running_job_and_the_kill_route_ends_it(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    script: list[Step] = [
        calls(call("shell", "s1", command="sleep 60", background=True)),
        answer("j1 is up"),
    ]
    with live(monkeypatch, repo, script) as h:
        h.client.post(f"/session/{SID}/turn", json={"text": "start it"}, headers=h.headers())
        h.until(SID, "turn", count=3)  # the words, the call, the answer
        listed = h.client.get(f"/session/{SID}/jobs", headers=h.headers())
        assert listed.status_code == 200, listed.text
        (job,) = listed.json()
        assert (job["job_id"], job["kind"], job["running"]) == ("j1", "task", True)
        assert job["state"] == "running"
        killed = h.client.post(f"/session/{SID}/jobs/j1/kill", headers=h.headers())
        assert killed.status_code == 202, killed.text
        deadline = time.monotonic() + 20
        (ended,) = h.client.get(f"/session/{SID}/jobs", headers=h.headers()).json()
        while ended["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
            (ended,) = h.client.get(f"/session/{SID}/jobs", headers=h.headers()).json()
        assert not ended["running"]
        assert (ended["report"]["ended"], ended["report"]["by"]) == ("killed", "person")
        # Worded by the engine, as `job list` words it for Zeus: no client words it again.
        assert ended["state"].startswith("killed by person after ")
        assert ended["row"].startswith("j1 · task · killed by person after ")
        assert not ended["report"]["notify"]
        again = h.client.post(f"/session/{SID}/jobs/j1/kill", headers=h.headers())
        assert again.status_code >= 400, "a job that is not running is not killed twice"


def test_a_job_s_exit_reaches_the_stream_as_a_harness_turn(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    major, minor = (int(n) for n in WIRE_VERSION.split("."))
    assert (major, minor) >= (4, 2), "the routes and JobView came with 4.2, a minor step (D201)"
    script: list[Step] = [
        calls(call("shell", "s1", command="sleep 1; echo ok", background=True)),
        answer("started"),
        answer("j1 passed"),
    ]
    with live(monkeypatch, repo, script) as h:
        h.client.post(f"/session/{SID}/turn", json={"text": "run it"}, headers=h.headers())
        h.until(SID, "turn", count=5)  # words, call, answer, the notice, the wake's answer
        turns = [f.payload for f in h.journal(SID) if f.kind == "turn"]
        notices = [t for t in turns if t.get("role") == "harness"]
        assert [t.get("trigger") for t in notices] == [8]
        assert "job j1" in str(notices[0].get("text")) and turns[-1].get("text") == "j1 passed"
