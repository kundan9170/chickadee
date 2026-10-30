"""`/jobs` and `/jobs kill`, and a job's exit in the stream: the REPL over a
hosted engine, its model scripted, as `test_repl_turns.py` drives it."""

from __future__ import annotations

import shutil

import pytest
from engines import Step, answer, call, calls
from repl_driver import Typed, printed, repl

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None and shutil.which("sandbox-exec") is None,
    reason="a sealed shell command needs a sandbox backend, as CI installs one",
)


def test_jobs_lists_a_running_job_and_kill_ends_it(monkeypatch: pytest.MonkeyPatch) -> None:
    script: list[Step] = [
        calls(call("shell", "s1", command="sleep 60", background=True)),
        answer("j1 is up"),
    ]
    lines = [
        Typed("start the server"),
        Typed("/jobs", after=printed("j1 is up")),
        Typed("/jobs kill j1", after=printed("j1 · task · running")),
        Typed("/exit", after=printed("j1 killed")),
    ]
    with repl(monkeypatch, script, lines) as (source, code):
        out = "\n".join(source.out)
    assert code == 0
    assert "j1 is up" in out
    assert "j1 · task · running" in out and "j1 killed" in out


def test_a_job_s_exit_shows_as_a_job_line_and_the_turn_it_woke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list[Step] = [
        calls(call("shell", "s1", command="sleep 1; echo all green", background=True)),
        answer("started"),
        answer("the suite passed"),
    ]
    lines = [
        Typed("run the suite"),
        Typed("/exit", after=printed("the suite passed")),
    ]
    with repl(monkeypatch, script, lines) as (source, _code):
        out = "\n".join(source.out)
    assert "[job] job j1 · task" in out and "all green" in out
    assert out.index("[job] job j1") < out.index("the suite passed")


def test_jobs_with_no_jobs_says_so_and_a_bad_line_says_its_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [Typed("/jobs"), Typed("/jobs stop j1"), Typed("/exit")]
    with repl(monkeypatch, [], lines) as (source, _code):
        out = "\n".join(source.out)
    assert "no jobs in this session" in out and "usage: /jobs [kill <id>]" in out
