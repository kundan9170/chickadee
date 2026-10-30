"""BatchIO parity: one turn followed to its end without a REPL, on the
wire; `--json` is the journal's own bytes; a question is answered by default, by a person, or
stops the turn, never answered silently; and the host-side `replay` and `gc` exit by §6a."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from engines import Step, answer, call, calls
from hosting import SID, hosted, repository, short_root
from recordings import install_run
from repos import init_repo

from chickadee.client.errors import TransportExhausted
from chickadee.client.request import ChatRequest
from chickadee.journal.paths import runs_root
from chickadee.schema.adapter import Completion
from chickadee.state.roots import roots
from chickadee_cli import gc, replay
from chickadee_cli.batch import BatchIO, follow, prepend
from chickadee_cli.render.style import Style
from chickadee_cli.surface import DONE, STOPPED
from chickadee_serve.frames import RecordFrame, TokenClass, encode_frame
from chickadee_wire.client import Client
from chickadee_wire.errors import Disconnected
from chickadee_wire.events import Frame

ASKS: list[Step] = [calls(call("ask_user", "q1", question="which branch?")), answer("on it")]


class Captured:
    """An output stream, kept as lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _batch(*, json_out: bool = False) -> tuple[BatchIO, Captured, Captured]:
    out, err = Captured(), Captured()
    return BatchIO(json=json_out, out=out, err=err, style=Style(tty=False)), out, err


def test_a_turn_is_followed_until_the_session_is_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    how, out, err = _batch()
    with hosted(monkeypatch, [answer("all done")]) as host, host.client() as client:
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="tidy up") == DONE
        assert client.liveness(SID).kind == "idle"
    assert "> tidy up" in out.text and "all done" in out.text
    assert err.lines == []


def test_json_is_the_journals_bytes_and_people_read_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    how, out, err = _batch(json_out=True)
    with hosted(monkeypatch, [answer("all done")]) as host, host.client() as client:
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="tidy up") == DONE
        journal = [
            encode_frame(f).decode("utf-8").rstrip("\n")
            for f in client.session_journal(SID, after)
        ]
    assert out.lines == journal, "stdout is the record stream, nothing else"
    assert all(RecordFrame.model_validate(json.loads(line)) for line in out.lines)
    assert "all done" in err.text, "the human lines go to stderr"


def test_yes_answers_a_question_with_its_default_as_automation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        hosted(monkeypatch, ASKS) as host,
        host.client() as client,
        host.client(TokenClass.AUTOMATION) as answerer,
    ):
        out, err = Captured(), Captured()
        how = BatchIO(json=False, out=out, err=err, style=Style(tty=False), answerer=answerer)
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="go") == DONE
    assert "? which branch?" in out.text
    assert "answered q1: no answer: proceed on the default the question states, by default" in (
        out.text
    ), "an automation token's answer is recorded as the default it agreed with (§6.1)"
    assert "on it" in out.text


def test_a_person_answers_on_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    class Terminal:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def read(self, prompt_text: str, default: str = "") -> str:
            self.asked.append(prompt_text)
            return "main"

    terminal = Terminal()
    with hosted(monkeypatch, ASKS) as host, host.client() as client:
        out, err = Captured(), Captured()
        how = BatchIO(json=False, out=out, err=err, style=Style(tty=False), source=terminal)
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="go") == DONE
    assert terminal.asked == ["? "], "asked once, not again while its answer is in flight"
    assert "answered q1: main, by human" in out.text


def test_a_question_nobody_can_answer_stops_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    how, out, _err = _batch()
    with hosted(monkeypatch, ASKS) as host, host.client() as client:
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="go") == STOPPED
        assert client.liveness(SID).kind != "awaiting"
    assert "rerun with --yes" in out.text
    assert "on it" not in out.text, "never answered by default without --yes"


def test_a_turn_failing_before_its_reply_ends_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model unreachable: the turn ends with no record of it. Liveness reads idle with the
    reason, so the command exits 2 at once and says why."""

    def unreachable(_request: ChatRequest) -> Completion:
        raise TransportExhausted("connection refused", 3)

    how, out, _err = _batch()
    with hosted(monkeypatch, [unreachable]) as host, host.client() as client:
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="hello") == STOPPED
    assert "connection refused" in out.text


def test_piped_input_comes_before_the_goal() -> None:
    assert prepend("diff --git a b\n+x\n", "review this") == "diff --git a b\n+x\n\nreview this"
    assert prepend("", "review this") == "review this"


# -- host-side commands ----------------------------------------------------------------------


def _repo_with(monkeypatch: pytest.MonkeyPatch, run_id: str) -> tuple[Path, Path]:
    """A repository holding the recorded task run (`recordings.RUN_FIXTURE`) as
    run `run_id`: the packet loop's recorded journals are v1, which no reader takes since
    D196."""
    repo = repository(monkeypatch, short_root())
    return repo, install_run(runs_root(roots(repo)), run_id)


def test_replay_folds_a_clean_journal_and_says_it_did_not_re_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _journal = _repo_with(monkeypatch, "R-1")
    code, text = replay.run(repo, "R-1")
    assert code == replay.IDENTICAL, text
    assert "folded, never re-executed" in text and "identical" in text


def test_replay_names_each_defect_and_exits_diverged(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, journal = _repo_with(monkeypatch, "R-1")
    lines = journal.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    after_end = {**last, "seq": last["seq"] + 1}
    journal.write_text("\n".join([*lines, json.dumps(after_end)]) + "\n", encoding="utf-8")
    code, text = replay.run(repo, "R-1")
    assert code == replay.DIVERGED, text
    assert "event follows run_end" in text


def test_replay_of_no_run_or_an_unreadable_one(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, journal = _repo_with(monkeypatch, "R-1")
    assert replay.run(repo, "R-none")[0] == replay.NO_SUCH_RUN
    lines = journal.read_text(encoding="utf-8").splitlines()
    journal.write_text("\n".join([lines[0], "{not json", *lines[1:]]) + "\n", encoding="utf-8")
    assert replay.run(repo, "R-1")[0] == replay.UNREADABLE


def test_gc_dry_run_removes_nothing_and_refuses_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, journal = _repo_with(monkeypatch, "R-1")
    before = sorted(p.name for p in journal.parent.rglob("*"))
    code, text = gc.run(repo, older_than_days=0, classes=(), run_id=None, dry_run=True)
    assert code == 0, text
    assert sorted(p.name for p in journal.parent.rglob("*")) == before
    code, text = gc.run(repo, older_than_days=0, classes=("record",), run_id=None, dry_run=True)
    assert code == 1 and "never pruned" in text


def test_ctrl_c_while_the_turn_is_sent_stops_it_with_2(monkeypatch: pytest.MonkeyPatch) -> None:
    stops: list[bool] = []

    def interrupted(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    def interrupt(_self: Client, _sid: str, *, hard: bool = False) -> None:
        stops.append(hard)

    monkeypatch.setattr(Client, "turn", interrupted)
    monkeypatch.setattr(Client, "interrupt", interrupt)
    how, out, _err = _batch()
    with hosted(monkeypatch) as host, host.client() as client:
        assert follow(client, SID, how, after=0, send="go") == STOPPED
    assert stops == [False], "the turn is stopped at its boundary, not left running"
    assert "stopping the turn" in out.text


def test_a_lost_session_stream_ends_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(*_args: object, **_kwargs: object) -> Iterator[Frame]:
        raise Disconnected("the stream dropped more times than its budget allows")
        yield  # pragma: no cover

    monkeypatch.setattr(Client, "session_events", gone)
    how, out, _err = _batch()
    with hosted(monkeypatch, [answer("done")]) as host, host.client() as client:
        assert follow(client, SID, how, after=0, send="go") == STOPPED
    assert "lost the session's stream" in out.text


def test_gc_refuses_a_negative_age(tmp_path: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from chickadee_cli.main import main; main()",
            *("gc", "--older-than", "-1", "--dry-run"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1, done.stderr


def test_a_config_that_does_not_load_names_the_next_step(tmp_path: Path) -> None:
    """No engine config at all is the first thing a new person meets."""
    repo, home = init_repo(tmp_path / "r", commit=None), tmp_path / "xdg"
    home.mkdir()
    done = subprocess.run(
        [sys.executable, "-c", "from chickadee_cli.main import main; main()", "ask", "hi"],
        cwd=repo,
        env={**os.environ, "XDG_CONFIG_HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1, done.stderr
    assert "the config does not load" in done.stderr
    assert "`chickadee setup` writes the engine config's template" in done.stderr
