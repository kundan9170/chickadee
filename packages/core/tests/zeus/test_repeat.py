"""The repetition detector.

Three identical shell calls at one generation: the third result carries `repetition{count:3}`;
the sixth is `BOUND_REACHED{repetition}`; a different call between them resets; the same calls
with a generation bump between the second and third do not count; three `coder(design)` calls
with the same design text and no bump are the same key; `zeus_repeat == 0` never
fires. The commands run for real, unwrapped (see `zeus_helpers.rig`)."""

from __future__ import annotations

from pathlib import Path

from refusals import stated

from chickadee.context.session import SessionState
from chickadee.context.work_line import WorkLine
from chickadee.schema import ToolCall, sha256_hex
from chickadee.work import record_work
from chickadee.zeus.loop import LoopConfig

from .zeus_helpers import answer, call, calls, payloads, rig


def _poll(tmp_path: Path, cid: str) -> ToolCall:
    """The same command every time; each run appends a line, so the file counts dispatches. In
    the workspace: under the lift a redirect outside it is asked."""
    counter = tmp_path / "ws" / "runs.txt"
    return call("shell", cid, command=f"echo run >> {counter}")


def _runs(tmp_path: Path) -> int:
    counter = tmp_path / "ws" / "runs.txt"
    return len(counter.read_text().splitlines()) if counter.exists() else 0


def _still(_: SessionState) -> int:
    """A workspace that does not move — what the engine's counter reads when nothing lands."""
    return 7


def test_the_third_identical_call_says_so_and_the_sixth_is_refused(tmp_path: Path) -> None:
    script = [calls(_poll(tmp_path, f"p{i}")) for i in range(1, 7)] + [
        answer("it never came up")
    ]
    r = rig(tmp_path, script, shell=True, generation=_still)
    events = r.run("start the server and wait for it")
    state = r.session.state
    texts = {i: state.result(f"p{i}").text for i in range(1, 7)}  # type: ignore[union-attr]
    assert "repetition" not in texts[1] and "repetition" not in texts[2]
    for n in (3, 4, 5):
        assert f"repetition{{count: {n}, since_generation: 7}}" in texts[n]
    assert stated(texts[6]) == (
        "BOUND_REACHED{resource: repetition, used: 6, limit: 6, scope: turn, "
        "reason: 'identical call, world unchanged'}"
    )
    assert _runs(tmp_path) == 5, "the refused sixth call was not dispatched"
    assert state.ran("p5") and not state.ran("p6")
    assert events[-1].kind == "answer", "the turn goes on in prose"
    bounds = payloads(r.session, "bound")
    assert [(b["resource"], b["outcome"]) for b in bounds] == [
        ("repetition", "noticed"),
        ("repetition", "refused"),
    ]
    assert bounds[0]["used"] == 3 and bounds[0]["tool"] == "shell"
    assert bounds[0]["generation"] == 7 and bounds[0]["args_digest"] == bounds[1]["args_digest"]


def test_a_different_call_between_resets_the_count(tmp_path: Path) -> None:
    other = call("shell", "x", command="echo other")
    seq = [_poll(tmp_path, "a1"), _poll(tmp_path, "a2"), other]
    seq += [_poll(tmp_path, f"b{i}") for i in range(1, 4)]
    r = rig(tmp_path, [calls(c) for c in seq], shell=True, generation=_still)
    r.run("poll")
    state = r.session.state
    assert "repetition" not in state.result("b2").text  # type: ignore[union-attr]
    assert "repetition{count: 3" in state.result("b3").text  # type: ignore[union-attr]


def test_a_generation_bump_between_identical_calls_is_a_recheck(tmp_path: Path) -> None:
    moved = {"at": 0}

    def generation(_: SessionState) -> int:
        return moved["at"]

    def bump_then(c: ToolCall):  # a response after the world moved
        def step(_request: object):
            moved["at"] += 1
            return calls(c)

        return step

    seq = [
        calls(_poll(tmp_path, "p1")),
        calls(_poll(tmp_path, "p2")),
        bump_then(_poll(tmp_path, "p3")),
        calls(_poll(tmp_path, "p4")),
    ]
    r = rig(tmp_path, seq, shell=True, generation=generation)
    r.run("rebuild and check")
    state = r.session.state
    for cid in ("p1", "p2", "p3", "p4"):
        assert "repetition" not in state.result(cid).text, cid  # type: ignore[union-attr]
    assert payloads(r.session, "bound") == []


def test_a_coder_episode_between_identical_calls_is_a_recheck(tmp_path: Path) -> None:
    """With the loop's own generation, an episode moves it (and is a different call)."""
    delegate = call("coder", "c1", title="fix the flaky test", brief="fix the flaky test")
    seq = [_poll(tmp_path, "p1"), _poll(tmp_path, "p2"), delegate, _poll(tmp_path, "p3")]
    r = rig(tmp_path, [calls(c) for c in seq], shell=True)
    r.run("fix it and re-run")
    assert "repetition" not in r.session.state.result("p3").text  # type: ignore[union-attr]
    assert len(r.session.state.episodes()) == 1


def test_three_delegations_of_one_design_are_one_key(tmp_path: Path) -> None:
    """`coder`'s digest is the work it names, however the call around it is worded; its
    files are the workspace's generation (was the design's text)."""
    work = "2026-10-18-rate-limiter"
    delegations = [
        call("coder", "c1", work=work),
        call("coder", "c2", work=work),
        call("coder", "c3", work=work, workspace="worktree"),
    ]
    r = rig(tmp_path, [calls(c) for c in delegations], generation=_still)
    record_work(r.session, WorkLine(name=work, dir="docs/chickadee", state="open"))
    r.run("build the rate limiter")
    state = r.session.state
    assert len(r.box.started) == 3, "a notice is served, not refused"
    assert "repetition{count: 3, since_generation: 7}" in state.result("c3").text  # type: ignore[union-attr]
    # Each episode names what its stop showed (the fake box's: the delegation's text).
    sha = sha256_hex(r.box.started[0].text)
    assert [sha256_hex(d.text) for d in r.box.started] == [sha] * 3
    assert [e.design_sha for e in state.episodes()] == [sha] * 3


def test_zeus_repeat_zero_never_fires(tmp_path: Path) -> None:
    seq = [calls(_poll(tmp_path, f"p{i}")) for i in range(1, 9)]
    cfg = LoopConfig(max_result_tokens=0, repeat=0)
    r = rig(tmp_path, seq, shell=True, config=cfg, generation=_still)
    r.run("poll forever")
    assert _runs(tmp_path) == 8
    assert all("repetition" not in r.session.state.result(f"p{i}").text for i in range(1, 9))  # type: ignore[union-attr]
    assert payloads(r.session, "bound") == []


def test_reads_are_counted_too(tmp_path: Path) -> None:
    seq = [calls(call("grep", f"f{i}", pattern="same")) for i in range(1, 4)]
    r = rig(tmp_path, seq, generation=_still)
    r.run("find it")
    assert "repetition{count: 3" in r.session.state.result("f3").text  # type: ignore[union-attr]


def test_a_read_that_records_itself_carries_the_notice_too(tmp_path: Path) -> None:
    """`read` records its own line, in one hold with the read; the notice goes in with it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("".join(f"x_{n} = {n}\n" for n in range(1, 80)), encoding="utf-8")
    seq = [calls(call("read", f"r{i}", path="a.py", lines="1-5")) for i in range(1, 4)]
    r = rig(tmp_path, seq, workspace=ws, generation=_still)
    r.run("read it again")
    assert "repetition{count: 3, since_generation: 7}" in r.session.state.result("r3").text  # type: ignore[union-attr]
