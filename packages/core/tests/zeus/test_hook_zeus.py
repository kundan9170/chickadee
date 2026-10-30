"""Zeus's points: his turn's, fired by `zeus/loop.py`, and his
calls', through his dispatcher.

What these pin: a `Deny` at `prompt_submitted` refuses the words before anything is recorded or
sent; a hook's words where no answer of his carries them are a harness notice (trigger 9) his
next request reads; a `Continue` at his stop is a notice and his turn goes on, at most
`hooks.max_continues` a turn, the one past it a bounded line; a `Halt` before a request sends
nothing, and after a batch ends his turn; a `Deny` at `permission_asked` refuses a call before
the person is asked; and with no hook bound his turn is as it was.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from chickadee.context.session import Session
from chickadee.context.standing import NoticeTrigger
from chickadee.hooks.fire import BoundHook, HookLimits, Hooks, HookSet, RoleHooks
from chickadee.hooks.payloads import AfterBatch, HookEvent
from chickadee.hooks.sinks import SessionHookSink
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Ask, Continue, Deny, Halt, HookVerdict, Note

from .zeus_helpers import SESSION_ID, answer, call, calls, payloads, rig

Hook = Callable[[HookEvent], HookVerdict]


def _hooks(session: Session, limits: HookLimits | None = None, **at: Hook) -> RoleHooks:
    bound = tuple(
        BoundHook(
            HookSpec.model_validate({"event": point, "name": point, "use": point}), run, point
        )
        for point, run in at.items()
    )
    sink = SessionHookSink(session)
    return RoleHooks(Hooks(HookSet(bound, limits=limits or HookLimits()), sink), "zeus")


def _session(tmp_path: Path) -> Session:
    return Session.create(tmp_path / "sessions", SESSION_ID)


def _notices(session: Session) -> list[str]:
    return [
        str(p["text"])
        for p in payloads(session, "turn")
        if p.get("role") == "harness" and p.get("trigger") == int(NoticeTrigger.HOOK)
    ]


def test_with_no_hook_bound_the_turn_is_as_it_was(tmp_path: Path) -> None:
    session = _session(tmp_path)
    r = rig(tmp_path, [answer("hi")], session=session, hooks=_hooks(session))
    events = r.run("hello")
    assert [e.kind for e in events] == ["answer"]
    assert _notices(session) == [] and payloads(session, "hook") == []


def test_a_deny_at_prompt_submitted_refuses_the_words_before_anything_is_sent(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    hooks = _hooks(session, prompt_submitted=lambda e: Deny("no secrets in prompts", None))
    r = rig(tmp_path, [answer("hi")], session=session, hooks=hooks)
    events = r.run("my key is hunter2")
    assert [e.kind for e in events] == ["error"]
    assert "HOOK_DENIED" in str(events[0].payload)
    assert r.model.requests == [], "nothing was sent"
    assert not any(p.get("role") == "user" for p in payloads(session, "turn"))


def test_a_note_at_prompt_submitted_is_a_notice_his_request_reads(tmp_path: Path) -> None:
    session = _session(tmp_path)
    hooks = _hooks(session, prompt_submitted=lambda e: Note("the build is red on main"))
    r = rig(tmp_path, [answer("ok")], session=session, hooks=hooks)
    r.run("hello")
    assert _notices(session) == ["the build is red on main"]
    assert "the build is red on main" in r.model.wire(0)


def test_a_continue_at_his_stop_goes_on_and_is_bounded(tmp_path: Path) -> None:
    session = _session(tmp_path)
    hooks = _hooks(session, before_stop=lambda e: Continue("you have not run the tests"))
    r = rig(
        tmp_path,
        [answer("done"), answer("ran them"), answer("x")],
        session=session,
        hooks=hooks,
    )
    events = r.run("fix it")
    assert [e.kind for e in events] == ["answer", "answer"]
    assert len(r.model.requests) == 2
    assert "you have not run the tests" in r.model.wire(1)
    lines = payloads(session, "hook")
    assert [line["hook"]["bounded"] for line in lines] == [False, True], (  # type: ignore[index]
        "the second is past `hooks.max_continues` and not taken"
    )


def test_a_note_after_a_batch_reaches_his_next_request(tmp_path: Path) -> None:
    session = _session(tmp_path)

    def ran(event: HookEvent) -> HookVerdict:
        assert isinstance(event.payload, AfterBatch)
        return Note(f"you ran {', '.join(event.payload.tools)}")

    hooks = _hooks(session, after_batch=ran)
    (r_ws := tmp_path / "ws").mkdir()
    (r_ws / "a.py").write_text("x = 1\n", encoding="utf-8")
    r = rig(
        tmp_path,
        [calls(call("read", "c1", path="a.py")), answer("read it")],
        session=session,
        hooks=hooks,
        workspace=r_ws,
    )
    r.run("look")
    assert _notices(session) == ["you ran read"]
    assert "you ran read" in r.model.wire(1)


def test_a_halt_before_a_request_sends_nothing(tmp_path: Path) -> None:
    session = _session(tmp_path)
    hooks = _hooks(session, before_request=lambda e: Halt("over the hour's budget"))
    r = rig(tmp_path, [answer("hi")], session=session, hooks=hooks)
    events = r.run("hello")
    assert [e.kind for e in events] == ["error"]
    assert "HOOK_HALTED{hook: before_request" in str(events[0].payload)
    assert r.model.requests == [], "the halt names its hook, as a tool point's does"


def test_a_halt_after_a_batch_ends_his_turn(tmp_path: Path) -> None:
    session = _session(tmp_path)
    hooks = _hooks(session, after_batch=lambda e: Halt("enough reading"))
    (r_ws := tmp_path / "ws").mkdir()
    (r_ws / "a.py").write_text("x = 1\n", encoding="utf-8")
    r = rig(
        tmp_path,
        [calls(call("read", "c1", path="a.py")), answer("never asked")],
        session=session,
        hooks=hooks,
        workspace=r_ws,
    )
    events = r.run("look")
    assert events[-1].kind == "error"
    assert "HOOK_HALTED{hook: after_batch" in str(events[-1].payload)
    assert len(r.model.requests) == 1


def test_permission_asked_refuses_before_the_person_is_asked(tmp_path: Path) -> None:
    """A hook's `Ask` at a call puts the question to the person; a `Deny` at
    `permission_asked` refuses it first, so nobody is asked, and never answers for them."""
    session = _session(tmp_path)
    hooks = _hooks(
        session,
        before_tool=lambda e: Ask("read a.py?"),
        permission_asked=lambda e: Deny("nobody approves reads here", None),
    )
    (r_ws := tmp_path / "ws").mkdir()
    (r_ws / "a.py").write_text("x = 1\n", encoding="utf-8")
    r = rig(
        tmp_path,
        [calls(call("read", "c1", path="a.py")), answer("refused")],
        session=session,
        hooks=hooks,
        workspace=r_ws,
    )
    r.run("look")
    assert r.person.asked == [], "the person is never asked"
    assert "HOOK_DENIED" in r.model.wire(1) and "nobody approves reads here" in r.model.wire(1)
