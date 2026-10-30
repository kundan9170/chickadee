"""The projection may select, never invent (invariant 19).

A superseded result is marked, never rewritten; a call's arguments are byte-identical between
the transcript and the journal — and are the arguments the model sent, `findings` included,
which are read out beside them and never edited out of them."""

from __future__ import annotations

from pathlib import Path

from refusals import stated

from chickadee.context.lines import canonical_json
from chickadee.context.results import SUPERSEDED_STUB
from chickadee.schema import ToolCall

from .zeus_helpers import ScriptedPerson, call, calls, payloads, rig

SOURCE = "".join(f"line_{n} = {n}\n" for n in range(1, 61))


def test_arguments_are_byte_identical_between_what_was_sent_the_log_and_the_wire(
    tmp_path: Path,
) -> None:
    first = call("grep", "f1", pattern="retry")
    second = call(
        "symbol",
        "s1",
        findings={"f1": "retry is configured in src/a.py"},
        name="retry",
    )
    r = rig(tmp_path, [calls(first), calls(second)])
    r.run("how does retry work?")
    logged = [p for p in payloads(r.session, "turn") if p["role"] == "assistant" and p["calls"]]
    sent = [first, second]
    for turn, original in zip(logged, sent, strict=True):
        (recorded,) = turn["calls"]  # type: ignore[misc]
        assert canonical_json(recorded["arguments"]) == canonical_json(original.arguments)  # type: ignore[index]
    # And on the wire, in the request after both: the same bytes, meta arguments included.
    wire = [m for m in r.model.requests[-1].messages if m.role == "assistant" and m.tool_calls]
    for message, original in zip(wire, sent, strict=True):
        (echoed,) = message.tool_calls
        assert canonical_json(echoed.arguments) == canonical_json(original.arguments)
    # The finding is recorded beside the call, where layer 1 reads it.
    state = r.session.state
    assert state.finding("f1") == "retry is configured in src/a.py"


def test_a_superseded_result_is_marked_and_never_rewritten(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "big.py").write_text(SOURCE, encoding="utf-8")
    one = call("read", "r1", path="big.py", lines="1-10")
    two = call("read", "r2", path="big.py", lines="11-20")
    r = rig(tmp_path, [calls(one), calls(two)], workspace=ws)
    r.run("read the top of big.py")
    state = r.session.state
    merged = state.result("r2")
    assert merged is not None and merged.supersedes == ("r1",), "the two views merged"
    original = state.result("r1")
    assert original is not None and "line_1 = 1" in original.text, "the log keeps what was read"
    request = r.model.requests[-1]
    by_call = {m.tool_call_id: m for m in request.messages if m.role == "tool"}
    assert by_call["r1"].content == SUPERSEDED_STUB.format(call_id="r2")
    assert "line_1 = 1" in str(by_call["r2"].content) and "line_20 = 20" in str(
        by_call["r2"].content
    )
    # The call the stub answers is untouched: it still asks for exactly what it asked for.
    calls_on_wire = [c for m in request.messages if m.role == "assistant" for c in m.tool_calls]
    assert [c.arguments.get("lines") for c in calls_on_wire] == ["1-10", "11-20"]


def test_a_finding_the_fold_would_refuse_is_refused_as_an_argument(tmp_path: Path) -> None:
    """A finding about a call that has no result would corrupt the record; the call carrying
    it is INVALID_ARGUMENTS on `findings`, and the turn is still recorded as sent."""
    bad = call("grep", "f9", findings={"nope": "made up"}, pattern="x")
    r = rig(tmp_path, [calls(bad)])
    r.run("go")
    state = r.session.state
    text = state.result("f9").text  # type: ignore[union-attr]
    assert text.startswith("INVALID_ARGUMENTS{field: findings")
    assert state.call("f9").args["findings"] == {"nope": "made up"}  # type: ignore[union-attr]
    assert state.finding("nope") is None


def test_a_call_without_an_intent_runs_and_is_recorded_as_sent(tmp_path: Path) -> None:
    """D208, moved: such a call was refused (`INVALID_ARGUMENTS{field: why}`)."""
    bare = call("grep", "f1", pattern="x")
    r = rig(tmp_path, [calls(bare)])
    r.run("go")
    state = r.session.state
    assert state.result("f1").ran  # type: ignore[union-attr]
    assert state.call("f1").args == {"pattern": "x"}  # type: ignore[union-attr]


def test_an_unknown_tool_is_named_never_substituted(tmp_path: Path) -> None:
    r = rig(tmp_path, [calls(call("grepp", "u1", pattern="x"))])
    r.run("go")
    state = r.session.state
    assert stated(state.result("u1").text) == "UNKNOWN_TOOL{name: 'grepp', nearest: 'grep'}"  # type: ignore[union-attr]
    assert state.call("u1").tool == "grepp", "the call is recorded as sent"  # type: ignore[union-attr]
    assert r.graph.asked == [], "never corrected to the nearest"


def test_a_person_s_answer_is_their_words_and_a_default_is_not(tmp_path: Path) -> None:
    """Only what the person said is numbered `[turn N]` and citable: a default, taken
    because nobody answered, is recorded as a call that did not run."""
    asked = call("ask_user", "q1", question="keep the old endpoint? default: yes")
    again = call("ask_user", "q2", question="and the docs? default: leave them")
    r = rig(tmp_path, [calls(asked), calls(again)], person=ScriptedPerson("no — drop it"))
    r.run("clean up the API")
    state = r.session.state
    human, default = state.result("q1"), state.result("q2")
    assert human is not None and default is not None
    assert human.ran and human.text == "no — drop it"
    assert state.is_user_words(human.seq)
    assert not default.ran and default.text.startswith("[default] ")
    assert not state.is_user_words(default.seq)
    wire = r.model.wire(-1)
    assert f"[turn {human.seq}]\nno — drop it" in wire
    assert f"[turn {default.seq}]" not in wire
    assert [q.id for q in r.person.asked] == ["q1", "q2"]


def test_a_call_id_the_log_cannot_take_is_relabelled_and_the_call_kept_as_sent(
    tmp_path: Path,
) -> None:
    """Some servers send no id, or number each response's calls from zero; the log refuses an
    id that is empty or already used. Such a call gets a harness id — a relabel, which
    invariant 19 permits — and its name and arguments are recorded exactly as sent."""
    first = call("grep", "call_0", pattern="a")
    reused = call("grep", "call_0", pattern="b")
    empty = ToolCall(id="", name="grep", arguments={"pattern": "c"})
    twice = (call("grep", "dup", pattern="d"), call("grep", "dup", pattern="e"))
    r = rig(tmp_path, [calls(first), calls(reused, empty), calls(*twice)])
    r.run("look around")
    logged = [
        c
        for p in payloads(r.session, "turn")
        if p["role"] == "assistant"
        for c in p["calls"]  # type: ignore[attr-defined]
    ]
    ids = [c["id"] for c in logged]
    assert ids[0] == "call_0" and ids[3] == "dup", "an id the log can take is kept"
    assert len(set(ids)) == len(ids) == 5
    sent = [first, reused, empty, *twice]
    for recorded, original in zip(logged, sent, strict=True):
        assert recorded["name"] == original.name
        assert canonical_json(recorded["arguments"]) == canonical_json(original.arguments)
    state = r.session.state
    assert state.pending_calls() == ()
    assert all(state.ran(i) for i in ids)
    request = r.model.requests[-1]
    on_wire = [c.id for m in request.messages if m.role == "assistant" for c in m.tool_calls]
    answered = [m.tool_call_id for m in request.messages if m.role == "tool"]
    assert on_wire == ids == answered, "each result pairs with its call on the wire"


def test_invalid_arguments_carry_the_schema_the_model_was_shown(tmp_path: Path) -> None:
    """Validated with `findings` set aside, answered with it in: the schema Zeus is shown, with
    no purpose line."""
    r = rig(tmp_path, [calls(call("grep", "f1", pattern=5))])
    r.run("go")
    text = r.session.state.result("f1").text  # type: ignore[union-attr]
    assert text.startswith("INVALID_ARGUMENTS: grep")
    (schema,) = [line for line in text.splitlines() if line.startswith("schema: ")]
    assert "'why'" not in schema and "'findings'" in schema
    assert "'required': ['pattern']" in schema


def test_a_call_that_names_no_tool_ends_the_turn_and_records_nothing_of_it(
    tmp_path: Path,
) -> None:
    """The log refuses a call with no tool, and `UNKNOWN_TOOL` has no name to give back. The
    response is treated as malformed: an `error` event, nothing of it on the log, and the
    person's words still there for the next turn."""
    nameless = ToolCall(id="n1", name="", arguments={"pattern": "x"})
    r = rig(tmp_path, [calls(call("grep", "f1", pattern="a"), nameless, text="looking")])
    events = r.run("look")
    errors = [e.payload for e in events if e.kind == "error"]
    assert len(errors) == 1 and str(errors[0]).startswith("MALFORMED_RESPONSE")
    assert "n1" in str(errors[0])
    state = r.session.state
    assert [t.role for t in state.turns()] == ["user"], "nothing of the response was recorded"
    assert state.pending_calls() == ()
    assert r.graph.asked == [], "no call of it was dispatched"
