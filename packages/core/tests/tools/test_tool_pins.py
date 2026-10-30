"""Pins of every role's tools as they behave today, in older versions moves them onto one
dispatcher.


Three dispatchers moved onto one: Zeus's (`zeus/loop.py` and the family modules), the scout's
(`scout/tools.py`) and the packet worker's, whose pins retired with the packet loop (the task
worker's tools are `test_edit_tools.py`'s and their neighbours'). The
refactor tasks must keep every assertion here passing unchanged: the
bytes a model reads, each refusal and whether it used up a call, the order refusals are
checked in, each record written, and what each role's counters are after a restart. A task may
change these tests' setup when the code they call changes shape, never their assertions; a pin
that needs its assertion edited is a behaviour change, and a task of its own.

Already pinned elsewhere, and cited rather than copied:
- Zeus: the call bound, the read bound at dispatch and the second `coder` in one response
  (`tests/zeus/test_bound.py`); repetition as a note at 3 and a refusal at 6
  (`tests/zeus/test_repeat.py`); what each tool hands its port
  (`tests/zeus/test_argument_pins.py`);
  `INVALID_ARGUMENTS`' shape (`tests/zeus/test_validate.py`).
- The scout: its caps ending the search `truncated`, and each `locate` starting fresh
  (`tests/scout/test_loop.py`); `request_exec`'s refusals (`tests/scout/test_escapes.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from refusals import stated
from scout.scout_helpers import ListSink
from zeus.zeus_helpers import SESSION_ID, ScriptedGraph, answer, call, calls, payloads, rig

from chickadee.context.session import Session
from chickadee.schema import ToolCall, sha256_hex
from chickadee.schema.journal import EventKind, ScoutToolP
from chickadee.scout.tools import ScoutTools
from chickadee.zeus.loop import LoopConfig

UNBOUNDED = LoopConfig(max_result_tokens=1_000_000)

# =============================================================================================
# Zeus
# =============================================================================================

READ_PREVIEW = (
    "src/a.py — 6 lines · 50 B · python [h_a9eb]\n"
    "1│def go():\n2│    return 1\n3│\n4│\n5│def stop():\n6│    return 2\n"
    '[no index for this file, so no outline: at= and depth="signatures" need one; '
    "read(lines=…) and grep_in work]\n"
    "ops: read(lines=…) · grep_in(pattern)\n"
    "[diagnostics: unavailable — no diagnostics index is attached to this read]\n"
    "[the whole file: all 6 lines fit the preview]"
)


def test_zeus_each_read_and_navigate_tool_answers_these_bytes(
    tmp_path: Path, workspace: Path
) -> None:
    r = rig(
        tmp_path,
        [
            calls(
                call("atlas", "z1"),
                call("grep", "z2", pattern="retry"),
                call("symbol", "z3", name="a.go"),
                call("neighbors", "z4", target="a.go"),
                call("impact", "z5", symbols=["a.go"]),
                call("recall", "z6", query="nothing"),
                call("tools", "z7", query="read"),
                call("read", "z8", path="src/a.py"),
                call("read", "z9", path="src/a.py", lines="1-2"),
            ),
            answer(),
        ],
        workspace=workspace,
    )
    r.run("look")
    expected: dict[str, tuple[bool, str]] = {
        "z1": (True, "components: chickadee (python) · edges: 3"),
        "z2": (True, "src/a.py:1: retry"),
        "z3": (True, "a.go · def at src/a.py:1 · callers: 0"),
        "z4": (True, "a.go: calls nothing"),
        "z5": (True, "no dependents"),
        # #571: a search that found nothing ran.
        "z6": (True, "UNRESOLVED{query: 'nothing', searched: 1, uncovered: []}"),
        "z7": (
            True,
            "read — Read a file or a handle: a preview and a handle with no span, or the span "
            "itself with `lines=` or `at=`. Give exactly one of `path` and `handle`.\n"
            "grep_in — Search inside a file already opened by `read`; every hit is addressable "
            "as a slice. `handle` and `pattern` are required.",
        ),
        "z8": (True, READ_PREVIEW),
        "z9": (
            True,
            "src/a.py — 6 lines · 50 B · python [h_a9eb]\n[you already hold view 1-6 (h_a9eb)]",
        ),
    }
    for cid, (ran, text) in expected.items():
        result = r.session.state.result(cid)
        assert result is not None, cid
        assert (result.ran, stated(result.text)) == (ran, text), cid


def test_zeus_a_deny_listed_read_is_secret_path_and_does_not_run(
    tmp_path: Path, workspace: Path
) -> None:
    r = rig(tmp_path, [calls(call("read", "s1", path=".env")), answer()], workspace=workspace)
    r.run("look")
    result = r.session.state.result("s1")
    assert result is not None
    assert (result.ran, stated(result.text)) == (
        False,
        "SECRET_PATH{path: .env, pattern: '.env'}",
    )


def test_zeus_a_handle_from_another_session_is_handle_expired(
    tmp_path: Path, workspace: Path
) -> None:
    r = rig(
        tmp_path,
        [calls(call("grep_in", "g1", handle="h_a9eb", pattern="return")), answer()],
        workspace=workspace,
    )
    r.run("look")
    result = r.session.state.result("g1")
    assert result is not None
    assert (result.ran, stated(result.text)) == (
        False,
        "HANDLE_EXPIRED{handle: h_a9eb, reason: not_in_session}",
    )


def test_zeus_navigation_answers_are_cut_by_lines_or_bytes_and_kept_behind_one_handle(
    tmp_path: Path,
) -> None:
    """`find` is bounded in lines (`result_cap`, 200), `atlas` in bytes (4096): one producer,
    two units. The whole answer is stored once; both name the same handle."""
    lines = [f"line {i} " + "x" * 60 for i in range(300)]
    r = rig(
        tmp_path,
        [calls(call("grep", "b1", pattern="q"), call("atlas", "b2")), answer()],
        graph=ScriptedGraph(big="\n".join(lines)),
    )
    r.run("look")
    found, atlas = r.session.state.result("b1"), r.session.state.result("b2")
    assert found is not None and atlas is not None
    assert found.text == "\n".join(lines[:200]) + (
        "\n\n[100 of 300 matches not shown; narrow the pattern, or read them by line: "
        'read {"handle": "h_5b02", "lines": "start-end"}]'
    )
    assert atlas.text == "\n".join(lines[:59]) + (
        '\n\n[showing lines 1-59 of 300; next: read {"handle": "h_5b02", "lines": "60-118"}]'
    )


def test_zeus_refusals_are_checked_in_this_order(tmp_path: Path) -> None:
    """`_plan`'s order, pairwise, for the checks no other suite orders: an unknown tool before
    bad arguments, bad findings before bad arguments. (The call bound before the tool is known
    is `test_bound.py`'s.) D208, moved: a missing `why` came between, and was refused."""
    r = rig(
        tmp_path,
        [
            calls(
                ToolCall(id="p1", name="nosuch", arguments={}),
                call("grep", "p2", pattern=5),
                ToolCall(
                    id="p3",
                    name="grep",
                    arguments={"findings": {"zz": "x"}, "query": 5},
                ),
            ),
            answer(),
        ],
    )
    r.run("x")
    texts = {cid: r.session.state.result(cid) for cid in ("p1", "p2", "p3")}
    got = {cid: (res.ran, stated(res.text)) for cid, res in texts.items() if res is not None}
    assert got["p2"][0] is False
    assert got["p2"][1].startswith("INVALID_ARGUMENTS: grep\n- pattern: expected string")
    del got["p2"]
    assert got == {
        "p1": (False, "UNKNOWN_TOOL{name: 'nosuch', nearest: None}"),
        "p3": (
            False,
            "INVALID_ARGUMENTS{field: findings, reason: "
            "'zz has no result to draw a finding from'}",
        ),
    }


def test_zeus_unknown_intent_findings_and_invalid_refusals_each_use_a_call(
    tmp_path: Path,
) -> None:
    """Four refusals of the kinds that count, then a valid call: with a cap of 4 the fifth is
    `BOUND_REACHED`, so each of the four was counted. A bound refusal is not
    (`test_bound.py`)."""
    r = rig(
        tmp_path,
        [
            calls(
                ToolCall(id="c1", name="nosuch", arguments={}),
                call("grep", "c2", pattern=5),
                ToolCall(
                    id="c3",
                    name="grep",
                    arguments={"findings": {"zz": "x"}, "query": "a"},
                ),
                call("grep", "c4", pattern=5),
            ),
            calls(call("atlas", "c5")),
            answer(),
        ],
        config=LoopConfig(max_tool_calls=4, max_result_tokens=1_000_000),
    )
    r.run("x")
    fifth = r.session.state.result("c5")
    assert fifth is not None
    assert (fifth.ran, stated(fifth.text)) == (
        False,
        "BOUND_REACHED{resource: tool_calls, used: 4, limit: 4, scope: turn, "
        "reason: 'bounded for safety, not for spend'}",
    )


def test_zeus_records_one_tool_line_per_call_and_a_refusal_whole(
    tmp_path: Path, workspace: Path
) -> None:
    r = rig(
        tmp_path,
        [
            calls(
                call("grep", "l1", pattern="q"),
                call("read", "l2", path="src/a.py"),
                call("nosuch", "l3"),
            ),
            answer(),
        ],
        workspace=workspace,
    )
    r.run("go")
    lines = {str(p["call_id"]): p for p in payloads(r.session, "tool")}
    assert sorted(lines) == ["l1", "l2", "l3"]
    assert {**lines["l3"], "text": stated(str(lines["l3"]["text"]))} == {
        "call_id": "l3",
        "handle": None,
        "image": None,
        "ran": False,
        "supersedes": [],
        "text": "UNKNOWN_TOOL{name: 'nosuch', nearest: None}",
        "view": [],
    }
    found = lines["l1"]["handle"]
    read = lines["l2"]["handle"]
    assert isinstance(found, dict) and isinstance(read, dict)
    assert (found["kind"], found["path"]) == ("findings", None)
    assert (read["kind"], read["path"], read["id"]) == ("file", "src/a.py", "h_a9eb")


def test_zeus_counters_restart_every_turn_and_after_a_resume(tmp_path: Path) -> None:
    """The call bound and the repetition streak are a turn's (`TurnBounds`): a new turn, and a
    session reopened after a restart, start both at zero. Nothing is replayed from the log."""
    cfg = LoopConfig(max_tool_calls=2, max_result_tokens=1_000_000)
    r = rig(
        tmp_path,
        [
            calls(call("atlas", "t1"), call("atlas", "t2")),
            calls(call("grep", "t3", pattern="a")),
            answer(),
            calls(call("grep", "t4", pattern="a")),
            answer(),
        ],
        config=cfg,
    )
    r.run("one")
    r.run("two")
    outcome = {c: r.session.state.result(c) for c in ("t3", "t4")}
    assert [o.ran for o in outcome.values() if o is not None] == [False, True]
    r.session.close()
    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    again = rig(
        tmp_path,
        [calls(call("atlas", "t5"), call("atlas", "t6"), call("atlas", "t7")), answer()],
        config=cfg,
        session=reopened,
    )
    again.run("three")
    ran = [again.session.state.result(c) for c in ("t5", "t6", "t7")]
    assert [x.ran for x in ran if x is not None] == [True, True, False]

    # The repetition streak (3 by default) is a turn's too: two identical calls in one turn
    # and one in the next carry no notice.
    tmp = tmp_path / "rep"
    rep = rig(
        tmp,
        [
            calls(call("grep", "r1", pattern="a")),
            calls(call("grep", "r2", pattern="a")),
            answer(),
            calls(call("grep", "r3", pattern="a")),
            answer(),
        ],
        config=UNBOUNDED,
    )
    rep.run("a")
    rep.run("b")
    third = rep.session.state.result("r3")
    assert third is not None and third.text == "src/a.py:1: a"


# =============================================================================================
# The scout
# =============================================================================================


def _args_sha(name: str, args: dict[str, object]) -> str:
    """How a `scout_tool` line's `args_sha` is formed today (`ScoutTools.call`)."""
    return sha256_hex(f"{name}\0{json.dumps(args, sort_keys=True, default=str)}")


def test_scout_offers_its_tools_in_this_order_and_request_exec_only_on_offer(
    scout_tools: ScoutTools,
) -> None:
    shared = ("symbol", "grep", "glob", "neighbors", "impact", "read", "grep_in")
    assert scout_tools.names(offer_exec=False) == shared
    assert scout_tools.names(offer_exec=True) == (*shared, "request_exec")


def test_scout_answers_these_bytes_counts_every_call_and_records_each(
    scout_tools: ScoutTools, sink: ListSink
) -> None:
    script: list[tuple[str, dict[str, object], bool]] = [
        ("atlas", {}, True),
        ("symbol", {"name": "HttpClient"}, False),
        ("symbol", {"name": "HttpClient"}, False),
        ("neighbors", {"target": "x", "depth": "2"}, False),
        ("grep", {"pattern": "timeout"}, False),
        ("glob", {"pattern": "*.toml"}, False),
        ("read", {"path": "config/app.yaml"}, False),
        ("request_exec", {"command": "ls", "reason": "r"}, False),
    ]
    answers = [scout_tools.call(n, a, sink, offer_exec=o) for n, a, o in script]
    assert scout_tools.calls == 8, "every call counts: unknown, dedup and invalid ones too"
    assert [stated(a.text) for a in answers] == [
        "UNKNOWN_TOOL{name: 'atlas', offered: [symbol, grep, glob, neighbors, impact, read, "
        "grep_in, request_exec]}",
        "pkg/net/client.py:6-11 pkg.net.client.HttpClient",
        "[dedup: the same symbol call as call #2; nothing re-run]",
        "INVALID_ARGUMENTS: neighbors\n- depth: expected integer, got string\nschema: "
        "{'type': 'object', 'properties': {'target': {'type': 'string', 'description': "
        "'a symbol or a path', 'minLength': 1}, 'depth': {'type': 'integer', 'description': "
        "'how many hops out; 1 is the immediate neighbourhood', 'minimum': 1, 'maximum': 3, "
        "'default': 1}}, 'additionalProperties': False, 'required': ['target']}",
        "grep 'timeout' · in-process, literal (no rg)\n"
        "config/app.yaml:2: timeout: 30\n"
        "pkg/net/client.py:9: def request(self, url: str, timeout: int = DEFAULT_TIMEOUT) "
        "-> str:",
        "glob '*.toml' · paths\npyproject.toml",
        "1: http:\n2:   timeout: 30",
        "UNKNOWN_TOOL{name: 'request_exec', offered: [symbol, grep, glob, neighbors, impact, "
        "read, grep_in]}",
    ]
    # D221 moved the three refusals' sizes (31, 112, 29): each now ends with its instruction.
    assert [(a.dedup_hit, a.tokens) for a in answers] == [
        (False, 49),
        (False, 14),
        (True, 16),
        (False, 134),
        (False, 47),
        (False, 11),
        (False, 8),
        (False, 47),
    ]
    records = [p for p in sink.of(EventKind.SCOUT_TOOL) if isinstance(p, ScoutToolP)]
    assert [r.model_dump() for r in records] == [
        {
            "tool": n,
            "args_sha": _args_sha(n, a),
            "result_tokens": ans.tokens,
            "dedup_hit": ans.dedup_hit,
        }
        for (n, a, _), ans in zip(script, answers, strict=True)
    ]


def test_scout_a_new_locate_call_starts_its_counters_at_zero(
    scout_tools: ScoutTools, sink: ListSink
) -> None:
    """`begin()` opens one `locate` call: its call count, its dedup cache and its streak start
    again. Nothing is replayed from the journal."""
    for _ in range(2):
        scout_tools.call("symbol", {"name": "HttpClient"}, sink, offer_exec=False)
    assert (scout_tools.calls, scout_tools.dedup_streak) == (2, 1)
    scout_tools.begin()
    assert (scout_tools.calls, scout_tools.dedup_streak) == (0, 0)
    again = scout_tools.call("symbol", {"name": "HttpClient"}, sink, offer_exec=False)
    assert not again.dedup_hit
