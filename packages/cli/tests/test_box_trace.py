"""The REPL shows every call the box makes: Athena's and each worker's tool
calls with their argument and result previews, each model turn, live, at `normal`; whole at
`v`; nothing at `q`. Each names the task it was made in. The design gate's `box_call`
session lines retired with the gate: the box's calls are all on its run's journal."""

from __future__ import annotations

from chickadee_cli.render.stream import Verbosity, frame_lines
from chickadee_cli.render.style import Style
from chickadee_serve.frames import JsonValue, RecordFrame

PIPE = Style(tty=False)
RUN = "2026-10-13-0a1b2c3d"


def run_frame(kind: str, payload: dict[str, JsonValue], task: str | None = "T1") -> RecordFrame:
    return RecordFrame(
        scope=RUN, run_id=RUN, seq=1, at="t", kind=kind, payload=payload, task_id=task
    )


def session_frame(kind: str, payload: dict[str, JsonValue]) -> RecordFrame:
    return RecordFrame(scope="S", seq=1, at="t", kind=kind, payload=payload)


WORKER_READ = run_frame(
    "tool_call",
    {
        "role": "worker",
        "tool": "read",
        "args_sha": "a" * 64,
        "outcome": "served",
        "result_tokens": 12,
        "args_preview": '{"path": "src/calc.py"}',
        "result_preview": "1│def add(a, b):\n2│    return a + b",
    },
)

WORKER_REFUSED = run_frame(
    "tool_call",
    {
        "role": "worker",
        "tool": "edit",
        "args_sha": "b" * 64,
        "outcome": "served",
        "result_tokens": 0,
        "denied_reason": "outside the task's workspace",
        "args_preview": '{"path": "other.py"}',
        "result_preview": "outside the task's workspace",
    },
)

WORKER_TURN = run_frame(
    "model_call",
    {
        "role": "worker",
        "request": "task",
        "tokens_in": 1900,
        "tokens_out": 120,
        "cost_usd": 0.0002,
        "stop_reason": "tool_use",
        "tools": ["read"],
        "text_preview": "I'll read calc.py first.\nThen edit add.",
    },
)

ATHENA_CALL = {
    "role": "athena",
    "request": "episode",
    "tokens_in": 2140,
    "tokens_out": 310,
    "cost_usd": 0.0123,
    "stop_reason": "tool_use",
    "tools": ["read"],
    "text_preview": "Looking at the module first.",
}

ATHENA_SEARCH = {
    "role": "athena",
    "tool": "search",
    "args_sha": "c" * 64,
    "outcome": "served",
    "result_tokens": 4,
    "args_preview": '{"query": "def add"}',
    "result_preview": "src/calc.py:1:def add(a, b):",
}


def test_a_workers_tool_call_shows_what_it_asked_and_what_came_back() -> None:
    assert frame_lines(WORKER_READ, Verbosity.NORMAL, PIPE) == [
        '    worker (T1) → read {"path": "src/calc.py"}',
        "        ← 1│def add(a, b): 2│ return a + b",
    ]


def test_a_refused_call_says_why() -> None:
    assert frame_lines(WORKER_REFUSED, Verbosity.NORMAL, PIPE) == [
        '    worker (T1) → edit {"path": "other.py"}',
        "        ✗ refused: outside the task's workspace",
    ]


def test_a_workers_turn_shows_its_usage_and_first_line() -> None:
    assert frame_lines(WORKER_TURN, Verbosity.NORMAL, PIPE) == [
        "    worker (T1) · task · 1,900 in / 120 out · $0.0002 · tool_use → read",
        "        “I'll read calc.py first. Then edit add.”",
    ]


def test_athenas_model_call_and_tool_call_in_her_run() -> None:
    assert frame_lines(run_frame("model_call", ATHENA_CALL), Verbosity.NORMAL, PIPE) == [
        "    athena (T1) · episode · 2,140 in / 310 out · $0.0123 · tool_use → read",
        "        “Looking at the module first.”",
    ]
    assert frame_lines(run_frame("tool_call", ATHENA_SEARCH), Verbosity.NORMAL, PIPE) == [
        '    athena (T1) → search {"query": "def add"}',
        "        ← src/calc.py:1:def add(a, b):",
    ]


def test_verbose_shows_each_preview_whole_and_quiet_shows_none() -> None:
    assert frame_lines(WORKER_READ, Verbosity.VERBOSE, PIPE) == [
        '    worker (T1) → read {"path": "src/calc.py"}',
        "        ← 1│def add(a, b):",
        "        2│    return a + b",
    ]
    for frame in (WORKER_READ, WORKER_REFUSED, WORKER_TURN):
        assert frame_lines(frame, Verbosity.QUIET, PIPE) == []


def test_an_older_logs_gate_call_is_opaque_and_never_raised() -> None:
    """A `box_call` session line, Athena's call at the design gate, is in logs written
    before the gate retired. It is the session's opaque kind now: named at `v`, and
    nothing at `normal` or `q`."""
    old = session_frame("box_call", {"event": "tool_call", "payload": ATHENA_SEARCH})
    broken = session_frame("box_call", {"event": "tool_call", "payload": {"role": "athena"}})
    for line in (old, broken):
        assert frame_lines(line, Verbosity.QUIET, PIPE) == []
        assert frame_lines(line, Verbosity.NORMAL, PIPE) == []
        assert frame_lines(line, Verbosity.VERBOSE, PIPE) == ["  [box_call]"]


def test_a_line_written_before_previews_reads_with_what_it_has() -> None:
    old = run_frame(
        "tool_call",
        {
            "role": "worker",
            "tool": "search",
            "args_sha": "d" * 64,
            "outcome": "served",
            "result_tokens": 40,
        },
    )
    assert frame_lines(old, Verbosity.NORMAL, PIPE) == [
        "    worker (T1) → search",
        "        ← 40 tokens",
    ]


def test_a_preview_is_collapsed_to_one_line_then_clipped() -> None:
    """#561: a JSON answer read as its first line, a lone `{`."""

    def shown(text: str) -> str:
        frame = run_frame(
            "tool_call",
            {
                "role": "worker",
                "tool": "read",
                "args_sha": "f" * 64,
                "outcome": "served",
                "result_tokens": 1,
                "result_preview": text,
            },
        )
        return frame_lines(frame, Verbosity.NORMAL, PIPE)[-1]

    assert shown("\n\nok\n\n") == "        ← ok"
    assert shown("ok\nmore") == "        ← ok more"
    assert (
        shown('{\n  "stubs": [\n    {"id": "s1"}\n  ]\n}')
        == '        ← { "stubs": [ {"id": "s1"} ] }'
    )
    assert shown("y" * 300).endswith("…")


def test_a_re_asked_call_says_why_under_it() -> None:
    """#560: each refused answer cost her a call, and nothing said why."""
    refused = {
        **ATHENA_CALL,
        "stop_reason": "end_turn",
        "tools": [],
        "refusal": "bad_format: x",
    }
    assert frame_lines(run_frame("model_call", refused), Verbosity.NORMAL, PIPE) == [
        "    athena (T1) · episode · 2,140 in / 310 out · $0.0123 · end_turn",
        "        “Looking at the module first.”",
        "        ✗ re-asked: bad_format: x",
    ]


def test_an_empty_search_reads_as_a_result_not_a_refusal() -> None:
    """#561: her `find` with no hits rendered `✗ refused`; its record now names no denial."""
    empty = {**ATHENA_SEARCH, "result_preview": "UNRESOLVED{query: 'SegmentTree'}"}
    assert frame_lines(run_frame("tool_call", empty), Verbosity.NORMAL, PIPE)[1] == (
        "        ← UNRESOLVED{query: 'SegmentTree'}"
    )


# -- the task loop ------------------------------------------------------------

TREE = "a" * 40


def test_a_plan_its_tasks_their_reports_reviews_and_steps_have_lines() -> None:
    plan = run_frame(
        "plan_written", {"sha": "b" * 64, "by": "athena", "version": 1, "asked": True}
    )
    revision = run_frame(
        "plan_written", {"sha": "c" * 64, "by": "athena", "version": 2, "asked": False}
    )
    delegated = run_frame(
        "task_delegated",
        {
            "task_id": "T1",
            "title": "add the parser",
            "brief_sha": "d" * 64,
            "by": "athena",
            "start_tree": TREE,
        },
    )
    reported = run_frame(
        "task_reported",
        {
            "task_id": "T1",
            "round": 1,
            "stopped": "replied",
            "facts": {
                "start_tree": TREE,
                "end_tree": "e" * 40,
                "files": [{"path": "app.py", "added": 1, "removed": 1, "kind": "edited"}],
                "commands": [
                    {
                        "operation_id": "step:T1:1:c2:abcd",
                        "command": "pytest -q",
                        "exit_code": 0,
                        "timed_out": False,
                        "tail": "1 passed",
                        "test_outcome": "passed",
                    }
                ],
            },
        },
    )
    reviewed = run_frame(
        "task_reviewed", {"task_id": "T1", "call_id": "c9", "answer": "accept"}
    )
    step = run_frame(
        "effect_completed",
        {
            "operation_id": "step:T1:1:c1:abcd",
            "outcome": "succeeded",
            "observed_surface": ["app.py"],
            "post_state_sha": "e" * 40,
        },
    )
    idle = run_frame(
        "effect_completed",
        {"operation_id": "step:T1:1:c3:abcd", "outcome": "succeeded", "observed_surface": []},
    )
    assert frame_lines(plan, Verbosity.NORMAL, PIPE) == [
        "plan v1 by athena · approved at stop 2"
    ]
    assert frame_lines(revision, Verbosity.NORMAL, PIPE) == [
        "plan v2 by athena · a revision, shown and not asked"
    ]
    assert frame_lines(delegated, Verbosity.NORMAL, PIPE) == [
        "> T1 delegated by athena: add the parser"
    ]
    assert frame_lines(reported, Verbosity.NORMAL, PIPE) == [
        "  T1 round 1 replied · 1 file(s) · 1 command(s) · tests passed"
    ]
    assert frame_lines(reviewed, Verbosity.NORMAL, PIPE) == ["  T1 reviewed: accept"]
    assert frame_lines(step, Verbosity.NORMAL, PIPE) == ["    step succeeded: app.py"]
    assert frame_lines(idle, Verbosity.NORMAL, PIPE) == []
    assert frame_lines(idle, Verbosity.VERBOSE, PIPE) == ["    step succeeded: nothing changed"]
