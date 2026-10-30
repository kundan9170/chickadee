"""Python tool calling, the demo arm (`chickadee/pycall.py`).

The comparison tests run the same task through the task worker twice, once with native tool
calls and once with Python blocks, and check that the worktree, the report, the recorded steps
and what the model was shown are the same. The rest pin the arm's own rules: one tool per
block, the checks before a block runs, the loop, `finish`, and a reply with no block.

No sandbox runs on every test machine, so the arm's children here run under the explicit
opt-in (`[sandbox] allow_unsandboxed_python`): still a separate process, never this one.
"""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from box_helpers import delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee import pycall, pycall_tools
from chickadee.client.request import ToolSchema
from chickadee.isolation.none import Passthrough
from chickadee.journal.effects import EffectKind
from chickadee.schema.journal import BoxMessageP, ToolCallP
from chickadee.tools.loop import Reply

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


py_rig = partial(worker_rig, python_calls=True)
"""The task worker with `calling = "python"`, its blocks in a plain child."""


def block(code: str, prose: str = "") -> Any:
    """A reply as a model in the Python arm writes it: prose, then one block."""
    return answer(f"{prose}\n```python\n{code}\n```\n" if prose else f"```python\n{code}\n```")


def shown_results(journal: Any) -> list[str]:
    """Every tool result the model was shown, in order: the content, not the call's id."""
    lines = journal.projection().lines(BoxMessageP)
    blobs = [journal.get_blob(p.payload.sha) for p in lines if p.payload.side == "tool"]
    return [json.loads(b)["content"] for b in blobs]


def step_surfaces(journal: Any) -> list[tuple[str, ...]]:
    run = journal.projection()
    return [
        op.intended.declared_surface
        for op in run.operations.values()
        if op.kind == EffectKind.WORKER_STEP.value and op.intended is not None
    ]


# -- native and Python, side by side -----------------------------------------------------------


def _native(root: Path) -> tuple[Any, Any]:
    worker, journal, _ = worker_rig(
        root,
        [
            calls(call("read", "c1", path="app.py")),
            calls(call("edit", "c2", path="app.py", old="return 1", new="return 2")),
            calls(call("finish", "c3", report="go returns 2")),
        ],
    )
    return worker.start("T1", "Make go return 2.", delegate(journal, root)), journal


def _python(root: Path) -> tuple[Any, Any, Any]:
    worker, journal, rung = py_rig(
        root,
        [
            block('read(path="app.py")', prose="Reading the file first."),
            block('edit(path="app.py", old="return 1", new="return 2")'),
            block('finish(report="go returns 2")'),
        ],
    )
    return worker.start("T1", "Make go return 2.", delegate(journal, root)), journal, rung


def test_the_same_task_ends_the_same_under_both(tmp_path: Path) -> None:
    files = {"app.py": APP, "tests/test_app.py": "x = 1\n"}
    native_root = init_repo(tmp_path / "native", files)
    native, native_journal = _native(native_root)

    py_root = init_repo(tmp_path / "python", files)
    py, py_journal, _ = _python(py_root)

    assert (py.stopped, py.report) == (native.stopped, native.report)
    assert py.report == "go returns 2"
    assert (py_root / "app.py").read_bytes() == (native_root / "app.py").read_bytes()
    assert py.facts.files == native.facts.files
    # Every call is the same step through the same dispatcher.
    assert step_surfaces(py_journal) == step_surfaces(native_journal)
    # A one-call block that prints nothing is shown exactly what the native call is shown.
    assert shown_results(py_journal) == shown_results(native_journal)


def test_the_model_is_sent_no_native_tools_and_reads_them_as_python(root: Path) -> None:
    _, _, rung = _python(root)
    first = rung.requests[0]
    assert first.tools == ()
    system = str(first.messages[0].content)
    assert system.startswith("You are the worker.")
    assert "def read(path: str" in system and "def finish(report: str" in system
    # An earlier call goes back as the text that made it, its result as a user message.
    later = rung.requests[1].messages
    assert all(m.role != "tool" and not m.tool_calls for m in later)
    assert "Reading the file first." in str(later[-2].content)
    assert str(later[-1].content).startswith("Result:\n")


# -- the arm's own rules -----------------------------------------------------------------------


def test_one_tool_called_in_a_loop_is_one_reply_and_each_call_a_line(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block(
                'for p in ["app.py", "tests/test_app.py"]:\n'
                "    text = read(path=p)\n"
                '    print(p, "go" in text)'
            ),
            block('finish(report="looked")'),
        ],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert seen.startswith("app.py True\ntests/test_app.py False")
    assert "[2 read call(s) ran]" in seen


def test_a_block_with_two_tools_is_refused_and_runs_nothing(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block('read(path="app.py")\nedit(path="app.py", old="return 1", new="return 2")'),
            block('finish(report="stopped")'),
        ],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    # A change uses no other tool, and the reads are named first.
    assert (
        "a block that changes something uses no other tool, and yours uses edit, read" in seen
    )
    assert 'send this block first:\n```python\nread(path="app.py")\n```' in seen
    assert (root / "app.py").read_text(encoding="utf-8") == APP
    assert step_surfaces(journal) == []


def test_a_script_error_is_shown_with_what_ran_before_it(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block('edit(path="app.py", old="return 1", new="return 2")\n1 / 0'),
            block('finish(report="stopped")'),
        ],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert "ZeroDivisionError" in seen and "[1 edit call(s) ran]" in seen
    assert "pycall_runner" not in seen
    # The edit before the error stands, as a recorded step.
    assert "return 2" in (root / "app.py").read_text(encoding="utf-8")
    assert step_surfaces(journal) == [("edit",)]


def test_a_reply_with_no_block_is_never_taken_as_the_report(root: Path) -> None:
    """Was (2026-10-23): the third reply with no block was the task's report, so a bare
    `{"path":...}` reached Athena as one. Now the round ends saying no tool was called."""
    worker, journal, _ = py_rig(root, [answer("hm"), answer("hm"), answer("my report")])
    report = worker.start("T1", "Do it.", delegate(journal, root))
    assert all("no ```python block" in s for s in shown_results(journal))
    assert report.stopped == "replied"
    assert report.report == (
        "(no report: the worker wrote no ```python block in 3 replies in a row, so it called "
        "no tool. Its last reply: 'my report')"
    )


def test_a_block_after_a_refusal_carries_on(root: Path) -> None:
    worker, journal, _ = py_rig(root, [answer("hm"), block("finish(report='go is unchanged')")])
    report = worker.start("T1", "Do it.", delegate(journal, root))
    assert report.report == "go is unchanged"


def test_bare_arguments_are_answered_with_the_block_to_send(root: Path) -> None:
    """2026-10-23: the model wrote a native call's arguments as text. The refusal names the
    block they mean, as a read (the reads before the writes), and runs nothing."""
    worker, journal, _ = py_rig(
        root, [answer('\n{\n  "path": "app.py"\n}'), block("finish(report='done')")]
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    [refusal] = shown_results(journal)[:1]
    assert "a JSON object is not a call" in refusal
    assert "Do: send exactly this as your next reply:\n```python\nread(path='app.py')\n```" in (
        refusal
    )
    assert step_surfaces(journal) == []


def test_a_native_call_is_answered_with_its_block(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [calls(call("read", "c1", path="app.py", lines="1-2")), block("finish(report='done')")],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    assert "read(path='app.py', start_line=1, end_line=2)" in shown_results(journal)[0]


def test_a_reply_with_nothing_to_convert_is_shown_a_whole_reply(root: Path) -> None:
    worker, journal, _ = py_rig(root, [answer("\n"), block("finish(report='done')")])
    worker.start("T1", "Do it.", delegate(journal, root))
    first = shown_results(journal)[0]
    assert "Do: reply with a ```python block that calls a tool, for example:" in first
    assert 'read(path="README.md")' in first and "finish(report=" in first


def _refused_lines(journal: Any, *, served: bool = False) -> list[ToolCallP]:
    """The `python` lines on the trace: refused ones, or with `served` the ones that ran."""
    return [
        p.payload
        for p in journal.projection().lines(ToolCallP)
        if p.payload.tool == "python" and bool(p.payload.denied_reason) != served
    ]


def test_every_refusal_no_dispatcher_saw_is_on_the_trace(root: Path) -> None:
    """Was: a reply with no block, and a call its signature rejected, answered the model and
    wrote no line, so the trace showed a task with no mistakes that had three."""
    worker, journal, _ = py_rig(
        root,
        [
            answer('{"path": "app.py"}'),
            block("grep(pattern='go', path='app.py')"),
            block("finish(report='done')"),
        ],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    first, second = _refused_lines(journal)
    assert "no ```python block" in (first.denied_reason or "")
    assert '"reply"' in (first.args_preview or "") and "app.py" in (first.args_preview or "")
    assert "unexpected keyword argument 'path'" in (second.denied_reason or "")


def test_an_indented_block_copied_from_the_examples_runs(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            answer("```python\n    read(path='app.py')\n    ```"),
            block("finish(report='done')"),
        ],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    assert "return 1" in shown_results(journal)[0]


@pytest.mark.parametrize(
    ("native", "args", "python"),
    [
        (
            "read",
            {"path": "a.py", "lines": "3-9"},
            "read(path='a.py', start_line=3, end_line=9)",
        ),
        ("read", {"path": "a.py", "at": "go"}, "read(path='a.py', symbol='go')"),
        (
            "grep_in",
            {"handle": "h_1", "pattern": "x", "context": 0},
            "grep_in(handle='h_1', pattern='x', context_lines=0)",
        ),
        ("edit", {"path": "a", "old": "x", "new": "y"}, "edit(path='a', old='x', new='y')"),
        ("read", {"path": "a.py", "lines": "three"}, None),
        ("edit", {"path": "a"}, None),
        ("read", {"nonsense": 1}, None),
    ],
)
def test_native_arguments_back_to_python(
    native: str, args: dict[str, object], python: str | None
) -> None:
    [tool] = [t for t in pycall_tools.PY_TOOLS if t.native == native and t.name == native]
    back = pycall_tools.from_native(tool, args)
    assert (None if back is None else pycall_tools.as_python(tool, back)) == python


FUNCTIONS = {"read": "read", "edit": "edit", "edit_many": "edit"}


@pytest.mark.parametrize(
    ("code", "says"),
    [
        ("import os\nread(path='a')", "`os` cannot be imported"),
        ("import os.path\nread(path='a')", "`os.path` cannot be imported"),
        ("from subprocess import run\nread(path='a')", "`subprocess` cannot be imported"),
        ("open('a').read()", "`open` is not available"),
        ("print(1)", "calls no tool"),
        ("read(path=", "not valid Python"),
        ("read('a')\nedit('a', 'x', 'y')", "uses edit, read"),
        ("x = read\nx(path='a')", None),
        ("import re\nprint(re.findall('x', read(path='a')))", None),
        ("from collections import Counter\nread(path='a')", None),
    ],
)
def test_what_a_block_may_hold(code: str, says: str | None) -> None:
    got = pycall.tools_of(code, FUNCTIONS)
    if says is None:
        assert got == {"read"}
    else:
        assert isinstance(got, pycall.Refused) and says in got.why


def test_edit_and_edit_many_are_one_tool() -> None:
    code = "edit('a', 'x', 'y')\nedit_many('a', [('p', 'q')])"
    assert pycall.tools_of(code, FUNCTIONS) == {"edit"}


def test_a_block_may_chain_tools_that_change_nothing() -> None:
    """Reads compose; a change stands alone."""
    functions = {**FUNCTIONS, "grep": "grep", "shell": "shell"}
    chained = "for line in grep('x').splitlines():\n    print(read(path=line.split(':')[0]))"
    assert pycall.tools_of(chained, functions) == {"grep", "read"}
    mixed = pycall.tools_of("print(read(path='a'))\nshell(command='ls')", functions)
    assert isinstance(mixed, pycall.Refused)
    assert "send this block first:\n```python\nprint(read(path='a'))\n```" in mixed.why


READ = pycall_tools.surface((ToolSchema("read", "", {}),))


def test_without_a_sandbox_a_block_is_not_run_unless_opted_in(tmp_path: Path) -> None:
    ran: list[dict[str, object]] = []

    def one(native: str, arguments: dict[str, object], n: int) -> Reply:
        ran.append(arguments)
        return Reply("x")

    refused = pycall.ScriptRunner(tmp_path, Passthrough()).run("read('a')", READ, one)
    assert "no sandbox" in str(refused.content) and ran == []
    allowed = pycall.ScriptRunner(tmp_path, Passthrough(), allow_unsandboxed=True)
    assert allowed.run("read('a')", READ, one).content == "x"
    assert ran == [{"path": "a"}]


# -- the typed surface -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function", "typed", "native"),
    [
        ("read", {"path": "a.py"}, {"path": "a.py"}),
        (
            "read",
            {"path": "a.py", "start_line": 5, "end_line": 178},
            {"path": "a.py", "lines": "5-178"},
        ),
        ("read", {"path": "a.py", "symbol": "go"}, {"path": "a.py", "at": "go"}),
        ("read", {"path": "a.py", "depth": "full"}, {"path": "a.py", "depth": "full"}),
        ("read", {"handle": "h_1"}, {"handle": "h_1"}),
        ("grep_in", {"handle": "h", "pattern": "x"}, {"handle": "h", "pattern": "x"}),
        (
            "grep_in",
            {"handle": "h", "pattern": "x", "context_lines": 0},
            {"handle": "h", "pattern": "x", "context": 0},
        ),
        ("grep", {"pattern": "x", "paths": ["*.py"]}, {"pattern": "x", "paths": ["*.py"]}),
        ("impact", {"symbols": ["go"]}, {"symbols": ["go"]}),
        ("tests_for", {"paths": ["a.py"]}, {"paths": ["a.py"]}),
        ("neighbors", {"target": "go", "depth": 2}, {"target": "go", "depth": 2}),
        ("edit", {"path": "a", "old": "x", "new": "y"}, {"path": "a", "old": "x", "new": "y"}),
        (
            "edit",
            {"path": "a", "old": "x", "new": "y", "replace_all": True},
            {"path": "a", "old": "x", "new": "y", "replace_all": True},
        ),
        (
            "edit_many",
            {"path": "a", "edits": [["x", "y"], ["p", "q"]]},
            {"path": "a", "edits": [{"old": "x", "new": "y"}, {"old": "p", "new": "q"}]},
        ),
        ("shell", {"command": "ls"}, {"command": "ls"}),
        (
            "shell",
            {"command": "ls", "network": True, "sandbox": False},
            {"command": "ls", "network": True, "sandbox": False},
        ),
        ("shell", {"command": "ls", "network": False, "sandbox": True}, {"command": "ls"}),
        ("finish", {"report": "done"}, {"report": "done"}),
    ],
)
def test_a_typed_call_is_the_native_call(
    function: str, typed: dict[str, object], native: dict[str, object]
) -> None:
    [tool] = [t for t in pycall_tools.PY_TOOLS if t.name == function]
    assert tool.bind(typed) == native


@pytest.mark.parametrize(
    ("function", "typed", "says"),
    [
        (
            "read",
            {"path": "a", "start_line": "five", "end_line": 9},
            "'start_line' must be int",
        ),
        (
            "read",
            {"path": "a", "start_line": True, "end_line": 9},
            "must be int | None, not bool",
        ),
        ("read", {"path": "a", "start_line": 5}, "together"),
        ("read", {"path": "a", "start_line": 9, "end_line": 5}, "start_line <= end_line"),
        ("read", {"path": "a", "handle": "h"}, "exactly one"),
        ("read", {}, "exactly one"),
        ("read", {"path": "a", "depth": "deep"}, "'depth' must be Literal"),
        ("read", {"path": "a", "lines": "1-2"}, "unexpected keyword argument 'lines'"),
        ("grep", {"pattern": "x", "paths": 5}, "'paths' must be list[str]"),
        ("edit", {"path": "a", "old": "x"}, "missing required argument 'new'"),
        ("edit_many", {"path": "a", "edits": [["x"]]}, "list[tuple[str, str]]"),
        ("edit_many", {"path": "a", "edits": []}, "at least one"),
    ],
)
def test_a_typed_call_that_does_not_bind_is_rejected(
    function: str, typed: dict[str, object], says: str
) -> None:
    [tool] = [t for t in pycall_tools.PY_TOOLS if t.name == function]
    with pytest.raises(pycall_tools.Rejected, match=re.escape(says)):
        tool.bind(typed)


def test_a_typed_read_shows_what_the_native_read_shows(tmp_path: Path) -> None:
    files = {"app.py": APP}
    native_root = init_repo(tmp_path / "native", files)
    worker, native_journal, _ = worker_rig(
        native_root,
        [
            calls(call("read", "c1", path="app.py", lines="2-2")),
            calls(call("finish", "c2", report="r")),
        ],
    )
    worker.start("T1", "Look.", delegate(native_journal, native_root))

    py_root = init_repo(tmp_path / "python", files)
    worker, py_journal, _ = py_rig(
        py_root,
        [block('read("app.py", start_line=2, end_line=2)'), block('finish(report="r")')],
    )
    worker.start("T1", "Look.", delegate(py_journal, py_root))
    assert shown_results(py_journal) == shown_results(native_journal)
    assert "return 1" in shown_results(py_journal)[0]


def test_a_wrong_type_raises_in_the_block_and_runs_nothing(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block(
                "try:\n"
                '    read("app.py", start_line="one", end_line=2)\n'
                "except TypeError as e:\n"
                "    print('caught:', e)"
            ),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert "caught: read() argument 'start_line' must be int | None, not str" in seen
    assert "no call ran; 1 rejected before running (TypeError)" in seen


def test_the_model_is_shown_the_typed_signatures(root: Path) -> None:
    _, _, rung = _python(root)
    system = str(rung.requests[0].messages[0].content)
    assert (
        "def read(path: str | None = None, start_line: int | None = None, "
        "end_line: int | None = None, symbol: str | None = None, "
        'depth: Literal["outline", "signatures", "full"] | None = None, *, '
        "handle: str | None = None) -> str:"
    ) in system
    assert "def edit_many(path: str, edits: list[tuple[str, str]]) -> str:" in system
    assert "def grep(pattern: str, paths: list[str] | None = None) -> GrepResult:" in system
    assert 'lines="' not in system


def test_a_tool_with_no_typed_entry_keeps_its_schema_typed() -> None:
    tool = ToolSchema(
        "lookup",
        "Look something up.",
        {
            "properties": {
                "mode": {"type": "string", "enum": ["a", "b"]},
                "names": {"type": "array", "items": {"type": "string"}},
                "key": {"type": "string"},
            },
            "required": ["key"],
        },
    )
    [py] = pycall_tools.surface((tool,))
    assert py.stub() == (
        "def lookup(key: str, mode: Literal['a', 'b'] | None = None, "
        "names: list[str] | None = None) -> str:\n"
        '    """\n    Look something up.\n    """'
    )
    assert py.bind({"key": "k", "mode": "a"}) == {"key": "k", "mode": "a"}
    with pytest.raises(pycall_tools.Rejected):
        py.bind({"key": "k", "mode": "c"})


def test_the_flags_are_not_read_as_config(tmp_path: Path) -> None:
    """A `CHICKADEE_*` variable is a config key (the environment layer), so a flag named
    that way refused the whole config as an unknown key. The flags sit outside that prefix."""
    from chickadee.state.layers import load_layers
    from chickadee.state.roots import Roots

    for name in ("project", "user", "repo_local", "checkout_local"):
        (tmp_path / name).mkdir()
    roots = Roots(
        project=tmp_path / "project",
        repo_local=tmp_path / "repo_local",
        checkout_local=tmp_path / "checkout_local",
        user=tmp_path / "user",
    )
    environ: dict[str, str] = dict.fromkeys(pycall.RETIRED_FLAGS, "1")
    environment = next(
        layer for layer in load_layers(roots, environ=environ) if layer.name == "environment"
    )
    assert environment.body == {}


def test_the_prose_is_shown_as_the_workers_words_and_the_block_kept_for_the_model(
    root: Path,
) -> None:
    """The worker's line shows a reply's text (`model_call.text_preview`): its prose alone,
    as a native reply's text, not the prose and the code run together. The model is sent its
    reply back whole, block included."""
    from chickadee.schema.journal import ModelCallP

    _, journal, rung = _python(root)
    previews = [p.payload.text_preview for p in journal.projection().lines(ModelCallP)]
    assert previews[0] == "Reading the file first."
    assert all(p is None or "```" not in p for p in previews)
    said = str(rung.requests[1].messages[-2].content)
    assert said == 'Reading the file first.\n```python\nread(path="app.py")\n```'


def test_a_forced_tool_call_is_never_sent_with_no_native_tools() -> None:
    """D220 forces the worker's native call where the provider enforces it. Under Python calls
    no native tool is offered, so nothing is forced: the block is the call."""
    from engines import ScriptedRung

    from chickadee.client.request import ChatRequest, Message
    from chickadee.schema import RetryParams

    rung = ScriptedRung([block('finish(report="r")')])
    client = pycall.PythonCallsClient(rung, pycall.Shared())  # type: ignore[arg-type]
    client.send(
        ChatRequest(
            messages=(Message(role="system", content="s"), Message(role="user", content="u")),
            params=RetryParams(temperature=0.2, max_tokens=100),
            tools=(
                ToolSchema("finish", "End.", {"properties": {"report": {"type": "string"}}}),
            ),
            force_tool=True,
        )
    )
    [sent] = rung.requests
    assert (sent.tools, sent.force_tool) == ((), False)


# -- D226: reading more replies ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "found"),
    [
        ("```Python\nread(path='a')\n```", ["read(path='a')"]),
        ("~~~py3\nread(path='a')\n~~~", ["read(path='a')"]),
        ("```python3\nread(path='a')\n```", ["read(path='a')"]),
        ("```\nread(path='a')\n```", ["read(path='a')"]),
        ("```bash\nls\n```\n```\nread(path='a')\n```", []),
        ("Reading.\n```python\nread(path='a')", ["read(path='a')"]),
        ("read(path='a')", ["read(path='a')"]),
        ("I will read(path='a') now.", []),
        (
            "```python\nread(path='a')\n```\n```python\nread(path='b')\n```",
            ["read(path='a')", "read(path='b')"],
        ),
    ],
)
def test_the_blocks_a_reply_is_read_as(reply: str, found: list[str]) -> None:
    """Any case, ~~~, an untagged only fence, an unclosed last fence, and a
    reply that is all Python calling a tool."""
    assert pycall.blocks(reply, {"read"}) == found


def test_several_blocks_run_in_order_and_stop_at_the_first_failure(root: Path) -> None:
    reply = answer(
        "```python\nprint(read(path='app.py', depth='outline'))\n```\n"
        "```python\nif read(path='app.py'):\n    raise ValueError('no')\n```\n"
        "```python\nprint(read(path='app.py'))\n```"
    )
    worker, journal, rung = py_rig(root, [reply, block('finish(report="r")')])
    worker.start("T1", "Look.", delegate(journal, root))
    first, second, third, _finished = shown_results(journal)
    assert "go" in first
    assert "ValueError: no" in second
    assert third == "not run: block 2 of this reply failed, and this block may depend on it"
    results = str(rung.requests[1].messages[-1].content)
    assert results.startswith("Result of block 1:") and "Result of block 3:" in results


def test_a_reply_with_more_blocks_than_the_cap_runs_none(root: Path) -> None:
    many = "\n".join(f"```python\nprint({n})\n```" for n in range(pycall.MAX_BLOCKS + 1))
    worker, journal, _ = py_rig(root, [answer(many), block('finish(report="r")')])
    worker.start("T1", "Look.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert f"has {pycall.MAX_BLOCKS + 1} ```python blocks" in seen and "none ran" in seen


# -- D226: the only smallest repair ------------------------------------------------------------


def test_a_repaired_block_runs_when_its_tools_change_nothing(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [block('read(path="app.py", start_line=1, end_line=2'), block('finish(report="r")')],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert seen.startswith("Ran with a fix: added `)` at line 1, column 45.")
    assert "return 1" in seen
    [noted] = [
        p for p in _refused_lines(journal, served=True) if "fixed" in (p.args_preview or "")
    ]
    assert noted.denied_reason is None


def test_a_repaired_block_that_changes_something_is_refused_with_the_block(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block('edit(path="app.py", old="return 1", new="return 2"'),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert "'(' was never closed" in seen and "^" in seen
    fixed = 'edit(path="app.py", old="return 1", new="return 2")'
    assert f"send exactly this as your next reply:\n```python\n{fixed}\n```" in seen
    assert (root / "app.py").read_text() == APP


def test_an_ambiguous_block_is_shown_its_error(root: Path) -> None:
    worker, journal, _ = py_rig(
        root, [block("x = (1 + (2\nread(path='app.py')"), block('finish(report="r")')]
    )
    worker.start("T1", "Look.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert seen.startswith("not run: your block is not valid Python:\nline 1:")
    assert "Ran with a fix" not in seen


def test_curly_quotes_are_straightened_for_any_tool(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block("edit(path=“app.py”, old=“return 1”, new=“return 2”)"),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Do it.", delegate(journal, root))
    assert "return 2" in (root / "app.py").read_text()


# -- D226: what a block shows ------------------------------------------------------------------


def test_a_last_bare_expression_is_shown(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block("text = read(path='app.py')\nlen(text.splitlines())"),
            block("print('x')"),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    value, printed, _finished = shown_results(journal)
    assert "Value: " in value
    assert "Value" not in printed


def test_a_name_from_an_earlier_block_is_explained(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block("src = read(path='app.py')"),
            block("print(src.count('def'), read(path='app.py'))"),
            block("print(nothing_here, read(path='app.py'))"),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    _first, lost, unknown, _finished = shown_results(journal)
    assert "`src` was set by an earlier block: variables do not carry over" in lost
    assert "NameError" in unknown and "earlier block" not in unknown


# -- D226: structured results ------------------------------------------------------------------


def test_grep_hands_its_matches_to_the_next_read(root: Path) -> None:
    code = (
        "for m in grep(pattern='return 1').matches:\n"
        "    print(m.path, m.line, read(path=m.path, start_line=m.line, end_line=m.line))"
    )
    worker, journal, _ = py_rig(root, [block(code), block('finish(report="r")')])
    worker.start("T1", "Look.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert "app.py 2 " in seen and "return 1" in seen
    assert "1 grep call(s) ran, 1 read call(s) ran" in seen


def test_shell_says_its_returncode(root: Path) -> None:
    code = "r = shell(command='python3 -m pytest -q')\nprint(r.returncode, r.stdout.strip())"
    worker, journal, _ = py_rig(
        root, [block(code), block('finish(report="r")')], stdout=b"3 failed\n", exit_code=1
    )
    worker.start("T1", "Test.", delegate(journal, root))
    seen, _finished = shown_results(journal)
    assert "1 3 failed" in seen


# -- D226: arguments ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function", "typed", "native"),
    [
        ("grep", {"pattern": "x", "paths": "src/"}, {"pattern": "x", "paths": ["src/"]}),
        (
            "read",
            {"path": "a", "start_line": "5", "end_line": 9},
            {"path": "a", "lines": "5-9"},
        ),
    ],
)
def test_the_two_coercions_with_one_meaning(
    function: str, typed: dict[str, object], native: dict[str, object]
) -> None:
    [tool] = [t for t in pycall_tools.PY_TOOLS if t.name == function]
    assert tool.bind(typed) == native


@pytest.mark.parametrize(
    ("function", "typed", "says"),
    [
        ("edit", {"path": "a", "old_str": "x", "new_str": "y"}, "here it is 'old'"),
        ("create", {"path": "a", "file_text": "x"}, "here it is 'content'"),
        ("create", {"path": "a", "content": "x", "command": "create"}, "do not send it"),
        ("read", {"path": "a", "lines": "1-5"}, "here it is start_line=1, end_line=5"),
    ],
)
def test_another_harnesss_argument_is_named_in_ours(
    function: str, typed: dict[str, object], says: str
) -> None:
    [tool] = [t for t in pycall_tools.PY_TOOLS if t.name == function]
    with pytest.raises(pycall_tools.Rejected, match=re.escape(says)):
        tool.bind(typed)


# -- D226: the switch is the config's ----------------------------------------------------------


def test_the_retired_flags_are_said_to_be_ignored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pycall, "_warned", __import__("threading").Event())
    said = pycall.warn_retired_flags({"CKD_PYTHON_CALLS": "1"})
    assert said is not None and 'calling = "python"' in said
    assert pycall.warn_retired_flags({"CKD_PYTHON_CALLS": "1"}) is None
    assert pycall.warn_retired_flags({}) is None


def test_the_flag_alone_turns_nothing_on(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CKD_PYTHON_CALLS", "1")
    worker, journal, rung = worker_rig(root, [calls(call("finish", "c1", report="r"))])
    worker.start("T1", "Do it.", delegate(journal, root))
    assert rung.requests[0].tools, "native tools are still sent: the config says native"


# -- review fixes ----------------------------------------------------------------------------


def test_a_non_ascii_digit_is_rejected_not_raised() -> None:
    """`"²".isdigit()` is True and `int("²")` raises: only ASCII digits are coerced."""
    [tool] = [t for t in pycall_tools.PY_TOOLS if t.name == "read"]
    with pytest.raises(pycall_tools.Rejected, match="'start_line' must be int"):
        tool.bind({"path": "a", "start_line": "²", "end_line": 3})


def test_a_literal_that_cannot_be_built_is_no_crash() -> None:
    from chickadee.tools.builtin import BUILTINS

    offered = {n: t.spec for n in ("read", "glob") if (t := BUILTINS.get(n)) is not None}
    got = pycall.tools_of(
        "str_replace_editor(command='view', path={[1]: 2})", FUNCTIONS, offered=offered
    )
    assert isinstance(got, pycall.Refused) and "calls no tool" in got.why


def test_a_refusal_for_want_of_a_sandbox_is_a_problem(tmp_path: Path) -> None:
    problems: list[str] = []

    def one(native: str, arguments: dict[str, object], n: int) -> Reply:
        return Reply("x")

    pycall.ScriptRunner(tmp_path, Passthrough()).run("read('a')", READ, one, None, problems)
    assert problems and "no sandbox" in problems[0]


def test_a_note_is_written_after_a_clean_run_and_never_counted(root: Path) -> None:
    from chickadee.journal.metrics import python_note as _python_note

    worker, journal, _ = py_rig(
        root,
        [
            block("read(path=“app.py”)"),
            block("print(read(path=“app.py”))\nraise ValueError('after')"),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    notes = [
        p for p in _refused_lines(journal, served=True) if "fixed" in (p.args_preview or "")
    ]
    assert len(notes) == 1, "the block that raised after its cleanup has no note"
    assert all(_python_note(n) for n in notes)
    assert not any(_python_note(p) for p in _refused_lines(journal))


# -- second review fixes ---------------------------------------------------------------------


def test_a_refused_call_fails_its_block_and_stops_the_rest(root: Path) -> None:
    """An `edit` whose `old` does not match is no exception, but its block failed: the
    `finish` after it must not report a change that never happened."""
    reply = answer(
        "```python\nedit(path='app.py', old='not in the file', new='x')\n```\n"
        "```python\nfinish(report='changed it')\n```"
    )
    worker, journal, _ = py_rig(root, [reply, block('finish(report="nothing changed")')])
    result = worker.start("T1", "Do it.", delegate(journal, root))
    _edit, after, _finished = shown_results(journal)
    assert after == "not run: block 1 of this reply failed, and this block may depend on it"
    assert result.report == "nothing changed"


def test_a_report_with_a_fence_is_the_report() -> None:
    """At report time no call is dispatched: a report holding a fence is the report, and a
    block that only calls `finish` gives its report."""
    from engines import ScriptedRung

    from chickadee.client.request import ChatRequest, Message
    from chickadee.schema import RetryParams

    request = ChatRequest(
        messages=(Message(role="system", content="s"), Message(role="user", content="u")),
        params=RetryParams(temperature=0.2, max_tokens=100),
    )
    with_output = "Done.\n```\n3 passed\n```"
    for written, report in [
        (with_output, with_output),
        ('```python\nfinish(report="all done")\n```', "all done"),
    ]:
        rung = ScriptedRung([answer(written)])
        client = pycall.PythonCallsClient(rung, pycall.Shared(wrap_up=True))  # type: ignore[arg-type]
        got = client.send(request)
        assert (got.text, got.tool_calls) == (report, ())


class _Ending:
    """A role whose every call ends its tool use, as the call cap does."""

    def call(self, call: Any, cancel: Any) -> Reply:
        return Reply("stopped: the call cap", ends=True)


def test_blocks_after_tool_use_ended_do_not_run(tmp_path: Path) -> None:
    shared = pycall.Shared(tools=(ToolSchema("read", "", {}),))
    runner = pycall.ScriptRunner(tmp_path, Passthrough(), allow_unsandboxed=True)
    role = pycall.PythonCallsRole(_Ending(), shared, runner)  # type: ignore[arg-type]

    def block_call(n: int) -> Any:
        from chickadee.schema.adapter import ToolCall

        args = {"code": "read(path='a')", "turn": "t1", "block": n}
        return ToolCall(id=f"c{n}", name="python", arguments=args)

    first = role.call(block_call(1), None)
    assert first.ends
    second = role.call(block_call(2), None)
    assert second.content == "not run: tool use ended at block 1 of this reply"


def test_a_block_that_failed_set_no_names(root: Path) -> None:
    worker, journal, _ = py_rig(
        root,
        [
            block("x = read(path='app.py')\nraise ValueError('after')"),
            block("print(x, read(path='app.py'))"),
            block('finish(report="r")'),
        ],
    )
    worker.start("T1", "Look.", delegate(journal, root))
    _failed, lost, _finished = shown_results(journal)
    assert "NameError" in lost and "earlier block" not in lost


def test_the_metrics_do_not_count_a_note_as_a_call(root: Path) -> None:
    from chickadee.journal.metrics import derive_metrics

    worker, journal, _ = py_rig(
        root, [block("read(path=“app.py”)"), block('finish(report="r")')]
    )
    worker.start("T1", "Look.", delegate(journal, root))
    run = journal.projection()
    lines = [p.payload for p in run.lines(ToolCallP)]
    notes = sum(p.tool == "python" and p.denied_reason is None for p in lines)
    assert notes == 1
    assert derive_metrics(run).tool_calls == len(lines) - notes


# -- third review fixes ---------------------------------------------------------------------


def test_a_block_skipped_after_a_failure_is_on_the_trace(root: Path) -> None:
    reply = answer(
        "```python\nif read(path='app.py'):\n    raise ValueError('no')\n```\n"
        "```python\nprint(read(path='app.py'))\n```"
    )
    worker, journal, _ = py_rig(root, [reply, block('finish(report="r")')])
    worker.start("T1", "Look.", delegate(journal, root))
    reasons = [p.denied_reason or "" for p in _refused_lines(journal)]
    assert any(r.startswith("not run: block 1 of this reply failed") for r in reasons)


def test_a_read_that_finds_nothing_fails_no_block(root: Path) -> None:
    """A probing loop's miss is an answer the script handled, not a failed block."""
    reply = answer(
        "```python\nfor p in ['nope.py', 'app.py']:\n    print(p, read(path=p)[:9])\n```\n"
        "```python\nprint(read(path='app.py', depth='outline'))\n```"
    )
    worker, journal, _ = py_rig(root, [reply, block('finish(report="r")')])
    worker.start("T1", "Look.", delegate(journal, root))
    _first, second, _finished = shown_results(journal)
    assert not second.startswith("not run")


def test_an_untagged_fence_of_output_is_no_block() -> None:
    assert pycall.blocks("Tests pass:\n```\n5 passed\n```", {"read"}) == []
    assert pycall.blocks("```\nread(path='a')\n```", {"read"}) == ["read(path='a')"]


def test_a_change_before_the_reads_is_never_named_to_send_first() -> None:
    functions = {**FUNCTIONS, "grep": "grep"}
    got = pycall.tools_of("edit('a', 'x', 'y')\ngrep('y')", functions)
    assert isinstance(got, pycall.Refused) and "send this block first" not in got.why
