"""Python tool calling: the worker calls its tools by Python blocks instead of
native tool calls, when `[tools.worker] calling = "python"`.

A reply is optional prose and up to five ```python blocks, run in order. A
block may loop and branch. It may use any number of tools that change nothing, or one tool
that changes something, alone.

Two thin wrappers around what exists, and nothing else changes:

- `PythonCallsClient` wraps the `ModelClient`. It sends no native tools: their schemas go into
  the system message as Python stubs. Each block of a reply becomes one `python` tool call,
  so the loop (`tools/loop.ask`) runs them as it runs several native calls; prose outside the
  blocks is the reply's text, shown as it always was. Earlier turns are sent back as text.
- `PythonCallsRole` wraps the role. A `python` call is read (`pycall_repair.read_block`),
  checked (`tools_of`), then run by `ScriptRunner` in its own process under the sandbox
  backend. Each call the script makes comes back over a pipe and goes to the wrapped role's
  own `call`: the same dispatcher, refusals, steps and journal lines a native call gets.

**The security boundary** is the child process, never this one: the script runs in a separate
`python -I` under `isolation` (bubblewrap on Linux): read-only filesystem, no network,
credentials scrubbed from its env, spawned only through `ProcessSupervisor`. It can change
nothing except through the tools it names, and every such call is the harness's to refuse.
Without a working sandbox the script is not run, unless `[sandbox] allow_unsandboxed_python`
says otherwise (then it is still a separate process, never `exec` here). The static checks
below give the model clear errors; they are not the boundary.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import textwrap
import threading
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from chickadee import pycall_tools
from chickadee.adapters.base import SHARED_TEMPLATES, load_template
from chickadee.client.request import ChatRequest, Message, ModelClient, TextPart, ToolSchema
from chickadee.context.result_data import Completed, Matches, Paths, ResultData
from chickadee.edits.lines import split_lines
from chickadee.isolation.base import SandboxBackend, level_of
from chickadee.isolation.base import select as select_backend
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.journal import metrics
from chickadee.journal.previews import args_preview
from chickadee.process import ProcessSession, ProcessSupervisor, scrub_env
from chickadee.process.supervisor import Cancelled
from chickadee.pycall_repair import Broken, Repaired, read_block
from chickadee.pycall_tools import PyTool, Rejected
from chickadee.schema.adapter import Completion, ParseError, StopReason, ToolCall
from chickadee.schema.journal import ToolCallP
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import args_sha
from chickadee.tools.foreign import translate
from chickadee.tools.loop import Reply, Role

if TYPE_CHECKING:
    from chickadee.box.sink import BoxSink

__all__ = [
    "IMPORTS",
    "MAX_BLOCKS",
    "RETIRED_FLAGS",
    "SCRIPT_TOOL",
    "PythonCallsClient",
    "PythonCallsRole",
    "ScriptRunner",
    "Shared",
    "blocks",
    "changes_something",
    "instructions",
    "split",
    "suggest",
    "tools_of",
    "traced",
    "traced_fixes",
    "warn_retired_flags",
]

SCRIPT_TOOL: Final = metrics.SCRIPT_TOOL
MAX_BLOCKS: Final = 5
"""Blocks one reply may carry; a reply with more runs none."""
RETIRED_FLAGS: Final = ("CKD_PYTHON_CALLS", "CKD_PYTHON_CALLS_UNSANDBOXED")
"""D225's switches, no longer read: the config holds the arm's switch."""
IMPORTS: Final = frozenset(
    {
        "re",
        "json",
        "textwrap",
        "difflib",
        "itertools",
        "collections",
        "functools",
        "math",
        "string",
    }
)
"""Modules a block may import: each only computes. `os`, `pathlib`, `io` and
`subprocess` would reach files around the tools and the journal."""
RUNNER: Final = Path(__file__).with_name("pycall_runner.py")

_OPEN = re.compile(r"^[ \t]*(```|~~~)[ \t]*([A-Za-z0-9_+.-]*)[ \t]*$")
"""A fence's opening line: its marker and its tag."""
_PYTHON_TAGS: Final = frozenset({"python", "python3", "py", "py3"})
_OBJECT = re.compile(r"\{.*\}", re.S)
"""A JSON object in a reply's text: a native call's arguments written out bare."""
_PREFERRED: Final = (
    "read",
    "grep",
    "glob",
    "grep_in",
    "symbol",
    "neighbors",
    "impact",
    "tests_for",
    "edit",
    "edit_many",
    "create",
    "delete",
    "move",
    "shell",
    "finish",
)
"""Which tool bare arguments are read as, first match first: the reads before the writes, so
an ambiguous `{"path":...}` is named as a `read`, never a `delete`."""
_NAME_ERROR = re.compile(r"NameError: name '(\w+)' is not defined")
_log = logging.getLogger(__name__)
_BANNED_NAMES: Final = frozenset({"open", "exec", "eval", "compile", "__import__"})
_warned = threading.Event()
"""Set once the retired flags have been said to be ignored: a process says it once."""


# -- what the model is shown ------------------------------------------------------------------


def instructions(tools: tuple[ToolSchema, ...]) -> str:
    stubs = pycall_tools.stubs(pycall_tools.surface(tools))
    return (
        load_template(SHARED_TEMPLATES, "python_calls.md")
        .replace("{imports}", ", ".join(sorted(IMPORTS)))
        .replace("{max_blocks}", str(MAX_BLOCKS))
        .replace("{stubs}", stubs)
    )


# -- reading a reply --------------------------------------------------------------------------


def split(text: str, functions: Collection[str] = ()) -> tuple[str, list[str]]:
    """A reply's prose and its Python blocks. A fence tagged `python`,
    `python3`, `py` or `py3`, in any case, opened by ``` or ~~~, is a block; an untagged fence
    is one when it is the reply's only fence; a fence never closed runs to the reply's end. A
    reply with no fence that parses whole as Python and calls one of `functions` is one block.
    Blocks are dedented: one copied from the instructions' examples keeps their indent."""
    lines = text.split("\n")
    fences: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        opened = _OPEN.match(lines[i])
        if opened is None:
            i += 1
            continue
        marker, tag = opened.group(1), opened.group(2).lower()
        close = next((j for j in range(i + 1, len(lines)) if lines[j].strip() == marker), None)
        end = len(lines) if close is None else close
        fences.append((i, end, tag))
        i = end + 1
    # An untagged fence is a block only when it is the only one and holds Python that calls
    # something: one of test output or a traceback is the reply's words.
    taken = [
        (start, end)
        for start, end, tag in fences
        if tag in _PYTHON_TAGS
        or (tag == "" and len(fences) == 1 and _calls_any("\n".join(lines[start + 1 : end])))
    ]
    if not taken:
        if not fences and _calls_one_of(text, functions):
            return "", [textwrap.dedent(text).strip("\n")]
        return text.strip(), []
    found = [textwrap.dedent("\n".join(lines[s + 1 : e])) for s, e in taken]
    inside = {n for s, e in taken for n in range(s, min(e + 1, len(lines)))}
    prose = "\n".join(line for n, line in enumerate(lines) if n not in inside)
    return prose.strip(), found


def blocks(text: str, functions: Collection[str] = ()) -> list[str]:
    return split(text, functions)[1]


def _calls_any(code: str) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(n, ast.Call) for n in ast.walk(tree))


def _calls_one_of(text: str, functions: Collection[str]) -> bool:
    if not functions or not text.strip():
        return False
    try:
        tree = ast.parse(textwrap.dedent(text))
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions
        for n in ast.walk(tree)
    )


@dataclass(frozen=True)
class Refused:
    why: str


def changes_something(native: str) -> bool:
    """Whether a native tool changes anything: `ToolSpec.mutates`, the one definition. A tool
    the registry does not know is taken to change something."""
    tool = BUILTINS.get(native)
    return tool is None or tool.spec.mutates != "nothing"


def tools_of(
    code: str,
    functions: Mapping[str, str],
    *,
    changes: Callable[[str], bool] = changes_something,
    offered: Mapping[str, Any] | None = None,
) -> frozenset[str] | Refused:
    """The native tools a parsed block uses, or why it may not run:
    any number that change nothing, or one that changes something, alone. `functions` maps
    each Python function to its native tool: `edit` and `edit_many` are one tool. `offered`,
    the role's specs by name, lets a refusal name the call for another harness's tool. A hint
    for the model, not the boundary: the runner gives the script only these tools' functions,
    the parent checks every call, and the sandbox holds the rest."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return Refused(
            f"not run: your block is not valid Python: {exc.msg} (line {exc.lineno})"
        )
    used: set[str] = set()
    for node in ast.walk(tree):
        refused = _import_refused(node)
        if refused is not None:
            return refused
        if isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                return Refused(f"not run: `{node.id}` is not available; use the tool")
            if node.id in functions:
                used.add(node.id)
    natives = frozenset(functions[name] for name in used)
    if not natives:
        foreign = _foreign_call(tree, functions, offered)
        call_one = "Call one of: " + ", ".join(sorted(functions))
        if foreign is not None:
            return Refused(f"not run: your block calls no tool here. {foreign}")
        return Refused(f"not run: your block calls no tool. {call_one}")
    changing = {n for n in natives if changes(n)}
    if changing and len(natives) > 1:
        names = ", ".join(sorted(used))
        why = (
            f"not run: a block that changes something uses no other tool, and yours uses "
            f"{names}. Send the reads first, then the change in a block after them."
        )
        first = _first_part(code, tree, functions, changes)
        if first is not None:
            why += f"\nDo: send this block first:\n```python\n{first}\n```"
        return Refused(why)
    return natives


def _import_refused(node: ast.AST) -> Refused | None:
    modules: list[str] = []
    if isinstance(node, ast.Import):
        modules = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        modules = [node.module or ""] if node.level == 0 else ["."]
    for module in modules:
        if module.split(".")[0] not in IMPORTS:
            return Refused(
                f"not run: `{module}` cannot be imported in a block; only "
                f"{', '.join(sorted(IMPORTS))}. Use the tools for files and commands."
            )
    return None


def _first_part(
    code: str, tree: ast.Module, functions: Mapping[str, str], changes: Callable[[str], bool]
) -> str | None:
    """The statements before the one that makes a block mix a changing tool with another, when
    they parse alone and use a tool: the block to send first."""
    seen: set[str] = set()
    lines = code.split("\n")
    for statement in tree.body:
        here = {
            functions[n.id]
            for n in ast.walk(statement)
            if isinstance(n, ast.Name) and n.id in functions
        }
        together = seen | here
        if any(changes(n) for n in together) and len(together) > 1:
            first = "\n".join(lines[: statement.lineno - 1]).strip("\n")
            # Only reads go first: a change before them is never the block to send first.
            if not seen or not first or any(changes(n) for n in seen):
                return None
            try:
                ast.parse(first)
            except SyntaxError:
                return None
            return first
        seen = together
    return None


def _foreign_call(
    tree: ast.Module, functions: Mapping[str, str], offered: Mapping[str, Any] | None
) -> str | None:
    """Another harness's tool called as a function (`str_replace_editor(command="view",...)`):
    the call to send instead, from `tools/foreign.py`'s one table, as Python."""
    if not offered:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        args: dict[str, object] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            try:
                args[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                # Not a literal, or one `literal_eval` cannot build (`{[1]: 2}`): no call to
                # name, never a crash of the turn.
                return None
        found = translate(node.func.id, args, offered)
        if found is None:
            continue
        if found.tool is None:
            return f"`{node.func.id}` has no equivalent here: {found.note}"
        for tool in pycall_tools.PY_TOOLS:
            if tool.native == found.tool and tool.name in functions:
                back = pycall_tools.from_native(tool, found.arguments)
                if back is not None:
                    call = pycall_tools.as_python(tool, back)
                    return f"Do: send exactly this:\n```python\n{call}\n```"
        return f"Call {found.tool} instead."
    return None


# -- the shared state of one round ------------------------------------------------------------


@dataclass
class Shared:
    """What the client and the role of one round both read: the tools offered, whether the
    loop now wants the report rather than another call, and what earlier blocks did."""

    tools: tuple[ToolSchema, ...] = ()
    wrap_up: bool = False
    blockless: int = 0
    refused: Callable[[str, str], None] | None = None
    """Records a block refused before any tool ran (its text, the refusal) on the role's
    trace, as a refused call is: without it, the refusal is seen by the model and no one
    else."""
    noted: Callable[[str, str], None] | None = None
    """Records a block that ran after a cleanup or a repair (its text as written, what was
    done), so the count D226 decision 10 asks for is on the trace."""
    failed: dict[str, str] = field(default_factory=dict[str, str])
    """Reply id -> what each of its later blocks is answered, once one failed or ended tool
    use: the blocks after it do not run."""
    assigned: set[str] = field(default_factory=set[str])
    """Names the round's earlier blocks assigned: a `NameError` for one is explained."""


# -- the client -------------------------------------------------------------------------------


_NO_BLOCK: Final = (
    "not run: your reply has no ```python block, so no tool ran. There are no native tool "
    "calls here, and a JSON object is not a call."
)
_DO_EXACT: Final = "Do: send exactly this as your next reply:\n```python\n{call}\n```"
_DO_ANY: Final = (
    "Do: reply with a ```python block that calls a tool, for example:\n"
    '```python\nread(path="README.md")\n```\n'
    "and when the task is done:\n"
    '```python\nfinish(report="what you changed, and what you ran")\n```'
)
_GAVE_UP: Final = (
    "(no report: the worker wrote no ```python block in {n} replies in a row, so it called "
    "no tool. Its last reply: {last!r})"
)
_MANY_BLOCKS: Final = (
    "not run: your reply has {n} ```python blocks, and a reply may carry at most {cap}; none "
    "ran. Send at most {cap}."
)


class PythonCallsClient:
    """`ModelClient` that speaks Python blocks to the model and native calls to the loop."""

    def __init__(self, inner: ModelClient, shared: Shared, *, max_blockless: int = 3) -> None:
        self._inner = inner
        self._shared = shared
        self._max_blockless = max_blockless

    def close(self) -> None:
        self._inner.close()

    def send(self, request: ChatRequest) -> Completion:
        if request.tools:
            self._shared.tools = request.tools
        sent = replace(
            request,
            messages=as_text(request.messages, instructions(self._shared.tools)),
            tools=(),
            # No native tool is offered, so none can be forced: the block is the call.
            force_tool=False,
        )
        completion = self._inner.send(sent)
        if completion.stop_reason is StopReason.MAX_TOKENS:
            return completion
        if self._shared.wrap_up:
            # The loop asked for the report, and dispatches no call now (`loop._after_the_end`):
            # the reply is the report, as a reply with no tool call is under native calling.
            # A report holding a fence (test output, a diff) is never turned into a call.
            return completion.model_copy(
                update={"text": _report(completion.text), "tool_calls": ()}
            )
        prose, found = split(completion.text, _function_names(self._shared.tools))
        if found and len(found) <= MAX_BLOCKS:
            self._shared.blockless = 0
            # The reply's text is its prose alone, so it is shown as the worker's words, as a
            # native reply's text is; the code travels in the calls, and `as_text` puts the
            # blocks back after the prose.
            turn = uuid.uuid4().hex[:12]
            return _calling(
                completion,
                [{"code": code, "turn": turn, "block": n} for n, code in enumerate(found, 1)],
                text=prose,
            )
        if found:
            many = _MANY_BLOCKS.format(n=len(found), cap=MAX_BLOCKS)
            return _calling(completion, [{"error": many}])
        _log.warning(
            "python calls: a reply with no block (%d in a row): text %r, native calls %s, "
            "%d tokens out",
            self._shared.blockless + 1,
            completion.text[:500],
            [(c.name, c.arguments) for c in completion.tool_calls],
            completion.usage.completion_tokens,
        )
        self._shared.blockless += 1
        if self._shared.blockless >= self._max_blockless:
            # The model will not write a block: the round ends saying so, never on its text
            # taken as a report (2026-10-23: a bare `{"path":...}` was Athena's report).
            last = " ".join(completion.text.split())[:300]
            gave_up = _GAVE_UP.format(n=self._shared.blockless, last=last)
            return completion.model_copy(
                update={"text": gave_up, "tool_calls": (), "stop_reason": StopReason.END_TURN}
            )
        instead = suggest(completion, pycall_tools.surface(self._shared.tools))
        do = _DO_ANY if instead is None else _DO_EXACT.format(call=instead)
        return _calling(completion, [{"error": f"{_NO_BLOCK}\n{do}", "reply": completion.text}])


def _report(text: str) -> str:
    """A report-time reply as the report: the text as written, or, when it is one block that
    only calls `finish` with a written-out report, that report."""
    _, found = split(text)
    if len(found) != 1:
        return text
    try:
        tree = ast.parse(found[0])
    except (SyntaxError, ValueError):
        return text
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return text
    call = tree.body[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        return text
    given = [*call.args, *(k.value for k in call.keywords if k.arg == "report")]
    if call.func.id != "finish" or len(given) != 1:
        return text
    report = given[0]
    if isinstance(report, ast.Constant) and isinstance(report.value, str):
        return report.value
    return text


def _function_names(tools: tuple[ToolSchema, ...]) -> frozenset[str]:
    return frozenset(f.name for f in pycall_tools.surface(tools))


def suggest(completion: Completion, functions: tuple[PyTool, ...]) -> str | None:
    """The block a reply with no block most likely meant, as one line of Python: its first
    native call, or bare JSON arguments in its text, converted. A hint in the refusal, never
    run: the model sends it, or does not (a guess is never executed)."""
    for call in completion.tool_calls:
        for tool in functions:
            if tool.native == call.name:
                args = pycall_tools.from_native(tool, call.arguments)
                if args is not None:
                    return pycall_tools.as_python(tool, args)
    found = _OBJECT.search(completion.text)
    if found is None:
        return None
    try:
        raw = json.loads(found.group(0))
    except ValueError:
        return None
    if not isinstance(raw, dict) or not raw:
        return None
    named = cast("dict[str, object]", raw)
    if isinstance(named.get("name"), str) and isinstance(named.get("arguments"), dict):
        # A whole native call written out: `{"name": "read", "arguments": {...}}`.
        wanted = cast("str", named["name"])
        named = cast("dict[str, object]", named["arguments"])
        order = [t for t in functions if t.native == wanted]
    else:
        rank = {n: i for i, n in enumerate(_PREFERRED)}
        order = sorted(functions, key=lambda t: rank.get(t.name, len(rank)))
    for tool in order:
        args = pycall_tools.from_native(tool, named)
        if args is not None:
            return pycall_tools.as_python(tool, args)
    return None


def _calling(
    completion: Completion, arguments: list[dict[str, object]], *, text: str | None = None
) -> Completion:
    calls = tuple(
        ToolCall(id=f"py-{uuid.uuid4().hex[:12]}", name=SCRIPT_TOOL, arguments=a)
        for a in arguments
    )
    update: dict[str, object] = {"tool_calls": calls, "stop_reason": StopReason.TOOL_USE}
    if text is not None:
        update["text"] = text
    return completion.model_copy(update=update)


def as_text(messages: tuple[Message, ...], preamble: str) -> tuple[Message, ...]:
    """The conversation with no native tool messages: the instructions appended to the
    system message, each call as the text that made it, each turn's results as one user
    message, each labelled by its block when there are several."""
    out: list[Message] = []
    pending: list[str] = []

    def flush() -> None:
        if len(pending) == 1:
            out.append(Message(role="user", content="Result:\n" + pending[0]))
        elif pending:
            labelled = (f"Result of block {n}:\n{text}" for n, text in enumerate(pending, 1))
            out.append(Message(role="user", content="\n\n".join(labelled)))
        pending.clear()

    for m in messages:
        if m.role == "tool":
            pending.append(_text(m.content))
            continue
        flush()
        if m.role == "assistant" and m.tool_calls:
            text = _text(m.content)
            for c in m.tool_calls:
                code = c.arguments.get("code") if c.name == SCRIPT_TOOL else _python(c)
                # A script's block is put back after its prose, as the model wrote it; a call
                # made natively before the switch is written as its Python. A `python` call
                # with no code (a reply refused for its blocks) kept its text whole.
                if isinstance(code, str):
                    text = f"{text}\n```python\n{code.rstrip()}\n```".lstrip("\n")
            out.append(Message(role="assistant", content=text))
            continue
        out.append(m)
    flush()
    for i, m in enumerate(out):
        if m.role == "system" and isinstance(m.content, str):
            out[i] = replace(m, content=f"{m.content}\n\n{preamble}")
            break
    else:
        out.insert(0, Message(role="system", content=preamble))
    return tuple(out)


def _python(call: ToolCall) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return f"{call.name}({args})"


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, tuple):
        parts = cast("tuple[object, ...]", content)
        return "\n".join(p.text if isinstance(p, TextPart) else "[image]" for p in parts)
    return str(content)


# -- the role ---------------------------------------------------------------------------------


_FIXED: Final = "Ran with a fix: {what}. Write the block complete next time."
_NOT_VALID: Final = "not run: your block is not valid Python:\n{error}"
_AFTER_FAILED: Final = (
    "not run: block {n} of this reply failed, and this block may depend on it"
)
_AFTER_ENDED: Final = "not run: tool use ended at block {n} of this reply"


class PythonCallsRole[A]:
    """`tools.loop.Role` that runs a `python` call's script, each of its calls through the
    wrapped role. Every other call (one owed from before the switch) goes straight through."""

    def __init__(self, inner: Role[A], shared: Shared, runner: ScriptRunner) -> None:
        self._inner = inner
        self._shared = shared
        self._runner = runner

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        if call.name != SCRIPT_TOOL:
            return self._inner.call(call, cancel)
        error = call.arguments.get("error")
        if error is not None:
            self._refuse(str(call.arguments.get("reply", "")), str(error))
            return Reply(str(error))
        code = str(call.arguments.get("code", ""))
        turn = call.arguments.get("turn")
        block = call.arguments.get("block")
        if isinstance(turn, str) and turn in self._shared.failed:
            return self._refuse(code, self._shared.failed[turn])
        reply, failed = self._run(code, call.id, cancel)
        if isinstance(turn, str) and isinstance(block, int) and (failed or reply.ends):
            said = _AFTER_ENDED if reply.ends else _AFTER_FAILED
            self._shared.failed[turn] = said.format(n=block)
        return reply

    def _run(
        self, code: str, call_id: str, cancel: threading.Event | None
    ) -> tuple[Reply, bool]:
        functions = pycall_tools.surface(self._shared.tools)
        params = {f.name: frozenset(p.name for p in f.params) for f in functions}
        read = read_block(code, params)
        if isinstance(read, Broken):
            return self._refuse(code, _NOT_VALID.format(error=read.error)), True
        offered = {
            t.name: tool.spec for t in self._shared.tools if (tool := BUILTINS.get(t.name))
        }
        natives = tools_of(read.code, {f.name: f.native for f in functions}, offered=offered)
        if isinstance(natives, Refused):
            return self._refuse(code, natives.why), True
        note: str | None = None
        if isinstance(read, Repaired):
            if any(changes_something(n) for n in natives):
                # A guess is run only for tools that change nothing.
                why = (
                    f"{_NOT_VALID.format(error=read.error)}\n{_DO_EXACT.format(call=read.code)}"
                )
                return self._refuse(code, why), True
            note = _FIXED.format(what="; ".join(read.what))
        done = ", ".join((*read.cleaned, *([note] if note else [])))

        answered: list[tuple[str, Reply]] = []

        def one(native: str, arguments: dict[str, object], n: int) -> Reply:
            reply = self._inner.call(
                ToolCall(id=f"{call_id}.{n}", name=native, arguments=arguments), cancel
            )
            answered.append((native, reply))
            return reply

        group = tuple(f for f in functions if f.native in natives)
        problems: list[str] = []
        reply = self._runner.run(
            read.code, group, one, cancel, problems, lost=frozenset(self._shared.assigned)
        )
        # A change the dispatcher refused, or that failed (an `edit` whose `old` did not match),
        # fails its block as an exception does: the reply's later blocks may depend on it. A
        # read that found nothing is an answer the script can act on, not a failure.
        failed = bool(problems) or any(
            r.refused and changes_something(native) for native, r in answered
        )
        if problems:
            self._refuse(code, "; ".join(problems))
        elif done and not failed and self._shared.noted is not None:
            # After the run, and only for a block that ran clean: a note says what ran.
            self._shared.noted(code, done)
        if not failed:
            # Only a block that ran through set its names.
            self._shared.assigned |= _assigned(read.code)
        if reply.ends:
            self._shared.wrap_up = True
        if note is not None:
            reply = Reply(
                f"{note}\n\n{_text(reply.content)}", ends=reply.ends, final=reply.final
            )
        return reply, failed

    def _refuse(self, code: str, why: str) -> Reply:
        if self._shared.refused is not None:
            self._shared.refused(code, why)
        return Reply(why)

    def after_end(self) -> str:
        self._shared.wrap_up = True
        return self._inner.after_end() + " (as plain text, with no python block)"

    def parse(self, text: str) -> A | ParseError:
        return self._inner.parse(text)

    def reask(self, error: ParseError) -> str:
        self._shared.wrap_up = True
        return self._inner.reask(error)

    def cut(self) -> ParseError:
        return self._inner.cut()


def _assigned(code: str) -> set[str]:
    """The names a block binds: an assignment, a loop's target, a definition or an import."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
    return out


# -- the runner -------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ran:
    """One call the script made: as written in Python, and what it was answered."""

    function: str
    arguments: dict[str, object]
    reply: Reply
    rejected: bool = False
    """Its arguments did not bind (`Rejected`): raised in the script, never run."""

    def label(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.function}({args})"


@dataclass(frozen=True)
class ScriptRunner:
    """Runs one script in a sandboxed child, its tool calls answered by `call`."""

    root: Path
    backend: SandboxBackend
    allow_unsandboxed: bool = False
    timeout_s: int = 900
    """The whole script, its tool calls (and any approval they wait on) included."""
    max_calls: int = 50
    max_output_chars: int = 12000
    python: str = field(default=sys.executable)

    def run(
        self,
        code: str,
        functions: tuple[PyTool, ...],
        call: Callable[[str, dict[str, object], int], Reply],
        cancel: threading.Event | None = None,
        problems: list[str] | None = None,
        *,
        lost: frozenset[str] = frozenset(),
    ) -> Reply:
        """Run `code` with `functions`. `call` is handed each call's native tool and native
        arguments, already checked against its Python signature. `problems`, when given,
        gains what no dispatcher saw: each rejected call, and the script's error. `lost`
        holds the names earlier blocks assigned, so a `NameError` for one is explained."""
        if level_of(self.backend) is SandboxLevel.NONE and not self.allow_unsandboxed:
            why = (
                "not run: no sandbox is available on this machine to run the block in (the "
                "person may set [sandbox] allow_unsandboxed_python = true and require_sandbox "
                "= false to run it in a plain child process)"
            )
            # A refusal like any other: traced, and the reply's later blocks not run.
            if problems is not None:
                problems.append(why)
            return Reply(why)
        argv = self.backend.wrap(
            Isolation(argv=(self.python, "-I", str(RUNNER)), cwd=self.root, network_off=True)
        )
        by_name = {f.name: f for f in functions}
        ran: list[_Ran] = []
        stopped: Reply | None = None
        capped = False
        done: dict[str, Any] = {}
        try:
            with ProcessSupervisor().session(
                argv, cwd=self.root, timeout_s=self.timeout_s, env=scrub_env(), cancel=cancel
            ) as session:
                lines = _Lines(session)
                spec = {"code": code, "functions": {f.name: f.positional() for f in functions}}
                session.write((json.dumps(spec) + "\n").encode())
                while True:
                    msg = lines.next()
                    if msg is None:
                        done = {"error": "the block's process exited without finishing"}
                        break
                    if msg.get("done"):
                        done = msg
                        break
                    if len(ran) >= self.max_calls:
                        capped = True
                        session.write(b'{"stop": true}\n')
                        continue
                    raw = msg.get("call")
                    arguments: dict[str, object] = (
                        dict(cast("dict[str, object]", raw)) if isinstance(raw, dict) else {}
                    )
                    function = by_name.get(str(msg.get("fn")))
                    if function is None:  # a forged line: it names no function it was given
                        session.write(b'{"stop": true}\n')
                        continue
                    try:
                        native = function.bind(arguments)
                    except Rejected as exc:
                        ran.append(_Ran(function.name, arguments, Reply(str(exc)), True))
                        answer: dict[str, object] = {"raise": "TypeError", "message": str(exc)}
                        session.write((json.dumps(answer) + "\n").encode())
                        continue
                    reply = call(function.native, native, len(ran))
                    ran.append(_Ran(function.name, arguments, reply))
                    stop = reply.final is not None or reply.ends
                    if stop:
                        stopped = reply
                    answer = {
                        "result": _text(reply.content),
                        "stop": stop,
                        "tool": function.native,
                        "data": _encoded(reply.data),
                    }
                    session.write((json.dumps(answer) + "\n").encode())
        except TimeoutError:
            done = {"error": f"the block was stopped after {self.timeout_s}s"}
        except Cancelled:
            done = {"error": "the block was stopped: the run was cancelled"}
        if done.get("error") and lost:
            done = {**done, "error": _explained(str(done["error"]), lost)}
        if problems is not None:
            problems.extend(f"{r.label()}: {_text(r.reply.content)}" for r in ran if r.rejected)
            if done.get("error"):
                problems.append(_tail(str(done["error"]), 3))
        return self._reply(ran, done, stopped, capped)

    def _reply(
        self,
        ran: list[_Ran],
        done: Mapping[str, Any],
        stopped: Reply | None,
        capped: bool,
    ) -> Reply:
        printed = str(done.get("output") or "")
        value = done.get("value")
        error = done.get("error")
        ends = stopped is not None and stopped.ends
        final = stopped.final if stopped is not None else None
        plain = not printed and not value and not error and not capped
        if len(ran) == 1 and not ran[0].rejected and plain:
            # One call that printed nothing: exactly the native call's answer.
            only = ran[0].reply
            return Reply(only.content, ends=only.ends, final=only.final)
        parts: list[str] = []
        if printed:
            parts.append(printed.rstrip())
        else:
            parts.extend(
                f"[{i}] {r.label()}\n{_text(r.reply.content)}" for i, r in enumerate(ran, 1)
            )
        if value:
            # A block's last bare expression, as a Python prompt shows it.
            parts.append(f"Value: {value}")
        counts: dict[str, int] = {}
        for r in ran:
            if not r.rejected:
                counts[r.function] = counts.get(r.function, 0) + 1
        notes = [
            ", ".join(f"{n} {name} call(s) ran" for name, n in counts.items()) or "no call ran"
        ]
        rejected = sum(r.rejected for r in ran)
        if rejected:
            notes.append(f"{rejected} rejected before running (TypeError)")
        if capped:
            notes.append(f"stopped at the cap of {self.max_calls} calls")
        if stopped is not None:
            notes.append("the block was ended by its last call")
        if error:
            parts.append(f"Error:\n{_tail(str(error), 20)}")
        text = "\n\n".join(parts).strip() or "(the block printed nothing)"
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n[output cut; print less]"
        return Reply(f"{text}\n\n[{'; '.join(notes)}]", ends=ends, final=final)


def _encoded(data: ResultData | None) -> dict[str, object] | None:
    """A result's data as the child reads it (`pycall_runner._wrapped`)."""
    match data:
        case Matches(rows=rows):
            return {"kind": "matches", "rows": [[m.path, m.line, m.text] for m in rows]}
        case Paths(paths=paths):
            return {"kind": "paths", "paths": list(paths)}
        case Completed(returncode=code, stdout=out, stderr=err):
            return {"kind": "completed", "returncode": code, "stdout": out, "stderr": err}
        case None:
            return None


def _explained(error: str, lost: frozenset[str]) -> str:
    """A `NameError` for a name an earlier block assigned, explained: each
    block runs in a fresh process, and a model trained on a notebook expects its state kept."""
    found = _NAME_ERROR.search(error)
    if found is None or found.group(1) not in lost:
        return error
    return (
        f"{error}\n(`{found.group(1)}` was set by an earlier block: variables do not carry "
        "over between blocks; get the value again in this block)"
    )


def traced(sink: BoxSink) -> Callable[[str, str], None]:
    """`refused` for `python_calls`: a block refused before any tool ran, as a refused
    `python` call on the role's trace, with what the model wrote as its arguments. The REPL
    shows it as it shows any refusal (`→ python … ✗ refused: …`)."""

    def refused(reply: str, why: str) -> None:
        args = {"reply": reply}
        sink.line(
            ToolCallP(
                role=sink.role,
                tool=SCRIPT_TOOL,
                args_sha=args_sha(args),
                outcome="served",
                result_tokens=0,
                denied_reason=why,
                args_preview=args_preview(args),
            )
        )

    return refused


def traced_fixes(sink: BoxSink) -> Callable[[str, str], None]:
    """`noted` for `python_calls`: a block that ran after a cleanup or a repair, as a served
    `python` line with the block as written and what was done."""

    def note(written: str, done: str) -> None:
        args = {"reply": written, "fixed": done}
        sink.line(
            ToolCallP(
                role=sink.role,
                tool=SCRIPT_TOOL,
                args_sha=args_sha(args),
                outcome="served",
                result_tokens=0,
                args_preview=args_preview(args),
            )
        )

    return note


def warn_retired_flags(environ: Mapping[str, str] | None = None) -> str | None:
    """D225's environment switches are no longer read. A process that has
    either set says so once, naming the config keys, so nobody believes the arm is on when it
    is not. Returns what it said, the first time."""
    env = os.environ if environ is None else environ
    set_ = [name for name in RETIRED_FLAGS if env.get(name, "") not in ("", "0")]
    if not set_ or _warned.is_set():
        return None
    _warned.set()
    said = (
        f'{", ".join(set_)} is no longer read (D226): set [tools.worker] calling = "python" '
        "and, to run a block with no sandbox, [sandbox] allow_unsandboxed_python = true"
    )
    _log.warning("python calls: %s", said)
    return said


def python_calls[A](
    client: ModelClient,
    role: Role[A],
    *,
    root: Path,
    refused: Callable[[str, str], None] | None = None,
    noted: Callable[[str, str], None] | None = None,
    allow_unsandboxed: bool = False,
) -> tuple[PythonCallsClient, PythonCallsRole[A]]:
    """The two wrappers for one round, sharing its state. The sandbox is the strongest
    backend this machine runs (`isolation.base.select`). `refused` records a block refused
    before any tool ran on the role's trace, and `noted` one that ran after a cleanup or a
    repair. `allow_unsandboxed` is `[sandbox] allow_unsandboxed_python`."""
    shared = Shared(refused=refused, noted=noted)
    runner = ScriptRunner(
        root=root, backend=select_backend(), allow_unsandboxed=allow_unsandboxed
    )
    return PythonCallsClient(client, shared), PythonCallsRole(role, shared, runner)


def _tail(text: str, n: int) -> str:
    return "\n".join(split_lines(text)[-n:])


class _Lines:
    """JSON lines read from a session's stdout."""

    def __init__(self, session: ProcessSession) -> None:
        self._session = session
        self._buf = b""

    def next(self) -> dict[str, Any] | None:
        while b"\n" not in self._buf:
            chunk = self._session.read(65536)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        try:
            msg: object = json.loads(line)
        except json.JSONDecodeError:
            msg = None
        if not isinstance(msg, dict):
            return {"done": True, "error": "the block's process wrote something unreadable"}
        return cast("dict[str, Any]", msg)
