"""The child side of Python tool calling (`pycall.py`): runs one model script, in its own
sandboxed process, never in the harness's.

Standard library only, and it imports nothing of `chickadee`: it is started as a plain file
(`python -I pycall_runner.py`). It speaks JSON lines with the harness:

- harness -> child, first line: `{"code", "functions": {name: [positional names]}}`;
- child -> harness, per tool call: `{"fn": name, "call": {...arguments}}`, answered by
  `{"result": str, "stop": bool, "tool": str, "data": {...} | null}`, or `{"raise":
  "TypeError", "message"}` when its arguments do not fit the function's signature (raised in
  the script, as Python would). `data` is what the result was made from:
  `grep`, `glob` and `shell` answer a `str` subclass carrying it;
- child -> harness, last line: `{"done": true, "output": str, "error": str | null, "value":
  str | null}`, `value` the `repr` of the block's last bare expression.

The script sees the functions of one tool (`edit` and `edit_many` are one), and nothing else
of the harness. What it
prints is captured and handed back as its output.
"""

import ast
import io
import json
import sys
import traceback
from collections.abc import Callable
from typing import Any, NamedTuple, cast


class Match(NamedTuple):
    path: str
    line: int
    text: str


_GONE = (
    "{name} is not available for this result: the call was refused or failed, or its text was "
    "cut or redacted before you saw it, so only the text is; read the text, or narrow the call"
)


class GrepResult(str):
    """`grep`'s text, and `.matches`: one `Match(path, line, text)` per hit shown."""

    _matches: list[Match] | None

    def __new__(cls, text: str, matches: list[Match] | None) -> "GrepResult":
        made = super().__new__(cls, text)
        made._matches = matches
        return made

    @property
    def matches(self) -> list[Match]:
        if self._matches is None:
            raise ValueError(_GONE.format(name=".matches"))
        return list(self._matches)


class GlobResult(str):
    """`glob`'s text, and `.paths`: the paths shown, as `glob.glob` returns them."""

    _paths: list[str] | None

    def __new__(cls, text: str, paths: list[str] | None) -> "GlobResult":
        made = super().__new__(cls, text)
        made._paths = paths
        return made

    @property
    def paths(self) -> list[str]:
        if self._paths is None:
            raise ValueError(_GONE.format(name=".paths"))
        return list(self._paths)


class ShellResult(str):
    """`shell`'s text, and `.returncode`, `.stdout`, `.stderr`, as `subprocess.run` names
    them; `.stdout` and `.stderr` are None when the output was cut."""

    _done: tuple[int, str | None, str | None] | None

    def __new__(
        cls, text: str, done: tuple[int, str | None, str | None] | None
    ) -> "ShellResult":
        made = super().__new__(cls, text)
        made._done = done
        return made

    def _held(self, name: str) -> tuple[int, str | None, str | None]:
        if self._done is None:
            raise ValueError(_GONE.format(name=f".{name}"))
        return self._done

    @property
    def returncode(self) -> int:
        return self._held("returncode")[0]

    @property
    def stdout(self) -> str | None:
        return self._held("stdout")[1]

    @property
    def stderr(self) -> str | None:
        return self._held("stderr")[2]


def _wrapped(text: str, tool: object, data: object) -> str:
    """A result as the script reads it: `grep`, `glob` and `shell` as `str` subclasses carrying
    their data, every other tool's as a plain `str`."""
    held: dict[str, Any] = cast("dict[str, Any]", data) if isinstance(data, dict) else {}
    kind = held.get("kind")
    if tool == "grep":
        rows = cast("list[list[Any]]", held["rows"]) if kind == "matches" else None
        found = None if rows is None else [Match(str(p), int(n), str(t)) for p, n, t in rows]
        return GrepResult(text, found)
    if tool == "glob":
        paths = cast("list[Any]", held["paths"]) if kind == "paths" else None
        return GlobResult(text, None if paths is None else [str(p) for p in paths])
    if tool == "shell":
        if kind != "completed":
            return ShellResult(text, None)
        out, err = held.get("stdout"), held.get("stderr")
        return ShellResult(
            text,
            (
                int(held["returncode"]),
                None if out is None else str(out),
                None if err is None else str(err),
            ),
        )
    return text


VALUE_CHARS = 4000
"""A last expression's `repr` is cut here: the harness bounds the whole answer too."""


class _Stopped(BaseException):
    """The harness ended the script: the call finished the task or hit a bound. A
    BaseException, so a script's `except Exception` does not swallow it."""


def main() -> None:
    proto_in = sys.stdin
    proto_out = sys.stdout
    spec = json.loads(proto_in.readline())
    functions: dict[str, list[str]] = spec["functions"]

    def function(name: str, params: list[str]) -> Callable[..., str]:
        def call(*args: object, **kwargs: object) -> str:
            if len(args) > len(params):
                raise TypeError(f"{name}() takes at most {len(params)} positional arguments")
            arguments = dict(zip(params, args, strict=False))
            for key, value in kwargs.items():
                if key in arguments:
                    raise TypeError(f"{name}() got multiple values for argument {key!r}")
                arguments[key] = value
            proto_out.write(json.dumps({"fn": name, "call": arguments}) + "\n")
            proto_out.flush()
            answer = json.loads(proto_in.readline())
            if answer.get("stop"):
                raise _Stopped()
            if "raise" in answer:
                raise TypeError(answer["message"])
            return _wrapped(str(answer["result"]), answer.get("tool"), answer.get("data"))

        call.__name__ = name
        return call

    namespace: dict[str, object] = {"__name__": "__script__"}
    for name, params in functions.items():
        namespace[name] = function(name, params)
    captured = io.StringIO()
    sys.stdout = captured
    sys.stdin = io.StringIO("")
    error = None
    value = None
    try:
        value = _run(spec["code"], namespace, frozenset(functions))
    except _Stopped:
        pass
    except BaseException as exc:  # the script's own failure is its answer, said
        # From the script's frame on: the model wrote only `<script>`, not this file.
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename != "<script>":
            tb = tb.tb_next
        error = "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()
    sys.stdout = proto_out
    done = {"done": True, "output": captured.getvalue(), "error": error, "value": value}
    proto_out.write(json.dumps(done))
    proto_out.write("\n")
    proto_out.flush()


def _run(code: str, namespace: dict[str, object], functions: frozenset[str]) -> str | None:
    """Run the block; when its last statement is a bare expression, the `repr` of its value,
    as a Python prompt shows it. Not for a call to a tool or to `print`: a tool's answer is
    already shown when the block prints nothing, and `print` shows its own."""
    tree = ast.parse(code, "<script>")
    last = tree.body[-1] if tree.body else None
    if not isinstance(last, ast.Expr) or _shown_already(last.value, functions):
        exec(compile(tree, "<script>", "exec"), namespace)
        return None
    head = ast.Module(body=tree.body[:-1], type_ignores=[])
    exec(compile(head, "<script>", "exec"), namespace)
    result = eval(compile(ast.Expression(last.value), "<script>", "eval"), namespace)
    if result is None:
        return None
    return repr(result)[:VALUE_CHARS]


def _shown_already(node: ast.expr, functions: frozenset[str]) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    return node.func.id == "print" or node.func.id in functions


if __name__ == "__main__":
    main()
