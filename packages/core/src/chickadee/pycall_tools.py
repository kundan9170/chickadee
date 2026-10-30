"""The typed Python surface of the worker's tools, for the Python tool-calling demo
(`pycall.py`).

Each `PyTool` is one Python function the model may call: its signature, written for Python
(`read(path, start_line, end_line)`, not `lines="5-178"`), the native tool it stands for, and
the conversion of its arguments to that tool's native arguments. The tools themselves are
unchanged; a typed call becomes exactly the native call a model could have made.

Arguments are checked here, in the parent, before the native call: a wrong type is raised in
the script as a `TypeError`, as Python would, and never reaches the tool. A tool the worker is
offered that has no entry here (a configured extra) keeps its stub generated from its schema.

Results stay strings, as the native tools return them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, cast

from chickadee.adapters.base import SHARED_TEMPLATES, load_template
from chickadee.client.request import ToolSchema
from chickadee.tools.foreign import foreign_argument, ours_for

__all__ = [
    "PY_TOOLS",
    "Param",
    "PyTool",
    "Rejected",
    "as_python",
    "from_native",
    "from_schema",
    "stubs",
    "surface",
]

DOCS: Final = SHARED_TEMPLATES / "python_tools"


class Rejected(Exception):
    """A typed call that does not bind: raised in the script as a `TypeError`."""


Check = Callable[[object], bool]


def _str(v: object) -> bool:
    return isinstance(v, str)


def _int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _bool(v: object) -> bool:
    return isinstance(v, bool)


def _strs(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in cast("list[object]", v))


def _pairs(v: object) -> bool:
    """`list[tuple[str, str]]`; a tuple arrives over JSON as a two-item list."""
    if not isinstance(v, list):
        return False
    for item in cast("list[object]", v):
        if not isinstance(item, list | tuple):
            return False
        pair = cast("list[object]", item)
        if len(pair) != 2 or not all(isinstance(x, str) for x in pair):
            return False
    return True


def _coerced(param: Param, value: object) -> object:
    """The two coercions with one meaning: a string where a list of strings
    is wanted is that one string, and a string of digits where an int is wanted is that int.
    Nothing else is changed: anything less certain stays the model's to fix."""
    if param.check is _strs and isinstance(value, str):
        return [value]
    digits = value.strip() if isinstance(value, str) else ""
    if param.check is _int and digits.isascii() and digits.isdigit():
        # ASCII alone: `"²".isdigit()` is True, and `int("²")` raises.
        return int(digits)
    return value


DEPTHS: Final = ("outline", "signatures", "full")


def _depth(v: object) -> bool:
    return v in DEPTHS


@dataclass(frozen=True)
class Param:
    name: str
    annotation: str
    """As the stub shows it."""
    check: Check
    default: str | None = None
    """The default as the stub shows it; None means required."""
    kw_only: bool = False

    @property
    def required(self) -> bool:
        return self.default is None


Args = Mapping[str, object]
Native = dict[str, object]


def _as_is(a: Args) -> Native | None:
    """The `back` of a tool whose Python arguments are its native ones."""
    return dict(a)


@dataclass(frozen=True)
class PyTool:
    name: str
    native: str
    params: tuple[Param, ...]
    to_native: Callable[[Args], Native]
    doc: str | None = None
    """The docstring; None reads the tool's template (`python_tools/<name>.md`)."""
    back: Callable[[Args], Native | None] = field(default=_as_is)
    """`to_native`'s inverse: native arguments as this function's, None when one cannot be
    (`from_native`). A rename lives beside its tool, never in a test of the tool's name."""
    returns: str = "str"
    """The return type the stub shows: a `str`, or for `grep`, `glob` and `shell` a `str`
    that also carries what the result was made from."""

    def positional(self) -> list[str]:
        """The names a positional argument binds to, in order."""
        return [p.name for p in self.params if not p.kw_only]

    def stub(self) -> str:
        parts: list[str] = []
        star = False
        for p in self.params:
            if p.kw_only and not star:
                parts.append("*")
                star = True
            parts.append(
                f"{p.name}: {p.annotation}"
                if p.required
                else f"{p.name}: {p.annotation} = {p.default}"
            )
        doc = self.doc if self.doc is not None else load_template(DOCS, f"{self.name}.md")
        doc = doc.strip().replace('"""', "'''")
        body = "\n".join(f"    {line}" if line else "" for line in doc.splitlines())
        return (
            f'def {self.name}({", ".join(parts)}) -> {self.returns}:\n    """\n{body}\n    """'
        )

    def bind(self, args: Args) -> Native:
        """`args` checked against the signature, then as the native tool's arguments.
        Raises `Rejected` with the message Python would give, and for another harness's
        argument, ours (`tools/foreign.py`). The two coercions with one meaning are made
        first."""
        known = {p.name: p for p in self.params}
        for key in args:
            if key not in known:
                raise Rejected(self._unexpected(key, args[key], known))
        bound: dict[str, object] = {}
        for p in self.params:
            if p.name not in args or args[p.name] is None:
                if p.required:
                    raise Rejected(f"{self.name}() missing required argument {p.name!r}")
                continue
            value = _coerced(p, args[p.name])
            if not p.check(value):
                raise Rejected(
                    f"{self.name}() argument {p.name!r} must be {p.annotation}, "
                    f"not {type(value).__name__} ({value!r:.60})"
                )
            bound[p.name] = value
        return self.to_native(bound)

    def _unexpected(self, key: str, value: object, known: Mapping[str, Param]) -> str:
        said = f"{self.name}() got an unexpected keyword argument {key!r}"
        # The native tool's own name for it (`read(lines="5-9")`), as this function takes it.
        back = self.back({key: value})
        if back is not None and back != {key: value} and back and set(back) <= set(known):
            return f"{said}; here it is {', '.join(f'{k}={v!r}' for k, v in back.items())}"
        ours = ours_for(self.native, key)
        if ours is not None and ours in known:
            return f"{said}; here it is {ours!r}"
        if foreign_argument(key) is not None:
            return f"{said}; do not send it. {self.name}() takes {', '.join(known)}"
        return said


def _set(out: Native, args: Args, name: str, native: str | None = None) -> None:
    if name in args:
        out[native or name] = args[name]


# -- the conversions --------------------------------------------------------------------------


def _read(a: Args) -> Native:
    if ("path" in a) == ("handle" in a):
        raise Rejected("read() takes path or handle, exactly one")
    start, end = a.get("start_line"), a.get("end_line")
    out: Native = {}
    _set(out, a, "path")
    _set(out, a, "handle")
    if (start is None) != (end is None):
        raise Rejected(
            "read() takes start_line and end_line together (one line: both the same)"
        )
    if start is not None and end is not None:
        s, e = cast("int", start), cast("int", end)
        if s < 1 or e < s:
            raise Rejected(f"read() needs 1 <= start_line <= end_line, got {s} and {e}")
        out["lines"] = f"{s}-{e}"
    _set(out, a, "symbol", "at")
    _set(out, a, "depth")
    return out


def _grep_in(a: Args) -> Native:
    out: Native = {"handle": a["handle"], "pattern": a["pattern"]}
    _set(out, a, "context_lines", "context")
    return out


def _same(a: Args) -> Native:
    return dict(a)


def _edit(a: Args) -> Native:
    out: Native = {"path": a["path"], "old": a["old"], "new": a["new"]}
    if a.get("replace_all"):
        out["replace_all"] = True
    return out


def _edit_many(a: Args) -> Native:
    pairs = cast("list[list[str]]", a["edits"])
    if not pairs:
        raise Rejected("edit_many() needs at least one (old, new) pair")
    return {"path": a["path"], "edits": [{"old": o, "new": n} for o, n in pairs]}


def _shell(a: Args) -> Native:
    out: Native = {"command": a["command"]}
    _set(out, a, "cwd")
    if a.get("network"):
        out["network"] = True
    if a.get("sandbox") is False:
        out["sandbox"] = False
    return out


def _read_back(n: Args) -> Native | None:
    out: Native = {k: v for k, v in n.items() if k not in ("lines", "at")}
    lines = n.get("lines")
    if lines is not None:
        start, sep, end = str(lines).partition("-")
        if not (sep and start.strip().isdigit() and end.strip().isdigit()):
            return None
        out["start_line"], out["end_line"] = int(start), int(end)
    if "at" in n:
        out["symbol"] = n["at"]
    return out


def _grep_in_back(n: Args) -> Native | None:
    out: Native = {k: v for k, v in n.items() if k != "context"}
    if "context" in n:
        out["context_lines"] = n["context"]
    return out


def _edit_many_back(n: Args) -> Native | None:
    edits = n.get("edits")
    if not isinstance(edits, list):
        return None
    pairs = cast("list[Mapping[str, object]]", edits)
    return {"path": n.get("path"), "edits": [(e.get("old"), e.get("new")) for e in pairs]}


def _p(name: str, annotation: str, check: Check, default: str | None = None) -> Param:
    return Param(name, annotation, check, default)


_DEPTH_T: Final = 'Literal["outline", "signatures", "full"] | None'

PY_TOOLS: Final[tuple[PyTool, ...]] = (
    PyTool(
        "read",
        "read",
        (
            _p("path", "str | None", _str, "None"),
            _p("start_line", "int | None", _int, "None"),
            _p("end_line", "int | None", _int, "None"),
            _p("symbol", "str | None", _str, "None"),
            _p("depth", _DEPTH_T, _depth, "None"),
            Param("handle", "str | None", _str, "None", kw_only=True),
        ),
        _read,
        back=_read_back,
    ),
    PyTool(
        "grep_in",
        "grep_in",
        (
            _p("handle", "str", _str),
            _p("pattern", "str", _str),
            _p("context_lines", "int", _int, "2"),
        ),
        _grep_in,
        back=_grep_in_back,
    ),
    PyTool(
        "grep",
        "grep",
        (_p("pattern", "str", _str), _p("paths", "list[str] | None", _strs, "None")),
        _same,
        returns="GrepResult",
    ),
    PyTool("glob", "glob", (_p("pattern", "str", _str),), _same, returns="GlobResult"),
    PyTool("symbol", "symbol", (_p("name", "str", _str),), _same),
    PyTool(
        "neighbors",
        "neighbors",
        (_p("target", "str", _str), _p("depth", "int", _int, "1")),
        _same,
    ),
    PyTool("impact", "impact", (_p("symbols", "list[str]", _strs),), _same),
    PyTool("tests_for", "tests_for", (_p("paths", "list[str]", _strs),), _same),
    PyTool(
        "edit",
        "edit",
        (
            _p("path", "str", _str),
            _p("old", "str", _str),
            _p("new", "str", _str),
            _p("replace_all", "bool", _bool, "False"),
        ),
        _edit,
    ),
    PyTool(
        "edit_many",
        "edit",
        (_p("path", "str", _str), _p("edits", "list[tuple[str, str]]", _pairs)),
        _edit_many,
        back=_edit_many_back,
    ),
    PyTool("create", "create", (_p("path", "str", _str), _p("content", "str", _str)), _same),
    PyTool("delete", "delete", (_p("path", "str", _str),), _same),
    PyTool("move", "move", (_p("path", "str", _str), _p("destination", "str", _str)), _same),
    PyTool(
        "shell",
        "shell",
        (
            _p("command", "str", _str),
            _p("cwd", "str | None", _str, "None"),
            _p("network", "bool", _bool, "False"),
            _p("sandbox", "bool", _bool, "True"),
        ),
        _shell,
        returns="ShellResult",
    ),
    PyTool("finish", "finish", (_p("report", "str", _str),), _same),
)


def stubs(tools: tuple[PyTool, ...]) -> str:
    return "\n\n".join(t.stub() for t in tools)


def surface(offered: tuple[ToolSchema, ...]) -> tuple[PyTool, ...]:
    """The Python functions for the tools offered now: the typed ones whose native tool is
    offered, then any offered tool with no typed entry, as its schema says."""
    names = frozenset(t.name for t in offered)
    typed = tuple(t for t in PY_TOOLS if t.native in names)
    covered = {t.native for t in typed}
    return typed + tuple(from_schema(t) for t in offered if t.name not in covered)


# -- back from native: the block to send for a native call -------------------------------------


def from_native(tool: PyTool, native: Mapping[str, object]) -> dict[str, object] | None:
    """`native`, a call's native arguments, as `tool`'s Python arguments (its `back`): for a
    refusal that names the block to send instead (the model wrote a native call, or its bare
    arguments, where a block was wanted). None when one does not map, or a required argument
    is missing: then no block is named, rather than a wrong one."""
    out = tool.back(native)
    if out is None:
        return None
    params = {p.name for p in tool.params}
    if any(k not in params for k in out):
        return None
    if any(p.required and p.name not in out for p in tool.params):
        return None
    return out


def as_python(tool: PyTool, args: Mapping[str, object]) -> str:
    """`tool` called with `args`, as one line of Python."""
    return f"{tool.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


# -- a tool with no typed entry ---------------------------------------------------------------


def _num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _dict(v: object) -> bool:
    return isinstance(v, dict)


def _list(v: object) -> bool:
    return isinstance(v, list)


def _any(v: object) -> bool:
    return True


_JSON: Final[Mapping[str, tuple[str, Check]]] = {
    "string": ("str", _str),
    "integer": ("int", _int),
    "number": ("float", _num),
    "boolean": ("bool", _bool),
    "object": ("dict", _dict),
}


def _typed(prop: Mapping[str, object]) -> tuple[str, Check]:
    """A JSON schema property as a Python annotation and its check."""
    enum = prop.get("enum")
    if isinstance(enum, list):
        values = tuple(cast("list[object]", enum))
        return "Literal[" + ", ".join(repr(v) for v in values) + "]", values.__contains__
    kind = prop.get("type")
    if kind == "array":
        items = prop.get("items")
        typed = isinstance(items, Mapping)
        if typed and cast("Mapping[str, object]", items).get("type") == "string":
            return "list[str]", _strs
        return "list", _list
    if isinstance(kind, str) and kind in _JSON:
        return _JSON[kind]
    return "object", _any


def from_schema(tool: ToolSchema) -> PyTool:
    """A tool as its native schema says: every argument by its native name and type."""
    props = cast("Mapping[str, Mapping[str, object]]", tool.schema.get("properties") or {})
    required = {str(r) for r in cast("list[object]", tool.schema.get("required") or [])}
    params: list[Param] = []
    for name, prop in props.items():
        annotation, check = _typed(prop)
        if name in required:
            params.append(Param(name, annotation, check))
        else:
            params.append(Param(name, f"{annotation} | None", check, "None"))
    params.sort(key=lambda p: not p.required)
    return PyTool(tool.name, tool.name, tuple(params), _same, doc=tool.description)
