"""The tools guard: one dispatcher, one registry, ports between the tools and
the world, and a `tools/` package that no role is imported into.

D164 moved every role's tools onto `chickadee/tools/`: a registry of `Tool`s, one `Dispatcher`
whose rules every role's policy orders, and ports each role supplies.
The old shape was a role deciding what a call was by its name, in its own loop. This fails the
next copy, by seven rules, over `packages/core/src/chickadee/**/*.py`:

- **T1, a tool's name compared as a string outside `chickadee/tools/`:** `name == "read"`,
  `tool in {"find", "symbol"}`, `match call.name: case "shell"`, where the other side is a name
  (an expression whose text holds `name`, `tool` or `kind`: the worker's old dispatcher read
  `request.kind`). Choosing what a call does by its name is
  dispatching it, and the dispatcher is where that lives: a rule, a spec field or a port.
- **T1b, a model name compared outside `config.py`**: a comparison (`==`, `!=`, `in`,
  `not in`), a `match`, or a `.startswith` / `.endswith`, where one side is a string or strings
  and the other is model-named: a name, attribute or string key whose last word is `model`,
  `model_id` or `model_name` (`rung.model_id`, `served_model`, `cfg["model"]`), through
  `.lower()`, `.casefold()` or `.strip()`. Per-model choice is configuration, resolved in
  `config.py`; code that branches on which model it is talking to is a second policy.
  `model_calls`, a spec field, is not a model name.
- **T2, a built-in reaching the world itself:** a module under `tools/builtin/` that opens a
  file, a process, a socket or the graph store (`open(`, `os.open(`, `read_text(`,
  `subprocess`, `ProcessSupervisor`, `sqlite3`, `socket`, `graph.store`) instead of asking a
  port, which is where what a call may touch is decided. A port's own
  `open` (`Opener.open`) is a port, and passes.
- **T3, a closed `Literal` of tool names outside `schema/`:** a second list of the tools that
  drifts from the registry. Two or more members, every one a registered tool's name.
- **T4, `chickadee/tools/` importing a role** (`zeus`, `scout`, `box`, `boxctx`, `engine`;
  decision 12): ports are `Protocol`s in `tools/`, implemented by their owners, so the
  dependency runs one way.
- **T5, a tier computed under `chickadee/tools/`**: inside a function, a `Tier(…)`,
  or a `Tier.<member>` that is not an operand of a comparison. A tier a function makes is a
  tier it decided; the router and the command classifier decide, and a rule reads their answer
  and may compare it (`tier is Tier.T0`). A `Tier.<member>` at module or class level, or as a
  parameter's default, is declared, not computed. D165 §2 also names `fnmatch` under `tools/`:
  the globs guard's G1 fails it everywhere, so T5 does not repeat it.
- **T6, a role's default rung read outside `config.py` and `roles.py`**: an
  attribute read of `.default_rung`. Which rung a role runs on is `EngineConfig.assignment`,
  computed once from the defaults and `[roles]`; a reader of the default is a second answer
  that ignores `[roles]`, the defect D176 fixed in the driver.

A registered tool's name is a `name="…"` given to `ToolSpec(` in `tools/spec.py` or under
`tools/builtin/`, read from the source: a guard imports only the standard library.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. What it cannot see: a name compared through a variable named none of
those, a name held in a constant (`== REQUEST_EXEC`), or a dispatcher written as a dict of
handlers keyed by name; a model name held in a constant, or used as a dict key; a tier chosen by
indexing a table. That is the review checklist's first question.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]

SRC = "packages/core/src/chickadee/"
TOOLS = f"{SRC}tools/"
CONFIG = f"{SRC}config.py"
"""Where a model name is resolved: T1b's one home."""
DEFAULT_RUNG_HOMES = frozenset({CONFIG, f"{SRC}roles.py"})
"""Where a role's default rung may be read: the cast, and the config's assignment."""
BUILTIN = f"{TOOLS}builtin/"
SCHEMA = f"{SRC}schema/"

ROLES = frozenset({"zeus", "scout", "box", "boxctx", "engine"})
"""The packages that implement a role (decision 12). `tools/` imports none of them."""

_PATH_IO = frozenset({"read_text", "read_bytes", "write_text", "write_bytes"})
"""`Path`'s own reads and writes, by method name."""
_OPENERS = frozenset({"os", "io", "builtins"})
"""Modules whose `open` opens a file. A port's own `open` (`context/opener.py` `Opener.open`)
is a port, which is where a built-in is meant to reach, so `x.open(` alone is not T2."""
_WORLD_MODULES = frozenset({"subprocess", "sqlite3", "socket", "chickadee.graph.store"})
_WORLD_NAMES = frozenset({"ProcessSupervisor"})

_MODEL_WORD = re.compile(r"(?:^|_)model(?:_id|_name)?$", re.IGNORECASE)
"""T1b: an identifier whose last word is a model, its id or its name (`served_model`), never one
that merely begins with it (`model_calls`)."""
_NORMALIZERS = frozenset({"lower", "casefold", "strip"})
"""T1b: methods that leave a model name a model name."""
_PREFIX_TESTS = frozenset({"startswith", "endswith"})

_NAMED = re.compile(r"name|tool|kind", re.IGNORECASE)
"""What the other side of a T1 comparison is: a call's name, its tool, or its kind (the
worker's old `request.kind == "read"`)."""

EXEMPT: dict[tuple[str, str], str] = {
    ("packages/core/src/chickadee/edits/draft.py", "<module>"): (
        "`DraftMode` is how a draft holds a path, an existing file or a new one; its two words "
        "match the `edit` and `create` tools only since D196 named the tools plainly (T-3.157)"
    ),
    ("packages/core/src/chickadee/context/compaction/receipts.py", "supersede"): (
        "compaction reads the session's history, not a call: an `ask_user` answer is what a "
        "person said, never an observation a newer call makes obsolete (28-context.md §5.7)"
    ),
    ("packages/core/src/chickadee/context/standing.py", "journaled_effects"): (
        "a fork's notice reads the session's history, not a call: the `shell` calls that ran "
        "and declared network hosts or effects, which a fork cannot undo (28-context.md §5.1.1)"
    ),
}


def tool_names(root: Path) -> frozenset[str]:
    """Every `name="…"` given to `ToolSpec(` in `tools/spec.py` and under `tools/builtin/`."""
    names: set[str] = set()
    for path in sorted((root / TOOLS).glob("spec.py")) + sorted((root / BUILTIN).glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ToolSpec"
            ):
                for kw in node.keywords:
                    if (
                        kw.arg == "name"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        names.add(kw.value.value)
    return frozenset(names)


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST
    what: str


def _strings(node: ast.expr) -> list[str] | None:
    """A string, or a set, tuple, list or `frozenset({…})` of them; None for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set", "tuple")
        and len(node.args) == 1
    ):
        return _strings(node.args[0])
    if isinstance(node, ast.Set | ast.Tuple | ast.List):
        out: list[str] = []
        for elt in node.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return None
            out.append(elt.value)
        return out
    return None


def _compared_name(node: ast.AST, names: frozenset[str]) -> str | None:
    """T1: the tool name a comparison holds against a name, or None."""
    if not isinstance(node, ast.Compare):
        return None
    if not all(isinstance(op, ast.Eq | ast.NotEq | ast.In | ast.NotIn) for op in node.ops):
        return None
    sides = [node.left, *node.comparators]
    for i, side in enumerate(sides):
        found = _strings(side)
        if not found:
            continue
        hit = next((s for s in found if s in names), None)
        if hit is None:
            continue
        others = sides[:i] + sides[i + 1 :]
        if any(_strings(o) is None and _NAMED.search(ast.unparse(o)) for o in others):
            return hit
    return None


def _matched_name(node: ast.AST, names: frozenset[str]) -> str | None:
    """T1: the tool name a `match <name>: case "…"` holds, or None."""
    if not (isinstance(node, ast.Match) and _NAMED.search(ast.unparse(node.subject))):
        return None
    for case in node.cases:
        for sub in ast.walk(case.pattern):
            if (
                isinstance(sub, ast.MatchValue)
                and isinstance(sub.value, ast.Constant)
                and sub.value.value in names
            ):
                return str(sub.value.value)
    return None


def _model_named(node: ast.expr) -> bool:
    """T1b: whether an expression is a model name: `model_id`, `rung.served_model`,
    `cfg["model"]`, or one of those through `.lower()` and the like."""
    while (
        isinstance(node, ast.Call)
        and not node.args
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _NORMALIZERS
    ):
        node = node.func.value
    if isinstance(node, ast.Name):
        return bool(_MODEL_WORD.search(node.id))
    if isinstance(node, ast.Attribute):
        return bool(_MODEL_WORD.search(node.attr))
    if isinstance(node, ast.Subscript):
        key = node.slice
        return (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and bool(_MODEL_WORD.search(key.value))
        )
    return False


def _model_compared(node: ast.AST) -> str | None:
    """T1b: the string a model name is compared, matched or prefix-tested against, or None."""
    if isinstance(node, ast.Compare):
        if not all(isinstance(op, ast.Eq | ast.NotEq | ast.In | ast.NotIn) for op in node.ops):
            return None
        sides = [node.left, *node.comparators]
        for i, side in enumerate(sides):
            found = _strings(side)
            if found and any(_model_named(o) for o in sides[:i] + sides[i + 1 :]):
                return found[0]
        return None
    if isinstance(node, ast.Match) and _model_named(node.subject):
        for case in node.cases:
            for sub in ast.walk(case.pattern):
                if (
                    isinstance(sub, ast.MatchValue)
                    and isinstance(sub.value, ast.Constant)
                    and isinstance(sub.value.value, str)
                ):
                    return sub.value.value
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _PREFIX_TESTS
        and _model_named(node.func.value)
        and node.args
    ):
        found = _strings(node.args[0])
        if found:
            return found[0]
    return None


def _tier_made(node: ast.AST, compared: set[int]) -> str | None:
    """T5: a tier a function makes, or None. `compared` holds the ids of the nodes that are a
    comparison's operands, which read a tier rather than make one."""
    func = node.func if isinstance(node, ast.Call) else None
    if isinstance(func, ast.Name) and func.id == "Tier":
        return "Tier("
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Tier"
        and id(node) not in compared
    ):
        return f"Tier.{node.attr}"
    return None


def _world(node: ast.AST) -> str | None:
    """T2: what a built-in reaches the world through, or None."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            return "open("
        if isinstance(func, ast.Attribute):
            if func.attr in _PATH_IO:
                return f"{func.attr}("
            if (
                func.attr == "open"
                and isinstance(func.value, ast.Name)
                and func.value.id in _OPENERS
            ):
                return f"{func.value.id}.open("
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in _WORLD_MODULES:
                return f"import {alias.name}"
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        if node.module in _WORLD_MODULES:
            return f"from {node.module} import"
        for alias in node.names:
            if alias.name in _WORLD_NAMES:
                return alias.name
    return None


def _literal_of_names(node: ast.AST, names: frozenset[str]) -> str | None:
    """T3: a `Literal[…]` whose two or more members are all tool names, or None."""
    if not isinstance(node, ast.Subscript):
        return None
    head = node.value
    if not (
        (isinstance(head, ast.Name) and head.id == "Literal")
        or (isinstance(head, ast.Attribute) and head.attr == "Literal")
    ):
        return None
    inner = node.slice
    members = _strings(inner) if isinstance(inner, ast.Tuple) else None
    if members and len(members) >= 2 and all(m in names for m in members):
        return f"Literal[{', '.join(repr(m) for m in members)}]"
    return None


def _role_import(node: ast.AST) -> str | None:
    """T4: the role package an import names, or None."""
    modules: list[str] = []
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
        modules = [node.module]
    for module in modules:
        parts = module.split(".")
        if len(parts) >= 2 and parts[0] == "chickadee" and parts[1] in ROLES:
            return module
    return None


def _default_rung(node: ast.AST) -> str | None:
    """T6: an attribute read of a role's `default_rung`."""
    if isinstance(node, ast.Attribute) and node.attr == "default_rung":
        return "default_rung"
    return None


def _sites(tree: ast.Module, rel: str, names: frozenset[str]) -> Iterator[_Site]:
    in_tools = rel.startswith(TOOLS)
    compared = {
        id(operand)
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for operand in (node.left, *node.comparators)
    }
    # T5 looks only inside function bodies: a tier at module or class level, or as a
    # parameter's default, is declared, not computed.
    in_bodies = {
        id(inner)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
        for stmt in fn.body
        for inner in ast.walk(stmt)
    }
    for scope, node in functions(tree):
        if not in_tools:
            hit = _compared_name(node, names) or _matched_name(node, names)
            if hit is not None:
                yield _Site("T1", scope, node, repr(hit))
        if rel != CONFIG:
            model = _model_compared(node)
            if model is not None:
                yield _Site("T1b", scope, node, repr(model))
        if in_tools and id(node) in in_bodies:
            made = _tier_made(node, compared)
            if made is not None:
                yield _Site("T5", scope, node, made)
        if rel.startswith(BUILTIN):
            reach = _world(node)
            if reach is not None:
                yield _Site("T2", scope, node, reach)
        if not rel.startswith(SCHEMA):
            closed = _literal_of_names(node, names)
            if closed is not None:
                yield _Site("T3", scope, node, closed)
        if in_tools:
            role = _role_import(node)
            if role is not None:
                yield _Site("T4", scope, node, role)
        if rel not in DEFAULT_RUNG_HOMES:
            read = _default_rung(node)
            if read is not None:
                yield _Site("T6", scope, node, read)


def sites(tree: ast.Module, rel: str, names: frozenset[str]) -> list[_Site]:
    """Every T1-T6 site in `tree`, the module at repo-relative path `rel`."""
    return list(_sites(tree, rel, names))


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one."""
    names = tool_names(REPO_ROOT)
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for site in sites(ast.parse(source), rel, names):
            key = (rel, site.function)
            if key in EXEMPT:
                used.add(key)
                continue
            line = getattr(site.node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(
                Violation(path=rel, line=line, text=f"{site.rule} ({site.what}): {text}")
            )
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="tools",
    rule=(
        "one dispatcher (D164): choose what a call does by a rule, a spec field or a port, "
        "never by comparing its name outside chickadee/tools/ (T1); a built-in reaches the "
        "world through a port (T2); no closed Literal of tool names outside schema/ (T3); "
        "chickadee/tools/ imports no role (T4) (D164 §12, §13); no model name compared outside "
        "config.py (T1b); no tier made under chickadee/tools/, only read (T5) (D165 §1, §2); "
        "a role's rung read from EngineConfig.assignment, never its default_rung outside "
        "config.py and roles.py (T6, D176)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
