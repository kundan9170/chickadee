"""The file-access guard: one read rule and one containment answer.

**F1, one read rule.** Each reader asked the deny list its own question, and two asked none.
`T-3.87` moved them onto `secrets.name_denied` and `secrets.read_denied`. This fails
the next caller that asks the list directly: outside `chickadee/secrets/`, a call of
`.match(...)` or `.readable(...)` on a deny list. The receiver is a deny list when it is named
`*deny*` or `listed` (`self.denylist`, `deny`), is a `Denylist(...)` or
`Denylist.configure(...)` call, or is a conditional choosing between those
(`(denylist or Denylist()).match`), since a `re` match is none of them.

**F2, one containment answer.** D154 counted twelve places in ten files asking whether a path
is inside a root, in five spellings, and they disagreed on links, on `..` and on the root
itself. `T-3.101` moved them onto `paths.within`. This fails the next copy: outside
`chickadee/paths.py`,

- a call of `.is_relative_to(...)`;
- an `in` or `not in` comparison against `.parents` (`root in path.parents`), while a `for`
  loop over `.parents` is a walk, not a membership test, and is not counted;
- `.startswith(...)` whose argument appends `os.sep` (or `os.path.sep`) to a string, by `+`
  or in an f-string: the string spelling of the same question.

What neither rule can see is the same question spelled another way: a deny list held under
another name, or containment as `relative_to` inside a `try`. That is the review checklist's
first question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation
from guards._resolve import name_uses

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = "packages/core/src/chickadee/paths.py"
"""F2's home."""
READ_HOME = "packages/core/src/chickadee/secrets/"
"""F1's home: the rule, `resolve` and `write_denied` ask the list directly."""

_ASKS = frozenset({"match", "readable"})


def sites(tree: ast.Module) -> list[ast.AST]:
    """Every node in `tree` asking containment the way only `paths.py` should, each once."""
    seps = {id(n) for n in name_uses(tree, "os", "sep")}
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        call = isinstance(node, ast.Call) and _calls_containment(node, seps)
        if call or (isinstance(node, ast.Compare) and _tests_parents(node)):
            found.append(node)
    return sorted(found, key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))


def read_sites(tree: ast.Module) -> list[ast.AST]:
    """Every node in `tree` asking the deny list directly, as only `secrets/` should."""
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _ASKS
        and _is_deny_list(node.func.value)
    ]
    return sorted(found, key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))


def _is_deny_list(node: ast.expr) -> bool:
    """`denylist`, `self.denylist`, `deny`, `listed`, `Denylist(...)`,
    `Denylist.configure(...)`, or a conditional over them."""
    if isinstance(node, ast.Name):
        return _names_a_deny_list(node.id)
    if isinstance(node, ast.Attribute):
        return _names_a_deny_list(node.attr)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "configure":
            # `Denylist.configure(...)`, the list a config resolves to.
            return _is_deny_list(func.value) or getattr(func.value, "id", "") == "Denylist"
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return name == "Denylist"
    if isinstance(node, ast.IfExp):
        return _is_deny_list(node.body) or _is_deny_list(node.orelse)
    if isinstance(node, ast.BoolOp):
        return any(_is_deny_list(v) for v in node.values)
    return False


def _names_a_deny_list(name: str) -> bool:
    return "deny" in name.casefold() or name == "listed"


def _calls_containment(node: ast.Call, seps: set[int]) -> bool:
    """`.is_relative_to(...)`, or `.startswith(...)` of a string with the separator appended."""
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "is_relative_to":
        return True
    return node.func.attr == "startswith" and any(_appends_sep(a, seps) for a in node.args)


def _tests_parents(node: ast.Compare) -> bool:
    """`x in y.parents` or `x not in y.parents`, anywhere in a chained comparison."""
    return any(
        isinstance(op, ast.In | ast.NotIn)
        and isinstance(right, ast.Attribute)
        and right.attr == "parents"
        for op, right in zip(node.ops, node.comparators, strict=True)
    )


def _appends_sep(arg: ast.expr, seps: set[int]) -> bool:
    """`root + os.sep` or `f"{root}{os.sep}"`: a string with the separator appended."""
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
        return _is_sep(arg.right, seps)
    if isinstance(arg, ast.JoinedStr):
        return any(
            isinstance(part, ast.FormattedValue) and _is_sep(part.value, seps)
            for part in arg.values
        )
    return False


def _is_sep(node: ast.expr, seps: set[int]) -> bool:
    if id(node) in seps:
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "sep"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "path"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
    )


def find(files: Sequence[Path]) -> list[Violation]:
    """Every F2 site outside `paths.py` and every F1 site outside `secrets/`.

    Parses core's source, which is Python 3.12: run it with 3.12 or later, as CI's
    `python3` is."""
    violations: list[Violation] = []
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = [] if rel == HOME else sites(tree)
        if not rel.startswith(READ_HOME):
            found += read_sites(tree)
        lines = source.splitlines()
        for node in sorted(found, key=lambda n: getattr(n, "lineno", 0)):
            line = getattr(node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=text))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="file_access",
    rule=(
        "whether a path may be read is `secrets.read_denied` or `name_denied`, never the deny "
        "list's own `.match` or `.readable` outside `secrets/`; whether a path is inside a "
        "root is `chickadee.paths.within`, never `is_relative_to`, `in .parents` or "
        "`startswith(root + os.sep)` (D154)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
