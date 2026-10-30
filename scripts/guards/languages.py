"""The languages guard: which language a file is has one home, the profiles.

*Which language is this path* was answered by three tables: the profiles' own `extensions`, the
graph's `parse.EXTENSIONS` and the reader's `outline.LANGUAGES`. They disagreed: the graph
parsed `.ts` and `.tsx`, which no profile owned, and the reader labelled Go code the harness
could not read (#276's shape). `T-3.164` moved every caller to `ProfileSet.language_of`, which
answers a path's profile and the grammar that parses it, and deleted both tables. This fails the
next one, over every package's source:

- **L1, an extension table:** a dict literal outside `chickadee/profiles/` with two or more keys
  that are file extensions (`".py"`, `".tsx"`) mapped to a name: a string, or a constant that
  holds one (`".py": PYTHON`). That is how both copies were written. A table whose values are
  not names (`outline._TEXT_FORMATS`, extension to a format and its label as a tuple) says how
  to show a document, not which language it is, and passes.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. The baseline is empty.

What it cannot see: a set of one language's extensions, or an `if` chain over a suffix
(`graph/lsp.py` picks a server and an LSP `languageId` that way; that resolver is not wired,
and the profile's `[lsp].server` is its home when it is). That is the review checklist's first
question.
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

HOME = "packages/core/src/chickadee/profiles/"
"""The one home: the profiles, read through `ProfileSet.language_of`."""

_EXTENSION = re.compile(r"\.[A-Za-z0-9][A-Za-z0-9+_-]*")
"""A file extension as a profile writes one: a dot, then a name (`.py`, `.c++`, `.d-ts`)."""

_MIN_KEYS = 2
"""One extension mapped to a name is a special case (`{".min.js":...}`); two is a table."""

EXEMPT: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class _Site:
    function: str
    node: ast.Dict
    keys: tuple[str, ...]


def _a_name(value: ast.expr) -> bool:
    """A string, or a constant that holds one: `"python"`, `PYTHON`, `parse.PYTHON`."""
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    return isinstance(value, ast.Name | ast.Attribute)


def _extension_table(node: ast.AST) -> tuple[str, ...] | None:
    """The extension keys of a dict literal mapping extensions to names, or None."""
    if not isinstance(node, ast.Dict):
        return None
    keys: list[str] = []
    for key, value in zip(node.keys, node.values, strict=True):
        if (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and _EXTENSION.fullmatch(key.value)
            and _a_name(value)
        ):
            keys.append(key.value)
    return tuple(keys) if len(keys) >= _MIN_KEYS else None


def _sites(tree: ast.Module) -> Iterator[_Site]:
    for scope, node in functions(tree):
        keys = _extension_table(node)
        if keys is not None and isinstance(node, ast.Dict):
            yield _Site(scope, node, keys)


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every L1 site in `tree`, the module at repo-relative path `rel`."""
    return [] if rel.startswith(HOME) else list(_sites(tree))


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one."""
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for site in sites(ast.parse(source), rel):
            key = (rel, site.function)
            if key in EXEMPT:
                used.add(key)
                continue
            line = site.node.lineno
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            shown = ", ".join(site.keys[:4]) + (", ..." if len(site.keys) > 4 else "")
            violations.append(Violation(path=rel, line=line, text=f"L1 ({shown}): {text}"))
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="languages",
    rule=(
        "one home for which language a file is: ask ProfileSet.language_of (the profile and "
        "the grammar that parses it), and add a language as a profile in profiles/, never an "
        "extension table in code (L1) (D203 decision 2)"
    ),
    corpus=("packages/*/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
