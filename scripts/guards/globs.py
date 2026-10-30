"""The globs guard: there is one path-glob matcher,
`paths.glob_match`, reading a glob as `.gitignore` does.

*Does this path match this glob* was answered in five places that read some globs
differently: `paths.glob_match`, the scout's `search`, the executor's effect bracket, the
`always_ask` / `never_ask` preferences and the graph's `.gitignore` reader (#276's shape).
D168 moved every one to `paths.glob_match`. This fails the next copy, by two rules, over
`packages/core/src/chickadee/**/*.py`:

- **G1, `fnmatch`:** a call into `fnmatch`, an import from it, or `glob.translate`, outside
  `paths.py`. `fnmatch` reads a glob as a file name, so its `*` crosses `/`; a path glob that
  reaches it is read differently from every other.
- **G2, a glob translated to a regex:** a string holding `[^/]`, the regex a glob's `*` or
  `?` becomes, outside `paths.py` and outside a docstring. That is how the scout's and the
  `.gitignore` reader's copies were written.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. The baseline is empty: the secret deny list, held here by D168,
matches through `glob_match` since, and the preference `deny` list since, which made
it the path globs its spec says and denies a command by its class instead.

What it cannot see: a matcher written without either (a `startswith` walk over segments, a
hand-rolled `*` loop). That is the review checklist's first question.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import docstrings, functions

REPO_ROOT = Path(__file__).resolve().parents[2]

HOME = "packages/core/src/chickadee/paths.py"
"""The one path-glob matcher."""

_TRANSLATED = "[^/]"

EXEMPT: dict[tuple[str, str], str] = {
    ("packages/core/src/chickadee/isolation/macos.py", "<module>"): (
        "the Seatbelt profile's regex rows for the deny list's bare-name patterns: "
        "`paths.segment_tokens`' one reading of a glob, written in the sandbox's own regex "
        "dialect rather than Python's (D170)"
    ),
}


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST
    what: str


def _fnmatch(node: ast.AST) -> str | None:
    """`fnmatch.<name>`, `from fnmatch import …` or `glob.translate`, as written."""
    if isinstance(node, ast.ImportFrom) and node.module == "fnmatch":
        return "from fnmatch import"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (
            node.func.value.id == "fnmatch"
            or (node.func.value.id == "glob" and node.func.attr == "translate")
        )
    ):
        return f"{node.func.value.id}.{node.func.attr}"
    return None


def _sites(tree: ast.Module) -> Iterator[_Site]:
    prose = docstrings(tree)
    for scope, node in functions(tree):
        called = _fnmatch(node)
        if called is not None:
            yield _Site("G1", scope, node, called)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _TRANSLATED in node.value
            and id(node) not in prose
        ):
            yield _Site("G2", scope, node, repr(node.value))


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every G1 and G2 site in `tree`, the module at repo-relative path `rel`."""
    return [] if rel == HOME else list(_sites(tree))


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
    name="globs",
    rule=(
        "one path-glob matcher: ask chickadee.paths.glob_match (and spelled_out for a tool "
        "that reads a list as .gitignore does), never fnmatch or a glob translated to a "
        "regex outside paths.py (G1, G2) (D167 §4, D168)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
