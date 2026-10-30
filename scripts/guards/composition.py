"""The composition guard: a session's lifecycle has one home.

A session's lifecycle was six flags on `engine._ZeusSession`, its single-thread worker made
beside them, and five engine methods that changed the flags (#276 T9.2). `T-3.81` moved them
into `zeus/conversation.py`'s `Conversation`, whose phases are D158 decision 2's table. This
fails the next one, over `packages/core/src/chickadee/**/*.py`:

- **C1, one lifecycle:** a `ThreadPoolExecutor(...)` constructed outside
  `zeus/conversation.py`, however it is spelled (`ThreadPoolExecutor`, reached through
  `concurrent.futures`, or renamed on import). A session's worker is its conversation's; a
  second executor somewhere else is a second lifecycle, with its own flags beside it. One
  exemption, keyed by file and function: `Dispatcher.execute_group`, whose pool lives for one
  group of one response and is joined before it returns.
- **C2, one cache of each per engine**: a `ReplayCache(...)` or `RunProjections(...)`
  constructed outside `engine.py`, the composition root. A reader that builds its own cache
  reads every log or journal again from the first line; the engine's pair is passed to it.
  `sessions.py`'s is baselined: `Sessions` is unwired, and D159 decides it.
- **C3, a journal folded once**: a `read_run(...)` call outside `journal/runs.py`,
  or a `read_tail(...)` call outside `journal/runs.py` and `journal/reader.py`. One exemption,
  keyed by file and function as the projection guard keys its own:
  `JournalRecords.journal_text`, since a journal handle shows the lines as written, not a fold
  of them. An exemption that no longer exempts anything fails.

What it cannot see: a lifecycle kept with `threading.Thread` and hand-rolled flags, or a
journal read with `open()`. Those are the review checklist's first and fourth questions.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]

CORE = "packages/core/src/chickadee/"
HOME = CORE + "zeus/conversation.py"
"""The one home of a session's worker and its phases."""
ROOT = CORE + "engine.py"
"""The composition root: the one place the engine's two read caches are built."""
RUNS = CORE + "journal/runs.py"
READER = CORE + "journal/reader.py"

RULES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    ("C1", frozenset({"ThreadPoolExecutor"}), frozenset({HOME})),
    ("C2", frozenset({"ReplayCache", "RunProjections"}), frozenset({ROOT})),
    ("C3", frozenset({"read_run"}), frozenset({RUNS})),
    ("C3", frozenset({"read_tail"}), frozenset({RUNS, READER})),
)
"""Each rule, the names whose call it guards, and the modules that are those names' homes.
C3 has two entries, since `read_tail` has a second home in the reader that defines it."""

EXEMPT: dict[tuple[str, str, str], str] = {
    ("C1", CORE + "tools/dispatch.py", "Dispatcher.execute_group"): (
        "threaded tool calls (D213 Amendment 1 decision 1): a pool made for one group of one "
        "response and joined before the call returns, so no lifecycle and no flags outlive it"
    ),
    ("C3", CORE + "context/project.py", "JournalRecords.journal_text"): (
        "a journal handle's bytes are the lines as written, not a fold of them (D158 "
        "decision 8): what a handle shows is the journal itself"
    ),
}
"""The exemptions, keyed by rule, file and qualified function name, each with its reason. An
entry that exempts nothing fails."""


def _aliases(tree: ast.Module, names: frozenset[str]) -> dict[str, str]:
    """Every name the module binds to one of `names`: the name itself, and any `as` name an
    import gives it, so a renamed import is still seen."""
    bound = {name: name for name in names}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in names and alias.asname is not None:
                    bound[alias.asname] = alias.name
    return bound


def _called(node: ast.AST, bound: dict[str, str]) -> str | None:
    """The guarded name `node` calls, by name or through a module, or None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return bound.get(func.id)
    if isinstance(func, ast.Attribute) and func.attr in bound.values():
        return func.attr
    return None


def sites(tree: ast.Module, rel: str) -> list[tuple[str, str, ast.AST]]:
    """Every site in `tree`, the module at repo-relative path `rel`: its rule, the function
    it sits in, and the call."""
    out: list[tuple[str, str, ast.AST]] = []
    for rule, names, homes in RULES:
        if rel in homes:
            continue
        bound = _aliases(tree, names)
        for scope, node in functions(tree):
            if _called(node, bound) is not None:
                out.append((rule, scope, node))
    return out


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one."""
    violations: list[Violation] = []
    used: set[tuple[str, str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for rule, scope, node in sites(ast.parse(source), rel):
            if (rule, rel, scope) in EXEMPT:
                used.add((rule, rel, scope))
                continue
            line = getattr(node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=f"{rule}: {text}"))
    for _rule, rel, scope in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {scope}"))
    return violations


BASELINE: dict[str, int] = {
    "packages/core/src/chickadee/sessions.py": 1,
}

GUARD = Guard(
    name="composition",
    rule=(
        "a session's lifecycle has one home: a session's worker and its phases live in "
        "zeus/conversation.py; never a ThreadPoolExecutor elsewhere (C1). The engine builds "
        "the one ReplayCache and RunProjections every reader shares (C2), and a run journal "
        "is folded only through journal/runs.py (C3) (D158)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
