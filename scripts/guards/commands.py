"""The commands guard: there is one command classifier, the command
table in `approval/commands.py`, and one place a command line is taken apart,
`shell_line.py`.

*What kind of command is this* was answered in four places that drifted apart: the table,
`prefilter_floor`'s regexes beside it, the scout's `action_in_disguise` patterns and the
executor's `_MUTATING` (#276's shape). `T-3.122` moved every caller to
`CommandTable.classify` / `classify_line`, and every `shlex.split` to `shell_line`. This fails
the next copy, by two rules, over `packages/core/src/chickadee/**/*.py`:

- **C1, a command regex:** a regular expression (`re.compile`, `re.search`, `re.match`,
  `re.fullmatch`, `re.findall`, `re.finditer`, `re.sub`, `re.split`) whose pattern names a
  command as a word (`rm`, `git`, `curl`, `sudo`, `pip`, `make`, …), outside
  `approval/commands.py`. Recognising a command by regex is classifying it, and the table is
  where that lives: add a row, or a pattern row for what is not a command.
- **C2, a second splitter:** a `shlex.split`, `shlex.shlex` or `shlex.join` call outside
  `shell_line.py`. A declared command becomes argv through `shell_line.argv`, which refuses
  bad quoting; a reader takes words through `shell_line.split`, which never refuses.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. There are none at landing.

What it cannot see: a classifier written without a regex (a set of command names, a
`startswith` chain), or naming a command this list does not hold. That is the review
checklist's first question.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]

HOME = "packages/core/src/chickadee/approval/commands.py"
"""The one command classifier."""

SPLITTER = "packages/core/src/chickadee/shell_line.py"
"""The one place a command line is taken apart."""

_SHLEX_FUNCTIONS = frozenset({"split", "shlex", "join"})

COMMAND_WORD = re.compile(
    r"(?<![A-Za-z0-9_-])(?:rm|rmdir|shred|mkfs|curl|wget|sudo|chmod|chown|git|pip3?|npm|pnpm"
    r"|yarn|make|docker|podman|cargo|apt(?:-get)?|brew|ssh|scp|rsync|sed|perl|touch|mkdir"
    r"|terraform|kubectl)(?![A-Za-z0-9_-])"
)
"""The heads every copy so far recognised. A pattern naming one of them as a word is a
command regex."""

_ESCAPE = re.compile(r"\\[A-Za-z]")

_RE_FUNCTIONS = frozenset(
    {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "split"}
)

EXEMPT: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST
    word: str


def _pattern_text(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _pattern_text(node.left), _pattern_text(node.right)
        return None if left is None and right is None else (left or "") + (right or "")
    return None


def _command_regex(node: ast.AST) -> str | None:
    """The command word a `re.<function>(pattern, …)` call's pattern names, or `None`."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.func.attr in _RE_FUNCTIONS
        and node.args
    ):
        return None
    text = _pattern_text(node.args[0])
    # An escape is not a letter of a word: `\bgit` names `git`, and `\b` must not read as
    # the `b` before it.
    found = COMMAND_WORD.search(_ESCAPE.sub(" ", text)) if text is not None else None
    return found.group(0) if found is not None else None


def _shlex_call(node: ast.AST) -> str | None:
    """The `shlex` function a `shlex.<function>(…)` call names, when it splits or joins."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "shlex"
        and node.func.attr in _SHLEX_FUNCTIONS
    ):
        return f"shlex.{node.func.attr}"
    return None


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every C1 and C2 site in `tree`, the module at repo-relative path `rel`."""
    out: list[_Site] = []
    for scope, node in functions(tree):
        word = _command_regex(node) if rel != HOME else None
        if word is not None:
            out.append(_Site("C1", scope, node, word))
        call = _shlex_call(node) if rel != SPLITTER else None
        if call is not None:
            out.append(_Site("C2", scope, node, call))
    return out


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
                Violation(path=rel, line=line, text=f"{site.rule} ({site.word}): {text}")
            )
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="commands",
    rule=(
        "one command classifier and one splitter: ask chickadee.approval.commands' "
        "CommandTable.classify / classify_line, never a regex naming a command outside "
        "approval/commands.py (C1); take a line apart with chickadee.shell_line (argv, split, "
        "join), never shlex outside shell_line.py (C2) (D106 §2, D166)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
