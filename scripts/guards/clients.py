"""The clients guard: there is one client protocol, `ModelClient` in
`client/request.py`, and every caller reaches a client through it.

Zeus and the scout each declared the `send` protocol D101 specified and nobody wrote, the
engine closed clients through `getattr`, and the frozen `complete()` survived beside `send`
with one caller (#276 T7.1, T7.2, T7.4). `T-3.70` moved every one. This fails the next one, by
three rules, over every package's `src`:

- **C1, a second client protocol:** a class deriving from `Protocol` that defines `send`,
  outside `client/request.py`.
- **C2, closing by lookup:** a `getattr(..., "close",...)` with the literal name. `close` is
  part of the protocol; whoever holds a client calls it.
- **C3, the retired call:** a `.complete(` call whose receiver is named `client`, or is an
  attribute named `client` (`self.client.complete(...)`). The effect recorder's
  `OpenEffect.complete` is held as `fx` or `effect`, so it is never matched.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. There are none at landing.

What it cannot see: a protocol spelled without `Protocol` (an ABC, a duck-typed class used
as one). That is the review checklist's first question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]

HOME = "packages/core/src/chickadee/client/request.py"
"""The one home of the client protocol."""

EXEMPT: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST


def _is_protocol(base: ast.expr) -> bool:
    """`Protocol`, `typing.Protocol`, or `Protocol[...]`."""
    if isinstance(base, ast.Subscript):
        base = base.value
    return (isinstance(base, ast.Name) and base.id == "Protocol") or (
        isinstance(base, ast.Attribute) and base.attr == "Protocol"
    )


def _defines_send(node: ast.ClassDef) -> bool:
    return any(
        isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == "send"
        for item in node.body
    )


def _closes_by_lookup(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "close"
    )


def _calls_complete_on_a_client(node: ast.Call) -> bool:
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "complete"):
        return False
    owner = func.value
    return (isinstance(owner, ast.Name) and owner.id == "client") or (
        isinstance(owner, ast.Attribute) and owner.attr == "client"
    )


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every C1, C2 and C3 site in `tree`, the module at repo-relative path `rel`."""
    out: list[_Site] = []
    for scope, node in functions(tree):
        if (
            isinstance(node, ast.ClassDef)
            and rel != HOME
            and any(_is_protocol(b) for b in node.bases)
            and _defines_send(node)
        ):
            out.append(_Site("C1", scope, node))
        if isinstance(node, ast.Call):
            if _closes_by_lookup(node):
                out.append(_Site("C2", scope, node))
            if _calls_complete_on_a_client(node):
                out.append(_Site("C3", scope, node))
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
            violations.append(Violation(path=rel, line=line, text=f"{site.rule}: {text}"))
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="clients",
    rule=(
        "one client protocol: type a model client as chickadee.client.ModelClient (send, "
        "close); never declare another Protocol with send (C1), close a client through "
        "getattr (C2), or call the retired complete() (C3) (D155)"
    ),
    corpus=("packages/*/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
