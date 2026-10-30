"""The effects guard: every effect's lifecycle is written by
`journal/effects.py`, and every effect kind is spelled there.

The lifecycle was written by hand in four places, the three phases were listed three times,
and the kinds were bare strings read again wherever a reader needed one (#276 T4.1). `T-3.66`
moved every one onto the recorder. This fails the next one, by three rules, over
`packages/core/src/chickadee/**/*.py`:

- **E1, building an effect payload:** a call to `EffectIntendedP`, `EffectStartedP` or
  `EffectCompletedP` outside `journal/effects.py`. A payload read back (an `isinstance`, a
  `lines(EffectIntendedP)`, an annotation) is not a call and never a violation.
- **E2, naming an effect phase:** `EventKind.EFFECT_INTENDED`, `EFFECT_STARTED` or
  `EFFECT_COMPLETED` outside `journal/effects.py` and `schema/journal.py`. Every other module
  reaches them through `EFFECT_EVENTS` or the recorder.
- **E3, an effect kind as a string:** a string literal equal to an `EffectKind` value outside
  `journal/effects.py` and `schema/journal.py` (whose `EventKind.SCOUT_ESCAPE` is the journal
  line kind, a different thing with the same word).

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails, as the records and projection guards' do.

What it cannot see: a lifecycle written through a raw dict handed to `Journal.append`. That
form is already a records-guard violation, and otherwise the review checklist's first
question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import MODULE, docstrings, functions

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = "packages/core/src/chickadee/"

RECORDER = CORE + "journal/effects.py"
"""The one home: the only file that builds an effect payload or spells a kind."""

SCHEMA = CORE + "schema/journal.py"
"""Where the payloads and `EventKind` are defined."""

PAYLOADS = frozenset({"EffectIntendedP", "EffectStartedP", "EffectCompletedP"})

PHASES = frozenset({"EFFECT_INTENDED", "EFFECT_STARTED", "EFFECT_COMPLETED"})

KINDS = frozenset(
    {
        "bootstrap",
        "scout_escape",
        "shell_write",
        "worker_step",
        "task_commit",
        "task_known_good",
        "approval_commit",
        "shell_job",
    }
)
"""`EffectKind`'s values, restated: a guard imports only the standard library, so it
reads source text and cannot import the enum. `test_effects_guard.py` checks the two agree."""

EXEMPT: dict[tuple[str, str], str] = {
    (CORE + "profiles/loader.py", "executable_overrides"): (
        'the profile\'s TOML key "bootstrap" ([toolchain].bootstrap, D109), not an effect kind'
    ),
    (CORE + "profiles/model.py", "ToolchainSpec._parse"): (
        'the profile\'s TOML key "bootstrap" ([toolchain].bootstrap, D109), not an effect kind'
    ),
    (CORE + "run_header.py", MODULE): (
        "`HeaderKind`'s \"bootstrap\": a run-header line's kind, not an effect kind"
    ),
    (CORE + "run_header.py", "_bootstrap"): (
        'the "bootstrap" header line the function renders: a header kind, not an effect kind'
    ),
}
"""D153 decision 7's, each the word "bootstrap" meaning something other than an effect."""


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_phase(node: ast.Attribute) -> bool:
    """`EventKind.EFFECT_*`, however the module was imported (`schema.EventKind.X`)."""
    if node.attr not in PHASES:
        return False
    owner = node.value
    return (isinstance(owner, ast.Name) and owner.id == "EventKind") or (
        isinstance(owner, ast.Attribute) and owner.attr == "EventKind"
    )


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every E1, E2 and E3 site in `tree`, the module at repo-relative path `rel`."""
    if rel == RECORDER:
        return []
    prose = docstrings(tree)
    out: list[_Site] = []
    for scope, node in functions(tree):
        if isinstance(node, ast.Call) and _call_name(node) in PAYLOADS:
            out.append(_Site("E1", scope, node))
        if rel == SCHEMA:
            continue
        if isinstance(node, ast.Attribute) and _is_phase(node):
            out.append(_Site("E2", scope, node))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in KINDS
            and id(node) not in prose
        ):
            out.append(_Site("E3", scope, node))
    return out


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one.

    Parses core's source, which is Python 3.12: run it with 3.12 or later, as CI's
    `python3` is."""
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for site in sites(ast.parse(source), rel):
            key = (rel, site.function)
            if site.rule == "E3" and key in EXEMPT:
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
    name="effects",
    rule=(
        "every effect is recorded through journal/effects.py: perform() or reopen(), never an "
        "effect payload built (E1), an effect phase named (E2) or an effect kind spelled as a "
        "string (E3) anywhere else; use EffectKind and EFFECT_EVENTS (D153)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
