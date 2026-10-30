#!/usr/bin/env python3
"""The D148 guard runner: one concept, one home, enforced and ratcheted.

Each theme of the quality wave ends with a guard — a check that fails a second
implementation of the concept the theme just unified. Without one runner, every guard would
edit the same audit script and the same CI step, and themes meant to run in parallel would
queue behind one file. So each guard is its own module under `scripts/guards/`, and this
runs all of them.

    python3 scripts/guards/run.py # check
    python3 scripts/guards/run.py --tighten # lower every baseline to what is true now

Exit code 0 = clean, 1 = at least one guard failed.

**Why a guard may fail for improving things.** A baseline is the debt a rule tolerates today,
so the rule can land before its last caller moves. When a count drops below its baseline the
ratchet has worked — but the recorded number is now loose, and a check advertising protection
it is not giving is the failure this repository keeps meeting from the other side (invariant
20: a clean answer is only a fact when coverage was total). So slack fails, and `--tighten`
makes the fix one command instead of manual arithmetic.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# `run.py` lives inside the package whose types it needs, and is invoked as a file rather
# than with `-m`, so that package's parent has to reach `sys.path` before the import below.
# Doing it here, once, is what lets every function in this file be properly typed instead of
# reaching through `getattr`.
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from guards import Guard, Violation  # noqa: E402

BASELINE_NAME = "BASELINE"
"""The module-level assignment `--tighten` rewrites. Named here rather than matched by regex
because `ast` gives the exact line span, so the rewrite needs no markers in the source."""

GUARD_NAME = "GUARD"
"""The module-level `Guard` the runner reads. One fixed attribute, so discovery never has to
guess at a naming convention."""

NOT_GUARDS = frozenset({"__init__.py", "run.py"})

ROWS_SHOWN = 5
"""Violation rows printed per failing file; the rest are counted. The rows say where
to look and the rule says what to do; a size guard reporting one row per line past its budget
would otherwise print hundreds. Changes what is printed, never a count or a verdict."""
"""The package's own files. Named rather than inferred from `__file__`, so a fixture
directory behaves exactly as the real one does."""


PYPROJECT = "pyproject.toml"
"""Where the project declares its interpreter floor, as `[project].requires-python`. Read,
never restated: writing the version a second time here would give it a second home."""


class GuardError(Exception):
    """A guard that cannot run. Always a failure, never a skip: a guard that is skipped and a
    guard that passes must not print the same thing."""


def _running() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def floor(root: Path) -> tuple[int, int]:
    """The `>=` lower bound of `requires-python` in the root `pyproject.toml`."""
    try:
        import tomllib
    except ModuleNotFoundError:
        # Below 3.11, which is below any floor this repository has declared.
        raise GuardError(
            f"Python {_running()} cannot read {PYPROJECT}; "
            "run the guards on the project's Python"
        ) from None
    path = root / PYPROJECT
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GuardError(f"cannot read the interpreter floor from {path}: {exc}") from None
    spec = project.get("requires-python") if isinstance(project, dict) else None
    if isinstance(spec, str):
        for clause in spec.split(","):
            bound = clause.strip()
            if bound.startswith(">="):
                numbers = bound[2:].strip().split(".")
                if len(numbers) >= 2 and numbers[0].isdigit() and numbers[1].isdigit():
                    return int(numbers[0]), int(numbers[1])
    raise GuardError(f"{path}'s requires-python is {spec!r}, which names no '>=X.Y' floor")


def check_interpreter(root: Path) -> None:
    """Refuse to run any guard on an interpreter below the project's floor.

    A guard parses project source with `ast`, and that source uses syntax only the floor
    parses — PEP 695 generics, at `adapters/base.py` for one. Below it, the first such guard
    crashes inside `ast.parse` with `SyntaxError: expected '('`, naming neither the guard's rule
    nor the cause. Checked before discovery, so no guard module is even imported.
    """
    required = floor(root)
    if sys.version_info[:2] < required:
        need = f"{required[0]}.{required[1]}"
        raise GuardError(
            f"Python {_running()} is below the project's floor of {need} (requires-python in "
            f"{root / PYPROJECT}); the guards parse source written for {need}, so they refuse "
            "to run rather than fail inside ast.parse"
        )


@dataclass(frozen=True)
class Outcome:
    """What one guard found, measured against what it tolerates."""

    name: str
    rule: str
    files_read: int
    counts: Mapping[str, int]
    baseline: Mapping[str, int]
    over: tuple[tuple[str, int, int], ...]
    """(path, found, allowed) where found > allowed — a second implementation appeared."""
    slack: tuple[tuple[str, int, int], ...]
    """(path, found, allowed) where found < allowed — the baseline is now loose."""
    violations: Mapping[str, tuple[Violation, ...]]
    """Per path, the rows themselves, so a failure can name file and line."""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def failed(self) -> bool:
        return bool(self.over or self.slack)


def _purge(package: str) -> None:
    """Drop a previously imported guard package, so discovering a second directory does not
    serve the first one's modules from cache. The runner itself discovers once; the test
    discovers many fixture directories, and a function that only works once is a trap.

    The module that defines `Guard` is kept. Every guard imports its types from it, and a
    fresh copy would be a second `Guard` class, failing the `isinstance` below for every
    real guard: the real directory *is* that package, which fixture directories, named
    otherwise, never showed."""
    keep = Guard.__module__
    for name in [n for n in sys.modules if n == package or n.startswith(f"{package}.")]:
        if name != keep:
            del sys.modules[name]


def discover(guard_dir: Path) -> list[Guard]:
    """Import every guard module in `guard_dir` and return its `GUARD`.

    A candidate is `*.py` other than the package's own files and any `_`-prefixed stem. A
    candidate that does not yield a well-formed `Guard` raises rather than being passed over.
    """
    package = guard_dir.name
    candidates = sorted(
        p
        for p in guard_dir.glob("*.py")
        if p.name not in NOT_GUARDS and not p.name.startswith("_")
    )
    parent = str(guard_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    _purge(package)

    guards: list[Guard] = []
    for path in candidates:
        module = importlib.import_module(f"{package}.{path.stem}")
        guard = getattr(module, GUARD_NAME, None)
        if not isinstance(guard, Guard):
            raise GuardError(f"{path.name}: no module-level `{GUARD_NAME}`")
        if guard.name != path.stem:
            raise GuardError(
                f"{path.name}: {GUARD_NAME}.name is {guard.name!r}, expected {path.stem!r}"
            )
        guards.append(guard)
    return guards


def files_for(root: Path, corpus: Sequence[str]) -> list[Path]:
    """Resolve a guard's globs against the repository root.

    An empty result raises. A guard whose corpus matches nothing passes forever, which is how
    a check written against `context/` keeps reporting clean after that directory is split out
    from under it.
    """
    found: set[Path] = set()
    for pattern in corpus:
        found.update(p for p in root.glob(pattern) if p.is_file())
    if not found:
        raise GuardError(f"corpus matches no files: {', '.join(corpus)}")
    return sorted(found)


def evaluate(guard: Guard, root: Path) -> Outcome:
    """Run one guard and compare what it found against what it tolerates."""
    files = files_for(root, guard.corpus)

    grouped: dict[str, list[Violation]] = {}
    for v in guard.find(files):
        grouped.setdefault(v.path, []).append(v)
    counts = {path: len(rows) for path, rows in grouped.items()}
    baseline = guard.baseline

    over = tuple(
        (path, n, baseline.get(path, 0))
        for path, n in sorted(counts.items())
        if n > baseline.get(path, 0)
    )
    slack = tuple(
        (path, counts.get(path, 0), allowed)
        for path, allowed in sorted(baseline.items())
        if counts.get(path, 0) < allowed
    )
    return Outcome(
        name=guard.name,
        rule=guard.rule,
        files_read=len(files),
        counts=counts,
        baseline=baseline,
        over=over,
        slack=slack,
        violations={path: tuple(rows) for path, rows in grouped.items()},
    )


def render_baseline(counts: Mapping[str, int]) -> str:
    """The `BASELINE` assignment as source. Sorted, so tightening produces a stable diff."""
    live = {path: n for path, n in sorted(counts.items()) if n > 0}
    if not live:
        return f"{BASELINE_NAME}: dict[str, int] = {{}}\n"
    rows = "".join(f'    "{path}": {n},\n' for path, n in live.items())
    return f"{BASELINE_NAME}: dict[str, int] = {{\n{rows}}}\n"


def rewrite_baseline(path: Path, counts: Mapping[str, int]) -> None:
    """Replace the `BASELINE` assignment in a guard module with what is true now."""
    src = path.read_text(encoding="utf-8")
    node = _baseline_node(ast.parse(src), path)
    lines = src.splitlines(keepends=True)
    assert node.end_lineno is not None
    lines[node.lineno - 1 : node.end_lineno] = [render_baseline(counts)]
    path.write_text("".join(lines), encoding="utf-8")


def _baseline_node(tree: ast.Module, path: Path) -> ast.stmt:
    for stmt in tree.body:
        target = None
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        if isinstance(target, ast.Name) and target.id == BASELINE_NAME:
            return stmt
    raise GuardError(f"{path.name}: no module-level `{BASELINE_NAME}` to tighten")


def report(outcomes: Sequence[Outcome]) -> None:
    """One numbered line per guard, in the two audits' format.

    `files_read` is on every line so "ran 11 guards over 0 files" is visible rather than
    inferred from a green tick.
    """
    for i, o in enumerate(outcomes, start=1):
        allowed = sum(o.baseline.values())
        note = ""
        if o.over:
            note = f" — {sum(n - a for _, n, a in o.over)} ABOVE baseline"
        elif o.slack:
            note = " — SLACK, the baseline is looser than the code"
        print(
            f"{i}. {o.name:<18} {o.files_read} files, {o.total} violation(s) "
            f"(baseline {allowed}){note}"
        )


def failures_of(outcome: Outcome) -> list[str]:
    """Every reason this guard failed, each naming the rule that was broken."""
    out: list[str] = []
    for path, found, allowed in outcome.over:
        out.append(
            f"{outcome.name}: {path} has {found} violation(s), baseline allows {allowed}"
        )
        out.append(f"    rule: {outcome.rule}")
        rows = outcome.violations.get(path, ())
        for v in rows[:ROWS_SHOWN]:
            out.append(f"    {path}:{v.line}: {v.text}")
        if len(rows) > ROWS_SHOWN:
            out.append(f"    ... and {len(rows) - ROWS_SHOWN} more")
    for path, found, allowed in outcome.slack:
        out.append(
            f"{outcome.name}: {path} is down to {found} from a baseline of {allowed} — "
            "the ratchet worked, now lower the baseline"
        )
    return out


def run(guard_dir: Path = HERE, root: Path = REPO_ROOT, *, tighten: bool = False) -> int:
    """Check every guard, or lower every baseline to what is true now."""
    try:
        # First, and for both modes: `--tighten` evaluates every guard before it rewrites
        # anything, so on the wrong interpreter it would fail the same unhelpful way.
        check_interpreter(root)
        guards = discover(guard_dir)
        outcomes = [evaluate(g, root) for g in guards]
    except GuardError as exc:
        print("GUARDS FAILED")
        print(f"  - {exc}")
        return 1

    if tighten:
        for outcome in outcomes:
            rewrite_baseline(guard_dir / f"{outcome.name}.py", outcome.counts)
            print(f"tightened {outcome.name}: {sum(outcome.counts.values())} violation(s)")
        if not guards:
            print("no guards to tighten")
        return 0

    if not guards:
        print("0 guards — none has landed yet; each theme adds its own (D148 decision 4)")
        print()
        print("GUARDS PASSED")
        return 0

    report(outcomes)
    reasons = [line for o in outcomes for line in failures_of(o)]
    print()
    if reasons:
        print("GUARDS FAILED")
        for line in reasons:
            print(f"  - {line}")
        if any(o.slack for o in outcomes):
            print()
            print("  run `python3 scripts/guards/run.py --tighten` and commit the result")
        return 1
    print(f"GUARDS PASSED — {len(outcomes)} concept(s) each in one home")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tighten",
        action="store_true",
        help="rewrite every baseline to the counts found now, then exit 0",
    )
    args = parser.parse_args(argv)
    return run(tighten=args.tighten)


if __name__ == "__main__":
    sys.exit(main())
