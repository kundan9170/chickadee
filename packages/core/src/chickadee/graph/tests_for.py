"""Which tests reach a change: the test files within a few reverse edges of each
changed file, and the changed test files themselves. Moved from the verifier's
`affected_suite`, whose final checkpoint it chose, so the box's `tests_for` tool and
the facts read the same selection the verifier ran. Selected from the graph, never by a model;
deterministic over its inputs.

What it cannot see, the tool says every time: dynamic imports, fixtures, data files and
subprocesses are not graph edges.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from chickadee.profiles.model import LanguageProfile

__all__ = ["TESTS_FOR_DEPTH", "SuiteSelection", "tests_for"]

TESTS_FOR_DEPTH: Final = 3
"""How many graph steps back from a changed file a test is looked for: a test that imports a
module that imports the changed one is affected; one five steps away is the rest of the
suite."""


@dataclass(frozen=True)
class SuiteSelection:
    """The test files that reach what changed, and, for each, the changed file it reaches."""

    tests: tuple[str, ...]
    reached: Mapping[str, str]
    """Test file → the changed file it reaches; `""` for a test file that was itself changed."""
    depth: int = TESTS_FOR_DEPTH


def tests_for(
    changed: Iterable[str],
    edges: Iterable[tuple[str, str | None, str, int]],
    profile: LanguageProfile,
    *,
    exclude: Iterable[str] = (),
    depth: int = TESTS_FOR_DEPTH,
) -> SuiteSelection:
    """The test files within `depth` reverse edges of a changed file, and the changed test
    files themselves, in path order. `edges` are the graph's file edges (source, target,
    class, count); `exclude` holds tests that are not the repository's (the run's own),
    which are run and labelled apart."""
    skip = set(exclude)
    dependents: dict[str, set[str]] = {}
    for src, dst, _cls, _n in edges:
        if dst is not None and dst != src:
            dependents.setdefault(dst, set()).add(src)
    reached: dict[str, str] = {}
    for start in sorted(set(changed)):
        if profile.tests.matches(start):
            reached.setdefault(start, "")
        seen = {start}
        frontier = {start}
        for _ in range(depth):
            frontier = {s for f in frontier for s in dependents.get(f, ()) if s not in seen}
            seen |= frontier
            for f in sorted(frontier):
                if profile.tests.matches(f):
                    reached.setdefault(f, start)
    tests = tuple(sorted(t for t in reached if t not in skip))
    return SuiteSelection(tests, {t: reached[t] for t in tests}, depth)
