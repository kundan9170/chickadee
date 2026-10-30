"""Pins: which paths each glob matches, as the six matchers answered in older versions and D171.

Six functions answered "does this repository path match this glob", and read some globs
differently:

- `paths`: `paths.glob_match`, for the profile's test globs and the router's migration globs;

- `scout`: the scout's own `scout/search.py` `glob_match`, for `search`'s globs;
- `effects`: `executor/effects.py` `_declared`, for an action's declared effect globs;
- `preferences`: `state/preferences.py` `_matches` (`fnmatch`), for the `always_ask` and
  `never_ask` path globs;
- `ignore`: `graph/ignore.py`'s own glob-to-regex translation, for a `.gitignore` line, asked
  here as a root `.gitignore` holding the one glob, of a file;
- `deny`: `approval/prefs.py` `denied` (`fnmatch`), the preference `deny` list, recorded with
  `T-3.119`'s gap review in older versions moved it (its answers were the `preferences`
  matcher's).

These pins hold every answer each gave over the corpus below, so the one matcher that replaces
them shows here as the exact answers it moves.

**The golden** is `fixtures/pins/glob_match.json`, `{caller: {glob: [paths it matched]}}`,
recorded from the matchers in older versions. It is never written again: the
old matchers are gone. Every answer the one matcher gives differently is a row of
`fixtures/pins/glob_match.changes.tsv`, with the D168 rule that moves it, and nothing else
moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from test_glob_match import MIGRATION_GLOBS

from chickadee.graph.ignore import Ignore
from chickadee.paths import glob_match
from chickadee.profiles.loader import load_profile

PINS: Final = Path(__file__).resolve().parent / "fixtures" / "pins"
GOLDEN: Final = PINS / "glob_match.json"
CHANGES: Final = PINS / "glob_match.changes.tsv"
CALLERS: Final = ("paths", "scout", "effects", "preferences", "ignore", "deny")

_SHAPES: Final = (
    # a glob with no `/`
    "*",
    "*.py",
    "*.yaml",
    "test_?.py",
    "[!a]*.txt",
    "[ab]*.py",
    "**.py",
    "a**b.py",
    "tests",
    # a glob with a `/`
    "src/*",
    "src/*.py",
    "src/*/b.py",
    "*/*.py",
    "docs/*.md",
    "config/*",
    "/src/*.py",
    "src/**.py",
    # `**`
    "**",
    "**/*.py",
    "**/*.yaml",
    "**/test_*.py",
    "**/tests",
    "src/**",
    "src/**/*.py",
    "a/**/b",
    "tests/**",
    "test/**",
    "__tests__/**",
    # a trailing `/`
    "tests/",
    "src/a/",
)

PATHS: Final = (
    "a.py",
    "ab",
    "a/b",
    "a/x/b",
    "a/x/y/b",
    "app.yaml",
    "config/app.yaml",
    "a/b/c.yaml",
    "src/a.py",
    "src/a/b.py",
    "src/a/b/c.py",
    "src/.hidden.py",
    "src/a.test.js",
    "src/a.spec.ts",
    "src/a_test.py",
    "src/test/a.js",
    "tests",
    "tests/test_a.py",
    "tests/unit/test_b.py",
    "pkg/tests/helpers.py",
    "pkg/tests/test_x.py",
    "test_utils/helpers.py",
    "test_1.py",
    "test/a.js",
    "x/__tests__/a.js",
    "conftest.py",
    "pkg/conftest.py",
    "notes.txt",
    "app.txt",
    "bin.txt",
    "docs/x.md",
    "docs/a/y.md",
    "db/migrations/0001.py",
    "migrations",
    "migrations/x.py",
    "x/alembic/versions/a.py",
    "schema.sql",
    "sql/a.sql",
    "schema/a.py",
    "prisma/schema.prisma",
    ".env",
    ".github/workflows/ci.yml",
    "a**b.py",
    "axb.py",
)


def patterns() -> tuple[str, ...]:
    shipped = (
        *load_profile("python").tests.globs,
        *load_profile("javascript").tests.globs,
        *MIGRATION_GLOBS,
    )
    return tuple(dict.fromkeys((*shipped, *_SHAPES)))


def answers(match: object) -> dict[str, list[str]]:
    """`{glob: [paths it matches]}` for a `match(path, glob) -> bool`."""
    assert callable(match)
    return {g: [p for p in PATHS if match(p, g)] for g in patterns()}


def one_matcher(path: str, glob: str) -> bool:
    return glob_match(path, (glob,)) is not None


def ignored(root: Path) -> dict[str, list[str]]:
    """The `.gitignore` reader's answers, now over the one matcher: a root `.gitignore`
    holding each glob alone, asked of each path as a file."""
    out: dict[str, list[str]] = {}
    for i, glob in enumerate(patterns()):
        here = root / str(i)
        here.mkdir()
        (here / ".gitignore").write_text(glob + "\n", encoding="utf-8")
        reader = Ignore(here)
        out[glob] = [p for p in PATHS if reader.skipped(here / p)]
    return out


def _moves(root: Path) -> set[tuple[str, str, str, str, str]]:
    """`(caller, glob, path, before, after)` for every answer the one matcher moves."""
    golden = json.loads(GOLDEN.read_text())
    matched, by_ignore = answers(one_matcher), ignored(root)
    out: set[tuple[str, str, str, str, str]] = set()
    for caller in CALLERS:
        now = by_ignore if caller == "ignore" else matched
        for glob in patterns():
            was = set(golden[caller][glob])
            for path in PATHS:
                before, after = path in was, path in now[glob]
                if before != after:
                    out.add((caller, glob, path, _word(before), _word(after)))
    return out


def _word(hit: bool) -> str:
    return "match" if hit else "no match"


def _listed() -> dict[tuple[str, str, str, str, str], str]:
    rows: dict[tuple[str, str, str, str, str], str] = {}
    for line in CHANGES.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        caller, glob, path, before, after, why = line.split("\t")
        rows[(caller, glob, path, before, after)] = why
    return rows


def test_the_corpus_is_the_recorded_one() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(CALLERS)
    for caller in CALLERS:
        assert set(golden[caller]) == set(patterns()), caller
        recorded = {p for hits in golden[caller].values() for p in hits}
        assert recorded <= set(PATHS), caller


def test_every_moved_answer_is_listed_and_nothing_else_moved(tmp_path: Path) -> None:
    moved, listed = _moves(tmp_path), _listed()
    unlisted = sorted(moved - set(listed))
    stale = sorted(set(listed) - moved)
    assert not unlisted, f"{len(unlisted)} answers moved without a row: {unlisted[:10]}"
    assert not stale, f"{len(stale)} rows list an answer that did not move: {stale[:10]}"


def test_every_listed_move_names_its_rule() -> None:
    for row, why in _listed().items():
        assert why.startswith("D168 rule "), (row, why)


SAME_READING: Final = frozenset({"preferences", "deny"})
"""Two matchers that were one reading: both called `fnmatch.fnmatch` on the path."""


def test_the_matchers_disagreed_and_the_corpus_shows_it() -> None:
    """A pin over globs every matcher read alike pins nothing: each differed from the others,
    but for the two that were one reading, which answered alike."""
    golden = json.loads(GOLDEN.read_text())
    for a in CALLERS:
        for b in CALLERS:
            if a < b and {a, b} != SAME_READING:
                assert any(golden[a][g] != golden[b][g] for g in patterns()), (a, b)
    assert all(golden["preferences"][g] == golden["deny"][g] for g in patterns())
