"""`paths.glob_match`, the one path-glob matcher, and "is this a test file".

It replaced three copies of one matcher (`packer/tools.is_test_path`,
`applier/gates._matches_any`, `approval/facts._first_match`) and the graph's own list
(`graph/queries._is_test_path`). Their answers were pinned before the change (`5b0a02e`);
`MOVED` lists every one that changed and why, and everything else answers as it did.
"""

from __future__ import annotations

import pytest

from chickadee.paths import (
    AnyOne,
    AnyRun,
    Char,
    CharClass,
    glob_match,
    segment_tokens,
    spelled_out,
)
from chickadee.profiles.loader import load_profile

MIGRATION_GLOBS: tuple[str, ...] = (
    "**/migrations/**",
    "migrations/**",
    "**/migrate/**",
    "**/alembic/versions/**",
    "**/schema/**",
    "**/prisma/**",
    "*.sql",
)
"""The router's migration globs (`approval/facts.DEFAULT_MIGRATION_GLOBS`, fact row 4), as
they stood when the router's fact rows retired with the packet loop: kept
here as globs `glob_match` was pinned over, never as a rule anything enforces."""

OLD_PY = ("tests/**", "test_*.py", "*_test.py", "conftest.py")
OLD_JS = ("test/**", "tests/**", "__tests__/**", "*.test.js", "*.spec.js")
PY = load_profile("python").tests
JS = load_profile("javascript").tests

# path: (python test, javascript test, migration path)
ANSWERS: dict[str, tuple[int, int, int]] = {
    "tests/test_a.py": (1, 1, 0),
    "tests/helpers.py": (1, 1, 0),
    "./tests/helpers.py": (1, 1, 0),
    "pkg/tests/helpers.py": (1, 1, 0),
    "pkg/tests/test_x.py": (1, 1, 0),
    "test_utils/helpers.py": (0, 0, 0),
    "src/app.py": (0, 0, 0),
    "conftest.py": (1, 0, 0),
    "pkg/conftest.py": (1, 0, 0),
    "src/a_test.py": (1, 0, 0),
    "tests": (0, 0, 0),
    "x/__tests__/a.js": (0, 1, 0),
    "src/a.test.js": (0, 1, 0),
    "test/a.js": (0, 1, 0),
    "src/test/a.js": (0, 1, 0),
    "src/a.spec.ts": (0, 0, 0),
    "db/migrations/0001.py": (0, 0, 1),
    "x/alembic/versions/a.py": (0, 0, 1),
    "schema.sql": (0, 0, 1),
    "sql/a.sql": (0, 0, 1),
    "migrations/x.py": (0, 0, 1),
}

MOVED: dict[str, str] = {
    "pkg/tests/helpers.py": "the profiles name nested test directories (D167 decision 5)",
    "pkg/tests/test_x.py": "javascript: `**/tests/**` (decision 5)",
    "test_utils/helpers.py": "a glob with no `/` names a file, never a directory (decision 4)",
    "x/__tests__/a.js": "javascript: `**/__tests__/**` (decision 5)",
    "src/test/a.js": "javascript: `**/test/**` (decision 5)",
    "tests": "a file named `tests` is not inside a `tests` directory (D168 rule 2)",
}
"""Against the Applier's and the router's copies under the old profiles. The packer's copy
also did not normalise, so `./tests/helpers.py` moved for it alone; the graph's own list is
gone with the feature it served (decision 2)."""


@pytest.mark.parametrize("path", sorted(ANSWERS))
def test_each_path_answers(path: str) -> None:
    got = (
        PY.matches(path),
        JS.matches(path),
        glob_match(path, MIGRATION_GLOBS) is not None,
    )
    assert tuple(int(b) for b in got) == ANSWERS[path]


@pytest.mark.parametrize("path", sorted(ANSWERS))
def test_an_answer_moved_only_where_listed(path: str) -> None:
    """The old copies' answer, rebuilt from their rules, against the new one."""
    old = (_old(path, OLD_PY), _old(path, OLD_JS))
    new = ANSWERS[path][:2]
    assert (old != new) == (path in MOVED), (path, old, new)


def _old(path: str, globs: tuple[str, ...]) -> int:
    """The Applier's and the router's copy (`_matches_any`, `_first_match`) as it was:
    normalised, then whole path or base name by `fnmatch`, or a `prefix/**` walk."""
    import fnmatch

    parts = [s for s in path.split("/") if s and s != "."]
    whole, name = "/".join(parts), parts[-1]
    for glob in globs:
        if fnmatch.fnmatchcase(whole, glob) or fnmatch.fnmatchcase(name, glob):
            return 1
        prefix = glob.split("**", 1)[0].rstrip("/")
        if (
            glob.endswith("**")
            and prefix
            and (whole == prefix or whole.startswith(prefix + "/"))
        ):
            return 1
    return 0


@pytest.mark.parametrize(
    ("path", "glob", "hit"),
    [
        # 1. `*`, `?` and a class stay within one segment
        ("a/b/c.py", "a/*.py", False),
        ("a/c.py", "a/*.py", True),
        ("pkg/net/client.py", "pkg/*.py", False),
        ("a/b.py", "a?b.py", False),
        ("a/b.py", "a[/]b.py", False),
        # 2. `**` as a whole segment is any number of directories, none included
        ("app.yaml", "**/*.yaml", True),
        ("config/app.yaml", "**/*.yaml", True),
        ("pkg/client.py", "pkg/**/*.py", True),
        ("pkg/net/client.py", "pkg/**/*.py", True),
        ("a/b", "a/**/b", True),
        ("a/x/y/b", "a/**/b", True),
        ("a/b/c", "a/**", True),
        ("a", "a/**", False),
        ("ab/c.py", "a/**", False),
        ("x/a/b/y.py", "**/a/b/**", True),
        ("x/a/c/b/y.py", "**/a/b/**", False),
        ("any/thing", "**", True),
        # 3. no `/`: the base name, at any depth, never a directory
        ("a/b/c.py", "*.py", True),
        ("pkg/conftest.py", "conftest.py", True),
        ("test_utils/helpers.py", "test_*.py", False),
        # 4. a `/` at the start or middle anchors at the root
        ("src/a.py", "/src/*.py", True),
        ("x/src/a.py", "/src/*.py", False),
        ("x/src/a.py", "src/*.py", False),
        # 5. a trailing `/` is everything inside the directory
        ("pkg/tests/a.py", "tests/", True),
        ("tests", "tests/", False),
        ("src/a/b/c.py", "src/a/", True),
        ("x/src/a/b.py", "src/a/", False),
        # 6. `**` not a whole segment is `*`
        ("a.py", "**.py", True),
        ("src/a/b.py", "src/**.py", False),
        # 7. escapes, classes, and what is literal
        ("notes.txt", "[!a]*.txt", True),
        ("app.txt", "[!a]*.txt", False),
        ("bpp.txt", "[^a]*.txt", True),
        ("]x", "[]]x", True),
        ("a*b", "a\\*b", True),
        ("axb", "a\\*b", False),
        ("[x", "[x", True),
        ("!a.py", "!a.py", True),
        ("a.py", "!a.py", False),
        ("a.js", "*.{js,ts}", False),
        # a class is read as `fnmatch` reads its members: `-` first or last is itself,
        # a descending range matches nothing rather than failing to compile, and no range
        # reaches `/`
        ("-.key", "[^-a]*.key", False),
        ("b.key", "[^-a]*.key", True),
        ("B.key", "[^-a]*.key", True),
        ("-", "[a-]", True),
        ("bx", "[z-a]x", False),
        ("bx", "[!z-a]x", True),
        ("a/b", "a[--0]b", False),
        ("a.b", "a[--0]b", True),
        # a `\\` escapes inside a class too, as git reads it
        ("]x", "[\\]]x", True),
        ("\\]x", "[\\]]x", False),
        ("-", "[a\\-z]", True),
        ("b", "[a\\-z]", False),
        # case-sensitive, as git and ripgrep are; nothing matches an empty path or glob
        ("A.PY", "*.py", False),
        ("", "*", False),
        ("a", "", False),
        ("a", "/", False),
    ],
)
def test_the_rules(path: str, glob: str, hit: bool) -> None:
    assert (glob_match(path, (glob,)) is not None) is hit


@pytest.mark.parametrize(
    ("glob", "out"),
    [
        ("tests/", "**/tests/**"),
        ("/tests/", "/tests/**"),
        ("src/a/", "src/a/**"),
        ("!a.py", "\\!a.py"),
        ("*.py", "*.py"),
        ("/", "/"),
    ],
)
def test_spelled_out_is_what_ripgrep_must_be_handed(glob: str, out: str) -> None:
    assert spelled_out(glob) == out


def test_the_first_matching_glob_is_named() -> None:
    assert glob_match("db/migrations/a.sql", MIGRATION_GLOBS) == "**/migrations/**"


@pytest.mark.parametrize(
    ("glob", "path"),
    [
        ("**/**/**/**/**/**/**/**/x", "/".join(["d"] * 40) + "/y"),
        ("**/a*/**/a*/**/a*/**/a*/**/b", "/".join(["aa"] * 60)),
        ("*a*a*a*a*a*a*a*a*c", "a" * 255),
    ],
)
def test_a_glob_cannot_stall_the_match(glob: str, path: str) -> None:
    """The scout's `search` globs are a model's. Before each fixed run was taken atomically,
    the first of these took eleven seconds on one path; now each is under a millisecond."""
    import time

    started = time.perf_counter()
    assert glob_match(path, (glob,)) is None
    assert time.perf_counter() - started < 0.5


def test_one_reading_of_a_segment_serves_both_writers() -> None:
    """`segment_tokens` is the one reading of a glob's characters: `glob_match` writes it as a
    Python expression and the Seatbelt profile as its own."""
    tokens = segment_tokens("id_[!x]?*.p[k-m]")
    assert tokens == (
        Char("i"),
        Char("d"),
        Char("_"),
        CharClass((("x", "x"),), negated=True),
        AnyOne(),
        AnyRun(),
        Char("."),
        Char("p"),
        CharClass((("k", "m"),), negated=False),
    )
    assert segment_tokens("a**b") == (Char("a"), AnyRun(), Char("b"))  # rule 6
    assert segment_tokens("\\*") == (Char("*"),)  # rule 7
    assert segment_tokens("x[") == (Char("x"), Char("["))
