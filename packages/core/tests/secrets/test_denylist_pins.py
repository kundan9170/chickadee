"""Pins: what the secret deny list denies, as it answered before it matched through the one
path-glob matcher.

The deny list read its patterns its own way (`fnmatch`, a hand-written `dir/**` walk), and its
two entries disagreed with each other: `match` (the write rule) judged a path whole, while
`name_denied` (the read rule) judged every tail of it. These pins hold every answer, so the
change shows here as the exact answers it moves.

**The corpus** is each built-in pattern and each shape a project may add, alone on a list,
against paths and directories; the sandbox mask `resolve` builds over a tree; and the Seatbelt
row each bare-name pattern becomes.

**The golden** is `fixtures/pins/denylist.json`, recorded in older versions and never written
again.
Every answer that moved is a row of `fixtures/pins/denylist.changes.tsv` with the D170 rule
that moves it, and nothing else moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from chickadee.isolation.macos import deny_regexes
from chickadee.isolation.types import Isolation
from chickadee.secrets.denylist import DEFAULT_PATTERNS, Denylist, name_denied, resolve

PINS: Final = Path(__file__).resolve().parents[1] / "fixtures" / "pins"
GOLDEN: Final = PINS / "denylist.json"
CHANGES: Final = PINS / "denylist.changes.tsv"

ADDED: Final = (
    "*secret*",
    "secrets/*.yaml",
    "config/**",
    "**/prod.env",
    "[^a]*.key",
    "[!a]*.key",
    "*.[pk]em",
    ".docker/*.json",
    "keys/",
    "/root.key",
    "a/**/b.key",
    "vault/**/*.hcl",
    ".kube/config",
    "*.kdbx",
    "deploy/*",
    "**/*.tfstate",
    "Secret?.txt",
)
"""Shapes a project may add with `deny_add`: names, classes, anchored and nested paths."""

PATHS: Final = (
    (".env", False),
    (".ENV", False),
    (".env.example", False),
    ("config/.env", False),
    ("prod.pem", False),
    ("secrets/prod.pem", False),
    (".ssh", True),
    (".ssh", False),
    (".ssh/id_rsa", False),
    ("home/u/.ssh/id_rsa", False),
    ("a/.ssh", True),
    (".ssh2/x", False),
    (".docker/config.json", False),
    ("sub/.docker/config.json", False),
    (".docker/other.json", False),
    ("sub/.docker/other.json", False),
    ("secrets", True),
    ("secrets/a.yaml", False),
    ("secrets/x/b.yaml", False),
    ("my-secrets.yaml", False),
    ("config", True),
    ("config/app.yaml", False),
    ("config/a/b.yaml", False),
    ("x/config/app.yaml", False),
    ("prod.env", False),
    ("deploy/prod.env", False),
    ("b.key", False),
    ("a.key", False),
    ("^.key", False),
    ("x.pem", False),
    ("x.kem", False),
    ("keys", True),
    ("keys/a", False),
    ("sub/keys/a", False),
    ("root.key", False),
    ("sub/root.key", False),
    ("a/b.key", False),
    ("a/x/y/b.key", False),
    ("vault/a.hcl", False),
    ("vault/x/a.hcl", False),
    (".kube/config", False),
    ("db.kdbx", False),
    ("deploy/a", False),
    ("deploy/a/b", False),
    ("state.tfstate", False),
    ("env/state.tfstate", False),
    ("Secret1.txt", False),
    ("service-account-x.json", False),
    ("id_rsa.pub", False),
    ("src/app.py", False),
    ("/home/u/.aws/credentials", False),
)

_FILES: Final = tuple(p for p, is_dir in PATHS if not is_dir and not p.startswith("/"))
TREE: Final = tuple(p for p in _FILES if not any(q.startswith(f"{p}/") for q in _FILES))
"""The files `resolve` walks: every relative file of the corpus but those that are also a
directory of another (`.ssh`, `deploy/a`)."""


def alone(pattern: str) -> Denylist:
    """A list holding `pattern` and the template exemptions, and nothing else."""
    return Denylist(
        patterns=(pattern,),
        removed=tuple(p for p in DEFAULT_PATTERNS if p != pattern),
    )


def _key(path: str, is_dir: bool) -> str:
    return f"{path}/ (dir)" if is_dir else path


def answers() -> dict[str, object]:
    """Every answer: `{caller: {pattern: {path: the denying pattern or None}}}`, and the
    Seatbelt rows."""
    out: dict[str, object] = {}
    for caller in ("match", "name_denied"):
        by_pattern: dict[str, dict[str, str | None]] = {}
        for pattern in (*DEFAULT_PATTERNS, *ADDED):
            listed = alone(pattern)
            row: dict[str, str | None] = {}
            for path, is_dir in PATHS:
                if caller == "match":
                    denial = listed.match(path, is_dir=is_dir)
                else:
                    denial = name_denied(path, listed, is_dir=is_dir)
                row[_key(path, is_dir)] = None if denial is None else denial.pattern
            by_pattern[pattern] = row
        out[caller] = by_pattern
    out["rows"] = {p: seatbelt_row(p) for p in (*DEFAULT_PATTERNS, *ADDED) if "/" not in p}
    return out


def seatbelt_row(pattern: str) -> list[str]:
    """The Seatbelt regex row a bare-name pattern becomes (`isolation/macos.py`)."""
    return list(
        deny_regexes(Isolation(argv=("true",), cwd=Path("/"), deny_patterns=(pattern,)))
    )


def mask(root: Path) -> list[str]:
    """What `resolve` masks over `TREE`, under the built-ins and every addition at once."""
    for rel in TREE:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    listed = Denylist.configure(add=ADDED)
    return [p.relative_to(root).as_posix() for p in resolve(listed, [root])]


def _moves(root: Path) -> set[tuple[str, str, str, str, str]]:
    """`(caller, pattern, path, before, after)` for every answer that moved."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = answers()
    out: set[tuple[str, str, str, str, str]] = set()
    for caller in ("match", "name_denied"):
        was_c, now_c = golden[caller], now[caller]
        assert isinstance(now_c, dict)
        for pattern, row in was_c.items():
            for path, before in row.items():
                after = now_c[pattern][path]  # type: ignore[index]
                if before != after:
                    out.add((caller, pattern, path, str(before), str(after)))
    for pattern, before in golden["rows"].items():
        after = now["rows"][pattern]  # type: ignore[index]
        if before != after:
            out.add(("rows", pattern, "-", str(before), str(after)))
    was_mask, now_mask = set(golden["mask"]), set(mask(root))
    for path in sorted(was_mask ^ now_mask):
        out.add(("mask", "*", path, str(path in was_mask), str(path in now_mask)))
    return out


def _listed() -> dict[tuple[str, str, str, str, str], str]:
    rows: dict[tuple[str, str, str, str, str], str] = {}
    for line in CHANGES.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        caller, pattern, path, before, after, why = line.split("\t")
        rows[(caller, pattern, path, before, after)] = why
    return rows


def test_the_corpus_is_the_recorded_one() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for caller in ("match", "name_denied"):
        assert set(golden[caller]) == {*DEFAULT_PATTERNS, *ADDED}, caller
        for row in golden[caller].values():
            assert set(row) == {_key(p, d) for p, d in PATHS}, caller


def test_every_moved_answer_is_listed_and_nothing_else_moved(tmp_path: Path) -> None:
    moved, listed = _moves(tmp_path), _listed()
    unlisted = sorted(moved - set(listed))
    stale = sorted(set(listed) - moved)
    assert not unlisted, f"{len(unlisted)} answers moved without a row: {unlisted[:10]}"
    assert not stale, f"{len(stale)} rows list an answer that did not move: {stale[:10]}"


def test_every_listed_move_names_its_rule() -> None:
    for row, why in _listed().items():
        assert why.startswith("D170 rule "), (row, why)
