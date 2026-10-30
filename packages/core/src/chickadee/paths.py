"""Where a path lands, and how it is spelled under a root.

Three questions that look alike and are not:

- **Where a path lands** (`within`): links and `..` resolved, on disk. Whether a write, a
  working directory or a read may use the path turns on this, so it is asked of the real
  file, and a link that leaves the root is outside however it is spelled. `contained` is a
  write's whole rule on it, said in words: the Applier's gate 1 and the box's edit tool.
- **How a model spelled a path** (`repo_path`): lexical, resolving nothing. `./a.py` and
  `a//b.py` are the repository's `a.py` and `a/b.py`; the one normaliser for every path a
  model hands the harness. `repo_relative` is its refusal, for a reader that takes a
  model's path as given: absolute, `\\`-separated, or naming `.`, `..` or `.git`.
- **How a path is spelled** (`name_under`): lexical, resolving nothing. The name a handle
  carries, and how the read rule spells a landing it has already resolved. The caller passes
  what it means.
- **Which file a path that does not resolve meant** (`suggest`): lexical, over
  a list of files the caller passes, so it does no I/O. Every tool that refuses a path names
  the file it most likely meant through it; `suggest_under` is its form for a path naming no
  file yet, a create's.
- **Whether a file is an operating system's own** (`os_junk`): a file a file browser
  writes beside the person's files on its own (`.DS_Store`, `Thumbs.db`, `desktop.ini`). Nobody
  wrote it, so it never makes a folder the person's, the harness never protects it, and the
  harness never commits it.
- **Which path globs a repository path matches** (`glob_match`): lexical too, the one matcher
  for every glob list a repository path is held against, reading a glob as `.gitignore` does:
  the profile's test globs (`TestSpec.matches`), the router's migration globs,
  the scout's `search` globs, an action's declared effects, the
  `always_ask` / `never_ask` preferences, each line of a `.gitignore` the graph reads, and the
  secret deny list. `segment_tokens` is the one reading of a glob's characters, which
  the macOS sandbox writes into its profile too.

Imports nothing from `chickadee`, so any module may ask it, `secrets/` included.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "OS_JUNK",
    "ROOT_RULE",
    "AnyOne",
    "AnyRun",
    "Char",
    "CharClass",
    "GlobSet",
    "Place",
    "Token",
    "contained",
    "glob_match",
    "name_under",
    "os_junk",
    "repo_path",
    "repo_relative",
    "segment_tokens",
    "spelled_out",
    "suggest",
    "suggest_under",
    "within",
]


def within(root: Path, path: Path | str) -> Path | None:
    """Where `path` lands, links and `..` resolved: absolute, or relative to `root`. The
    resolved path when it is `root` or under the resolved `root`; None when it lands anywhere
    else or cannot be resolved (`OSError`, `ValueError`). The root itself counts as inside;
    a caller that refuses it says so."""
    try:
        real_root = Path(os.path.realpath(root))
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(root) / candidate
        landed = Path(os.path.realpath(candidate))
    except (OSError, ValueError):
        return None
    return landed if landed.is_relative_to(real_root) else None


def contained(raw: str, root: Path) -> str | None:
    """Why a file at `raw` may not be written under `root`, or None (moved from the Applier's
    `containment_of`): an absolute path, a `..` or a `.git` part, a path landing
    outside `root` (a link out included) or on `root` itself, or one inside a nested
    repository below `root`, whose history is not ours to rewrite."""
    if os.path.isabs(raw):
        return f"absolute path: {raw}"
    if ".." in Path(raw).parts:
        return f"path escapes with '..': {raw}"
    if ".git" in Path(raw).parts:
        return f"path touches git internals: {raw}"
    real_root = Path(os.path.realpath(root))
    resolved = within(root, raw)
    if resolved is None or resolved == real_root:
        # The root itself is not a file a write can land on: `within` counts it inside.
        return f"resolves outside the worktree: {raw}"
    for parent in resolved.parents:
        if parent == real_root:
            break
        if (parent / ".git").exists():
            return f"{raw} lies inside nested repository {parent.relative_to(real_root)}"
    return None


def repo_path(text: str, root: Path | None = None) -> str:
    """A path as a model wrote it, spelled as the repository names it: `.` and empty segments
    dropped (`./calc.py`, `src/./app.py`, `src//app.py`), and an absolute path lexically
    inside `root` made relative to it. Text only: a plan names files that do not exist yet,
    so nothing is resolved, checked for existence or matched loosely.

    Everything else comes back exactly as written, for the check that owns it to refuse with
    its reason: a `..` segment (collapsing one can change the meaning through a link, or climb
    out; containment is `within`'s), an absolute path outside `root`, a `\\` separator, and
    a spelling that would normalise to nothing."""
    if "\\" in text:
        return text
    rel = text
    if text.startswith("/"):
        if root is None:
            return text
        rel = next(
            (
                text[len(base) :]
                for base in {root.as_posix(), Path(os.path.realpath(root)).as_posix()}
                if text == base or text.startswith(base.rstrip("/") + "/")
            ),
            "",
        )
        if not rel:
            return text
    segments = [seg for seg in rel.split("/") if seg not in ("", ".")]
    if not segments or ".." in segments:
        return text
    return "/".join(segments)


ROOT_RULE = "paths are from the repository root"
"""What a refusal of a path that does not resolve says beside the file it meant."""


@dataclass(frozen=True)
class Place:
    """Where a path a model sent lands in the repository: the repository path, and whether it
    is a directory. What `FileView.place` answers, and what `tools.foreign` writes a suggested
    call's path from."""

    path: str
    directory: bool = False


def suggest(sent: str, files: Iterable[str]) -> str | None:
    """The file of `files` that `sent` most likely meant, or None: the longest
    suffix of `sent` that is one of them (`/workspaces/<repo>/config.py` is `config.py`), else
    the one file with its name, when exactly one has it.

    `sent` is read loosely first, as a model writes a path: quotes around it, `file://`, `\\`
    separators, `./` and empty segments dropped. `files` are repository paths, what `glob` would
    list: the caller passes no ignored or deny-listed file, so a suggestion never says a secret
    exists. Deterministic and without I/O."""
    segments = _loose(sent)
    if not segments:
        return None
    known = set(files)
    for n in range(len(segments), 0, -1):
        tail = "/".join(segments[-n:])
        if tail in known:
            return tail
    named = [f for f in known if f.rsplit("/", 1)[-1] == segments[-1]]
    return named[0] if len(named) == 1 else None


def suggest_under(sent: str, files: Iterable[str]) -> str | None:
    """For a path naming no file yet (a create's, a move's destination): the directory of
    `files` that its parent most likely meant, by `suggest`'s rules, with its name under it; or
    None. `/workspaces/<repo>/core/x.py` is `core/x.py` when `core/` holds a file."""
    segments = _loose(sent)
    if len(segments) < 2:
        return None
    directories = {f.rsplit("/", 1)[0] for f in files if "/" in f}
    parent = suggest("/".join(segments[:-1]), directories)
    return None if parent is None else f"{parent}/{segments[-1]}"


def _loose(sent: str) -> list[str]:
    """`sent`'s segments, read as `suggest` reads a path."""
    text = sent.strip().strip("'\"`").strip()
    text = text.removeprefix("file://").replace("\\", "/")
    return [seg for seg in text.split("/") if seg not in ("", ".")]


_FORBIDDEN_SEGMENTS = frozenset({".", "..", ".git"})


def repo_relative(path: str) -> bool:
    """Whether a path a *model* supplied may be read as a repository path at all: relative,
    `/`-separated, and naming no `.`, `..` or `.git` segment. Empty segments are dropped, so
    `src/` and `a//b` stay valid spellings of a real path; a path that is only separators has
    nothing left and is refused. Moved from the packer's view, whose tool requests
    it guarded; the scout's pointers are its reader now."""
    if not path or path.startswith("/") or "\\" in path:
        return False
    parts = [p for p in path.split("/") if p]
    return bool(parts) and not any(p in _FORBIDDEN_SEGMENTS for p in parts)


OS_JUNK: Final = frozenset({".ds_store", "thumbs.db", "ehthumbs.db", "desktop.ini"})
"""The names, case-folded, of the files macOS Finder and Windows Explorer write into a folder
the person opens. Folded because both systems' default disks ignore case."""


def os_junk(path: str) -> bool:
    """Whether `path`'s last segment is an operating system's own file (`OS_JUNK`). Lexical:
    `path` is `/`-separated, as a repository path is."""
    return path.rsplit("/", 1)[-1].casefold() in OS_JUNK


def name_under(root: Path, path: Path) -> str | None:
    """`path` relative to `root`, POSIX, when it is under it lexically; None otherwise. The
    root itself is `"."`. A `..` segment below the root climbs out of it, so a path that
    holds one is not under it: a caller normalises first, and one that did not is told so
    rather than handed a name beginning `../`."""
    if not path.is_relative_to(root):
        return None
    rel = path.relative_to(root)
    return None if ".." in rel.parts else rel.as_posix()


def glob_match(path: str, globs: Iterable[str]) -> str | None:
    """The first of `globs` that `path` matches, or None: the one path-glob matcher, reading a
    glob as `.gitignore` does. `path` is a repository-relative
    POSIX path; `.` and empty segments are dropped first, so a spelling cannot change the
    answer. Case-sensitive, as git and ripgrep are.

    1. `*`, `?` and a `[…]` class never match `/`: `src/*.py` is `src/a.py`, not
       `src/a/b.py`.
    2. `**` as a whole segment is any number of directories: `**/x` is `x` at any depth, the
       root included; `a/**/b` is `a/b`, `a/x/b`, `a/x/y/b`; `a/**` is everything inside
       `a`, not `a` itself.
    3. A glob with no `/` matches the base name, at any depth: `test_*.py` names a file
       wherever it is, never a directory.
    4. A glob with a `/` at its start or middle is anchored at the repository root; a
       leading `/` only says so.
    5. A trailing `/` names a directory, and the glob matches everything inside one it
       matches: `tests/` is `**/tests/**`, `src/a/` is `src/a/**`.
    6. `**` that is not a whole segment is `*`: `**.py` is `*.py`.
    7. `\\` makes the next character literal; `[!…]` and `[^…]` negate; a `[` with no `]`
       is literal. `!` at the start and `{a,b}` are literal too: a list is a union, with no
       exclusion and no alternation.
    """
    parts = [seg for seg in path.split("/") if seg and seg != "."]
    if not parts:
        return None
    whole = "/".join(parts)
    for glob in globs:
        compiled = _compiled(glob)
        if compiled is not None and compiled.fullmatch(whole + "/"):
            return glob
    return None


class GlobSet:
    """Many globs as one expression: whether a path matches ANY of them, read as
    `glob_match` reads each, in one regex match instead of one per glob. With
    `any_tail`, whether any of them matches some tail of the path: the path from one of its
    segments to its end, as a name judged wherever it sits is (`secrets.denylist`).

    For a caller asking the same question of many paths. `glob_match` still says *which*
    glob matched; this only says whether one did."""

    def __init__(self, globs: Iterable[str], *, any_tail: bool = False) -> None:
        bodies = [c.pattern for c in map(_compiled, globs) if c is not None]
        if not bodies:
            self._union: re.Pattern[str] | None = None
            return
        union = "|".join(f"(?:{b})" for b in bodies)
        self._union = re.compile(f"(?:[^/]+/)*(?:{union})" if any_tail else f"(?:{union})")

    def matches(self, path: str) -> bool:
        if self._union is None:
            return False
        parts = [seg for seg in path.split("/") if seg and seg != "."]
        return bool(parts) and self._union.fullmatch("/".join(parts) + "/") is not None


def spelled_out(glob: str) -> str:
    """`glob` as a tool that reads a list as `.gitignore` does (ripgrep's `--glob`) must be
    handed it to list the files `glob_match` matches. There a trailing `/` is directories
    only and a leading `!` an exclusion, so rule 5 is written out (`tests/` is
    `**/tests/**`, `src/a/` is `src/a/**`) and a leading `!` escaped (rule 7)."""
    if glob.startswith("!"):
        glob = "\\" + glob
    body = glob.removesuffix("/")
    if body == glob or not body.strip("/"):
        return glob
    return f"{body}/**" if "/" in body else f"**/{body}/**"


_STAR = None
"""A wildcard in a run: `**` among segments, `*` within one."""


@functools.lru_cache(maxsize=1024)
def _compiled(glob: str) -> re.Pattern[str] | None:
    """`glob` as one expression over the normalised path with a `/` after it, so every
    segment, the last included, ends in one; or None for a glob that names nothing (empty,
    or only `/`). Bounded: the scout's `search` globs are a model's."""
    glob = spelled_out(glob)
    segments = [seg for seg in glob.split("/") if seg and seg != "."]
    if not segments:
        return None
    if "/" not in glob:
        segments.insert(0, "**")  # rule 3: at any depth
    items: list[str | None] = []
    for i, seg in enumerate(segments):
        if seg != "**":
            items.append(_segment(seg) + "/")
        elif i == len(segments) - 1:
            items += ["[^/]+/", _STAR]  # `a/**` is inside `a`: at least one more segment
        elif items[-1:] != [_STAR]:
            items.append(_STAR)  # `**/**` is `**`
    return re.compile(_joined(items, "(?:[^/]+/)*"))


def _segment(seg: str) -> str:
    """One segment of a glob, which holds no `/`, as an expression that matches none."""
    items: list[str | None] = []
    for token in segment_tokens(seg):
        if isinstance(token, AnyRun):
            items.append(_STAR)
        elif isinstance(token, AnyOne):
            items.append("[^/]")
        elif isinstance(token, CharClass):
            items.append(_class_regex(token))
        else:
            items.append(re.escape(token.char))
    return _joined(items, "[^/]*")


def _class_regex(token: CharClass) -> str:
    if not token.spans:
        return "[^/]" if token.negated else "(?!)"  # a class with no member matches nothing
    body = "".join(
        re.escape(lo) if lo == hi else f"{re.escape(lo)}-{re.escape(hi)}"
        for lo, hi in token.spans
    )
    return f"[^/{body}]" if token.negated else f"[{body}]"


def _joined(items: Sequence[str | None], star: str) -> str:
    """A run of fixed pieces and wildcards as one expression that cannot backtrack without
    bound, as `fnmatch.translate` writes one. Each fixed run between two wildcards is taken
    atomically at its first place: a later place would leave the next wildcard less to
    absorb, never more, so the first is never wrong. Each fixed piece matches one segment
    (or one character), so a run's length is fixed. Without this, `**/**/…/x` or `a*a*…*b`
    against a long path tries every split, and one model's glob stalls a search."""
    out: list[str] = []
    i = 0
    while i < len(items) and items[i] is not _STAR:
        out.append(items[i] or "")
        i += 1
    while i < len(items):
        i += 1  # the wildcard
        fixed: list[str] = []
        while i < len(items) and items[i] is not _STAR:
            fixed.append(items[i] or "")
            i += 1
        run = "".join(fixed)
        # The last run ends the path, so it is not atomic: it must take the end.
        out.append(star + run if i == len(items) else f"(?>{star}?{run})")
    return "".join(out)


# -- reading one segment ------------------------------------------------------------------


@dataclass(frozen=True)
class AnyRun:
    """`*`: any characters within one segment, none included. `**` in a segment is one."""


@dataclass(frozen=True)
class AnyOne:
    """`?`: one character, never `/`."""


@dataclass(frozen=True)
class CharClass:
    """`[…]`: one character among `spans`, each an inclusive `(low, high)`, or, `negated`,
    one not among them. `/` is in no span, and a negated class never matches it either."""

    spans: tuple[tuple[str, str], ...]
    negated: bool


@dataclass(frozen=True)
class Char:
    """A character that means itself: anything else, or one a `\\` escapes."""

    char: str


Token = AnyRun | AnyOne | CharClass | Char


def segment_tokens(segment: str) -> tuple[Token, ...]:
    """One segment of a glob (no `/`), read: the one reading of a glob's characters.
    `glob_match` writes it as a Python expression here, and the macOS
    sandbox as its profile's (`isolation/macos.py`), so the two cannot read a glob apart."""
    out: list[Token] = []
    i = 0
    while i < len(segment):
        c = segment[i]
        if c == "\\" and i + 1 < len(segment):
            out.append(Char(segment[i + 1]))
            i += 2
            continue
        if c == "*":
            if not (out and isinstance(out[-1], AnyRun)):
                out.append(AnyRun())  # rule 6: `**` in a segment is `*`
        elif c == "?":
            out.append(AnyOne())
        elif c == "[" and (end := _class_end(segment, i)) is not None:
            negated = segment[i + 1] in "!^"
            out.append(CharClass(_spans(segment[i + 1 + negated : end]), negated))
            i = end + 1
            continue
        else:
            out.append(Char(c))
        i += 1
    return tuple(out)


def _class_end(seg: str, start: int) -> int | None:
    """The index of the `]` closing the class opened at `start`, or None, when the `[` is a
    literal. A `]` straight after the `[` (or the `[!`, `[^`) is a member, not the end, and so
    is one a `\\` escapes (rule 7, as git reads it)."""
    i = start + 1
    if i < len(seg) and seg[i] in "!^":
        i += 1
    if i < len(seg) and seg[i] == "]":
        i += 1
    while i < len(seg):
        if seg[i] == "\\":
            i += 2
        elif seg[i] == "]":
            return i
        else:
            i += 1
    return None


def _spans(body: str) -> tuple[tuple[str, str], ...]:
    """A class's members as `fnmatch` and git read them, each an inclusive `(low, high)`,
    with `/` taken out (rule 1).

    `x-y` is a range; a `-` first or last, or one a `\\` escapes, is itself; and the end of
    one range cannot start the next: `a-c-e` is `a` to `c`, then `-` and `e`. A descending
    range matches nothing and is dropped, as `fnmatch` drops it; written into an expression it
    would not compile (`[z-a]`)."""
    chars: list[tuple[str, bool]] = []  # each character, and whether a `\\` escaped it
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            chars.append((body[i + 1], True))
            i += 2
        else:
            chars.append((body[i], False))
            i += 1
    members: list[tuple[str, str]] = []
    i = 0
    while i < len(chars):
        low = chars[i][0]
        if i + 2 < len(chars) and chars[i + 1] == ("-", False):
            members.append((low, chars[i + 2][0]))
            i += 3
        else:
            members.append((low, low))
            i += 1
    below, above = chr(ord("/") - 1), chr(ord("/") + 1)
    out: list[tuple[str, str]] = []
    for low, high in members:
        pieces = [(low, below), (above, high)] if low <= "/" <= high else [(low, high)]
        out.extend((lo, hi) for lo, hi in pieces if lo <= hi)
    return tuple(out)
