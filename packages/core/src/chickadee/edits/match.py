"""The exact text matcher: the one place a text anchor is
matched against a file.

Exact text, in order, exactly one match per hunk. No whitespace or indentation
tolerance and no fuzzy match: a stale anchor fails loudly instead of editing the wrong line.
The file's newlines follow `lines.decode`: a uniformly `\\r\\n` file is matched as `\\n`, the
form the worker is shown, and written back as `\\r\\n`.

A hunk marked `lines` (every hunk a reply format writes) matches as whole lines: a run of
the file's lines equal to its old lines, never part of a line, the file's last line with or
without its final newline. Its replacement takes the run's line ending, and keeps the file's
final-newline state. A hunk without it is literal text, matched anywhere (Zeus's exact hunks).

`find` is the matcher; `resolve` applies hunks with it, for the Applier; `line_run` is the same
replacement for a run of lines given by number, a line tool's. `find` answers spans of
the text it was given and what goes in each, so the worker's draft (`edits/draft.py`)
writes them through its one write path; `replace_all` and V4A's `@@` scopes are its modes, the
draft's alone: no `Hunk` field carries either.

A failure carries the reason and a detail naming the hunk but not the path, so the Applier
and the worker's draft say the same thing and each adds the path it knows. A `NO_MATCH` also
names the lines nearest the anchor, so the next attempt, or the next call, re-anchors
from where the text probably is rather than guessing again.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Final

from chickadee.edits.hunk import Hunk, MatchReason
from chickadee.edits.lines import as_decoded, clip, decode, encode, row, split_lines

__all__ = [
    "NEAREST_LINES",
    "NEAREST_RATIO",
    "MatchFailure",
    "Region",
    "Replacement",
    "find",
    "line_run",
    "nearest_lines",
    "resolve",
    "splice",
]

NEAREST_LINES: Final = 3
"""How many lines a `NO_MATCH` names at most. A starting guess until E1 measures it (the owner's
item 2)."""
NEAREST_RATIO: Final = 0.6
"""How alike a line must be to the anchor's key line to be named, as `difflib` scores it on
both stripped: below it a line is noise that sends the retry somewhere wrong. A guess too."""


EMPTY_OLD_DO: Final = (
    "to add text, set old to the line it goes after and new to that line followed by the "
    "addition; to add at the end, old is the file's last line"
)
"""What an empty `old` is told: how adding text is written, what it was trying to do."""


ESCAPED_DO: Final = (
    "send old with each backslash as the file has it, so `\\n` is a backslash and an n, not a "
    'line break (in a Python block, a raw string r"""...""" keeps them)'
)
"""What an `old` that matches only with its escapes written back is told."""


@dataclass(frozen=True)
class MatchFailure:
    reason: MatchReason
    """`NO_MATCH` or `AMBIGUOUS`."""
    detail: str
    """`hunk i: …`, without the path."""
    do: str = field(compare=False)
    """What to do next, set where the cause is found: an empty `old`, one
    that occurs twice and a scope line that does are all `AMBIGUOUS`, and each is fixed
    differently. Not compared: the failure is its reason and detail."""


@dataclass(frozen=True)
class Region:
    """A span of a text, `[start, end)` in characters."""

    start: int
    end: int


@dataclass(frozen=True)
class Replacement:
    """What one match puts where: `region` of the text it was found in, and `content`, the
    replacement as the text holds it (a lines hunk's endings already applied)."""

    region: Region
    content: str


def splice(text: str, region: Region, content: str) -> str:
    """`text` with `region` replaced by `content`: the one way a matched region is written."""
    return text[: region.start] + content + text[region.end :]


def resolve(text: str, hunks: Sequence[Hunk]) -> str | MatchFailure:
    """`text` after `hunks` apply in order, or the first that cannot.

    Each hunk's `old` must occur exactly once in the result of the hunks before it, so a later
    hunk may anchor on text an earlier one wrote. Occurrences are counted the way `str.count`
    counts them — non-overlapping, scanning left to right — because that is what §3.1
    specifies."""
    decoded = decode(text)
    current = decoded.text
    for i, hunk in enumerate(hunks):
        found = find(current, hunk.old, hunk.new, lines=hunk.lines, crlf=decoded.crlf)
        if isinstance(found, MatchFailure):
            return MatchFailure(found.reason, f"hunk {i}: {found.detail}", found.do)
        for one in reversed(found):
            current = splice(current, one.region, one.content)
    return encode(current, decoded)


def find(
    text: str,
    old: str,
    new: str,
    *,
    lines: bool,
    replace_all: bool = False,
    scopes: Sequence[str] = (),
    crlf: bool = False,
) -> tuple[Replacement, ...] | MatchFailure:
    """Where `old` occurs in `text`, and what replaces each: one occurrence, or every one under
    `replace_all`, left to right and non-overlapping. `text` is as it is matched (decoded);
    `crlf` says it was decoded from a uniformly `\r\n` file, so `old` and `new` are read as
    `lines.as_decoded` reads them.

    `lines` matches whole lines; otherwise `old` is literal text. Each of `scopes` (V4A's
    `@@` lines) names one line, matched stripped and exactly once, and narrows the
    search to that line's block (`_block_end`). Each scope narrows within the one before it.
    The occurrences are counted before any is replaced, so two are `AMBIGUOUS` however their
    replacements would join."""
    old, new = as_decoded(old, crlf), as_decoded(new, crlf)
    if old == "":
        return MatchFailure(MatchReason.AMBIGUOUS, "empty old matches everywhere", EMPTY_OLD_DO)
    window = _scoped(text, scopes)
    if isinstance(window, MatchFailure):
        return window
    spans = _text_spans(text, old, window) if not lines else []
    runs = _line_runs(text, old, window) if lines else []
    found = len(runs) if lines else len(spans)
    if not found:
        escaped = _escaped_once(text, old)
        if escaped is not None:
            return MatchFailure(MatchReason.NO_MATCH, f"old not found; {escaped}", ESCAPED_DO)
        return MatchFailure(
            MatchReason.NO_MATCH,
            f"old not found{nearest_lines(text, old)}",
            "copy old exactly from the lines shown above (whitespace included), or read the "
            "file again, then send the edit once more",
        )
    if found > 1 and not replace_all:
        return MatchFailure(
            MatchReason.AMBIGUOUS,
            f"old occurs {found} times",
            "add the lines around old until it matches one place only, then send it again",
        )
    if not lines:
        return tuple(Replacement(span, new) for span in spans)
    return _line_replacements(text, runs, len(split_lines(old)), new)


@dataclass(frozen=True)
class _Window:
    """The lines a search may match in, `[first, end)`, and their character span."""

    first: int
    end: int
    span: Region


def _offsets(rows: Sequence[str]) -> list[int]:
    """Where each line starts, and the text's end last."""
    out = [0]
    for line in rows:
        out.append(out[-1] + len(line))
    return out


def _scoped(text: str, scopes: Sequence[str]) -> _Window | MatchFailure:
    rows = split_lines(text, keepends=True)
    at = _offsets(rows)
    first, end = 0, len(rows)
    for scope in scopes:
        want = scope.strip()
        hits = [j for j in range(first, end) if rows[j].strip() == want]
        if not hits:
            return MatchFailure(
                MatchReason.NO_MATCH,
                f"scope {scope.strip()!r} not found",
                "copy the scope line exactly as the file has it, or leave the scope out",
            )
        if len(hits) > 1:
            return MatchFailure(
                MatchReason.AMBIGUOUS,
                f"scope {scope.strip()!r} occurs {len(hits)} times",
                "name a scope line that occurs once, or put an outer scope before it",
            )
        first, end = hits[0], _block_end(rows, hits[0], end)
    return _Window(first, end, Region(at[first], at[end]))


def _block_end(rows: Sequence[str], head: int, limit: int) -> int:
    """The line after `head`'s block: the lines after it that are blank or indented deeper,
    through a multi-line header's closing line at its own indent (`) -> int:`), and then one
    line of closing brackets at its own indent (a brace language's `}`)."""
    indent = _indent(rows[head])
    k = head + 1
    while k < limit:
        bare = rows[k].strip()
        if not bare or _indent(rows[k]) > indent:
            k += 1
            continue
        if _indent(rows[k]) == indent and bare[0] in _CLOSERS and bare.endswith(_OPENERS):
            # A header spanning lines closes at its own indent and opens the body:
            # `) -> int:`, `) {`. The block goes on through it.
            k += 1
            continue
        if _indent(rows[k]) == indent and bare.strip(_CLOSERS) == "":
            k += 1
        break
    return k


_CLOSERS: Final = "})];,"
"""What a line that only closes its block holds."""
_OPENERS: Final = (":", "{")
"""How a line that opens a block ends."""


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _text_spans(text: str, old: str, window: _Window) -> list[Region]:
    """`old`'s occurrences as text inside `window`, non-overlapping left to right, as
    `str.count` counts them."""
    found: list[Region] = []
    pos, stop = window.span.start, window.span.end
    while (at := text.find(old, pos, stop)) != -1:
        found.append(Region(at, at + len(old)))
        pos = at + len(old)
    return found


def _line_runs(text: str, old: str, window: _Window) -> list[int]:
    """The first line of each run of whole lines inside `window` equal to `old`'s lines,
    non-overlapping left to right."""
    bare = [_bare(ln) for ln in split_lines(text, keepends=True)]
    want = list(split_lines(old))
    runs: list[int] = []
    j = window.first
    while j + len(want) <= window.end:
        if bare[j : j + len(want)] == want:
            runs.append(j)
            j += len(want)
        else:
            j += 1
    return runs


def _line_replacements(
    text: str, runs: Sequence[int], size: int, new: str
) -> tuple[Replacement, ...]:
    """Each run of `size` lines replaced by `new`'s lines, each taking its run's line ending;
    the file's final-newline state is kept."""
    rows = split_lines(text, keepends=True)
    at = _offsets(rows)
    found: list[tuple[Replacement, str]] = []
    for start in runs:
        end = start + size
        eol = _ending(rows[start]) or "\n"
        put = "".join(f"{ln}{eol}" for ln in split_lines(new))
        region = Region(at[start], at[end])
        if _ending(rows[end - 1]) == "":
            # The run held the file's last line, which had no newline: neither has the result.
            if put:
                put = put.removesuffix(eol)
            else:
                # Nothing replaces it, so the line written last before it becomes the last and
                # loses its newline: a run replaced just before it (`replace_all`), or the
                # file's own line before the run.
                while not put and found and found[-1][0].region.end == region.start:
                    before, before_eol = found.pop()
                    region = Region(before.region.start, region.end)
                    put = before.content.removesuffix(before_eol)
                line = at.index(region.start)
                if not put and line > 0:
                    region = Region(region.start - len(_ending(rows[line - 1])), region.end)
        found.append((Replacement(region, put), eol))
    return tuple(r for r, _ in found)


def line_run(text: str, first: int, last: int, new: str) -> Replacement:
    """Lines `first` to `last` of `text` (1-based, inclusive) replaced by `new`'s lines, by the
    rule a lines hunk is replaced by: each takes the run's line ending, and the file's
    final-newline state is kept; an empty `new` deletes them. `last = first - 1` is an empty
    run, before line `first`: `new`'s lines are inserted there, after line `last`."""
    rows = split_lines(text, keepends=True)
    if not (1 <= first <= last + 1 and last <= len(rows)):
        raise ValueError(f"lines {first}-{last} of a text of {len(rows)} lines")
    if last >= first:
        (one,) = _line_replacements(text, (first - 1,), last - first + 1, new)
        return one
    at = _offsets(rows)
    # An insertion takes the ending of the line it follows, else of the line it precedes.
    near = [rows[i] for i in (first - 2, first - 1) if 0 <= i < len(rows)]
    eol = next((e for e in map(_ending, near) if e), "\n")
    put = "".join(f"{ln}{eol}" for ln in split_lines(new))
    if put and first > len(rows) and rows and _ending(rows[-1]) == "":
        # After a last line with no newline: it gains one, and the inserted last line has none.
        put = eol + put.removesuffix(eol)
    return Replacement(Region(at[first - 1], at[first - 1]), put)


def _ending(line: str) -> str:
    return "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""


def _bare(line: str) -> str:
    """A line without its ending."""
    return line.removesuffix(_ending(line))


def _escaped_once(text: str, old: str) -> str | None:
    """Why `old` did not match, when it is the escapes: `old` occurs exactly
    once in `text` with some of its newlines and tabs written back as `\\n` and `\\t`, each
    one either way. A model writing `print("a\\n")` in a Python string sends a newline where
    the file has a backslash and an n, often on one line of several. A hint only: the match
    stays exact."""
    if ("\n" not in old and "\t" not in old) or old in text:
        # As written, it is in the text: the miss is not the escapes (a `lines` hunk inside a
        # line, a scope), so the hint would mislead.
        return None
    either = {"\n": r"(?:\n|\\n)", "\t": r"(?:\t|\\t)"}
    pattern = "".join(either.get(ch, re.escape(ch)) for ch in old)
    found = re.findall(pattern, text)
    if len(found) != 1:
        return None
    return (
        "it occurs once with some of its newlines or tabs written as backslash escapes "
        "(\\n, \\t)"
    )


def nearest_lines(text: str, anchor: str) -> str:
    """The lines of `text` most like `anchor`'s key line, best first, at most `NEAREST_LINES`,
    as numbered rows after `; nearest lines:`; empty when none is alike enough.

    The key is the anchor's most distinctive line, the one with the most letters and digits
    (the first of equals): its first line is often `}` or `else:`, which is alike everywhere and
    names nothing. Compared stripped, so a wrong indentation, the commonest miss, still finds
    its line; shown as the text holds it, clipped like any row, so the next anchor can be copied
    from it. The numbers are `text`'s: the file's, or the file after the hunks before this one.
    """
    key = _key_line(anchor)
    if not key:
        return ""
    lines = split_lines(text)
    # The key is analysed once: `difflib` caches its second sequence.
    matcher = SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(key)
    scored: list[tuple[float, int]] = []
    for n, line in enumerate(lines, start=1):
        matcher.set_seq1(line.strip())
        if matcher.real_quick_ratio() < NEAREST_RATIO or matcher.quick_ratio() < NEAREST_RATIO:
            continue
        score = matcher.ratio()
        if score >= NEAREST_RATIO:
            scored.append((score, n))
    best = sorted(scored, key=lambda s: (-s[0], s[1]))[:NEAREST_LINES]
    if not best:
        return ""
    width = len(str(max(n for _, n in best)))
    return "; nearest lines:\n" + "\n".join(row(n, width, clip(lines[n - 1])) for _, n in best)


def _key_line(anchor: str) -> str:
    """The anchor's line with the most letters and digits, stripped; the first of equals."""
    best, most = "", 0
    for line in split_lines(anchor):
        count = sum(ch.isalnum() for ch in line)
        if count > most:
            best, most = line.strip(), count
    return best
