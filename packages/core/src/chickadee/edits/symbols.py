"""A symbol's span in a text: what a symbol edit replaces and what `symbol`
answers for a target, read from the text the worker's draft holds, never the graph index of the
file on disk.

It composes the parser and the name rule, and adds neither: `graph/parse.parse_source` gives
each declaration and its `extent` (its whole lines, decorators and an `export` included), and
`graph/names.names_qualified` says whether a written name names it. A span is whole lines, so
a declaration's own line holds whatever its first line holds (`export const f = …`). A
declaration sharing its first or last line with other code (`}; function f() {`) is refused for
an edit by name: replacing whole lines would take that code with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from chickadee.edits.lines import split_lines
from chickadee.graph.grammars import PYTHON
from chickadee.graph.names import names_qualified
from chickadee.graph.parse import Symbol, parse_source
from chickadee.profiles.resolve import FileLanguage

__all__ = [
    "NAMED_MAX",
    "SymbolMiss",
    "SymbolSpan",
    "find_symbol",
    "symbols_named",
]

NAMED_MAX: Final = 20
"""How many of a file's symbols a `NO_MATCH` names, or matches an `AMBIGUOUS` does, before the
rest are counted: a generated file can hold thousands."""


@dataclass(frozen=True)
class SymbolSpan:
    """One declaration in a text: its qualified name, its signature, and its lines, first and
    last, decorators included."""

    qualified_name: str
    sig: str | None
    first: int
    last: int
    head: int
    """The byte column the declaration starts at on line `first`."""
    tail: int
    """The byte column it ends at on line `last`."""


_BY_TEXT: Final = "edit it by its text instead, giving old and new"
"""What a symbol edit that cannot resolve to one whole-line span does instead."""


@dataclass(frozen=True)
class SymbolMiss:
    """Why a name has no one span: no parser for the file's language (`UNSUPPORTED`), no
    declaration it names (`NO_MATCH`, naming the file's), or several (`AMBIGUOUS`, naming each).
    `detail` names the symbol, not the path."""

    reason: Literal["UNSUPPORTED", "NO_MATCH", "AMBIGUOUS"]
    detail: str
    do: str = field(compare=False)
    """What to do next, set where the cause is found. Not compared."""


def symbols_named(
    path: str, text: str, name: str, language: FileLanguage | None
) -> tuple[SymbolSpan, ...] | SymbolMiss:
    """Every declaration in `text` that `name` names, in the file's order; `UNSUPPORTED` when
    `path` has no `language` (the profile set's answer for it). A name equal to a declaration's
    whole qualified name names that one alone."""
    parsed = _parse(path, text, language)
    return parsed if isinstance(parsed, SymbolMiss) else _named(parsed[1], name)


def find_symbol(
    path: str, text: str, name: str, language: FileLanguage | None
) -> SymbolSpan | SymbolMiss:
    """The one declaration `name` names in `text`, read as `language`, for an edit to replace
    or follow, or why there is not one: a declaration whose first or last line also holds
    other code is `UNSUPPORTED` by name, since its whole lines are more than it."""
    found = _parse(path, text, language)
    if isinstance(found, SymbolMiss):
        return found
    grammar, parsed = found
    named = _named(parsed, name)
    if len(named) == 1:
        shared = _shared(split_lines(text), named[0], grammar)
        if shared is not None:
            return SymbolMiss(
                "UNSUPPORTED",
                f"{named[0].qualified_name} shares {shared}, and a symbol edit replaces whole "
                "lines",
                _BY_TEXT,
            )
        return named[0]
    if not named:
        listed = _listed(sorted({s.qualified_name for s in parsed})) or "none"
        return SymbolMiss(
            "NO_MATCH",
            f"no symbol {name!r}; the file declares: {listed}",
            "send it again with one of the names the file declares",
        )
    each = _listed([f"{s.qualified_name} (line {s.first})" for s in named])
    if len({s.qualified_name for s in named}) == 1:
        # A property and its setter, or overloads: no name picks one, so asking for the
        # qualified name would only be refused again.
        return SymbolMiss(
            "AMBIGUOUS",
            f"{name!r} is declared {len(named)} times: {each}; no name picks one",
            _BY_TEXT,
        )
    return SymbolMiss(
        "AMBIGUOUS",
        f"{name!r} names {len(named)} symbols: {each}",
        "send it again with the qualified name of the one you mean",
    )


def _parse(
    path: str, text: str, language: FileLanguage | None
) -> tuple[str, tuple[Symbol, ...]] | SymbolMiss:
    """`text`'s declarations, parsed as `language`, with the grammar that read them. A text
    that does not parse cleanly keeps what tree-sitter recovers, as the index does: a
    draft mid-edit is the normal case."""
    if language is None:
        return SymbolMiss("UNSUPPORTED", f"no symbol parser reads {path}'s language", _BY_TEXT)
    source = text.encode("utf-8")
    return language.grammar, parse_source(path, source, language.grammar, language.name).symbols


def _named(parsed: tuple[Symbol, ...], name: str) -> tuple[SymbolSpan, ...]:
    exact = [s for s in parsed if s.qualified_name == name]
    named = exact or [s for s in parsed if names_qualified(s.qualified_name, name)]
    ordered = sorted(named, key=lambda s: (s.extent.first, s.extent.last))
    return tuple(_span(s) for s in ordered)


def _span(symbol: Symbol) -> SymbolSpan:
    e = symbol.extent
    return SymbolSpan(symbol.qualified_name, symbol.sig, e.first, e.last, e.head, e.tail)


def _shared(lines: tuple[str, ...], span: SymbolSpan, grammar: str) -> str | None:
    """What else the declaration's first or last line holds, or None: code before it on its
    first line, or after it on its last, a trailing comment aside (it goes with the line)."""
    before = lines[span.first - 1].encode("utf-8")[: span.head].decode("utf-8", "replace")
    if before.strip():
        return f"line {span.first} with {before.strip()!r} before it"
    after = lines[span.last - 1].encode("utf-8")[span.tail :].decode("utf-8", "replace")
    if _code(after, grammar):
        return f"line {span.last} with {after.strip()!r} after it"
    return None


_HASH_COMMENTS: Final = frozenset({PYTHON, "ruby"})
"""The grammars whose comments are `#` alone; PHP takes `#` beside the C family's `//` and
`/* */`, which every other grammar takes."""


def _code(rest: str, grammar: str) -> bool:
    """Whether what follows a declaration on its last line is code: not its own `,` or `;`
    (an object's method, a statement's end), and not a comment, a block comment then
    nothing else included."""
    rest = rest.strip().lstrip(",;").strip()
    if grammar in _HASH_COMMENTS:
        return bool(rest) and not rest.startswith("#")
    while rest.startswith("/*"):
        end = rest.find("*/")
        if end == -1:
            return False  # a comment running on: nothing after it on this line
        rest = rest[end + 2 :].strip()
    hash_too = grammar == "php" and rest.startswith("#") and not rest.startswith("#[")
    return bool(rest) and not rest.startswith("//") and not hash_too


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:NAMED_MAX])
    more = len(items) - NAMED_MAX
    return f"{shown} and {more} more" if more > 0 else shown
