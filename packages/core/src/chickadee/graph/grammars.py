"""The tree-sitter grammars the harness parses with, by the name a profile gives them.

One home for a grammar: its loaded `Language`, the tags query its wheel ships, and the first
place a text fails to parse. Which file a grammar parses is the profiles' answer, never this
module's: there is no extension here. The graph's walkers (`graph/parse`), its
tags-query extractor (`graph/tags`) and the `parse` syntax check (`box/syntax`) all load a
grammar here, so a grammar the harness ships is named once.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import tree_sitter_c
import tree_sitter_c_sharp
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_kotlin
import tree_sitter_php
import tree_sitter_python
import tree_sitter_ruby
import tree_sitter_rust
import tree_sitter_scala
import tree_sitter_swift
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

__all__ = [
    "GRAMMARS",
    "JAVASCRIPT",
    "PYTHON",
    "TSX",
    "TYPESCRIPT",
    "ParseFault",
    "error_node",
    "first_error",
    "language",
    "wheel_tags_query",
]

PYTHON = "python"
JAVASCRIPT = "javascript"
TYPESCRIPT = "typescript"
TSX = "tsx"


@dataclass(frozen=True)
class _Grammar:
    load: Callable[[], object]
    """The wheel's language function: the grammar's compiled tables, as a capsule."""
    queries: Path
    """The wheel's `queries/` folder, where it ships a `tags.scm` if it ships one."""


def _wheel(module: object) -> Path:
    return Path(str(getattr(module, "__file__", ""))).parent / "queries"


_GRAMMARS: Final[dict[str, _Grammar]] = {
    PYTHON: _Grammar(tree_sitter_python.language, _wheel(tree_sitter_python)),
    JAVASCRIPT: _Grammar(tree_sitter_javascript.language, _wheel(tree_sitter_javascript)),
    TYPESCRIPT: _Grammar(
        tree_sitter_typescript.language_typescript, _wheel(tree_sitter_typescript)
    ),
    TSX: _Grammar(tree_sitter_typescript.language_tsx, _wheel(tree_sitter_typescript)),
    "go": _Grammar(tree_sitter_go.language, _wheel(tree_sitter_go)),
    "rust": _Grammar(tree_sitter_rust.language, _wheel(tree_sitter_rust)),
    "java": _Grammar(tree_sitter_java.language, _wheel(tree_sitter_java)),
    "c": _Grammar(tree_sitter_c.language, _wheel(tree_sitter_c)),
    "cpp": _Grammar(tree_sitter_cpp.language, _wheel(tree_sitter_cpp)),
    "c_sharp": _Grammar(tree_sitter_c_sharp.language, _wheel(tree_sitter_c_sharp)),
    "ruby": _Grammar(tree_sitter_ruby.language, _wheel(tree_sitter_ruby)),
    # `language_php` reads a file as PHP does: HTML until `<?php`. `language_php_only` would
    # call a template's markup an error.
    "php": _Grammar(tree_sitter_php.language_php, _wheel(tree_sitter_php)),
    "kotlin": _Grammar(tree_sitter_kotlin.language, _wheel(tree_sitter_kotlin)),
    "swift": _Grammar(tree_sitter_swift.language, _wheel(tree_sitter_swift)),
    "scala": _Grammar(tree_sitter_scala.language, _wheel(tree_sitter_scala)),
}

GRAMMARS: Final = frozenset(_GRAMMARS)
"""Every grammar the harness can parse with: a profile naming another is `init`'s missing
grammar (`init/toolchain.missing_grammars`)."""


@functools.cache
def language(grammar: str) -> Language | None:
    """`grammar` loaded, once per process, or None when the harness ships no such grammar."""
    found = _GRAMMARS.get(grammar)
    return Language(found.load()) if found is not None else None


def wheel_tags_query(grammar: str) -> Path | None:
    """The tags query `grammar`'s wheel ships (`queries/tags.scm`), or None: Kotlin's and C#'s
    ship none, and a repository-held query supplies it (`graph/tags`)."""
    found = _GRAMMARS.get(grammar)
    if found is None:
        return None
    path = found.queries / "tags.scm"
    return path if path.is_file() else None


@dataclass(frozen=True)
class ParseFault:
    """Where a text first fails to parse: a 1-based line and column (in characters), and what
    tree-sitter found there."""

    line: int
    column: int
    what: str

    def said(self, path: str) -> str:
        """In words, not as `path:N:`: that shape is a search hit's, read back as an anchor."""
        return f"{path}: line {self.line}, column {self.column}: {self.what}"


_SHOWN_MAX = 30
"""Characters of the unexpected text a parse error quotes."""


def first_error(grammar: str, source: bytes) -> ParseFault | None:
    """The first `ERROR` or `MISSING` node of `source` parsed by `grammar`, in the text's
    order, or None when it parses cleanly. A parse, never a compile: it says nothing of names,
    types or imports. Raises `KeyError` for a grammar the harness does not ship; the caller
    asked the profile set, which names only shipped ones once `init` has checked."""
    loaded = language(grammar)
    if loaded is None:
        raise KeyError(grammar)
    node = error_node(Parser(loaded).parse(source).root_node)
    if node is None:
        return None
    row = node.start_point[0]
    # The line's start without splitting the whole file: this runs after every edit call.
    start = source.rfind(b"\n", 0, node.start_byte) + 1
    column = len(source[start : node.start_byte].decode("utf-8", "replace")) + 1
    if node.is_missing:
        return ParseFault(row + 1, column, f"missing {node.type!r}")
    text = source[node.start_byte : node.end_byte].decode("utf-8", "replace").strip()
    shown = text.splitlines()[0][:_SHOWN_MAX] if text else ""
    return ParseFault(row + 1, column, f"unexpected {shown!r}" if shown else "syntax error")


def error_node(root: Node) -> Node | None:
    """The first `ERROR` or `MISSING` node under `root` that the text shows, or None. A tree
    can say `has_error` for nothing but a zero-width token the grammar inserts and hides
    (Kotlin's `_class_member_semi` after `fun m() {}` on one line, which is valid Kotlin): that
    is the grammar's own bookkeeping, not an error in the text, and is not one here."""
    return _first_bad(root) if root.has_error else None


def _first_bad(node: Node) -> Node | None:
    """The first node, in document order, that is `ERROR` or `MISSING`; a subtree with no
    error is never walked."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_error or current.is_missing:
            return current
        stack.extend(c for c in reversed(current.children) if c.has_error or c.is_missing)
    return None
