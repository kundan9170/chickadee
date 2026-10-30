"""tree-sitter parsing per profile: symbols, spans, signatures, doc first lines.

Parsing never raises on bad input. A file that will not parse is a **gap**: the file
row survives with `parse_ok=0`, the index completes, and the report says what it missed. An
indexer that dies on one malformed file is an indexer that cannot run on a real repository
mid-edit — which is exactly when the harness needs it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Final

from tree_sitter import Node, Parser

from chickadee.graph.grammars import JAVASCRIPT, PYTHON, TSX, TYPESCRIPT, error_node, language
from chickadee.graph.tags import tags
from chickadee.profiles.resolve import FileLanguage

_JS_FAMILY = frozenset({JAVASCRIPT, TYPESCRIPT, TSX})

WALKED: Final = frozenset({PYTHON, *_JS_FAMILY})
"""The grammars a hand-written walker reads, imports, exports and callbacks included. Every
other grammar is read by its tags query (`graph/tags`): symbols and the names
they call, no imports, so its edges are name-resolved."""

# Node types that declare something worth a symbol row, per language family.
_DECLARATIONS = {
    PYTHON: {
        "function_definition": "function",
        "class_definition": "class",
    },
    JAVASCRIPT: {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "function_signature": "function",
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "interface_declaration": "class",
        "enum_declaration": "class",
        "method_definition": "method",
    },
}

# Python callables commonly given a function as an argument. Recording every identifier
# argument as a reference would treat `max(a, b)` as uses of functions named `a` and `b`.
_PY_KEY_HOF = frozenset({"sorted", "max", "min", "sort"})
_PY_FIRST_HOF = frozenset({"map", "filter", "reduce"})
_PY_HOF = _PY_KEY_HOF | _PY_FIRST_HOF
_JS_HOF = frozenset(
    {"map", "filter", "reduce", "forEach", "find", "some", "every", "sort", "flatMap"}
)


@dataclass(frozen=True)
class Extent:
    """A declaration's whole extent: its first and last lines, and the byte columns it
    starts at on the first and ends at on the last, so a reader of whole lines can tell what
    else they hold."""

    first: int
    last: int
    head: int
    """The byte column the declaration starts at on line `first`."""
    tail: int
    """The byte column it ends at on line `last`: what follows is not part of it."""


@dataclass(frozen=True)
class Symbol:
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    sig: str | None
    doc_first_line: str | None
    extent: Extent
    """The declaration's whole extent: its decorators, its `export`, and a bound
    function's declaration included, which `line_start` and `line_end` (the declaration node's
    own, what the index stores) leave out. What a symbol edit replaces."""
    parent: str | None = None
    exported: bool = False


@dataclass(frozen=True)
class Import:
    module: str
    alias: str | None
    line: int


@dataclass(frozen=True)
class Global:
    name: str
    qualified_name: str
    line: int
    kind: str
    type_text: str | None


@dataclass(frozen=True)
class Reference:
    """A name used at a call/reference site, before any resolution has happened.

    Deliberately not an `Edge`: this carries no `edge_class`, because nothing here knows
    whether the name resolved. Handing resolution a type that already has a class field is
    how a `fallback` guess ends up recorded as `resolved`.
    """

    name: str
    line: int
    from_symbol: str
    kind: str  # calls | references | inherits
    column: int | None = None
    """The byte column `name` starts at on `line`, or None when the parse could not place it
    there (a callee split over lines). What a language server is asked about: the
    name's first spelling on the line can be another thing (`f=g.f()`'s keyword)."""


@dataclass(frozen=True)
class ParsedFile:
    path: str
    lang: str
    """The profile's name the file is recorded under (`FileLanguage.name`): a reference
    resolves among files of its own language only."""
    parse_ok: bool
    symbols: tuple[Symbol, ...] = ()
    imports: tuple[Import, ...] = ()
    globals: tuple[Global, ...] = ()
    references: tuple[Reference, ...] = ()
    error: str | None = None


@dataclass
class _Walker:
    """Accumulates one file's parse. A dataclass rather than nested closures so each
    extraction step is separately readable and testable."""

    source: bytes
    grammar: str
    module: str
    symbols: list[Symbol] = field(default_factory=list[Symbol])
    imports: list[Import] = field(default_factory=list[Import])
    globals: list[Global] = field(default_factory=list[Global])
    references: list[Reference] = field(default_factory=list[Reference])
    dunder_all: frozenset[str] | None = None
    export_names: set[str] = field(default_factory=set[str])

    def text(self, node: Node | None) -> str:
        if node is None:
            return ""
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def name_of(self, node: Node) -> str | None:
        child = node.child_by_field_name("name")
        return self.text(child) if child is not None else None


def module_name(path: str) -> str:
    """Repository-relative, extensionless, dotted: `a/util.py` → `a.util`. The extension is
    whatever the path's is: which extensions are source is the profiles' answer, not this.

    The basename alone is not enough. `a/util.py` and `b/util.py` both produced
    `util.helper`, which collides in `symbols.qualified_name` *and* defeats the D19
    ambiguity rule: `name_index` counts distinct qualified names, so two colliding
    declarations looked like one and emitted a confident `fallback` edge to a name that
    could mean either file.
    """
    return str(PurePosixPath(path).with_suffix("")).replace("/", ".")


def _module_name(path: str) -> str:
    return module_name(path)


def parse_source(path: str, source: bytes, grammar: str, lang: str) -> ParsedFile:
    """Parse one file's bytes with `grammar`, recorded under the language `lang` (both from
    the file's profile, `ProfileSet.language_of`). Never raises for malformed input."""
    loaded = language(grammar)
    if loaded is None:
        missing = f"no grammar for {grammar}"
        return ParsedFile(path=path, lang=lang, parse_ok=False, error=missing)
    try:
        tree = Parser(loaded).parse(source)
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - grammar-level failure
        return ParsedFile(path=path, lang=lang, parse_ok=False, error=str(exc))

    if grammar not in WALKED:
        return _from_tags(path, source, grammar, lang, tree.root_node)
    walker = _Walker(source=source, grammar=grammar, module=_module_name(path))
    _visit(tree.root_node, walker, parent=None)
    if grammar == PYTHON:
        symbols = _apply_python_exports(walker)
    elif grammar in _JS_FAMILY:
        symbols = _apply_js_exports(walker)
    else:
        symbols = tuple(walker.symbols)
    # `has_error` is the honest signal: tree-sitter recovers and still yields a tree, so a
    # file with a syntax error produces partial symbols. Keeping them is right — a file
    # being edited is the normal case — but the file is still recorded as not-parse-ok so
    # the gap is reported rather than implied by absence.
    return ParsedFile(
        path=path,
        lang=lang,
        parse_ok=not tree.root_node.has_error,
        symbols=symbols,
        imports=tuple(walker.imports),
        globals=tuple(walker.globals),
        references=tuple(walker.references),
        error="syntax error" if tree.root_node.has_error else None,
    )


# A tags query's definition kinds, as the index names a symbol's kind (`schema.sql`). A
# property, field or variable is not a symbol the graph keeps, as the walkers keep none.
_TAG_KINDS: Final = {
    "function": "function",
    "macro": "function",
    "method": "method",
    "class": "class",
    "interface": "class",
    "type": "class",
    "enum": "class",
    "object": "class",
    "module": "module",
    "constant": "const",
}

_SIG_MAX = 200
"""Characters of a declaration's first line kept as its signature."""

_TAG_NAME = re.compile(r"[^\W\d][\w$]*[?!=]?")
"""A name a tags query may mark: an identifier in any of the twelve, Ruby's `valid?`, `save!`
and `name=` and PHP's and Kotlin's `$` included. Anything else (`a.b`, `x[0]`, a literal) is
not one name, and is no reference."""

_BODIES: Final = frozenset(
    {
        "body",
        "class_body",
        "declaration_list",
        "field_declaration_list",
        "template_body",
        "enum_class_body",
        "protocol_body",
        "enum_body",
    }
)
"""The node types that open a class's members, where its header (what it extends) ends."""

_ATTRIBUTES: Final = frozenset({"attribute_item"})
"""Siblings before a declaration that belong to it (Rust's `#[test]`, `#[derive]`): its extent
reaches back over them, as the walkers' reaches back over a decorator, so a symbol edit takes
them with it. Java's, C#'s, Kotlin's and Swift's annotations are inside the declaration."""


def _header_end(node: Node) -> int:
    """Where a class definition's header ends: the start of its body, or its own end."""
    body = node.child_by_field_name("body")
    if body is None:
        body = next((c for c in node.named_children if c.type in _BODIES), None)
    return body.start_byte if body is not None else node.end_byte


def _receiver(node: Node, source: bytes) -> str | None:
    """The type a method defined outside its type's body belongs to: Go's receiver
    (`func (s *S) M()`), the type of the Rust `impl` around it, or C++'s `A::m` scope. None
    for any other definition, whose parent is where it is."""
    if node.type == "method_declaration" and node.child_by_field_name("receiver") is not None:
        receiver = node.child_by_field_name("receiver")
        found = next(_types(receiver), None) if receiver is not None else None
        return _text(found, source) if found is not None else None
    if node.type == "function_item":
        outer = node.parent
        while outer is not None and outer.type not in ("impl_item", "source_file"):
            outer = outer.parent
        if outer is not None and outer.type == "impl_item":
            impl = outer.child_by_field_name("type")
            found = next(_types(impl), None) if impl is not None else None
            return _text(found, source) if found is not None else None
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        inner = declarator.child_by_field_name("declarator") if declarator else None
        scope = inner.child_by_field_name("scope") if inner is not None else None
        if inner is not None and inner.type == "qualified_identifier" and scope is not None:
            return _text(scope, source).rsplit("::", 1)[-1] or None
    return None


def _types(node: Node) -> Iterator[Node]:
    """`node`'s type identifiers, outermost first: `*S` and `S[T]` name `S`."""
    if node.type == "type_identifier":
        yield node
    for child in node.named_children:
        yield from _types(child)


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace").strip()


def _from_tags(path: str, source: bytes, grammar: str, lang: str, root: Node) -> ParsedFile:
    """`ParsedFile` from `grammar`'s tags query: each definition a symbol, nested by the
    definitions that contain it (or the type its receiver names), its first line its signature;
    each reference a name used by the innermost definition around it, or the module, and one in
    a class's header (what it extends or implements) an `inherits`."""
    module = _module_name(path)
    symbols: list[Symbol] = []
    references: list[Reference] = []
    open_: list[tuple[Node, Symbol]] = []
    found = tags(grammar, root, source)
    # A query may mark a declaration's own name as a use too (Ruby's, Go's): a name is not a
    # reference to itself.
    declared = {t.name_node.start_byte for t in found if t.role == "definition"}
    named: set[int] = set()
    for tag in found:
        while open_ and open_[-1][0].end_byte <= tag.node.start_byte:
            open_.pop()
        around = open_[-1] if open_ else None
        parent = around[1] if around else None
        if tag.role == "reference":
            start = tag.name_node.start_byte
            if start in declared or not _TAG_NAME.fullmatch(tag.name):
                continue
            in_header = (
                around is not None
                and parent is not None
                and parent.kind == "class"
                and start < _header_end(around[0])
            )
            kind = "inherits" if in_header else "calls" if tag.kind == "call" else "references"
            owner = parent.qualified_name if parent else module
            line = tag.name_node.start_point[0] + 1
            references.append(Reference(tag.name, line, owner, kind))
            continue
        kind = _TAG_KINDS.get(tag.kind)
        if kind is None or tag.name_node.start_byte in named:
            # Two patterns marking one name (a method and a function) are one declaration.
            continue
        named.add(tag.name_node.start_byte)
        node = tag.node
        owner_type = _receiver(node, source)
        if owner_type is not None and (parent is None or parent.kind != "class"):
            parent_name = f"{module}.{owner_type}"
            kind = "method"
        else:
            parent_name = parent.qualified_name if parent else None
            if kind == "function" and parent is not None and parent.kind == "class":
                kind = "method"
        qualified = f"{parent_name}.{tag.name}" if parent_name else f"{module}.{tag.name}"
        first_line = source[node.start_byte : node.end_byte].split(b"\n", 1)[0]
        sig = first_line.decode("utf-8", "replace").strip()[:_SIG_MAX] or None
        first = node
        before = node.prev_named_sibling
        while before is not None and before.type in _ATTRIBUTES:
            first, before = before, before.prev_named_sibling
        symbol = Symbol(
            name=tag.name,
            qualified_name=qualified,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            sig=sig,
            doc_first_line=None,
            extent=Extent(
                first.start_point[0] + 1,
                node.end_point[0] + 1,
                first.start_point[1],
                node.end_point[1],
            ),
            parent=parent_name,
        )
        symbols.append(symbol)
        open_.append((node, symbol))
    broken = error_node(root) is not None
    return ParsedFile(
        path=path,
        lang=lang,
        parse_ok=not broken,
        symbols=tuple(symbols),
        references=tuple(references),
        error="syntax error" if broken else None,
    )


def parse_file(root: Path, path: Path, language: FileLanguage) -> ParsedFile:
    """`path` read and parsed as `language`, the profile set's answer for it."""
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_bytes()
    except OSError as exc:
        unreadable = f"unreadable: {exc}"
        return ParsedFile(path=rel, lang=language.name, parse_ok=False, error=unreadable)
    return parse_source(rel, source, language.grammar, language.name)


# Scopes that mean this declaration is *inside* another body, not the export itself.
# `export function outer() { function inner() {} }` must not mark `inner`.
_JS_NESTED_SCOPE = frozenset(
    {
        "statement_block",
        "class_body",
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    }
)


def _under_export(node: Node) -> bool:
    """True when this node is the declaration an ESM `export` statement wraps.

    Stops at a nested function/class body so an inner function inside
    `export function outer() { function inner() {} }` is not exported.
    `export { name }` is collected separately.
    """
    current = node.parent
    while current is not None:
        if current.type == "export_statement":
            return True
        if current.type in _JS_NESTED_SCOPE:
            return False
        current = current.parent
    return False


def _collect_export_names(node: Node, w: _Walker) -> None:
    """Local names in `export { x }` / `export default x`.

    Declaration forms (`export function`, `export const`) are marked via `_under_export`.
    """
    for child in node.named_children:
        if child.type != "export_clause":
            continue
        for spec in child.named_children:
            if spec.type != "export_specifier":
                continue
            name = spec.child_by_field_name("name")
            if name is None:
                continue
            text = w.text(name)
            if text and text != "default":
                w.export_names.add(text)
    value = node.child_by_field_name("value")
    if value is not None and value.type == "identifier":
        w.export_names.add(w.text(value))


def _literal_string_seq(node: Node, w: _Walker) -> tuple[str, ...] | None:
    """`["a", "b"]` / `("a",)` only. A computed `__all__` is not an export list."""
    if node.type not in ("list", "tuple"):
        return None
    names: list[str] = []
    for child in node.named_children:
        if child.type != "string":
            return None
        names.append(w.text(child).strip("\"'`"))
    return tuple(names)


def _apply_python_exports(w: _Walker) -> tuple[Symbol, ...]:
    """`__all__` wins when it is a literal; otherwise module-level and not `_private`."""
    listed = w.dunder_all
    out: list[Symbol] = []
    for symbol in w.symbols:
        if symbol.parent is not None or symbol.kind == "method":
            exported = False
        elif listed is not None:
            exported = symbol.name in listed
        else:
            exported = not symbol.name.startswith("_")
        out.append(replace(symbol, exported=exported))
    return tuple(out)


def _apply_js_exports(w: _Walker) -> tuple[Symbol, ...]:
    """Module-level ESM bindings only. Nested decls inside an exported body are not exports."""
    listed = w.export_names
    out: list[Symbol] = []
    for symbol in w.symbols:
        if symbol.parent is not None or symbol.kind == "method":
            exported = False
        else:
            exported = symbol.exported or symbol.name in listed
        out.append(replace(symbol, exported=exported))
    return tuple(out)


def _visit(node: Node, w: _Walker, parent: Symbol | None) -> None:  # noqa: C901
    # A tree-sitter visitor: one branch per node type it knows. 16 against a ceiling of
    # 15; the next node type added here should be a table lookup, not a seventeenth branch.
    if node.type == "export_statement":
        _collect_export_names(node, w)
    family = JAVASCRIPT if w.grammar in _JS_FAMILY else w.grammar
    kinds = _DECLARATIONS[family]
    handled = False

    if node.type in ("arrow_function", "function_expression") and family == JAVASCRIPT:
        symbol = _js_bound_function(node, w, parent)
        if symbol is not None:
            w.symbols.append(symbol)
            for child in node.children:
                _visit(child, w, parent=symbol)
            handled = True
    elif node.type in kinds:
        symbol = _declaration(node, w, kinds[node.type], parent)
        if symbol is not None:
            w.symbols.append(symbol)
            for child in node.children:
                _visit(child, w, parent=symbol)
            handled = True
    elif node.type in ("import_statement", "import_from_statement"):
        # Dispatch by language: the two grammars disagree about where the module lives.
        # Python uses `module_name`; JavaScript uses `source`, so routing an ESM import
        # through the Python extractor produced no `imports` row at all.
        if family == JAVASCRIPT:
            _js_import(node, w)
        else:
            _python_import(node, w)
    elif node.type == "lexical_declaration" or node.type == "variable_declaration":
        _js_binding(node, w, parent)
    elif node.type == "assignment":
        _python_assignment(node, w, parent)
    elif node.type == "new_expression":
        _new_expression(node, w, parent)
    elif node.type in ("call", "call_expression"):
        _reference(node, w, parent, "calls")
        _hof_callbacks(node, w, parent, family)

    if not handled:
        for child in node.children:
            _visit(child, w, parent)


def _declaration(node: Node, w: _Walker, kind: str, parent: Symbol | None) -> Symbol | None:
    name = w.name_of(node)
    if not name:
        return None
    qualified = f"{parent.qualified_name}.{name}" if parent else f"{w.module}.{name}"
    if parent is not None and kind == "function":
        kind = "method"
    _inherits(node, w, qualified)
    return Symbol(
        name=name,
        qualified_name=qualified,
        kind=kind,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        sig=_signature(node, w, name),
        doc_first_line=_doc_first_line(node, w),
        extent=_extent(node),
        parent=parent.qualified_name if parent else None,
        exported=_under_export(node) if w.grammar in _JS_FAMILY else False,
    )


_EXTENT_WRAPPERS = frozenset(
    {
        "decorated_definition",
        "export_statement",
        "variable_declarator",
        "lexical_declaration",
        "variable_declaration",
        "public_field_definition",
        "field_definition",
        "pair",
        "expression_statement",
    }
)
"""Nodes that hold a declaration and are part of it: a Python function's decorators, an
`export`, the binding a JS function is the value of (a declarator, a field, an object's key),
the statement a Python lambda is assigned in."""


def _extent(node: Node) -> Extent:
    """`node`'s declaration: up through what wraps it, and back over the `decorator`s before
    it (a TS method's are its siblings, not its children)."""
    outer = node
    while outer.parent is not None and outer.parent.type in _EXTENT_WRAPPERS:
        outer = outer.parent
    first = outer
    before = outer.prev_named_sibling
    while before is not None and before.type == "decorator":
        first, before = before, before.prev_named_sibling
    return Extent(
        first.start_point[0] + 1,
        outer.end_point[0] + 1,
        first.start_point[1],
        outer.end_point[1],
    )


def _signature(node: Node, w: _Walker, name: str) -> str | None:
    params = node.child_by_field_name("parameters")
    if params is None:
        return None
    returns = node.child_by_field_name("return_type")
    sig = f"{name}{w.text(params)}"
    return f"{sig} -> {w.text(returns)}" if returns is not None else sig


def _doc_first_line(node: Node, w: _Walker) -> str | None:
    """First line only. The graph indexes what a symbol *is*, not its prose — and a full
    docstring in the index is a full docstring in every packed context that names it."""
    body = node.child_by_field_name("body")
    if body is None or not body.children:
        return None
    first = body.children[0]
    if first.type == "expression_statement" and first.children:
        literal = first.children[0]
        if literal.type == "string":
            text = w.text(literal).strip("\"'` \n")
            return text.split("\n")[0].strip() or None
    return None


def _inherits(node: Node, w: _Walker, qualified: str) -> None:
    # Python exposes bases as the `superclasses` field; JavaScript has no field for it and
    # hangs a `class_heritage` child off the declaration instead. Checked against both
    # grammars rather than assumed — the field name that does not exist returns None
    # silently, which is how `extends Base` produced no edge at all.
    bases = node.child_by_field_name("superclasses")
    if bases is None:
        bases = next((c for c in node.children if c.type == "class_heritage"), None)
    if bases is None:
        return
    for child in bases.named_children if bases.named_children else []:
        text = w.text(child).strip()
        if text:
            name = text.rsplit(".", 1)[-1]
            line = child.start_point[0] + 1
            w.references.append(
                Reference(
                    name=name,
                    line=line,
                    from_symbol=qualified,
                    kind="inherits",
                    column=_last_segment_column(child, name, line),
                )
            )


def _python_import(node: Node, w: _Walker) -> None:
    line = node.start_point[0] + 1
    module_node = node.child_by_field_name("module_name")
    base = w.text(module_node) if module_node is not None else ""
    named = [c for c in node.named_children if c.type in ("dotted_name", "aliased_import")]
    if node.type == "import_statement":
        for child in named:
            _import_entry(child, w, line, prefix="")
        return

    # Identity by byte span, never `is`: the tree-sitter bindings build a fresh `Node`
    # object on every access, so `child is module_node` is False even for the same node.
    # That silently emitted `from a.b import thing` as both `a.b.a.b` and `a.b.thing`.
    span = (module_node.start_byte, module_node.end_byte) if module_node is not None else None
    imported = [c for c in named if (c.start_byte, c.end_byte) != span]
    if not imported:
        w.imports.append(Import(module=base, alias=None, line=line))
        return
    for child in imported:
        # A relative import's module name is already dotted (`.`, `..`), so appending a
        # separator would turn `from. import x` into `..x`.
        prefix = "" if not base else base if base.endswith(".") else f"{base}."
        _import_entry(child, w, line, prefix=prefix)


def _import_entry(node: Node, w: _Walker, line: int, prefix: str) -> None:
    if node.type == "aliased_import":
        name = w.text(node.child_by_field_name("name"))
        alias = w.text(node.child_by_field_name("alias"))
        w.imports.append(Import(module=f"{prefix}{name}", alias=alias or None, line=line))
    else:
        w.imports.append(Import(module=f"{prefix}{w.text(node)}", alias=None, line=line))


def _js_import(node: Node, w: _Walker) -> None:
    """ESM: `import {a, b as c} from "mod"` / `import d from "mod"`."""
    source = node.child_by_field_name("source")
    if source is None:
        return
    module = w.text(source).strip("\"'`")
    line = node.start_point[0] + 1
    clause = next((c for c in node.named_children if c.type == "import_clause"), None)
    if clause is None:
        w.imports.append(Import(module=module, alias=None, line=line))
        return
    entries = 0
    for child in clause.named_children:
        if child.type == "identifier":  # default import
            w.imports.append(Import(module=module, alias=w.text(child), line=line))
            entries += 1
        elif child.type == "named_imports":
            for spec in child.named_children:
                if spec.type != "import_specifier":
                    continue
                name = w.text(spec.child_by_field_name("name"))
                alias_node = spec.child_by_field_name("alias")
                w.imports.append(
                    Import(
                        module=f"{module}.{name}",
                        alias=w.text(alias_node) if alias_node is not None else None,
                        line=line,
                    )
                )
                entries += 1
    if not entries:
        w.imports.append(Import(module=module, alias=None, line=line))


def _require_target(node: Node, w: _Walker) -> str | None:
    """The module of a CommonJS `require("mod")` call, or None if this is not one."""
    if node.type != "call_expression":
        return None
    fn = node.child_by_field_name("function")
    if fn is None or w.text(fn) != "require":
        return None
    args = node.child_by_field_name("arguments")
    if args is None:
        return None
    literal = next((c for c in args.named_children if c.type == "string"), None)
    return w.text(literal).strip("\"'`") if literal is not None else None


def _js_binding(node: Node, w: _Walker, parent: Symbol | None) -> None:
    """`const x =...` at module level is module state; `require(...)` is an import."""
    for declarator in node.named_children:
        if declarator.type != "variable_declarator":
            continue
        name = w.name_of(declarator)
        if not name:
            continue
        value = declarator.child_by_field_name("value")
        if value is not None:
            module = _require_target(value, w)
            if module is not None:
                w.imports.append(
                    Import(module=module, alias=name, line=declarator.start_point[0] + 1)
                )
                continue
        if parent is None and name.isupper():
            w.globals.append(
                Global(
                    name=name,
                    qualified_name=f"{w.module}.{name}",
                    line=declarator.start_point[0] + 1,
                    kind="constant",
                    type_text=None,
                )
            )


def _python_assignment(node: Node, w: _Walker, parent: Symbol | None) -> None:
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
    if left is None or right is None or left.type != "identifier":
        if parent is None:
            _python_global(node, w)
        return
    name = w.text(left)
    if parent is None and name == "__all__":
        parsed = _literal_string_seq(right, w)
        # A later computed assignment must drop an earlier literal; otherwise `__all__`
        # would keep advertising names the file no longer claims.
        w.dunder_all = frozenset(parsed) if parsed is not None else None
    if right.type == "lambda":
        qualified = f"{parent.qualified_name}.{name}" if parent else f"{w.module}.{name}"
        params = right.child_by_field_name("parameters")
        w.symbols.append(
            Symbol(
                name=name,
                qualified_name=qualified,
                kind="function" if parent is None else "method",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                sig=f"{name}{w.text(params)}" if params is not None else name,
                doc_first_line=None,
                extent=_extent(node),
                parent=parent.qualified_name if parent else None,
                exported=False,
            )
        )
        return
    if parent is None:
        _python_global(node, w)
    if right.type == "identifier":
        w.references.append(
            Reference(
                name=w.text(right),
                line=right.start_point[0] + 1,
                from_symbol=parent.qualified_name if parent else w.module,
                kind="references",
                column=right.start_point[1],
            )
        )


def _js_bound_function(node: Node, w: _Walker, parent: Symbol | None) -> Symbol | None:
    """`const pad = () => {}` / class field arrows: the name lives on the binding."""
    name = _js_binding_name(node, w)
    if not name:
        return None
    qualified = f"{parent.qualified_name}.{name}" if parent else f"{w.module}.{name}"
    kind = "method" if parent is not None else "function"
    params = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
    return Symbol(
        name=name,
        qualified_name=qualified,
        kind=kind,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        sig=f"{name}{w.text(params)}" if params is not None else name,
        doc_first_line=_doc_first_line(node, w),
        extent=_extent(node),
        parent=parent.qualified_name if parent else None,
        exported=_under_export(node),
    )


# These mean the function is a *value* in a larger expression, not the binding itself.
# Walking through them names `const total = items.reduce((a, b) => a + b)` as `total`
# and duplicates `const App = () => { useEffect(() => {}) }` as a second `App`.
_JS_VALUE_BOUNDARY = frozenset(
    {
        "arguments",
        "call_expression",
        "new_expression",
        "return_statement",
        "statement_block",
        "array",
        "arrow_function",
        "function_expression",
    }
)


def _js_binding_name(node: Node, w: _Walker) -> str | None:
    current = node.parent
    while current is not None:
        if current.type in _JS_VALUE_BOUNDARY:
            return None
        if current.type == "variable_declarator":
            return w.name_of(current)
        if current.type in ("field_definition", "public_field_definition"):
            prop = current.child_by_field_name("property")
            if prop is None:
                prop = current.child_by_field_name("name")
            return w.text(prop) if prop is not None else None
        if current.type in ("pair", "property"):
            key = current.child_by_field_name("key")
            if key is not None:
                return w.text(key).strip("\"'`")
        current = current.parent
    return None


def _new_expression(node: Node, w: _Walker, parent: Symbol | None) -> None:
    ctor = node.child_by_field_name("constructor")
    if ctor is None and node.named_child_count:
        ctor = node.named_child(0)
    if ctor is None:
        return
    text = w.text(ctor)
    name = text.rsplit(".", 1)[-1].strip()
    if not name or not name.isidentifier():
        return
    line = node.start_point[0] + 1
    w.references.append(
        Reference(
            name=name,
            line=line,
            from_symbol=parent.qualified_name if parent else w.module,
            kind="calls",
            column=_last_segment_column(ctor, name, line),
        )
    )


def _hof_callbacks(node: Node, w: _Walker, parent: Symbol | None, family: str) -> None:
    """The callback argument only — not every identifier the HOF happens to take."""
    callee = node.child_by_field_name("function")
    if callee is None:
        return
    callee_name = w.text(callee).rsplit(".", 1)[-1].strip()
    hof = _JS_HOF if family == JAVASCRIPT else _PY_HOF
    if callee_name not in hof:
        return
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    owner = parent.qualified_name if parent else w.module
    idents = (
        _js_hof_idents(args) if family == JAVASCRIPT else _py_hof_idents(callee_name, args, w)
    )
    for ident in idents:
        name = w.text(ident)
        if name.isidentifier():
            w.references.append(
                Reference(
                    name=name,
                    line=ident.start_point[0] + 1,
                    from_symbol=owner,
                    kind="references",
                    column=ident.start_point[1],
                )
            )


def _js_hof_idents(args: Node) -> tuple[Node, ...]:
    """`items.map(fn, thisArg)` — only `fn` is the callback."""
    children = args.named_children
    if children and children[0].type == "identifier":
        return (children[0],)
    return ()


def _py_hof_idents(callee: str, args: Node, w: _Walker) -> tuple[Node, ...]:
    if callee in _PY_KEY_HOF:
        found: list[Node] = []
        for child in args.named_children:
            if child.type != "keyword_argument":
                continue
            if w.text(child.child_by_field_name("name")) != "key":
                continue
            ident = child.child_by_field_name("value")
            if ident is not None and ident.type == "identifier":
                found.append(ident)
        return tuple(found)
    positional = [c for c in args.named_children if c.type != "keyword_argument"]
    if positional and positional[0].type == "identifier":
        return (positional[0],)
    return ()


def _python_global(node: Node, w: _Walker) -> None:
    left = node.child_by_field_name("left")
    if left is None or left.type != "identifier":
        return
    name = w.text(left)
    annotation = node.child_by_field_name("type")
    w.globals.append(
        Global(
            name=name,
            qualified_name=f"{w.module}.{name}",
            line=node.start_point[0] + 1,
            kind="constant" if name.isupper() else "module_state",
            type_text=w.text(annotation) if annotation is not None else None,
        )
    )


def _reference(node: Node, w: _Walker, parent: Symbol | None, kind: str) -> None:
    fn = node.child_by_field_name("function")
    if fn is None:
        return
    text = w.text(fn)
    name = text.rsplit(".", 1)[-1].strip()
    if not name or not name.isidentifier():
        return
    line = node.start_point[0] + 1
    w.references.append(
        Reference(
            name=name,
            line=line,
            from_symbol=parent.qualified_name if parent else w.module,
            kind=kind,
            column=_last_segment_column(fn, name, line),
        )
    )


def _last_segment_column(node: Node, name: str, line: int) -> int | None:
    """The byte column `name`, the last segment of `node`'s text (`b` of `a.b`), starts at, when
    it ends on `line`; None when the expression runs onto another line."""
    row, end = node.end_point
    if row + 1 != line:
        return None
    return max(end - len(name.encode("utf-8")), 0)
