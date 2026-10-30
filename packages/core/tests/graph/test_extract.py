"""Richer tree-sitter census: JS arrows/exports/new, Python lambdas and HOF callbacks."""

from __future__ import annotations

from pathlib import Path

from chickadee.graph.indexer import Indexer
from chickadee.graph.parse import parse_source
from chickadee.profiles import shipped


def test_export_const_arrow_is_a_function() -> None:
    src = b"export const pad = (s) => s.trim();\n"
    parsed = parse_source("util.js", src, "javascript", "javascript")
    names = {s.qualified_name for s in parsed.symbols}
    assert "util.pad" in names
    assert next(s for s in parsed.symbols if s.name == "pad").exported is True


def test_const_function_expression_is_a_function() -> None:
    src = b"const pad = function(s) { return s; };\n"
    parsed = parse_source("util.js", src, "javascript", "javascript")
    names = {s.qualified_name for s in parsed.symbols}
    assert "util.pad" in names
    assert next(s for s in parsed.symbols if s.name == "pad").exported is False


def test_new_expression_is_a_call_of_the_class() -> None:
    src = b"class Formatter {}\nfunction go(){ return new Formatter(); }\n"
    parsed = parse_source("a.js", src, "javascript", "javascript")
    refs = [r for r in parsed.references if r.kind == "calls"]
    assert any(r.name == "Formatter" for r in refs)


def test_map_callback_is_a_reference_not_a_call_of_the_callback() -> None:
    src = b"const pad = (s) => s;\nfunction go(items){ return items.map(pad); }\n"
    parsed = parse_source("a.js", src, "javascript", "javascript")
    assert any(r.name == "pad" and r.kind == "references" for r in parsed.references)
    assert any(r.name == "map" and r.kind == "calls" for r in parsed.references)
    assert not any(r.name == "pad" and r.kind == "calls" for r in parsed.references)


def test_map_thisarg_is_not_a_callback() -> None:
    src = (
        b"function pad(s){ return s; }\n"
        b"function go(items, ctx){ return items.map(pad, ctx); }\n"
    )
    parsed = parse_source("a.js", src, "javascript", "javascript")
    assert any(r.name == "pad" and r.kind == "references" for r in parsed.references)
    assert not any(r.name == "ctx" and r.kind == "references" for r in parsed.references)


def test_python_lambda_bound_to_a_name_is_a_function() -> None:
    src = b"pad = lambda s: s.strip()\n"
    parsed = parse_source("a.py", src, "python", "python")
    assert any(s.name == "pad" and s.kind == "function" for s in parsed.symbols)


def test_python_sorted_key_is_a_callback_reference() -> None:
    src = b"def helper(x):\n    return x\n\ndef go(xs):\n    return sorted(xs, key=helper)\n"
    parsed = parse_source("a.py", src, "python", "python")
    assert any(r.name == "helper" and r.kind == "references" for r in parsed.references)
    assert not any(r.name == "xs" and r.kind == "references" for r in parsed.references)


def test_max_of_two_values_is_not_a_callback() -> None:
    src = b"def go(a, b):\n    return max(a, b)\n"
    parsed = parse_source("a.py", src, "python", "python")
    assert not any(r.kind == "references" for r in parsed.references)


def test_print_argument_is_not_a_callback() -> None:
    src = b"def go(x):\n    print(x)\n"
    parsed = parse_source("a.py", src, "python", "python")
    assert not any(r.name == "x" and r.kind == "references" for r in parsed.references)


def test_assigned_identifier_is_a_reference() -> None:
    src = b"def helper():\n    return 1\n\ndef go():\n    cb = helper\n    return cb\n"
    parsed = parse_source("a.py", src, "python", "python")
    assert any(r.name == "helper" and r.kind == "references" for r in parsed.references)


def test_typescript_function_is_indexed(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.ts").write_text(
        "export function pad(s: string): string { return s; }\n", encoding="utf-8"
    )
    ix = Indexer(root, profiles=shipped())
    ix.full_index()
    names = {r[0] for r in ix.conn.execute("SELECT qualified_name FROM symbols")}
    assert "a.pad" in names


def test_a_typescript_file_is_read_by_its_profile(tmp_path: Path) -> None:
    """Moved twice: T-3.164 made `.ts` a gap while no profile owned it; T-3.165's TypeScript
    profile owns it, so the index reads it under `typescript` and reports no gap."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.ts").write_text("export function pad(s: string) { return s; }\n")
    ix = Indexer(root, profiles=shipped())
    report = ix.full_index()
    assert report.gaps == ()
    rows = ix.conn.execute("SELECT path, lang FROM files").fetchall()
    assert [tuple(r) for r in rows] == [("a.ts", "typescript")]


def test_reduce_callback_is_not_named_for_the_const() -> None:
    src = b"const total = items.reduce((a, b) => a + b);\n"
    parsed = parse_source("a.js", src, "javascript", "javascript")
    assert not any(s.name == "total" for s in parsed.symbols)


def test_nested_arrow_inside_named_arrow_is_not_a_duplicate() -> None:
    src = b"const App = () => { useEffect(() => { return 1; }); };\n"
    parsed = parse_source("a.js", src, "javascript", "javascript")
    assert [s.qualified_name for s in parsed.symbols if s.name == "App"] == ["a.App"]


def test_inner_map_arrow_does_not_steal_the_binding() -> None:
    src = b"const pad = (items) => items.map(x => x);\n"
    parsed = parse_source("util.js", src, "javascript", "javascript")
    names = {s.qualified_name for s in parsed.symbols}
    assert "util.pad" in names
    assert "util.pad.pad" not in names


def test_object_property_arrow_keeps_its_key() -> None:
    src = b"const o = { pad: (s) => s };\n"
    parsed = parse_source("a.js", src, "javascript", "javascript")
    assert "a.pad" in {s.qualified_name for s in parsed.symbols}


def test_typescript_interface_and_enum_are_symbols() -> None:
    src = b"export interface Widget { x: number }\nexport enum Color { Red }\n"
    parsed = parse_source("a.ts", src, "typescript", "typescript")
    names = {s.name for s in parsed.symbols}
    assert "Widget" in names
    assert "Color" in names


def test_nested_function_inside_export_is_not_exported() -> None:
    src = b"export function outer(){ function inner(){ return 1; } }\n"
    parsed = parse_source("mod.js", src, "javascript", "javascript")
    by_name = {s.name: s.exported for s in parsed.symbols}
    assert by_name["outer"] is True
    assert by_name["inner"] is False


def test_export_clause_marks_the_local_binding() -> None:
    src = b"const api = () => {}; export { api };\n"
    parsed = parse_source("mod.js", src, "javascript", "javascript")
    assert next(s for s in parsed.symbols if s.name == "api").exported is True


def test_export_default_identifier_is_exported() -> None:
    src = b"const api = () => {}; export default api;\n"
    parsed = parse_source("mod.js", src, "javascript", "javascript")
    assert next(s for s in parsed.symbols if s.name == "api").exported is True
