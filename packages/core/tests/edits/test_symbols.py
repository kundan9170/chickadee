"""A symbol's span in a text: by the one parser and the one name rule,
decorators, an `export` and a bound function's declaration included; a name naming several is
`AMBIGUOUS` naming each, none `NO_MATCH` naming the file's, and a language with no parser
`UNSUPPORTED`."""

from __future__ import annotations

from chickadee.edits.symbols import (
    NAMED_MAX,
    SymbolMiss,
    SymbolSpan,
    find_symbol,
    symbols_named,
)
from chickadee.profiles import FileLanguage, shipped

_LANGUAGES = shipped()
"""What a symbol edit reads a path as is the profile set's answer for it; the span rule is the
subject here, TypeScript's included."""


def _lang(path: str) -> FileLanguage | None:
    return _LANGUAGES.language_of(path)


PY = (
    "import functools\n"  # 1
    "\n"
    "\n"
    "@functools.cache\n"  # 4
    "@other(\n"
    "    1)\n"
    "def build(name):\n"  # 7
    "    return name\n"
    "\n"
    "\n"
    "class Widget:\n"  # 11
    "    @property\n"
    "    def render(self):\n"
    "        return 1\n"
    "\n"
    "\n"
    "class Other:\n"  # 17
    "    def render(self):\n"
    "        return 2\n"
)

TS = (
    "@sealed\n"  # 1
    "export class Widget {\n"
    "  @log()\n"
    "  render(): number {\n"
    "    return 1;\n"
    "  }\n"
    "}\n"
    "export const pad = (s: string) => {\n"  # 8
    "  return s;\n"
    "};\n"
)


def _span(path: str, text: str, name: str) -> SymbolSpan:
    found = find_symbol(path, text, name, _lang(path))
    assert isinstance(found, SymbolSpan), found
    return found


def test_a_python_span_holds_its_decorators_multi_line_ones_too() -> None:
    span = _span("app.py", PY, "build")
    assert (span.first, span.last) == (4, 8)
    assert span.qualified_name == "app.build" and span.sig == "build(name)"


def test_a_method_span_holds_its_decorator() -> None:
    span = _span("app.py", PY, "Widget.render")
    assert (span.first, span.last) == (12, 14)


def test_a_ts_span_holds_its_decorators_and_export() -> None:
    assert (lambda s: (s.first, s.last))(_span("ui.ts", TS, "Widget")) == (1, 7)
    assert (lambda s: (s.first, s.last))(_span("ui.ts", TS, "Widget.render")) == (3, 6)


def test_a_bound_function_spans_its_whole_declaration() -> None:
    span = _span("ui.ts", TS, "pad")
    assert (span.first, span.last) == (8, 10)


def test_a_name_naming_several_is_ambiguous_naming_each() -> None:
    found = find_symbol("app.py", PY, "render", _lang("app.py"))
    assert isinstance(found, SymbolMiss) and found.reason == "AMBIGUOUS"
    assert "app.Widget.render (line 12)" in found.detail
    assert "app.Other.render (line 18)" in found.detail
    assert _span("app.py", PY, "Other.render").first == 18


def test_a_whole_qualified_name_names_that_one_alone() -> None:
    text = "def go():\n    pass\n\n\nclass go:\n    def go(self):\n        pass\n"
    assert _span("go.py", text, "go.go.go").first == 6


def test_a_name_must_begin_a_segment() -> None:
    found = find_symbol("app.py", PY, "ild", _lang("app.py"))
    assert isinstance(found, SymbolMiss) and found.reason == "NO_MATCH"
    assert "app.build" in found.detail and "app.Widget.render" in found.detail


def test_a_long_no_match_names_some_and_counts_the_rest() -> None:
    text = "".join(f"def f{i}():\n    pass\n" for i in range(NAMED_MAX + 5))
    found = find_symbol("many.py", text, "nothing", _lang("many.py"))
    assert isinstance(found, SymbolMiss) and found.detail.endswith("and 5 more")


def test_a_language_with_no_parser_is_unsupported() -> None:
    for path in ("notes.md", "main.zig", "Makefile"):
        found = find_symbol(path, "anything\n", "x", _lang(path))
        assert isinstance(found, SymbolMiss) and found.reason == "UNSUPPORTED", path
        assert isinstance(symbols_named(path, "x\n", "x", _lang(path)), SymbolMiss)


def test_a_text_mid_edit_still_yields_what_parses() -> None:
    broken = PY + "def half(:\n"
    assert _span("app.py", broken, "build").first == 4


def test_symbols_named_lists_every_match_in_file_order() -> None:
    named = symbols_named("app.py", PY, "render", _lang("app.py"))
    assert not isinstance(named, SymbolMiss)
    assert [s.qualified_name for s in named] == ["app.Widget.render", "app.Other.render"]


def test_one_name_declared_twice_is_ambiguous_without_asking_for_its_name() -> None:
    # T-3.133 review: "give the qualified name" sent the worker round again for a name no
    # spelling can pick (a property and its setter, overloads).
    text = (
        "class C:\n    @property\n    def x(self):\n        return 1\n\n"
        "    @x.setter\n    def x(self, v):\n        pass\n"
    )
    for name in ("x", "C.x", "a.C.x"):
        found = find_symbol("a.py", text, name, _lang("a.py"))
        assert isinstance(found, SymbolMiss) and found.reason == "AMBIGUOUS"
        assert "declared 2 times" in found.detail and "by its text" in found.do
        assert "qualified name" not in found.detail


def test_a_declaration_sharing_its_line_with_other_code_is_refused_by_name() -> None:
    # A span is whole lines: replacing one that holds other code would take it too.
    js = (
        "const a = {\n  x: 1\n}; function render() {\n  return 1;\n}\n"
        "function next() {} function last() {}\n"
    )
    for name, said in (
        ("render", "line 3 with '};' before it"),
        ("next", "line 6 with 'function last() {}' after it"),
        ("last", "line 6 with 'function next() {}' before it"),
    ):
        found = find_symbol("x.js", js, name, _lang("x.js"))
        assert isinstance(found, SymbolMiss) and found.reason == "UNSUPPORTED", name
        assert said in found.detail and "by its text" in found.do
    py = "x = 1; y = lambda: 2\n"
    found = find_symbol("x.py", py, "y", _lang("x.py"))
    assert isinstance(found, SymbolMiss) and "'x = 1;' before it" in found.detail


def test_indentation_a_trailing_comment_and_its_own_semicolon_are_its_line() -> None:
    js = "function ok() {} // trailing\nexport const p = () => {\n  return 1;\n};\n"
    assert _span("x.js", js, "ok").first == 1
    assert (_span("x.js", js, "p").first, _span("x.js", js, "p").last) == (2, 4)
    py = "class C:\n    async def f(self):\n        pass  # done\n"
    assert _span("x.py", py, "C.f").first == 2


def test_the_view_still_locates_a_declaration_sharing_a_line() -> None:
    named = symbols_named("x.js", "}; function render() {}\n", "render", _lang("x.js"))
    assert not isinstance(named, SymbolMiss) and named[0].first == 1


def test_an_objects_methods_and_keyed_arrows_are_edited_by_name() -> None:
    # T-3.133 review: their own `,` and their key are part of their lines, not other code.
    js = (
        "const api = {\n  load() {\n    return 1;\n  },\n  save: () => {\n    return 2;\n"
        "  }, // keyed\n};\n"
    )
    assert (_span("x.js", js, "load").first, _span("x.js", js, "load").last) == (2, 4)
    assert (_span("x.js", js, "save").first, _span("x.js", js, "save").last) == (5, 7)


def test_code_after_a_block_comment_still_shares_the_line() -> None:
    js = "function a() {} /* x */ function b() {}\nfunction c() {} /* note */\n"
    found = find_symbol("x.js", js, "a", _lang("x.js"))
    assert (
        isinstance(found, SymbolMiss) and "'/* x */ function b() {}' after it" in found.detail
    )
    assert _span("x.js", js, "c").first == 2
