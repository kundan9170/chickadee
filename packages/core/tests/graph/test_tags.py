"""The tags-query extractor: a grammar with no hand-written
walker is read by its tags query into the same `ParsedFile`: definitions nested by where they
are, and the names they call."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.graph.grammars import GRAMMARS
from chickadee.graph.parse import WALKED, parse_source
from chickadee.graph.tags import HELD_DIR, query_path, tags

# Each: a class (or type) with a method calling a top-level function, and that function.
SAMPLES: dict[str, tuple[str, bytes, str, str]] = {
    "go": (
        "a.go",
        b"package a\n\ntype T struct{}\n\nfunc (t T) M() int { return Helper() }\n\n"
        b"func Helper() int { return 1 }\n",
        "a.T.M",
        "a.Helper",
    ),
    "rust": (
        "a.rs",
        b"struct S;\nimpl S {\n    fn m(&self) -> i32 { helper() }\n}\n"
        b"fn helper() -> i32 { 1 }\n",
        "a.S.m",
        "a.helper",
    ),
    "java": (
        "A.java",
        b"class A {\n  int m() { return helper(); }\n  static int helper() { return 1; }\n}\n",
        "A.A.m",
        "A.A.helper",
    ),
    "c": (
        "a.c",
        b"int helper(void) { return 1; }\nint m(void) { return helper(); }\n",
        "a.m",
        "a.helper",
    ),
    "cpp": (
        "a.cpp",
        b"int helper() { return 1; }\n"
        b"class A {\n public:\n  int m() { return helper(); }\n};\n",
        "a.A.m",
        "a.helper",
    ),
    "c_sharp": (
        "A.cs",
        b"class A {\n  int M() { return Helper(); }\n  static int Helper() { return 1; }\n}\n",
        "A.A.M",
        "A.A.Helper",
    ),
    "ruby": (
        "a.rb",
        b"class A\n  def m\n    helper\n  end\nend\n\ndef helper\n  1\nend\n",
        "a.A.m",
        "a.helper",
    ),
    "php": (
        "a.php",
        b"<?php\nclass A {\n  function m() { return helper(); }\n}\n"
        b"function helper() { return 1; }\n",
        "a.A.m",
        "a.helper",
    ),
    "kotlin": (
        "a.kt",
        b"class A {\n    fun m(): Int = helper()\n}\n\nfun helper(): Int = 1\n",
        "a.A.m",
        "a.helper",
    ),
    "swift": (
        "a.swift",
        b"class A {\n  func m() -> Int { return helper() }\n}\n"
        b"func helper() -> Int { return 1 }\n",
        "a.A.m",
        "a.helper",
    ),
    "scala": (
        "a.scala",
        b"class A {\n  def m(): Int = helper()\n}\nobject O {\n  def helper(): Int = 1\n}\n",
        "a.A.m",
        "a.O.helper",
    ),
}


def test_every_grammar_without_a_walker_has_a_sample() -> None:
    assert set(SAMPLES) == GRAMMARS - WALKED


@pytest.mark.parametrize("grammar", sorted(SAMPLES))
def test_definitions_and_the_call_between_them(grammar: str) -> None:
    path, source, caller, callee = SAMPLES[grammar]
    parsed = parse_source(path, source, grammar, grammar)
    assert parsed.parse_ok, parsed.error
    names = {s.qualified_name: s for s in parsed.symbols}
    assert caller in names and callee in names, sorted(names)
    assert names[caller].sig is not None and names[caller].line_start >= 1
    calls = [(r.name, r.from_symbol) for r in parsed.references if r.kind == "calls"]
    assert (callee.rsplit(".", 1)[1], caller) in calls, calls
    assert parsed.imports == (), "a tags query reads no imports"


@pytest.mark.parametrize("grammar", sorted(SAMPLES))
def test_a_declarations_own_name_is_not_a_use_of_it(grammar: str) -> None:
    path, source, _caller, callee = SAMPLES[grammar]
    parsed = parse_source(path, source, grammar, grammar)
    helper = next(s for s in parsed.symbols if s.qualified_name == callee)
    own = [r for r in parsed.references if r.from_symbol == callee and r.name == helper.name]
    assert own == []


def test_a_method_is_one_symbol_however_many_patterns_mark_it() -> None:
    path, source, caller, _callee = SAMPLES["swift"]
    parsed = parse_source(path, source, "swift", "swift")
    assert [s.qualified_name for s in parsed.symbols].count(caller) == 1
    assert all(s.qualified_name.count(".m") <= 1 for s in parsed.symbols)


def test_a_function_definition_spans_its_body_not_its_declarator() -> None:
    source = b"int m(void)\n{\n  return 1;\n}\nint proto(void);\n"
    parsed = parse_source("a.c", source, "c", "c")
    symbols = {s.name: s for s in parsed.symbols}
    assert (symbols["m"].line_start, symbols["m"].line_end) == (1, 4)
    assert "proto" not in symbols, "a prototype is not a definition"


def test_a_broken_file_keeps_what_the_query_recovers() -> None:
    source = b"package a\n\nfunc Good() int { return 1 }\n\nfunc Bad( {\n"
    parsed = parse_source("a.go", source, "go", "go")
    assert not parsed.parse_ok
    assert "a.Good" in {s.qualified_name for s in parsed.symbols}


def test_the_held_query_replaces_the_wheels() -> None:
    for grammar in ("c", "cpp", "php", "swift", "kotlin", "c_sharp"):
        assert query_path(grammar) == HELD_DIR / f"{grammar}.scm"
    held = query_path("go")
    assert held is not None and held.name == "tags.scm" and HELD_DIR not in held.parents


def test_a_grammar_with_no_query_has_no_tags(tmp_path: Path) -> None:
    from tree_sitter import Parser

    from chickadee.graph.grammars import language

    loaded = language("kotlin")
    assert loaded is not None
    root = Parser(loaded).parse(b"fun f() = 1\n").root_node
    assert tags("kotlin", root, b"fun f() = 1\n", held=tmp_path) == ()


# -- the review of #605 ----------------------------------------------------------------------


def test_methods_of_two_types_in_one_file_are_two_symbols() -> None:
    """A Go receiver, a Rust `impl` and a C++ `A::m` name the type a method belongs to."""
    go = (
        b"package a\n\ntype S struct{}\ntype T struct{}\n\n"
        b"func (s *S) M() {}\nfunc (t T) M() {}\n"
    )
    names = {s.qualified_name: s for s in parse_source("a.go", go, "go", "go").symbols}
    assert {"a.S.M", "a.T.M"} <= set(names) and names["a.S.M"].parent == "a.S"
    rust = b"struct S;\nstruct T;\nimpl S { fn new() {} }\nimpl T { fn new() {} }\n"
    names = {s.qualified_name: s for s in parse_source("a.rs", rust, "rust", "rust").symbols}
    assert {"a.S.new", "a.T.new"} <= set(names)
    cpp = b"class A { int m(); };\nint A::m() { return 1; }\n"
    names = {s.qualified_name for s in parse_source("a.cpp", cpp, "cpp", "cpp").symbols}
    assert "a.A.m" in names


def test_a_rust_attribute_is_part_of_the_declaration() -> None:
    source = b"#[test]\n#[ignore]\nfn foo() {\n    assert!(true);\n}\n"
    foo = next(s for s in parse_source("a.rs", source, "rust", "rust").symbols)
    assert (foo.extent.first, foo.extent.last, foo.line_start) == (1, 5, 3)


def test_a_ruby_predicate_or_bang_call_is_a_reference() -> None:
    source = b"def valid?\n  true\nend\n\ndef go\n  valid?\n  save!\nend\n"
    refs = {r.name for r in parse_source("a.rb", source, "ruby", "ruby").references}
    assert {"valid?", "save!"} <= refs


@pytest.mark.parametrize(
    ("grammar", "path", "source", "base"),
    [
        ("java", "A.java", b"class A extends B implements C {}\n", {"B", "C"}),
        ("ruby", "a.rb", b"class A < B\nend\n", {"B"}),
        ("php", "a.php", b"<?php\nclass A extends B implements C {}\n", {"B", "C"}),
        ("scala", "a.scala", b"class A extends B\n", {"B"}),
        ("c_sharp", "A.cs", b"class A : B {}\n", {"B"}),
        ("kotlin", "a.kt", b"class A : B()\n", {"B"}),
    ],
)
def test_what_a_class_extends_is_inherited(
    grammar: str, path: str, source: bytes, base: set[str]
) -> None:
    parsed = parse_source(path, source, grammar, grammar)
    inherits = {r.name for r in parsed.references if r.kind == "inherits"}
    assert base <= inherits, parsed.references


def test_a_hidden_token_the_grammar_inserts_is_no_parse_error() -> None:
    """Kotlin's grammar marks a zero-width `_class_member_semi` MISSING on valid one-line
    code; nothing in the text is wrong."""
    source = b"class A : B() { fun m() { foo() } }\n"
    assert parse_source("a.kt", source, "kotlin", "kotlin").parse_ok
