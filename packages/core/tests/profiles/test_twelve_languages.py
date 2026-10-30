"""The twelve languages at every layer.

For each: the profile loads and states its own tools; a broken file fails the syntax check with
its line and a good one passes as parsed; `init` detects it from a manifest and a source file;
the graph answers a definition, its caller in another file, and `tests_for` the test that calls
it; and Zeus's shell treats a new file of it as source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from chickadee.approval.commands import CommandClass
from chickadee.approval.commands import shipped as command_table
from chickadee.box.syntax import SyntaxChecker
from chickadee.graph.grammars import GRAMMARS
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.graph.tests_for import tests_for as select_tests
from chickadee.init.detect import detect_profiles
from chickadee.profiles import LanguageProfile, ProfileSet, resolve_conventions, shipped
from chickadee.zeus.tools.graphnav import GraphNavigator


@dataclass(frozen=True)
class Fixture:
    manifest: str
    lib: tuple[str, str]
    """A file defining `compute` (a function) and a class with a method."""
    app: tuple[str, str]
    """Another file calling `compute`."""
    test: tuple[str, str]
    """A test file calling `compute`."""
    good: str
    broken: str
    """`good` with one syntax error, on line `broken_line` (or reported at the construct that
    holds it)."""
    broken_line: int
    method: str
    """The method's qualified name's last two parts: `Class.method`."""


LANGUAGES: dict[str, Fixture] = {
    "typescript": Fixture(
        "package.json",
        (
            "src/lib.ts",
            "export function compute(x: number): number {\n  return x + 1;\n}\n\n"
            "export class Calc {\n  add(a: number): number {\n    return compute(a);\n  }\n}\n",
        ),
        (
            "src/app.ts",
            "import { compute } from './lib';\n\nexport function run(): number {\n"
            "  return compute(1);\n}\n",
        ),
        (
            "src/lib.test.ts",
            "import { compute } from './lib';\n\ntest('compute', () => {\n"
            "  expect(compute(1)).toBe(2);\n});\n",
        ),
        "const x: number = 1;\n",
        "const x: number = 1;\nfunction f( {\n",
        2,
        "Calc.add",
    ),
    "go": Fixture(
        "go.mod",
        (
            "calc/lib.go",
            "package calc\n\nfunc compute(x int) int {\n\treturn x + 1\n}\n\n"
            "type Calc struct{}\n\nfunc (c Calc) Add(a int) int {\n\treturn compute(a)\n}\n",
        ),
        ("calc/app.go", "package calc\n\nfunc run() int {\n\treturn compute(1)\n}\n"),
        (
            "calc/lib_test.go",
            'package calc\n\nimport "testing"\n\nfunc TestCompute(t *testing.T) {\n'
            '\tif compute(1) != 2 {\n\t\tt.Fatal("compute")\n\t}\n}\n',
        ),
        "package a\n\nfunc f() int { return 1 }\n",
        "package a\n\nfunc f() int { return 1\n",
        3,
        "Calc.Add",
    ),
    "rust": Fixture(
        "Cargo.toml",
        (
            "src/lib.rs",
            "pub fn compute(x: i32) -> i32 {\n    x + 1\n}\n\npub struct Calc;\n\n"
            "impl Calc {\n    pub fn add(&self, a: i32) -> i32 {\n"
            "        compute(a)\n    }\n}\n",
        ),
        ("src/app.rs", "pub fn run() -> i32 {\n    compute(1)\n}\n"),
        (
            "tests/lib_test.rs",
            "#[test]\nfn computes() {\n    let got = compute(1);\n    assert_eq!(got, 2);\n}\n",
        ),
        "fn f() -> i32 { 1 }\n",
        "fn f() -> i32 { 1 }\nfn g( {\n",
        2,
        "Calc.add",
    ),
    "java": Fixture(
        "pom.xml",
        (
            "src/main/java/Lib.java",
            "class Lib {\n  static int compute(int x) {\n    return x + 1;\n"
            "  }\n}\n\nclass Calc {\n  int add(int a) {\n    return Lib.compute(a);\n  }\n}\n",
        ),
        (
            "src/main/java/App.java",
            "class App {\n  int run() {\n    return Lib.compute(1);\n  }\n}\n",
        ),
        (
            "src/test/java/LibTest.java",
            "class LibTest {\n  void computes() {\n    assert Lib.compute(1) == 2;\n  }\n}\n",
        ),
        "class A {}\n",
        "class A {\n  void f( {\n}\n",
        2,
        "Calc.add",
    ),
    "c": Fixture(
        "Makefile",
        (
            "src/lib.c",
            "int compute(int x) {\n  return x + 1;\n}\n\nstruct calc { int total; };\n\n"
            "int calc_add(struct calc *c, int a) {\n  return compute(a);\n}\n",
        ),
        ("src/app.c", "int compute(int x);\n\nint run(void) {\n  return compute(1);\n}\n"),
        (
            "tests/test_lib.c",
            "int compute(int x);\n\nint main(void) {\n  return compute(1) == 2 ? 0 : 1;\n}\n",
        ),
        "int f(void) { return 1; }\n",
        "int f(void) { return 1; }\nint g( {\n",
        2,
        "lib.calc_add",
    ),
    "cpp": Fixture(
        "CMakeLists.txt",
        (
            "src/lib.cpp",
            "int compute(int x) {\n  return x + 1;\n}\n\nclass Calc {\n public:\n"
            "  int add(int a) {\n    return compute(a);\n  }\n};\n",
        ),
        ("src/app.cpp", "int compute(int x);\n\nint run() {\n  return compute(1);\n}\n"),
        (
            "tests/lib_test.cpp",
            "int compute(int x);\n\nint main() {\n  return compute(1) == 2 ? 0 : 1;\n}\n",
        ),
        "int f() { return 1; }\n",
        "int f() { return 1; }\nint g( {\n",
        2,
        "Calc.add",
    ),
    "csharp": Fixture(
        "global.json",
        (
            "src/Lib.cs",
            "static class Lib {\n  public static int Compute(int x) {\n"
            "    return x + 1;\n  }\n}\n\nclass Calc {\n  int Add(int a) {\n"
            "    return Lib.Compute(a);\n  }\n}\n",
        ),
        ("src/App.cs", "class App {\n  int Run() {\n    return Lib.Compute(1);\n  }\n}\n"),
        (
            "tests/LibTests.cs",
            "class LibTests {\n  void Computes() {\n"
            "    System.Diagnostics.Debug.Assert(Lib.Compute(1) == 2);\n  }\n}\n",
        ),
        "class A {}\n",
        "class A {\n  void F( {\n}\n",
        2,
        "Calc.Add",
    ),
    "ruby": Fixture(
        "Gemfile",
        (
            "lib/calc.rb",
            "def compute(x)\n  x + 1\nend\n\nclass Calc\n  def add(a)\n"
            "    compute(a)\n  end\nend\n",
        ),
        ("lib/app.rb", "def run\n  compute(1)\nend\n"),
        (
            "spec/calc_spec.rb",
            "describe 'compute' do\n  it 'adds one' do\n"
            "    expect(compute(1)).to eq(2)\n  end\nend\n",
        ),
        "def f\n  1\nend\n",
        "def f\n  [1,\nend\n",
        3,
        "Calc.add",
    ),
    "php": Fixture(
        "composer.json",
        (
            "src/Lib.php",
            "<?php\nfunction compute($x) {\n  return $x + 1;\n}\n\nclass Calc {\n"
            "  function add($a) {\n    return compute($a);\n  }\n}\n",
        ),
        ("src/App.php", "<?php\nfunction run() {\n  return compute(1);\n}\n"),
        (
            "tests/LibTest.php",
            "<?php\nclass LibTest {\n  function testCompute() {\n"
            "    assert(compute(1) === 2);\n  }\n}\n",
        ),
        "<?php\nfunction f() { return 1; }\n",
        "<?php\nfunction f() { return 1; }\nfunction g( {\n",
        3,
        "Calc.add",
    ),
    "kotlin": Fixture(
        "build.gradle.kts",
        (
            "src/main/kotlin/Lib.kt",
            "fun compute(x: Int): Int {\n    return x + 1\n}\n\n"
            "class Calc {\n    fun add(a: Int): Int {\n        return compute(a)\n    }\n}\n",
        ),
        ("src/main/kotlin/App.kt", "fun run(): Int {\n    return compute(1)\n}\n"),
        (
            "src/test/kotlin/LibTest.kt",
            "class LibTest {\n    fun computes() {\n        check(compute(1) == 2)\n    }\n}\n",
        ),
        "fun f(): Int = 1\n",
        "fun f(): Int = 1\nfun g( {\n",
        2,
        "Calc.add",
    ),
    "swift": Fixture(
        "Package.swift",
        (
            "Sources/Calc/Lib.swift",
            "func compute(_ x: Int) -> Int {\n  return x + 1\n}\n\n"
            "class Calc {\n  func add(_ a: Int) -> Int {\n    return compute(a)\n  }\n}\n",
        ),
        ("Sources/Calc/App.swift", "func run() -> Int {\n  return compute(1)\n}\n"),
        (
            "Tests/CalcTests/LibTests.swift",
            "class LibTests {\n  func testCompute() {\n"
            "    precondition(compute(1) == 2)\n  }\n}\n",
        ),
        "func f() -> Int { return 1 }\n",
        "func f() -> Int { return 1 }\nfunc g( {\n",
        2,
        "Calc.add",
    ),
    "scala": Fixture(
        "build.sbt",
        (
            "src/main/scala/Lib.scala",
            "object Lib {\n  def compute(x: Int): Int = x + 1\n}\n\n"
            "class Calc {\n  def add(a: Int): Int = Lib.compute(a)\n}\n",
        ),
        ("src/main/scala/App.scala", "object App {\n  def run(): Int = Lib.compute(1)\n}\n"),
        (
            "src/test/scala/LibSpec.scala",
            "class LibSpec {\n  def computes(): Unit =\n    assert(Lib.compute(1) == 2)\n}\n",
        ),
        "object A { def f(): Int = 1 }\n",
        "object A { def f(): Int = 1 }\nobject B { def g( }\n",
        2,
        "Calc.add",
    ),
}

NAMES = sorted(LANGUAGES)


def _compute(name: str) -> str:
    return "Compute" if name == "csharp" else "compute"


def _ext(path: str) -> str:
    return Path(path).suffix


@pytest.fixture(scope="module")
def profiles() -> ProfileSet:
    return ProfileSet.load()


def _profile(profiles: ProfileSet, name: str) -> LanguageProfile:
    profile = profiles.by_name(name)
    assert profile is not None, name
    return profile


def test_the_twelve_ship() -> None:
    assert set(NAMES) | {"python", "javascript"} == set(shipped().names())


# -- the profile -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_each_profile_states_its_check_grammar_and_conventions(
    profiles: ProfileSet, name: str
) -> None:
    profile = _profile(profiles, name)
    assert profile.syntax is not None and profile.syntax.mode == "parse"
    assert set(profile.grammars()) <= GRAMMARS
    path, warning = resolve_conventions(Path("/nowhere"), profile)
    assert warning is None and path is not None
    assert path.read_text().startswith("# ")
    assert profile.deps.manifests and profile.lsp.server is not None
    assert profile.examples.snippet.strip()


@pytest.mark.parametrize("name", NAMES)
def test_each_profile_owns_its_fixtures_and_its_tests_are_tests(
    profiles: ProfileSet, name: str
) -> None:
    fixture = LANGUAGES[name]
    for path, _text in (fixture.lib, fixture.app, fixture.test):
        found = profiles.language_of(path)
        assert found is not None and found.name == name, path
    assert _profile(profiles, name).tests.matches(fixture.test[0])
    assert not _profile(profiles, name).tests.matches(fixture.lib[0])


@pytest.mark.parametrize("name", NAMES)
def test_each_tool_a_profile_runs_is_classed(profiles: ProfileSet, name: str) -> None:
    """Each profile command has its class in the shipped table, before any profile row is
    added: a runner is a test run, an install and a bootstrap a lockfile install."""
    profile = _profile(profiles, name)
    table = command_table()
    expected = (
        (profile.tests.runner, CommandClass.TESTS),
        (profile.build.compile_cmd, CommandClass.BUILD),
        (profile.deps.install_cmd, CommandClass.LOCKFILE_INSTALL),
        (profile.toolchain.bootstrap, CommandClass.LOCKFILE_INSTALL),
    )
    for template, cls in expected:
        if template is None:
            continue
        argv = [a for a in template.argv if "{" not in a]
        found = table.classify(argv)
        assert found is not None and found.cls == cls, (template.source, found)


# -- the syntax check ------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_a_broken_file_fails_with_its_line_and_a_good_one_parses(
    profiles: ProfileSet, tmp_path: Path, name: str
) -> None:
    fixture = LANGUAGES[name]
    checker = SyntaxChecker(tmp_path, profiles.all())
    path = "a" + _ext(fixture.lib[0])
    good = checker.check(path, fixture.good.encode())
    assert good is not None and good.ok and good.parsed_only, good
    broken = checker.check(path, fixture.broken.encode())
    assert broken is not None and not broken.ok
    assert broken.detail is not None and broken.detail.startswith(f"{path}: line ")
    line = int(broken.detail.removeprefix(f"{path}: line ").split(",")[0])
    # Where tree-sitter's recovery puts the error is its own: at the broken line, at the start
    # of the construct it could not close (Ruby's `def`), or, for a missing closer, at the end
    # of the file just after it (Go's `}`).
    assert 1 <= line <= fixture.broken_line + 1, broken.detail


# -- detection -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_init_detects_each_from_a_manifest_and_a_source_file(
    profiles: ProfileSet, tmp_path: Path, name: str
) -> None:
    fixture = LANGUAGES[name]
    (tmp_path / fixture.manifest).write_text("")
    lib, text = fixture.lib
    (tmp_path / lib).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / lib).write_text(text)
    evidence, _gaps = detect_profiles(tmp_path, profiles)
    found = {e.name: e for e in evidence}
    assert name in found, sorted(found)
    assert fixture.manifest in found[name].manifests


@pytest.mark.parametrize("name", NAMES)
def test_a_new_file_of_each_is_source(profiles: ProfileSet, name: str) -> None:
    """Zeus's shell puts back a source file it wrote, judged by every
    installed profile (`zeus/writes.ShellWrites._source` asks `owns`)."""
    path = "brand_new" + _ext(LANGUAGES[name].lib[0])
    assert any(p.owns(path) for p in profiles.all())


# -- the graph -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory, profiles: ProfileSet) -> Iterator[Graph]:
    """One repository holding every language's fixture, each under its own directory."""
    root = tmp_path_factory.mktemp("twelve")
    for name, fixture in LANGUAGES.items():
        (root / name).mkdir()
        (root / name / fixture.manifest).write_text("")
        for path, text in (fixture.lib, fixture.app, fixture.test):
            (root / name / path).parent.mkdir(parents=True, exist_ok=True)
            (root / name / path).write_text(text)
    indexer = Indexer(root, profiles=profiles)
    report = indexer.full_index()
    manifests = tuple(f.manifest for f in LANGUAGES.values())
    assert [g for g in report.gaps if not g.path.endswith(manifests)] == []
    graph = Graph(root, indexer.conn, profiles=profiles)
    yield graph
    indexer.close()


@pytest.mark.parametrize("name", NAMES)
def test_the_graph_answers_the_definition_the_method_and_the_caller(
    indexed: Graph, name: str
) -> None:
    fixture = LANGUAGES[name]
    lib, app = f"{name}/{fixture.lib[0]}", f"{name}/{fixture.app[0]}"
    symbols = {s.qualified_name: s for s in indexed.symbols_in(lib)}
    compute = [q for q in symbols if q.endswith(f".{_compute(name)}")]
    assert compute, sorted(symbols)
    assert any(q.endswith(fixture.method) for q in symbols), sorted(symbols)
    callers = {e.file for q in compute for e in indexed.callers(q)}
    assert app in callers, (compute, callers)


@pytest.mark.parametrize("name", NAMES)
def test_tests_for_names_the_test_that_calls_it(
    indexed: Graph, profiles: ProfileSet, name: str
) -> None:
    fixture = LANGUAGES[name]
    lib, test = f"{name}/{fixture.lib[0]}", f"{name}/{fixture.test[0]}"
    selected = select_tests([lib], indexed.file_edges(), _profile(profiles, name))
    assert test in selected.tests, selected


def test_atlas_says_which_languages_are_read_by_tags(
    indexed: Graph, profiles: ProfileSet
) -> None:
    nav = GraphNavigator(indexed, profiles.all())
    atlas = nav.atlas()
    assert isinstance(atlas, str)
    line = next(row for row in atlas.splitlines() if "read by a tags query" in row)
    assert "edges name-resolved and imports not read" in line
    for name, fixture in LANGUAGES.items():
        if name == "typescript":
            assert "typescript (" not in line, "TypeScript is read by its walker"
            continue
        # A manifest written in the language is one of its files: Package.swift,
        # build.gradle.kts.
        files = 3 + (profiles.language_of(fixture.manifest) is not None)
        assert f"{name} ({files} files)" in line, name
