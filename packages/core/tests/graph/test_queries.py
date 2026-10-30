"""GraphQuery goldens."""

from __future__ import annotations

from pathlib import Path

from chickadee.graph.dependents import python_relative_module, relative_import_file
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph


def test_symbols_in_lists_declarations_in_order(indexed) -> None:
    names = [s.qualified_name for s in indexed.symbols_in("app.py")]
    assert names == [
        "app.Base",
        "app.Base.setup",
        "app.Widget",
        "app.Widget.render",
        "app.build_widget",
    ]


def test_sig_and_doc_first_line(indexed) -> None:
    assert indexed.sig_of("app.Widget.render") == "render(self, x: int) -> str"
    assert indexed.doc_first_line("app.Widget.render") == "Render it."
    assert indexed.sig_of("nope.missing") is None


def test_callers_and_callees(indexed) -> None:
    callees = [e.dst for e in indexed.callees("app.Widget.render")]
    assert "helpers.helper" in callees
    callers = [e.src for e in indexed.callers("helpers.helper")]
    assert "app.Widget.render" in callers


def test_imports(indexed) -> None:
    assert [(i.module, i.alias) for i in indexed.imports("app.py")] == [
        ("helpers.helper", None)
    ]


def test_globals_are_declarations_not_values(indexed) -> None:
    by_name = {g.name: g for g in indexed.globals_in("app.py")}
    assert by_name["MAX_SIZE"].kind == "constant"
    assert by_name["counter"].kind == "module_state"
    assert by_name["counter"].type_text == "int"
    assert not hasattr(by_name["MAX_SIZE"], "value")


def test_resolve_returns_every_candidate(indexed) -> None:
    assert [s.qualified_name for s in indexed.resolve("helper")] == ["helpers.helper"]
    assert indexed.resolve("does_not_exist") == ()


def test_symbol_exists_and_file_known(indexed) -> None:
    assert indexed.symbol_exists("build_widget")
    assert not indexed.symbol_exists("build_widgets")
    assert indexed.file_known("app.py")
    assert not indexed.file_known("ghost.py")


def test_indexed_paths_lists_known_source_files(indexed) -> None:
    paths = indexed.indexed_paths()
    assert "app.py" in paths
    assert "helpers.py" in paths
    assert "tests/test_app.py" in paths
    assert "ghost.py" not in paths
    assert paths == tuple(sorted(paths))
    assert all(indexed.file_known(p) for p in paths)


def test_public_contracts_exclude_private_names(indexed) -> None:
    names = {s.name for s in indexed.public_contracts(["helpers.py"])}
    assert "helper" in names
    assert all(not n.startswith("_") for n in names)


def test_dependency_contracts_report_the_weakest_class(indexed) -> None:
    contracts = {c.dependency: c for c in indexed.dependency_contracts(["app.py"])}
    assert "helpers" in contracts
    assert contracts["helpers"].symbols == ("helper",)
    # The exact class, not membership in the set of all legal classes — that assertion
    # could not fail whatever the implementation did.
    assert contracts["helpers"].edge_class == "fallback"


def test_one_unresolved_import_makes_the_whole_contract_unknown(tmp_path: Path) -> None:
    """Weakest-wins: a contract is only as trustworthy as its least-resolved member."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "dep.py").write_text("def real():\n    return 1\n", encoding="utf-8")
    (root / "use.py").write_text("from dep import real, absent\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    contracts = {c.dependency: c for c in Graph(root, ix.conn).dependency_contracts(["use.py"])}
    assert contracts["dep"].symbols == ("absent", "real")
    assert contracts["dep"].edge_class == "unknown", "one unresolved member downgrades it"


def test_same_basename_in_two_directories_stays_ambiguous(tmp_path: Path) -> None:
    """`a/util.py` and `b/util.py` both produced `util.helper`, which collided in
    `qualified_name` and made D19's ambiguity check see one declaration where there are two.
    """
    root = tmp_path / "r"
    root.mkdir()
    for d in ("a", "b"):
        (root / d).mkdir()
        (root / d / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "use.py").write_text("def go():\n    return helper()\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()

    names = {r[0] for r in ix.conn.execute("SELECT qualified_name FROM symbols")}
    assert {"a.util.helper", "b.util.helper"} <= names
    row = ix.conn.execute("SELECT edge_class FROM edges WHERE file = 'use.py'").fetchone()
    assert row["edge_class"] == "unknown", "two candidates must not resolve to one"


def test_javascript_imports_are_extracted(tmp_path: Path) -> None:
    """ESM imports were routed through the Python extractor and produced no rows at all."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.js").write_text(
        'import {thing, other as o} from "./mod.js";\n'
        'import fallback from "pkg";\n'
        'const pad = require("left-pad");\n',
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    found = {
        (r["module"], r["alias"])
        for r in ix.conn.execute("SELECT module, alias FROM imports WHERE file = 'a.js'")
    }
    assert ("./mod.js.thing", None) in found
    assert ("./mod.js.other", "o") in found
    assert ("pkg", "fallback") in found
    assert ("left-pad", "pad") in found


def test_resolve_near_uses_import_distance(tmp_path: Path) -> None:
    """Folder-prefix sort would pick `src/parse.py` because it shares a directory with the
    caller. Path-lex would pick it too (`src/` < `z_dep/`). Import distance picks the file
    `src/app.py` actually imports.
    """
    root = tmp_path / "r"
    (root / "src").mkdir(parents=True)
    (root / "z_dep").mkdir()
    (root / "z_dep" / "parse.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "src" / "parse.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    (root / "src" / "app.py").write_text(
        "from z_dep.parse import helper\n\n\ndef go():\n    return helper()\n",
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    g = Graph(root, ix.conn)
    names = [s.qualified_name for s in g.resolve("helper", near="src/app.py")]
    assert names[0] == "z_dep.parse.helper"
    assert "src.parse.helper" in names


def test_callers_and_callees_default_to_calls_not_inherits(indexed) -> None:
    inherit_dsts = {e.dst for e in indexed.callees("app.Widget", kind="inherits")}
    assert "app.Base" in inherit_dsts
    call_dsts = {e.dst for e in indexed.callees("app.Widget")}
    assert "app.Base" not in call_dsts
    inherit_srcs = {e.src for e in indexed.callers("app.Base", kind="inherits")}
    assert "app.Widget" in inherit_srcs
    call_srcs = {e.src for e in indexed.callers("app.Base")}
    assert "app.Widget" not in call_srcs
    both = {e.type for e in indexed.callees("app.Widget", kind=None)}
    assert "inherits" in both


def test_relative_import_does_not_truncate_tsx_to_ts() -> None:
    files = {"src/util.tsx", "src/util.ts", "src/app.js"}
    assert relative_import_file("src/app.js", "./util.tsx", files) == "src/util.tsx"
    assert relative_import_file("src/app.js", "./util.tsx.thing", files) == "src/util.tsx"


def test_relative_import_does_not_truncate_jsx_to_js() -> None:
    files = {"src/util.jsx", "src/util.js", "src/app.js"}
    assert relative_import_file("src/app.js", "./util.jsx", files) == "src/util.jsx"


def test_extensionless_relative_import_finds_the_file() -> None:
    files = {"src/util.ts", "src/app.js"}
    assert relative_import_file("src/app.js", "./util", files) == "src/util.ts"
    files = {"src/util/index.ts", "src/app.js"}
    assert relative_import_file("src/app.js", "./util", files) == "src/util/index.ts"


def test_python_relative_import_feeds_resolve_near(tmp_path: Path) -> None:
    """`from.foo import helper` must beat a same-named module that sorts first by path."""
    root = tmp_path / "r"
    (root / "pkg").mkdir(parents=True)
    (root / "other").mkdir()
    (root / "pkg" / "foo.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "other" / "foo.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    (root / "pkg" / "app.py").write_text(
        "from .foo import helper\n\n\ndef go():\n    return helper()\n",
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    g = Graph(root, ix.conn)
    names = [s.qualified_name for s in g.resolve("helper", near="pkg/app.py")]
    assert names == ["pkg.foo.helper", "other.foo.helper"]


def test_python_relative_import_beyond_the_package_is_none() -> None:
    assert python_relative_module("pkg/app.py", "..foo.helper") is None
    assert python_relative_module("pkg/app.py", ".foo.helper") == "pkg.foo.helper"


def test_relative_import_that_escapes_the_root_is_ignored() -> None:
    files = {"src/app.js", "foo.ts"}
    assert relative_import_file("src/app.js", "../../foo", files) is None
    assert relative_import_file("src/app.js", "../../foo.ts", files) is None
