"""BL-51 leftovers: search rank, sig_of P-6, parent_id, contracts, spawn policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.graph import fts
from chickadee.graph.indexer import Indexer
from chickadee.graph.parse import parse_source
from chickadee.graph.queries import Graph


def test_sig_of_ambiguous_bare_name_is_none(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def parse():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def parse():\n    return 2\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    graph = Graph(root, ix.conn)
    assert graph.sig_of("parse") is None
    assert graph.doc_first_line("parse") is None
    assert graph.sig_of("a.parse") == "parse()"
    assert graph.sig_of("b.parse") == "parse()"


def test_sig_of_unique_bare_name_still_resolves(indexed) -> None:
    assert indexed.sig_of("helper") == "helper(x: int) -> str"
    assert indexed.doc_first_line("helper") == "Help."


def test_parent_id_points_at_the_class(indexed) -> None:
    method = indexed.conn.execute(
        "SELECT parent_id FROM symbols WHERE qualified_name = ?",
        ("app.Widget.render",),
    ).fetchone()
    assert method is not None and method["parent_id"] is not None
    parent = indexed.conn.execute(
        "SELECT qualified_name FROM symbols WHERE id = ?",
        (method["parent_id"],),
    ).fetchone()
    assert parent["qualified_name"] == "app.Widget"
    free = indexed.conn.execute(
        "SELECT parent_id FROM symbols WHERE qualified_name = ?",
        ("app.build_widget",),
    ).fetchone()
    assert free["parent_id"] is None


def test_public_contracts_exclude_methods(indexed) -> None:
    names = {s.name for s in indexed.public_contracts(["app.py"])}
    assert "build_widget" in names
    assert "Widget" in names
    assert "render" not in names
    assert "setup" not in names


def test_dunder_all_is_the_public_list(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "api.py").write_text(
        '__all__ = ["shown"]\n\ndef shown():\n    return 1\n\ndef hidden():\n    return 2\n',
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    names = {s.name for s in Graph(root, ix.conn).public_contracts(["api.py"])}
    assert names == {"shown"}


def test_computed_dunder_all_clears_earlier_literal(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "api.py").write_text(
        '__all__ = ["shown"]\n'
        '__all__ = [n for n in ["shown", "hidden"]]\n'
        "def shown():\n    return 1\n"
        "def hidden():\n    return 2\n",
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    names = {s.name for s in Graph(root, ix.conn).public_contracts(["api.py"])}
    assert names == {"shown", "hidden"}


def test_js_non_exported_function_is_not_public() -> None:
    src = b"function helper(){ return 1; }\nexport function shown(){ return 2; }\n"
    parsed = parse_source("mod.js", src, "javascript", "javascript")
    by_name = {s.name: s.exported for s in parsed.symbols}
    assert by_name["helper"] is False
    assert by_name["shown"] is True


def test_search_ranks_bm25_before_recency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee.graph import queries as queries_mod

    root = tmp_path / "r"
    root.mkdir()
    (root / "old.py").write_text("def refresh_session():\n    return 1\n", encoding="utf-8")
    (root / "new.py").write_text("def session_adjacent():\n    return 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    monkeypatch.setattr(queries_mod, "git_recency", lambda _repo: {"new.py": 0, "old.py": 99})
    graph = Graph(root, ix.conn)
    ranked = [s.qualified_name for s in graph.search("refresh_session")]
    assert ranked[0] == "old.refresh_session"


def test_search_recency_breaks_equal_bm25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee.graph import queries as queries_mod

    root = tmp_path / "r"
    root.mkdir()
    (root / "old.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "new.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    graph = Graph(root, ix.conn)
    by_file = {
        str(r["file"]): int(r["id"]) for r in graph.conn.execute("SELECT id, file FROM symbols")
    }
    # Equal BM25, FTS order prefers old.py. Recency must still put new.py first.
    monkeypatch.setattr(
        fts,
        "search",
        lambda *_args, **_kwargs: (
            (by_file["old.py"], 0.0),
            (by_file["new.py"], 0.0),
        ),
    )
    monkeypatch.setattr(queries_mod, "git_recency", lambda _repo: {"new.py": 0, "old.py": 99})
    ranked = [s.file for s in graph.search("helper")]
    assert ranked[0] == "new.py"


def test_search_limit_applies_after_recency_tie_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee.graph import queries as queries_mod

    root = tmp_path / "r"
    root.mkdir()
    (root / "old.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "new.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    graph = Graph(root, ix.conn)
    by_file = {
        str(r["file"]): int(r["id"]) for r in graph.conn.execute("SELECT id, file FROM symbols")
    }

    def fake_search(
        _conn: object, _query: str, limit: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        assert limit is None
        return (
            (by_file["old.py"], 0.0),
            (by_file["new.py"], 0.0),
        )

    monkeypatch.setattr(fts, "search", fake_search)
    monkeypatch.setattr(queries_mod, "git_recency", lambda _repo: {"new.py": 0, "old.py": 99})
    ranked = graph.search("helper", limit=1)
    assert len(ranked) == 1
    assert ranked[0].file == "new.py"


def test_declared_version_from_package_json(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "package.json").write_text(
        '{"dependencies": {"react": "^18.2.0"}}\n', encoding="utf-8"
    )
    (root / "a.js").write_text(
        'import {useState} from "react";\nexport function go(){ return 1; }\n',
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    contracts = Graph(root, ix.conn).dependency_contracts(["a.js"])
    react = next(c for c in contracts if c.dependency == "react")
    assert react.declared_version == "^18.2.0"


def test_declared_version_from_pyproject(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["django>=4.2"]\n',
        encoding="utf-8",
    )
    (root / "a.py").write_text(
        "from django.db import models\n\ndef go():\n    return 1\n", encoding="utf-8"
    )
    ix = Indexer(root)
    ix.full_index()
    contracts = Graph(root, ix.conn).dependency_contracts(["a.py"])
    hit = next(c for c in contracts if c.dependency.startswith("django"))
    assert hit.declared_version == ">=4.2"


def test_declared_version_follows_source_language(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "package.json").write_text(
        '{"dependencies": {"demo": "^1.0.0"}}\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["demo>=2.0"]\n',
        encoding="utf-8",
    )
    (root / "a.py").write_text(
        "from demo import x\n\ndef go():\n    return 1\n", encoding="utf-8"
    )
    (root / "b.js").write_text(
        'import {x} from "demo";\nexport function go(){ return 1; }\n',
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    graph = Graph(root, ix.conn)
    py = next(c for c in graph.dependency_contracts(["a.py"]) if c.dependency == "demo")
    js = next(c for c in graph.dependency_contracts(["b.js"]) if c.dependency == "demo")
    assert py.declared_version == ">=2.0"
    assert js.declared_version == "^1.0.0"


def test_python_source_ignores_package_json(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "package.json").write_text(
        '{"dependencies": {"demo": "^1.0.0"}}\n', encoding="utf-8"
    )
    (root / "a.py").write_text(
        "from demo import x\n\ndef go():\n    return 1\n", encoding="utf-8"
    )
    ix = Indexer(root)
    ix.full_index()
    hit = next(
        c for c in Graph(root, ix.conn).dependency_contracts(["a.py"]) if c.dependency == "demo"
    )
    assert hit.declared_version is None


def test_nearest_manifest_without_pin_does_not_inherit(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "root"\nversion = "0"\ndependencies = ["demo>=2.0"]\n',
        encoding="utf-8",
    )
    pkg = root / "packages" / "app"
    pkg.mkdir(parents=True)
    (pkg / "pyproject.toml").write_text(
        '[project]\nname = "app"\nversion = "0"\ndependencies = []\n',
        encoding="utf-8",
    )
    (pkg / "a.py").write_text(
        "from demo import x\n\ndef go():\n    return 1\n", encoding="utf-8"
    )
    ix = Indexer(root)
    ix.full_index()
    hit = next(
        c
        for c in Graph(root, ix.conn).dependency_contracts(["packages/app/a.py"])
        if c.dependency == "demo"
    )
    assert hit.declared_version is None


def test_invalid_utf8_manifest_is_not_a_version(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "package.json").write_bytes(b"\xff\xfe not json")
    (root / "a.js").write_text(
        'import {x} from "demo";\nexport function go(){ return 1; }\n',
        encoding="utf-8",
    )
    ix = Indexer(root)
    ix.full_index()
    hit = next(
        c for c in Graph(root, ix.conn).dependency_contracts(["a.js"]) if c.dependency == "demo"
    )
    assert hit.declared_version is None


def test_undeclared_import_has_no_version(indexed) -> None:
    contracts = indexed.dependency_contracts(["app.py"])
    assert all(c.declared_version is None for c in contracts)
