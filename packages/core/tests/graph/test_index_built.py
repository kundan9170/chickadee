"""`init` builds the code graph.

Before this, init wrote a placeholder manifest and every graph query in production answered
`INDEX_UNAVAILABLE`. Each test reads the graph back through `Graph`, as a query would, rather
than counting rows: the promise is that a query answers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.graph import indexer as indexer_mod
from chickadee.graph import store
from chickadee.graph.parse import ParsedFile, parse_file
from chickadee.graph.queries import Graph
from chickadee.init import InitError, init, publish
from chickadee.profiles import FileLanguage

APP = "def go() -> int:\n    return 1\n"
LIB = "def helper() -> int:\n    return 2\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "lib.py": LIB})


@contextmanager
def graph(repo: Path) -> Iterator[Graph]:
    g = Graph(repo)
    try:
        yield g
    finally:
        g.close()


def test_init_leaves_a_graph_a_query_reads(repo: Path) -> None:
    assert not store.index_exists(repo)
    report = init(repo)
    assert store.index_exists(repo)
    assert report.index.files_indexed == 2
    assert report.index.symbols >= 2
    with graph(repo) as g:
        assert g.symbol_exists("go")
        assert g.file_known("lib.py")


def test_the_placeholder_manifest_is_gone(repo: Path) -> None:
    report = init(repo)
    published = publish.chickadee_dir(repo) / publish.GENERATIONS_DIR / report.generation
    assert not (published / "index.json").exists()


@pytest.mark.writes_worktree("lib.py")
def test_a_reinit_rebuilds_from_scratch(repo: Path) -> None:
    """The index is derived, so a re-run rebuilds it; a file gone from disk is gone."""
    init(repo)
    (repo / "lib.py").unlink()
    report = init(repo)
    assert report.index.files_indexed == 1
    with graph(repo) as g:
        assert not g.symbol_exists("helper")
        assert g.symbol_exists("go")


@pytest.mark.writes_worktree("app.py")
def test_a_crash_mid_build_leaves_the_previous_index_answering(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build is one transaction: a fault after some rows are written rolls them back,
    and the previous generation stays published."""
    first = init(repo)
    (repo / "app.py").write_text(APP + "\n\ndef later() -> int:\n    return 3\n")

    real = parse_file
    calls: list[Path] = []

    def parse_then_fail(root: Path, path: Path, language: FileLanguage) -> ParsedFile:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("injected fault mid-build")
        return real(root, path, language)

    monkeypatch.setattr(indexer_mod, "parse_file", parse_then_fail)
    with pytest.raises(OSError, match="mid-build"):
        init(repo, force=True)

    state = publish.read_state(repo)
    assert state is not None and state["generation"] == first.generation
    with graph(repo) as g:
        assert g.symbol_exists("go")
        assert g.symbol_exists("helper"), "the rows deleted before the fault came back"
        assert not g.symbol_exists("later"), "a row written before the fault was kept"


def test_a_file_the_repository_marks_secret_is_never_parsed(tmp_path: Path) -> None:
    """D154, the owner's call on `T-3.108`: `chickadee init` runs before any engine config,
    so it reads the repository's own additions, with its removals inert."""
    repo = init_repo(
        tmp_path / "repo",
        {
            "app.py": APP,
            "vault/keys.py": "def secret() -> str:\n    return 'x'\n",
            ".chickadee/config.toml": '[secrets]\ndeny_add = ["vault/**"]\n',
        },
    )
    report = init(repo)
    assert "vault/keys.py" in {gap.path for gap in report.index.gaps}
    with graph(repo) as g:
        assert not g.symbol_exists("secret")
        assert not g.file_known("vault/keys.py")


@pytest.mark.writes_worktree(".chickadee/config.toml")
def test_an_unreadable_config_refuses_init_rather_than_index_blind(repo: Path) -> None:
    """The deny list cannot be known, so nothing is indexed and nothing is published."""
    config = repo / ".chickadee" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[secrets\n")
    with pytest.raises(InitError, match="deny list could not be read"):
        init(repo)
    assert not store.index_exists(repo)
    assert publish.read_state(repo) is None


def test_a_file_nothing_filled_is_not_an_index(repo: Path) -> None:
    """`index_exists` means a committed pass: opening the graph creates its file, and a
    reindex over that file would silently be the full build `init` announces."""
    with graph(repo):
        pass
    assert not store.index_exists(repo)


def test_a_reader_opens_the_graph_while_a_build_is_in_progress(repo: Path) -> None:
    """WAL's promise, kept by `connect` taking no write lock on a current database: a
    reader neither waits for the build nor sees it before its commit."""
    init(repo)
    builder = store.connect(repo)
    try:
        builder.execute("DELETE FROM files")  # the build's transaction, still open
        reader = store.connect(repo)
        reader.execute("PRAGMA busy_timeout=0")  # waiting would be the failure
        try:
            assert Graph(repo, reader).symbol_exists("go")
        finally:
            reader.close()
    finally:
        builder.rollback()
        builder.close()
