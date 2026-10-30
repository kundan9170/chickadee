"""What T-3.164's review found, each fixed and pinned: one answer for a file's language across
the index, the edit port and the reader."""

from __future__ import annotations

from pathlib import Path

from chickadee.box.syntax import SyntaxChecker
from chickadee.context.outline import sniff
from chickadee.graph import store
from chickadee.graph.indexer import Indexer
from chickadee.graph.parse import parse_file
from chickadee.graph.queries import Graph
from chickadee.profiles import LanguageProfile, ProfileSet, shipped
from chickadee.zeus.tools.graphnav import indexed_profiles


def test_an_index_from_before_the_move_is_rebuilt_not_read(tmp_path: Path) -> None:
    """A version-2 index holds `.ts` rows no profile owns now, and reindex never removes a
    row for a file it skips: the version moved so the index is rebuilt."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.ts").write_text("export function pad(s: string) { return s; }\n")
    old = Indexer(root, profiles=shipped())
    old.full_index()
    old.conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    old.conn.commit()
    old.close()

    # A set with no TypeScript stands in for the one before the move: the `.ts` row is gone.
    before = ProfileSet.of(p for p in shipped().all() if p.name != "typescript")
    ix = Indexer(root, profiles=before)
    ix.reindex()
    assert ix.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert store.SCHEMA_VERSION > 2, "moved past the version before T-3.164"


def test_an_unreadable_file_keeps_its_language(tmp_path: Path) -> None:
    language = shipped().language_of("gone.py")
    assert language is not None
    parsed = parse_file(tmp_path, tmp_path / "gone.py", language)
    assert (parsed.lang, parsed.parse_ok) == ("python", False)
    assert parsed.error is not None and parsed.error.startswith("unreadable")


def _two_claim_h() -> tuple[LanguageProfile, ...]:
    return tuple(
        LanguageProfile.model_validate(
            {
                "name": name,
                "extensions": [".h"],
                "tree_sitter_grammar": name,
                "deps": {"manifests": [manifest]},
            }
        )
        for name, manifest in (("c", "Makefile"), ("cpp", "CMakeLists.txt"))
    )


def test_the_edit_port_breaks_a_tie_as_the_index_does(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("")
    profiles = _two_claim_h()
    indexed = ProfileSet.of(profiles, repo=tmp_path).language_of("a.h")
    checked = SyntaxChecker(tmp_path, profiles).language_of("a.h")
    assert indexed is not None and checked is not None
    assert checked.name == indexed.name == "cpp"


def test_a_preview_reads_the_repositorys_own_profiles(tmp_path: Path) -> None:
    """An override that adds `.pyx` to Python makes it code in the preview, as in the index."""
    overrides = tmp_path / ".chickadee" / "profiles"
    overrides.mkdir(parents=True)
    (overrides / "python.toml").write_text('extensions = [".py", ".pyi", ".pyx"]\n')
    repo_set = ProfileSet.load(repo=tmp_path)
    assert sniff("fast.pyx", b"x = 1\n", repo_set).kind == "python"
    assert sniff("fast.pyx", b"x = 1\n").kind == "plain text"


def test_a_pin_is_read_from_the_manifests_the_profile_names(tmp_path: Path) -> None:
    """No language-to-manifest table in the graph: the TypeScript profile naming
    `package.json` gets its pins read with no change to `queries.py`."""
    profiles = shipped()
    root = tmp_path / "r"
    root.mkdir()
    (root / "package.json").write_text('{"dependencies": {"react": "^18.2.0"}}')
    (root / "a.ts").write_text('import React from "react";\nexport const x = React;\n')
    ix = Indexer(root, profiles=profiles)
    ix.full_index()
    contracts = Graph(root, ix.conn, profiles=profiles).dependency_contracts(["a.ts"])
    assert {c.dependency: c.declared_version for c in contracts} == {"react": "^18.2.0"}


def test_the_checker_judges_by_the_index_s_set_not_the_detected_few(tmp_path: Path) -> None:
    """With no manifest to pick, the index calls `a.h` a gap; a run that detected only `cpp`
    must not judge `a.h` as C++ (the second review of #598)."""
    profiles = _two_claim_h()
    full = ProfileSet.of(profiles, repo=tmp_path)
    assert full.language_of("a.h") is None
    detected = (profiles[1],)
    assert SyntaxChecker(tmp_path, detected, languages=full).language_of("a.h") is None
    # Without the index's set the detected few answer alone, as before.
    alone = SyntaxChecker(tmp_path, detected).language_of("a.h")
    assert alone is not None and alone.name == "cpp"


def test_the_checker_judges_only_the_detected_languages(tmp_path: Path) -> None:
    checker = SyntaxChecker(tmp_path, (), languages=shipped())
    assert checker.language_of("a.py") is None


def test_a_preview_matches_an_extension_as_written_as_the_index_does(tmp_path: Path) -> None:
    """`X.PY` is a gap in the index, so the preview does not label it Python."""
    assert shipped().language_of("X.PY") is None
    assert sniff("X.PY", b"x = 1\n").kind == "plain text"
    assert sniff("x.py", b"x = 1\n").kind == "python"


def test_the_profiles_in_use_are_the_languages_the_index_recorded(tmp_path: Path) -> None:
    """Read off the index's `lang` column, never guessed again by extension."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.ts").write_text("export const x = 1;\n")
    profiles = shipped()
    ix = Indexer(root, profiles=profiles)
    ix.full_index()
    graph = Graph(root, ix.conn, profiles=profiles)
    assert graph.indexed_languages() == (("python", 1), ("typescript", 1))
    assert [p.name for p in indexed_profiles(graph, profiles)] == ["python", "typescript"]
    # A language the installed set no longer holds is not offered.
    without = ProfileSet.of(p for p in profiles.all() if p.name != "typescript")
    assert [p.name for p in indexed_profiles(graph, without)] == ["python"]


def test_a_bare_name_shaped_like_a_source_file_is_a_citation_whatever_its_case() -> None:
    """The review of #605: `.PY` is not Python to a profile, but `Utils.PY:12` is still shaped
    like a file, as it was when a hand-written suffix list was asked case-blind."""
    from chickadee.context.citations import _names_a_file  # pyright: ignore[reportPrivateUsage]
    from chickadee.secrets.denylist import Denylist

    for name in ("Utils.PY", "Main.JAVA", "lib.Rs"):
        assert _names_a_file(name, lambda _p: None, Denylist()), name
    assert not _names_a_file("main.zig", lambda _p: None, Denylist())


def test_a_ruby_or_php_trailing_comment_is_not_code_after_a_method() -> None:
    """The review of #605: symbol edits reach Ruby and PHP now, where `#` starts a comment."""
    from chickadee.edits.symbols import SymbolSpan, find_symbol

    ruby = shipped().language_of("a.rb")
    found = find_symbol("a.rb", "class A\n  def m\n    1\n  end # m\nend\n", "m", ruby)
    assert isinstance(found, SymbolSpan) and (found.first, found.last) == (2, 4)
    php = shipped().language_of("a.php")
    text = "<?php\nfunction m() {\n  return 1;\n} # m\n"
    assert isinstance(find_symbol("a.php", text, "m", php), SymbolSpan)
