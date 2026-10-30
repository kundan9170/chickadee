"""`[lsp].language_id`: the LSP `languageId` of each extension, one name or a table, loaded as
`tree_sitter_grammar` is, through the one normaliser."""

from __future__ import annotations

import pathlib

import pytest

from chickadee.profiles import ProfileError, load_profile

BASE = (
    'name = "web"\nextensions = [".js", ".jsx"]\ntree_sitter_grammar = "javascript"\n'
    '[syntax]\nmode = "parse"\n'
)


def _web(builtin: pathlib.Path, lsp: str) -> None:
    (builtin / "web.toml").write_text(BASE + "[lsp]\n" + lsp)


def test_absent_it_is_the_profiles_name(builtin: pathlib.Path) -> None:
    _web(builtin, 'server = "srv --stdio"\n')
    p = load_profile("web", builtin_dir=builtin)
    assert (p.language_id_for("a.js"), p.language_id_for("a.jsx")) == ("web", "web")
    assert p.language_id_for("a.py") is None


def test_one_name_or_a_table(builtin: pathlib.Path) -> None:
    _web(builtin, 'language_id = "javascript"\n')
    assert load_profile("web", builtin_dir=builtin).language_id_for("a.jsx") == "javascript"
    _web(builtin, 'language_id = { ".jsx" = "javascriptreact", ".js" = "javascript" }\n')
    p = load_profile("web", builtin_dir=builtin)
    assert (p.language_id_for("a.js"), p.language_id_for("a.jsx")) == (
        "javascript",
        "javascriptreact",
    )
    assert p.lsp.language_id == ((".js", "javascript"), (".jsx", "javascriptreact"))


def test_a_table_that_misses_an_extension_is_refused(builtin: pathlib.Path) -> None:
    _web(builtin, 'language_id = { ".jsx" = "javascriptreact" }\n')
    with pytest.raises(ProfileError, match=r"\[lsp\]\.language_id .*none for \['\.js'\]"):
        load_profile("web", builtin_dir=builtin)


def test_the_shipped_profiles_name_theirs() -> None:
    js = load_profile("javascript")
    assert js.language_id_for("view.jsx") == "javascriptreact"
    assert js.language_id_for("a.mjs") == "javascript"
    assert load_profile("python").language_id_for("a.py") == "python"
