"""A grammar per extension, and the profile set as the one home for a file's language."""

from __future__ import annotations

import pathlib

import pytest

from chickadee.profiles import LanguageProfile, ProfileError, ProfileSet, load_profile

TS_TOML = """
name = "typescript"
extensions = [".ts", ".tsx"]
tree_sitter_grammar = { ".ts" = "typescript", ".tsx" = "tsx" }

[syntax]
mode = "parse"
"""


def _typescript(builtin: pathlib.Path, toml: str = TS_TOML) -> None:
    (builtin / "typescript.toml").write_text(toml)


def test_one_grammar_names_every_extension(builtin: pathlib.Path) -> None:
    p = load_profile("python", builtin_dir=builtin)
    assert p.grammar_for("a.py") == "python"
    assert p.grammar_for("README.md") is None
    assert p.grammars() == ("python",)


def test_a_table_names_a_grammar_per_extension(builtin: pathlib.Path) -> None:
    _typescript(builtin)
    p = load_profile("typescript", builtin_dir=builtin)
    assert p.grammar_for("src/a.ts") == "typescript"
    assert p.grammar_for("src/view.tsx") == "tsx"
    assert p.grammars() == ("typescript", "tsx")


def test_a_table_round_trips_and_the_profile_stays_a_value(builtin: pathlib.Path) -> None:
    _typescript(builtin)
    p = load_profile("typescript", builtin_dir=builtin)
    revived = LanguageProfile.model_validate(p.model_dump())
    assert revived == p
    # Hashable, as a dict field would not be: callers dedupe profiles with `dict.fromkeys`.
    assert hash(revived) == hash(p)


@pytest.mark.parametrize(
    ("table", "said"),
    [
        ('{ ".ts" = "typescript" }', r"none for \['\.tsx'\]"),
        (
            '{ ".ts" = "typescript", ".tsx" = "tsx", ".js" = "javascript" }',
            r"\['\.js'\] are not its extensions",
        ),
        ('{ ".ts" = "typescript", ".tsx" = "" }', r"empty for \['\.tsx'\]"),
        ('""', r"empty for"),
    ],
)
def test_a_grammar_table_that_misses_or_overreaches_is_refused(
    builtin: pathlib.Path, table: str, said: str
) -> None:
    grammar = TS_TOML.splitlines()[3]
    _typescript(builtin, TS_TOML.replace(grammar, f"tree_sitter_grammar = {table}"))
    with pytest.raises(ProfileError, match=said):
        load_profile("typescript", builtin_dir=builtin)


def test_an_override_changes_one_extensions_grammar(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    """PR-2: tables deep-merge, so a repository renames one grammar and keeps the other."""
    _typescript(builtin)
    (repo / ".chickadee" / "profiles").mkdir(parents=True)
    (repo / ".chickadee" / "profiles" / "typescript.toml").write_text(
        'tree_sitter_grammar = { ".tsx" = "typescript" }\n'
    )
    p = load_profile("typescript", repo=repo, builtin_dir=builtin)
    assert p.grammar_for("a.ts") == "typescript"
    assert p.grammar_for("a.tsx") == "typescript"


def test_the_set_answers_a_paths_profile_and_grammar(builtin: pathlib.Path) -> None:
    _typescript(builtin)
    s = ProfileSet.load(builtin_dir=builtin)
    found = s.language_of("web/src/view.tsx")
    assert found is not None
    assert (found.name, found.grammar) == ("typescript", "tsx")
    assert s.language_of("README.md") is None


def test_two_profiles_claiming_one_extension_are_a_gap_until_a_manifest_picks(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    """PR-4 through the set: ambiguous is None, never the first one found."""
    for name, manifest in (("cpp", "CMakeLists.txt"), ("c", "Makefile")):
        (builtin / f"{name}.toml").write_text(
            f'name = "{name}"\nextensions = [".h"]\ntree_sitter_grammar = "{name}"\n'
            '[syntax]\nmode = "parse"\n'
            f'[deps]\nmanifests = ["{manifest}"]\n'
        )
    assert ProfileSet.load(repo=repo, builtin_dir=builtin).language_of("a.h") is None
    (repo / "CMakeLists.txt").write_text("")
    found = ProfileSet.load(repo=repo, builtin_dir=builtin).language_of("a.h")
    assert found is not None and (found.name, found.grammar) == ("cpp", "cpp")


def test_a_set_of_loaded_profiles_answers_as_a_loaded_set(builtin: pathlib.Path) -> None:
    _typescript(builtin)
    loaded = ProfileSet.load(builtin_dir=builtin)
    held = ProfileSet.of(loaded.all())
    for path in ("a.py", "a.ts", "a.tsx", "a.md"):
        assert held.language_of(path) == loaded.language_of(path)
