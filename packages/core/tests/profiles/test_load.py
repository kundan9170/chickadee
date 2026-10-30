"""Load, round-trip and strict validation."""

from __future__ import annotations

import pathlib

import pytest

from chickadee.profiles import LanguageProfile, ProfileError, ProfileSet, load_profile
from chickadee.profiles.loader import executable_overrides


def test_round_trip(builtin: pathlib.Path) -> None:
    p = load_profile("python", builtin_dir=builtin)
    assert p.name == "python"
    assert p.extensions == (".py", ".pyi")
    assert p.tests.timeout_s == 120
    revived = LanguageProfile.model_validate(p.model_dump())
    assert revived == p


def test_unknown_key_rejects_with_its_path(builtin: pathlib.Path) -> None:
    (builtin / "python.toml").write_text(
        "\n".join(
            [
                'name = "python"',
                'extensions = [".py"]',
                'tree_sitter_grammar = "python"',
                'verify_tier = "deep"',
            ]
        )
    )
    with pytest.raises(ProfileError, match="verify_tier"):
        load_profile("python", builtin_dir=builtin)


def test_missing_required_key_rejects(builtin: pathlib.Path) -> None:
    (builtin / "python.toml").write_text('name = "python"\nextensions = [".py"]\n')
    with pytest.raises(ProfileError, match=r"tree_sitter_grammar"):
        load_profile("python", builtin_dir=builtin)


def test_malformed_toml_rejects(builtin: pathlib.Path) -> None:
    (builtin / "python.toml").write_text("name = [unclosed")
    with pytest.raises(ProfileError, match=r"malformed TOML"):
        load_profile("python", builtin_dir=builtin)


def test_syntax_cmd_mode_requires_a_command(builtin: pathlib.Path) -> None:
    (builtin / "js.toml").write_text(
        'name = "js"\nextensions = [".js"]\ntree_sitter_grammar = "javascript"\n'
        '[syntax]\nmode = "cmd"\n'
    )
    with pytest.raises(ProfileError, match=r"requires syntax\.cmd"):
        load_profile("js", builtin_dir=builtin)


def test_inproc_mode_must_not_carry_a_command(builtin: pathlib.Path) -> None:
    (builtin / "x.toml").write_text(
        'name = "x"\nextensions = [".x"]\ntree_sitter_grammar = "g"\n'
        '[syntax]\nmode = "inproc"\ncmd = "parse {file}"\n'
    )
    with pytest.raises(ProfileError, match="must not carry"):
        load_profile("x", builtin_dir=builtin)


def test_no_extensions_rejects(builtin: pathlib.Path) -> None:
    (builtin / "x.toml").write_text('name = "x"\nextensions = []\ntree_sitter_grammar = "g"\n')
    with pytest.raises(ProfileError, match="no extensions"):
        load_profile("x", builtin_dir=builtin)


def test_extension_without_dot_rejects(builtin: pathlib.Path) -> None:
    (builtin / "x.toml").write_text(
        'name = "x"\nextensions = ["py"]\ntree_sitter_grammar = "g"\n'
    )
    with pytest.raises(ProfileError, match="start with a dot"):
        load_profile("x", builtin_dir=builtin)


def test_empty_compile_cmd_means_no_compile_step(builtin: pathlib.Path) -> None:
    """An empty string is a declaration ("this language has no compile step"), not an
    omission, and must not become an unsplittable template."""
    assert load_profile("python", builtin_dir=builtin).build.compile_cmd is None


def test_unknown_profile_name_raises(builtin: pathlib.Path) -> None:
    with pytest.raises(ProfileError, match="no builtin profile"):
        load_profile("cobol", builtin_dir=builtin)


def test_profile_set_indexes_by_extension(builtin: pathlib.Path) -> None:
    s = ProfileSet.load(builtin_dir=builtin)
    assert s.names() == ("python",)
    assert [p.name for p in s.for_extension(".py")] == ["python"]
    assert s.for_extension(".rs") == []


def test_shipped_profiles_load() -> None:
    """The profiles that actually ship, loaded from their real location."""
    s = ProfileSet.load()
    assert set(s.names()) >= {"python", "javascript"}
    js = s.by_name("javascript")
    assert js is not None and js.syntax is not None
    assert js.syntax.mode == "cmd"
    assert js.syntax.cmd is not None
    assert js.syntax.cmd.argv[0] == "node"


def test_shipped_test_globs_feed_the_lockdown() -> None:
    """The applier's forbidden-target set is this same list, so the two cannot disagree."""
    s = ProfileSet.load()
    py = s.by_name("python")
    assert py is not None
    assert "**/tests/**" in py.tests.globs  # nested test directories too
    assert "conftest.py" in py.tests.globs
    assert py.tests.matches("pkg/tests/helpers.py")


# --- D109 / T-2.30: [toolchain].bootstrap -------------------------------------------------


def _with_toolchain(builtin: pathlib.Path, table: str) -> None:
    """Append a `[toolchain]` table to the fixture's builtin python profile."""
    path = builtin / "python.toml"
    path.write_text(path.read_text() + "\n[toolchain]\n" + table + "\n")


def test_bootstrap_round_trips_as_argv(builtin: pathlib.Path) -> None:
    _with_toolchain(builtin, 'bootstrap = ["uv", "sync", "--frozen"]')
    p = load_profile("python", builtin_dir=builtin)
    assert p.toolchain.bootstrap is not None
    assert p.toolchain.bootstrap.argv == ("uv", "sync", "--frozen")
    assert p.toolchain.bootstrap.placeholders() == frozenset()
    revived = LanguageProfile.model_validate(p.model_dump())
    assert revived == p


def test_a_shell_string_bootstrap_refuses_the_load_naming_the_list_form(
    builtin: pathlib.Path,
) -> None:
    """A command that runs with the network up is never a shell string (the D104
    rule): the field's shape is the list, and a string says so rather than being split."""
    _with_toolchain(builtin, 'bootstrap = "uv sync --frozen && rm -rf /"')
    with pytest.raises(ProfileError, match=r"argv list.*\[\"uv\", \"sync\", \"--frozen\"\]"):
        load_profile("python", builtin_dir=builtin)


def test_an_empty_bootstrap_list_declares_no_bootstrap(builtin: pathlib.Path) -> None:
    _with_toolchain(builtin, "bootstrap = []")
    assert load_profile("python", builtin_dir=builtin).toolchain.bootstrap is None


def test_absent_toolchain_means_no_bootstrap(builtin: pathlib.Path) -> None:
    assert load_profile("python", builtin_dir=builtin).toolchain.bootstrap is None


def test_shipped_profiles_declare_their_bootstrap_and_print_provenance() -> None:
    s = ProfileSet.load()
    py, js = s.by_name("python"), s.by_name("javascript")
    assert py is not None and js is not None
    assert py.toolchain.bootstrap is not None
    assert py.toolchain.bootstrap.argv == ("uv", "sync", "--frozen")
    assert js.toolchain.bootstrap is not None
    assert js.toolchain.bootstrap.argv == ("npm", "ci")
    # The import smoke prints where the module came from, so the checkpoint can
    # compare it to the worktree.
    assert py.build.import_smoke is not None and "__file__" in py.build.import_smoke.source
    assert js.build.import_smoke is not None
    assert "require.resolve" in js.build.import_smoke.source


def test_a_project_override_of_bootstrap_moves_the_trust_digest(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    """A project profile's bootstrap is executable config: the digest a
    person accepts changes when the override does, and the shipped default is not in it."""
    from chickadee.state.trust import executable_subset

    assert executable_overrides(repo, builtin_dir=builtin) == {}
    baseline = executable_subset({"hooks": [], **executable_overrides(repo, builtin)}).digest

    override = repo / ".chickadee" / "profiles"
    override.mkdir(parents=True)
    (override / "python.toml").write_text('[toolchain]\nbootstrap = ["make", "deps"]\n')
    declared = executable_overrides(repo, builtin_dir=builtin)
    assert declared == {"toolchain": {"bootstrap": {"python": ["make", "deps"]}}}
    moved = executable_subset({"hooks": [], **declared}).digest
    assert moved != baseline
    assert "toolchain.bootstrap" in executable_subset({"hooks": [], **declared}).subset

    (override / "python.toml").write_text('[toolchain]\nbootstrap = ["make", "deps", "-j"]\n')
    again = executable_subset({"hooks": [], **executable_overrides(repo, builtin)}).digest
    assert again != moved


def test_an_override_that_declares_no_bootstrap_is_still_a_declaration(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    override = repo / ".chickadee" / "profiles"
    override.mkdir(parents=True)
    (override / "python.toml").write_text("[toolchain]\nbootstrap = []\n")
    assert executable_overrides(repo, builtin_dir=builtin) == {
        "toolchain": {"bootstrap": {"python": []}}
    }


def test_a_string_bootstrap_in_an_override_refuses_at_digest_time_too(
    builtin: pathlib.Path, repo: pathlib.Path
) -> None:
    override = repo / ".chickadee" / "profiles"
    override.mkdir(parents=True)
    (override / "python.toml").write_text('[toolchain]\nbootstrap = "curl x | sh"\n')
    with pytest.raises(ProfileError, match="argv list"):
        executable_overrides(repo, builtin_dir=builtin)
