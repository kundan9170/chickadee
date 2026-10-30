"""`init` asks the profile set which language a file is: detection
counts a file toward the profile that will read it, as the index does, and a grammar is missing
when the code graph cannot parse with it."""

from __future__ import annotations

import pathlib

from chickadee.graph.grammars import GRAMMARS
from chickadee.init import detect, toolchain
from chickadee.profiles import ProfileSet


def _profile(builtin: pathlib.Path, name: str, body: str) -> None:
    (builtin / f"{name}.toml").write_text(f'name = "{name}"\n{body}')


def test_a_profile_naming_a_grammar_the_graph_lacks_is_missing(tmp_path: pathlib.Path) -> None:
    builtin = tmp_path / "profiles"
    builtin.mkdir()
    _profile(
        builtin,
        "python",
        'extensions = [".py"]\ntree_sitter_grammar = "python"\n[syntax]\nmode = "inproc"\n',
    )
    _profile(
        builtin,
        "web",
        'extensions = [".ts", ".vue"]\n'
        'tree_sitter_grammar = { ".ts" = "typescript", ".vue" = "vue" }\n'
        '[syntax]\nmode = "parse"\n',
    )
    assert "vue" not in GRAMMARS
    ps = ProfileSet.load(builtin_dir=builtin)
    assert toolchain.missing_grammars(ps.names(), ps) == ("web",)
    assert toolchain.missing_grammars(("python",), ps) == ()


def test_a_file_counts_toward_the_profile_that_reads_it(tmp_path: pathlib.Path) -> None:
    """A `.h` two profiles claim, with no manifest to pick one, is a gap here as in the index;
    the profile a manifest picks counts it."""
    builtin = tmp_path / "profiles"
    builtin.mkdir()
    for name, manifest in (("c", "Makefile"), ("cpp", "CMakeLists.txt")):
        _profile(
            builtin,
            name,
            f'extensions = [".h"]\ntree_sitter_grammar = "{name}"\n'
            '[syntax]\nmode = "parse"\n'
            f'[deps]\nmanifests = ["{manifest}"]\n',
        )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.h").write_text("int f(void);\n")

    evidence, gaps = detect.detect_profiles(repo, ProfileSet.load(repo, builtin))
    assert evidence == ()
    assert gaps == ("a.h",)

    (repo / "CMakeLists.txt").write_text("")
    evidence, gaps = detect.detect_profiles(repo, ProfileSet.load(repo, builtin))
    assert [(e.name, e.file_count) for e in evidence] == [("cpp", 1)]
    assert gaps == ()
