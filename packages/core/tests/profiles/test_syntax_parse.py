"""The `parse` syntax mode and the mode with no default.

A profile's check is its grammar's parse: no toolchain, no spawn, and a pass is said `parsed`,
never more. A profile that does not state its mode is refused at load, naming the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.box.facts import render_facts
from chickadee.box.syntax import SyntaxChecker
from chickadee.edits.draft import CandidateDraft
from chickadee.edits.formats.base import Targets
from chickadee.edits.lines import decode
from chickadee.graph.grammars import GRAMMARS, first_error
from chickadee.profiles import LanguageProfile, ProfileError, load_profile
from chickadee.schema.journal import SyntaxFactP, TaskFactsP

GO = LanguageProfile.model_validate(
    {
        "name": "go",
        "extensions": [".go"],
        "tree_sitter_grammar": "go",
        "syntax": {"mode": "parse"},
    }
)

GOOD_GO = b"package main\n\nfunc main() {\n\tprintln(1)\n}\n"
BROKEN_GO = b"package main\n\nfunc main() {\n\tprintln(1\n}\n"


def _toml(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "go.toml").write_text(
        'name = "go"\nextensions = [".go"]\ntree_sitter_grammar = "go"\n' + body
    )
    return d


# -- the mode has no default -----------------------------------------------------------------


def test_a_profile_with_no_syntax_table_is_refused_naming_its_file(tmp_path: Path) -> None:
    builtin = _toml(tmp_path, "")
    with pytest.raises(ProfileError, match=r"go\.toml: \[syntax\]\.mode must be stated"):
        load_profile("go", builtin_dir=builtin)


def test_a_syntax_table_with_no_mode_is_refused_naming_its_file(tmp_path: Path) -> None:
    builtin = _toml(tmp_path, '[syntax]\ncmd = "go vet {file}"\n')
    with pytest.raises(ProfileError, match=r"(?s)go\.toml.*syntax\.mode"):
        load_profile("go", builtin_dir=builtin)


def test_parse_takes_no_command(tmp_path: Path) -> None:
    builtin = _toml(tmp_path, '[syntax]\nmode = "parse"\ncmd = "go vet {file}"\n')
    with pytest.raises(ProfileError, match=r'"parse" must not carry syntax\.cmd'):
        load_profile("go", builtin_dir=builtin)


def test_the_shipped_profiles_state_their_modes_unchanged() -> None:
    python, javascript = load_profile("python"), load_profile("javascript")
    assert python.syntax is not None and python.syntax.mode == "inproc"
    assert javascript.syntax is not None and javascript.syntax.mode == "cmd"


def test_a_profile_built_in_code_with_no_check_judges_nothing(tmp_path: Path) -> None:
    """No default mode: a profile with no `[syntax]` is never judged as Python."""
    bare = LanguageProfile.model_validate(
        {"name": "go", "extensions": [".go"], "tree_sitter_grammar": "go"}
    )
    assert bare.syntax is None
    assert SyntaxChecker(tmp_path, (bare,)).check("a.go", BROKEN_GO) is None


# -- the parse check -------------------------------------------------------------------------


def test_a_broken_file_fails_with_its_line_and_column(tmp_path: Path) -> None:
    verdict = SyntaxChecker(tmp_path, (GO,)).check("main.go", BROKEN_GO)
    assert verdict is not None and not verdict.ok
    assert verdict.detail is not None and verdict.detail.startswith("main.go: line ")
    line = int(verdict.detail.removeprefix("main.go: line ").split(",")[0])
    assert 4 <= line <= 5, verdict.detail
    assert ", column " in verdict.detail


def test_a_good_file_passes_as_parsed_only(tmp_path: Path) -> None:
    verdict = SyntaxChecker(tmp_path, (GO,)).check("main.go", GOOD_GO)
    assert verdict is not None and verdict.ok and verdict.parsed_only


def test_parse_never_spawns(tmp_path: Path) -> None:
    def spawn(*_args: object) -> object:
        raise AssertionError("a parse check spawned")

    checker = SyntaxChecker(tmp_path, (GO,), spawn)  # type: ignore[arg-type]
    assert checker.on_disk("main.go") is None
    assert checker.inproc("main.go", BROKEN_GO) is not None


def test_a_grammar_the_harness_lacks_is_unchecked_not_a_crash(tmp_path: Path) -> None:
    odd = GO.model_copy(update={"tree_sitter_grammar": ((".go", "golang"),)})
    assert SyntaxChecker(tmp_path, (odd,)).check("main.go", BROKEN_GO) is None


def test_the_edit_tool_says_parsed_never_syntax_ok() -> None:
    files = {"main.go": GOOD_GO.decode()}
    checker = SyntaxChecker(Path("/nowhere"), (GO,))
    draft = CandidateDraft(
        Targets(writable=("main.go",)),
        lambda p: decode(files[p]) if p in files else None,
        lambda _: None,
        judge=checker.inproc,
        edit_check="reject",
    )
    with draft.call() as call:
        assert call.apply("main.go", None, GOOD_GO.decode().replace("1", "2")) is None
        said = call.answer()
    assert isinstance(said, str)
    assert said.endswith("parsed: main.go")
    assert "syntax ok" not in said and "compiled" not in said


def test_the_facts_say_parsed_in_apart_from_ok_in() -> None:
    facts = TaskFactsP(
        start_tree="a" * 40,
        end_tree="b" * 40,
        syntax=(
            SyntaxFactP(path="app.py", ok=True, depth="compiled"),
            SyntaxFactP(path="main.go", ok=True, depth="parsed"),
        ),
    )
    rendered = render_facts(facts, "finished")
    assert "  syntax: ok in app.py; parsed in main.go" in rendered


def test_each_check_records_how_far_it_went(tmp_path: Path) -> None:
    """#606: the depth is a typed field of the fact, never a word in its detail."""
    from chickadee.box.facts import _syntax  # pyright: ignore[reportPrivateUsage]

    (tmp_path / "main.go").write_bytes(GOOD_GO)
    (tmp_path / "app.py").write_text("x = 1\n")
    checker = SyntaxChecker(tmp_path, (GO, load_profile("python")))
    facts = {f.path: f for f in _syntax(tmp_path, ["main.go", "app.py"], checker)}
    assert (facts["main.go"].depth, facts["main.go"].detail) == ("parsed", None)
    assert facts["app.py"].depth == "compiled"


@pytest.mark.parametrize("grammar", sorted(GRAMMARS))
def test_every_shipped_grammar_loads_and_finds_an_error(grammar: str) -> None:
    """Each grammar parses an empty file cleanly and calls an unbalanced bracket an error."""
    assert first_error(grammar, b"") is None
    broken = b"<?php\n(((\n" if grammar == "php" else b"(((\n"
    error = first_error(grammar, broken)
    assert error is not None and error.line >= 1 and error.column >= 1


# -- the review of #605 ----------------------------------------------------------------------


def test_valid_kotlin_the_grammar_flags_only_by_a_hidden_token_parses() -> None:
    assert first_error("kotlin", b"class A : B() { fun m() { foo() } }\n") is None


C_GUARD = (
    '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
    "int f(void);\n\n#ifdef __cplusplus\n}\n#endif\n"
)


def test_a_parse_failure_is_reported_never_refused() -> None:
    """A grammar's parse can fail where the compiler accepts the file (C's `#ifdef` around a
    brace): the call stands and says so, even under `reject`."""
    assert first_error("c", C_GUARD.encode()) is not None, "the grammar's own limit"
    c = LanguageProfile.model_validate(
        {
            "name": "c",
            "extensions": [".h"],
            "tree_sitter_grammar": "c",
            "syntax": {"mode": "parse"},
        }
    )
    checker = SyntaxChecker(Path("/nowhere"), (c,))
    draft = CandidateDraft(
        Targets(writable=("api.h",), creates=frozenset({"api.h"})),
        lambda _p: None,
        lambda _: None,
        judge=checker.inproc,
        edit_check="reject",
    )
    with draft.call() as call:
        assert call.apply("api.h", None, C_GUARD) is None
        said = call.answer()
    assert isinstance(said, str) and "parse error: api.h: line " in said


def test_a_parse_error_names_its_column_in_characters() -> None:
    error = first_error("go", 'package a\n\nvar s = "é" +\n'.encode())
    assert error is not None and error.line >= 3
