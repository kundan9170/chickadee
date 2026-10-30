"""Symbol edits on the worker's tools: `replace_symbol` and
`insert_after_symbol`, each naming a declaration resolved on the draft as it now reads,
decorators included, through the box's edit port on a real git worktree (it was the packet's
view)."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import init_repo

from .tools_helpers import EditWorker, refused, said, worktree_edits

APP = (
    "import functools\n"
    "\n"
    "\n"
    "@functools.cache\n"
    "def build_widget(name):\n"
    "    return name\n"
    "\n"
    "\n"
    "class Widget:\n"
    "    @property\n"
    "    def render(self):\n"
    "        return 1\n"
    "\n"
    "\n"
    "class Other:\n"
    "    def render(self):\n"
    "        return 2\n"
)
SYMBOL_TOOLS = ("edit", "replace_symbol", "insert_after_symbol")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    files = {"app.py": APP, "notes.txt": "a note\n", "ui.js": "const a = 1; function f() {}\n"}
    return init_repo(tmp_path / "repo", files)


@pytest.fixture
def run(root: Path) -> EditWorker:
    return EditWorker(worktree_edits(root), SYMBOL_TOOLS)


def _text(root: Path, path: str = "app.py") -> str:
    return (root / path).read_text(encoding="utf-8")


def test_a_decorated_function_is_replaced_without_duplicating_its_decorators(
    run: EditWorker, root: Path
) -> None:
    new = "@functools.lru_cache\ndef build_widget(name):\n    return name.upper()\n"
    answer = said(run.call("replace_symbol", path="app.py", symbol="build_widget", content=new))
    assert answer == "ok: app.py\n" + new.rstrip("\n")
    text = _text(root)
    assert "@functools.cache" not in text and text.count("@functools.lru_cache") == 1
    assert text == APP.replace("@functools.cache\n", "@functools.lru_cache\n").replace(
        "return name\n", "return name.upper()\n"
    )


def test_a_symbol_edited_earlier_is_found_at_its_new_span(run: EditWorker, root: Path) -> None:
    grown = (
        "@functools.cache\ndef build_widget(name):\n    name = name.strip()\n    return name\n"
    )
    said(run.call("replace_symbol", path="app.py", symbol="build_widget", content=grown))
    new = "    @property\n    def render(self):\n        return 3\n"
    said(run.call("replace_symbol", path="app.py", symbol="Widget.render", content=new))
    text = _text(root)
    assert "return 3" in text and "return 1" not in text and "return 2" in text
    assert "name = name.strip()" in text


def test_insert_after_a_symbol_follows_its_last_line(run: EditWorker, root: Path) -> None:
    content = "\n    def size(self):\n        return 0"
    said(
        run.call("insert_after_symbol", path="app.py", symbol="Widget.render", content=content)
    )
    assert (
        "        return 1\n\n    def size(self):\n        return 0\n\n\nclass Other:"
        in _text(root)
    )


def test_an_ambiguous_name_names_each_and_the_qualified_one_resolves(run: EditWorker) -> None:
    reason = refused(
        run.call("replace_symbol", path="app.py", symbol="render", content="x = 1"),
        "AMBIGUOUS",
    )
    assert "app.Widget.render (line 10)" in reason and "app.Other.render (line 16)" in reason
    said(run.call("replace_symbol", path="app.py", symbol="Other.render", content="    pass"))


def test_a_name_naming_nothing_names_the_files_symbols(run: EditWorker) -> None:
    reason = refused(
        run.call("replace_symbol", path="app.py", symbol="missing", content="x = 1"),
        "NO_MATCH",
    )
    assert reason.startswith("NO_MATCH: app.py: no symbol 'missing'")
    assert "app.build_widget" in reason and "app.Widget.render" in reason


def test_a_language_with_no_parser_is_unsupported_per_call(run: EditWorker) -> None:
    reason = refused(
        run.call("replace_symbol", path="notes.txt", symbol="a", content="b"),
        "UNSUPPORTED",
    )
    assert "notes.txt" in reason
    # The file stays editable by text.
    said(run.call("edit", path="notes.txt", old="a note", new="b note"))


def test_a_declaration_sharing_a_line_is_refused_and_the_file_kept(
    run: EditWorker, root: Path
) -> None:
    reason = refused(
        run.call("replace_symbol", path="ui.js", symbol="f", content="function f() {}"),
        "UNSUPPORTED",
    )
    assert "'const a = 1;' before it" in reason
    assert _text(root, "ui.js") == "const a = 1; function f() {}\n"
