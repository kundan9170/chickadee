"""`tools/foreign.py`: one test per row of OpenHands' table, the inference of
a missing `command`, and a target the role is not offered left out."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from chickadee.paths import Place
from chickadee.tools.builtin import create, edit, finish, glob, insert_lines, read, shell
from chickadee.tools.foreign import ROWS, foreign_argument, ours_for, translate
from chickadee.tools.spec import ToolSpec

WORKER: Mapping[str, ToolSpec] = {
    s.name: s
    for s in (read.SPEC, glob.SPEC, create.SPEC, edit.SPEC, insert_lines.SPEC, shell.SPEC)
}
"""What a worker with a line tool is offered, of the table's targets; `finish` beside it."""
WITH_FINISH = {**WORKER, "finish": finish.SPEC}


def _call(name: str, offered: Mapping[str, ToolSpec] = WORKER, **args: object) -> str | None:
    got = translate(name, args, offered)
    assert got is not None
    return got.call


def test_the_table_is_the_rows() -> None:
    assert [(r.tool, r.command, r.ours) for r in ROWS] == [
        ("str_replace_editor", "view", "read"),
        ("str_replace_editor", "view", "glob"),
        ("str_replace_editor", "create", "create"),
        ("str_replace_editor", "str_replace", "edit"),
        ("str_replace_editor", "insert", "insert_lines"),
        ("str_replace_editor", "undo_edit", None),
        ("execute_bash", None, "shell"),
        ("bash", None, "shell"),
        ("think", None, None),
        ("finish", None, "finish"),
    ]


def test_view_of_a_file_is_read_with_view_range_as_lines() -> None:
    got = translate(
        "str_replace_editor", {"command": "view", "path": "a.py", "view_range": [3, 9]}, WORKER
    )
    assert got is not None
    assert got.call == 'read {"path": "a.py", "lines": "3-9"}'
    assert got.takes == ("path", "handle", "lines", "at", "depth", "max_bytes")


def test_a_view_range_to_the_end_is_left_out() -> None:
    sent = {"command": "view", "path": "a.py", "view_range": [3, -1]}
    assert _call("str_replace_editor", **sent) == 'read {"path": "a.py"}'


def test_view_of_a_directory_is_glob_of_what_is_in_it() -> None:
    assert (
        _call("str_replace_editor", command="view", path="src/") == 'glob {"pattern": "src/*"}'
    )
    placed = translate(
        "str_replace_editor",
        {"command": "view", "path": "/workspaces/r/src"},
        WORKER,
        lambda sent: Place("src", directory=True),
    )
    assert placed is not None and placed.call == 'glob {"pattern": "src/*"}'


def test_create_is_create_with_file_text_as_content() -> None:
    got = _call("str_replace_editor", command="create", path="a.py", file_text="x = 1\n")
    assert got == 'create {"path": "a.py", "content": "x = 1\\n"}'


def test_str_replace_is_edit_with_old_and_new() -> None:
    got = _call(
        "str_replace_editor", command="str_replace", path="a.py", old_str="1", new_str="2"
    )
    assert got == 'edit {"path": "a.py", "old": "1", "new": "2"}'


def test_insert_is_insert_lines_after_the_line() -> None:
    got = translate(
        "str_replace_editor",
        {"command": "insert", "path": "a.py", "insert_line": 4, "new_str": "y = 2\n"},
        WORKER,
    )
    assert got is not None
    assert got.call == 'insert_lines {"path": "a.py", "after": 4, "content": "y = 2\\n"}'
    assert got.note == "use after as the line number read shows"


def test_undo_edit_has_nothing_to_call() -> None:
    got = translate("str_replace_editor", {"command": "undo_edit", "path": "a.py"}, WORKER)
    assert got is not None and got.tool is None and got.call is None
    assert got.note == "the harness keeps every step; there is nothing to call"


@pytest.mark.parametrize("name", ["execute_bash", "bash"])
def test_execute_bash_is_shell_and_says_what_has_no_equivalent(name: str) -> None:
    got = translate(name, {"command": "ls", "timeout": 5}, WORKER)
    assert got is not None
    assert got.call == 'shell {"command": "ls"}'
    assert got.note == "do not send timeout: no equivalent here"
    plain = translate(name, {"command": "ls"}, WORKER)
    assert plain is not None and plain.note == ""


def test_think_says_reason_in_your_reply() -> None:
    got = translate("think", {"thought": "hm"}, WORKER)
    assert got is not None and got.tool is None and got.note == "reason in your reply"


def test_finish_is_finish_with_message_as_report() -> None:
    assert _call("finish", WITH_FINISH, message="done") == 'finish {"report": "done"}'


@pytest.mark.parametrize(
    ("args", "tool"),
    [
        ({"path": "a.py", "old_str": "1", "new_str": "2"}, "edit"),
        ({"path": "a.py", "file_text": "x"}, "create"),
        ({"path": "a.py", "insert_line": 1, "new_str": "x"}, "insert_lines"),
        ({"path": "a.py"}, "read"),
    ],
)
def test_a_missing_command_is_inferred_from_the_arguments(
    args: dict[str, object], tool: str
) -> None:
    got = translate("str_replace_editor", args, WORKER)
    assert got is not None and got.tool == tool


def test_a_target_not_offered_is_left_out() -> None:
    zeus = {n: s for n, s in WORKER.items() if n not in ("edit", "create", "insert_lines")}
    sent = {"command": "str_replace", "path": "a.py", "old_str": "1", "new_str": "2"}
    assert translate("str_replace_editor", sent, zeus) is None
    assert translate("finish", {"message": "x"}, WORKER) is None


def test_a_name_or_command_not_in_the_table_is_none() -> None:
    assert translate("view_file", {"path": "a.py"}, WORKER) is None
    assert translate("str_replace_editor", {"command": "explode"}, WORKER) is None


def test_the_path_is_where_it_lands() -> None:
    got = translate(
        "str_replace_editor",
        {"command": "view", "path": "/workspaces/r/config.py"},
        WORKER,
        lambda sent: Place("config.py"),
    )
    assert got is not None and got.call == 'read {"path": "config.py"}'


def test_foreign_argument_names_the_openhands_tool() -> None:
    assert foreign_argument("command") == "str_replace_editor"
    assert foreign_argument("file_text") == "str_replace_editor"
    assert foreign_argument("is_input") == "execute_bash"
    assert foreign_argument("message") == "finish"
    assert foreign_argument("colour") is None


def test_a_name_ours_and_theirs_alike_is_not_named_as_theirs() -> None:
    """Review of #620: `path` on `glob` was called OpenHands' and told `here it is pattern`."""
    assert foreign_argument("path") is None
    assert ours_for("glob", "path") is None


def test_the_suggestion_names_the_tool_as_the_model_is_shown_it() -> None:
    """Review of #620: under a per-model name the suggestion was dropped."""
    from dataclasses import replace

    shown = {**WORKER, "read": replace(read.SPEC, name="view_file")}
    got = translate("str_replace_editor", {"command": "view", "path": "a.py"}, shown)
    assert got is not None and got.call == 'view_file {"path": "a.py"}'


def test_ours_for_names_our_argument_on_our_tool() -> None:
    assert ours_for("edit", "old_str") == "old"
    assert ours_for("create", "file_text") == "content"
    assert ours_for("create", "command") is None
