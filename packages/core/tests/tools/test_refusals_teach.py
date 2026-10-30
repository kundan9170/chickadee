"""A refusal names the call to send: the three refusals of the 2026-10-19 log,
each through the worker's real dispatch, as the model reads them.

Pinned first as they read in older versions (`OFFERED` was the whole answer to the first, `does
not
exist` the second's, `not an argument of this tool` the third's), then each moved to the text
below, which the task's Log lists."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from box.box_helpers import WORKER_RUN, delegate, worker_rig
from box.box_helpers import worker_call as call
from engines import answer, calls
from refusals import stated
from repos import init_repo

from chickadee.journal.paths import journal_path
from chickadee.schema.journal import BoxMessageP
from chickadee.state.roots import roots

CONFIG = "WIDTH = 800\n"
READ_TAKES = "path, handle, lines, at, depth, max_bytes"
CREATE_SCHEMA = (
    "schema: {'type': 'object', 'properties': {'path': {'type': 'string', 'description': "
    "'the new file, from the repository root'}, 'content': {'type': 'string', 'description': "
    "'its whole content', 'x-block': True}}, 'required': ['path', 'content'], "
    "'additionalProperties': False}"
)
MOUNT = "/workspaces/tekken-like-2d-fighting-game-mvp"
SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo", {"config.py": CONFIG, "core/__init__.py": "", "main.py": "x = 1\n"}
    )


def _said(tmp_path: Path, tool: str, **args: object) -> str:
    root = _repo(tmp_path)
    worker, journal, _ = worker_rig(root, [calls(call(tool, "c1", **args)), answer("done")])
    worker.start("T1", "Read the config.", delegate(journal, root))
    [text] = [
        json.loads(journal.get_blob(p.payload.sha))["content"]
        for p in journal.projection().lines(BoxMessageP)
        if p.payload.side == "tool"
    ]
    return text


# -- the log's three refusals, moved ----------------------------------------------------------


def test_an_openhands_view_names_the_read_to_send_with_the_repository_path(
    tmp_path: Path,
) -> None:
    said = _said(tmp_path, "str_replace_editor", command="view", path=f"{MOUNT}/config.py")
    assert stated(said) == (
        "UNKNOWN_TOOL: str_replace_editor is not a tool here. Do not call it again. "
        f'Call instead: read {{"path": "config.py"}}. read takes: {READ_TAKES}.'
    )


def test_a_mount_path_names_the_file_it_meant(tmp_path: Path) -> None:
    said = _said(tmp_path, "read", path=f"{MOUNT}/config.py")
    assert stated(said) == (
        f"INVALID_ARGUMENTS{{field: path, reason: '{MOUNT}/config.py does not exist', "
        "did_you_mean: config.py, rule: 'paths are from the repository root'}"
    )


def test_command_on_create_is_refused_with_what_create_takes(tmp_path: Path) -> None:
    said = _said(tmp_path, "create", command="create", content="X = 1\n", path="core/x.py")
    assert stated(said) == (
        "INVALID_ARGUMENTS: create\n"
        "- command: not an argument of create. Do not send it; create takes path, content.\n"
        'Send instead: create {"content": "X = 1\\n", "path": "core/x.py"}\n'
        f"{CREATE_SCHEMA}"
    )


def test_the_whole_call_to_send_carries_short_values_and_renames_ours(tmp_path: Path) -> None:
    said = _said(tmp_path, "edit", path="config.py", old_str="800", new_str="900")
    assert 'Send instead: edit {"path": "config.py", "old": "800", "new": "900"}' in said


def test_a_long_value_in_the_call_to_send_is_not_echoed(tmp_path: Path) -> None:
    body = "import random\n" * 20
    said = _said(tmp_path, "create", command="create", content=body, path="core/ai.py")
    assert 'Send instead: create {"content": <as you sent it>, "path": "core/ai.py"}' in said
    assert body not in said


def test_no_call_to_send_when_a_refusal_has_no_known_fix(tmp_path: Path) -> None:
    said = _said(
        tmp_path / "a", "create", command="create", colour="red", content="X", path="a.py"
    )
    assert "instead:" not in said
    said = _said(tmp_path / "b", "create", command="create", content="X")
    assert "instead:" not in said, "the fixed call would still miss path"


# -- the rest of D209 through the worker's dispatch ------------------------------------------


def test_a_suggested_call_is_never_run(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    sent = call(
        "str_replace_editor", "c1", command="create", path="core/new.py", file_text="N\n"
    )
    worker, journal, _ = worker_rig(root, [calls(sent), answer("done")])
    worker.start("T1", "Make it.", delegate(journal, root))
    assert not (root / "core" / "new.py").exists()
    [text] = [
        json.loads(journal.get_blob(p.payload.sha))["content"]
        for p in journal.projection().lines(BoxMessageP)
        if p.payload.side == "tool"
    ]
    assert 'Call instead: create {"path": "core/new.py", "content": "N\\n"}' in text


def test_an_openhands_argument_on_edit_is_named_with_ours(tmp_path: Path) -> None:
    said = _said(tmp_path, "edit", path="config.py", old_str="800", new_str="900")
    assert "- old_str: not an argument of edit. Do not send it; use old." in (said)
    assert "- new_str: not an argument of edit. Do not send it; use new." in (said)


def test_a_path_on_glob_is_refused_as_before(tmp_path: Path) -> None:
    said = _said(tmp_path, "glob", path="src", pattern="*")
    assert "- path: not an argument of this tool; known: ['pattern']" in said


def test_an_unknown_argument_from_no_table_is_as_before(tmp_path: Path) -> None:
    said = _said(tmp_path, "create", colour="red", content="X = 1\n", path="core/x.py")
    assert "- colour: not an argument of this tool; known: ['content', 'path']" in said


def test_a_name_from_no_table_is_as_before(tmp_path: Path) -> None:
    said = _said(tmp_path, "view_file", path="config.py")
    assert said.startswith("UNKNOWN_TOOL{name: 'view_file', offered: [read, ")


def test_think_says_nothing_is_to_be_called(tmp_path: Path) -> None:
    said = _said(tmp_path, "think", thought="hm")
    assert (
        said
        == "UNKNOWN_TOOL: think is not a tool here. Do not call it again. Reason in your reply."
    )


def test_an_edit_of_a_mount_path_names_the_file_it_meant(tmp_path: Path) -> None:
    said = _said(tmp_path, "edit", path=f"{MOUNT}/config.py", old="800", new="900")
    assert stated(said) == (
        f"DENIED: absolute path: {MOUNT}/config.py; did you mean config.py? "
        "paths are from the repository root"
    )


def test_an_edit_of_a_file_not_there_names_the_one_it_meant(tmp_path: Path) -> None:
    said = _said(tmp_path, "edit", path="src/config.py", old="800", new="900")
    assert stated(said) == (
        "FILE_ABSENT: src/config.py does not exist; did you mean "
        "config.py? paths are from the repository root"
    )


def test_a_create_under_a_mount_names_its_directory(tmp_path: Path) -> None:
    said = _said(tmp_path, "create", path=f"{MOUNT}/core/x.py", content="X = 1\n")
    assert stated(said).endswith("; did you mean core/x.py? paths are from the repository root")


def test_a_delete_and_a_move_of_a_mount_path_name_the_file(tmp_path: Path) -> None:
    deleted = _said(tmp_path / "d", "delete", path=f"{MOUNT}/main.py")
    assert stated(deleted).endswith(
        "; did you mean main.py? paths are from the repository root"
    )
    moved = _said(tmp_path / "m", "move", path=f"{MOUNT}/main.py", destination="app.py")
    assert stated(moved).endswith("; did you mean main.py? paths are from the repository root")


def test_a_path_refused_for_another_reason_names_nothing(tmp_path: Path) -> None:
    said = _said(tmp_path, "create", path="docs/chickadee/config.py", content="X = 1\n")
    assert said.startswith("DENIED: docs/chickadee/config.py is in the work folder")
    assert "did you mean" not in said


# -- the count ---------------------------------------------------------------


def test_the_count_reads_refusals_repeats_and_first_reads_from_the_journal(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    mount = call("read", "c1", path=f"{MOUNT}/config.py")
    script = [calls(mount), calls(mount), calls(call("read", "c3", path="config.py"))]
    worker, journal, _ = worker_rig(root, [*script, answer("done")])
    worker.start("T1", "Read the config.", delegate(journal, root))
    run = journal_path(roots(root), WORKER_RUN)
    counted = subprocess.run(
        [sys.executable, str(SCRIPTS / "refusal_count.py"), str(run.parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert counted.stdout.splitlines() == [
        "worker: 3 calls, 2 refused, 1 identical repeats",
        "  refused INVALID_ARGUMENTS: 2",
        "  first read of config.py: after 2 calls",
    ]
