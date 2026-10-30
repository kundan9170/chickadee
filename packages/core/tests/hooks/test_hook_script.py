"""A script hook's run (`hooks/script.py`).

What these pin: the argv runs as given, `~` and `$CHICKADEE_PROJECT_DIR` resolved and nothing
else; the script reads its event on stdin and the project directory in its environment, and its
answer is read through the protocol; a script past its timeout is a timed-out hook error; its
output is cut at `OUTPUT_MAX`; a script changed since the set was bound is not run; and where
bubblewrap runs, the script cannot write the repository.

The mechanics run over `Passthrough`, as a unit test may (`HarnessSpawn.require_sandbox`); the
episode never gives a script runner where nothing seals (`delegation._parts`).
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pytest

from chickadee.box.spawn import HarnessSpawn, run_argv
from chickadee.hooks.fire import HookError
from chickadee.hooks.payloads import BeforeTool, HookEvent
from chickadee.hooks.points import HookPoint
from chickadee.hooks.script import OUTPUT_MAX, ScriptSpawn, argv_of, script_runner
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Deny, Note
from chickadee.isolation.base import has
from chickadee.isolation.linux import BINARY, Bubblewrap
from chickadee.isolation.none import Passthrough
from chickadee.process import ProcessSupervisor

RUN = "2026-10-24-0d1e2f30"
bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")


def _spec(root: Path, body: str, **kw: object) -> HookSpec:
    (root / "hook.py").write_text(body, encoding="utf-8")
    fields: dict[str, object] = {
        "event": "before_tool",
        "name": "guard",
        "command": [sys.executable, "$CHICKADEE_PROJECT_DIR/hook.py"],
    }
    fields.update(kw)
    return HookSpec.model_validate(fields)


def _event() -> HookEvent:
    payload = BeforeTool(tool="read", tool_sent="read", input={"path": "a.py"})
    return HookEvent(HookPoint.BEFORE_TOOL, "worker", payload, run_id=RUN, call_id="c1")


def _passthrough() -> ScriptSpawn:
    return partial(run_argv, HarnessSpawn(Passthrough(), ProcessSupervisor()))


def test_the_argv_resolves_the_project_dir_and_home_and_nothing_else(tmp_path: Path) -> None:
    spec = HookSpec.model_validate(
        {
            "event": "before_tool",
            "name": "g",
            "command": ["~/bin/g", "$CHICKADEE_PROJECT_DIR/x.sh", "$HOME", "a b"],
        }
    )
    assert argv_of(spec, tmp_path) == (
        str(Path("~/bin/g").expanduser()),
        f"{tmp_path}/x.sh",
        "$HOME",
        "a b",
    )


def test_the_script_reads_its_event_and_its_answer_is_read(tmp_path: Path) -> None:
    body = (
        "import json, os, sys\n"
        "seen = json.load(sys.stdin)\n"
        "ctx = seen['tool_name'] + ' ' + os.environ['CHICKADEE_PROJECT_DIR']\n"
        "print(json.dumps({'hookSpecificOutput': {'additionalContext': ctx}}))\n"
    )
    run, _ = script_runner(_passthrough(), tmp_path)(_spec(tmp_path, body))
    assert run(_event()) == Note(f"read {tmp_path}")


def test_exit_2_denies_in_stderrs_words(tmp_path: Path) -> None:
    body = "import sys\nsys.stderr.write('use the edit tool')\nsys.exit(2)\n"
    run, _ = script_runner(_passthrough(), tmp_path)(_spec(tmp_path, body))
    assert run(_event()) == Deny("use the edit tool", None)


def test_a_script_past_its_timeout_is_a_timed_out_error(tmp_path: Path) -> None:
    body = "import time\ntime.sleep(30)\n"
    run, _ = script_runner(_passthrough(), tmp_path)(_spec(tmp_path, body, timeout_s=1))
    with pytest.raises(HookError) as raised:
        run(_event())
    assert raised.value.timed_out


def test_its_output_is_cut_at_the_cap(tmp_path: Path) -> None:
    body = f"import json\nprint(json.dumps({{'systemMessage': 'x' * {OUTPUT_MAX * 2}}}))\n"
    run, _ = script_runner(_passthrough(), tmp_path)(_spec(tmp_path, body))
    with pytest.raises(HookError, match="does not parse"):
        run(_event())  # cut mid-string: what is past the cap is never read


def test_a_script_changed_since_it_was_bound_is_not_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    spec = _spec(tmp_path, f"open({str(marker)!r}, 'w').close()\n")
    run, digest = script_runner(_passthrough(), tmp_path)(spec)
    (tmp_path / "hook.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(HookError, match="changed since the set was bound"):
        run(_event())
    assert not marker.exists()
    _, again = script_runner(_passthrough(), tmp_path)(spec)
    assert again != digest, "the digest covers the script's bytes"


@bwrap_only
def test_under_bubblewrap_the_script_cannot_write_the_repository(tmp_path: Path) -> None:
    body = (
        "import json\n"
        "try:\n"
        "    open('written', 'w').close()\n"
        "    said = 'wrote'\n"
        "except OSError:\n"
        "    said = 'refused'\n"
        "print(json.dumps({'systemMessage': said}))\n"
    )
    spawn = partial(
        run_argv, HarnessSpawn(Bubblewrap(), ProcessSupervisor(), require_sandbox=True)
    )
    run, _ = script_runner(spawn, tmp_path)(_spec(tmp_path, body))
    verdict = run(_event())
    assert "refused" in str(verdict) and not (tmp_path / "written").exists()
