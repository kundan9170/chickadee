"""Standing preferences in every shell, Zeus's and the box's alike.

What these pin, against `test_shell_class_tiers.py`'s answers with no preferences: a class set
to `ask` is asked, naming the rule, where it ran unasked; `lockfile_install = "allow"` runs an
install unasked, where it was asked, and `network: true` still makes it T2; a class set
to `deny` is refused as `PREFERENCE_DENIED` before its tier, with nothing asked and nothing run,
where it was asked or ran; a declared write to a path a `deny` glob names is refused at parse;
and the preferences are read per call, so an edited file applies at the next one. The box
shell, which the worker and Athena run, reads them the same way: a denied class is refused
attended or not, and an asked one names who asks and for which task.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from repos import init_repo
from sandboxes import Enforcing, Scripted
from tool_args import checked

from chickadee.approval.commands import GitLine
from chickadee.approval.prefs import merge, parse_level
from chickadee.approval.tier import Preferences
from chickadee.box.shell import BoxShell
from chickadee.context.homes import Scratch
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import ToolResult
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.executor.types import ExecRequest, ExecResult
from chickadee.journal.port import JournalPort
from chickadee.schema import sha256_hex
from chickadee.schema.authorization import Tier
from chickadee.schema.journal import EventKind
from chickadee.shell import PreferenceDenied, Shell, ShellClaim, Spawn, Yielded


class _PrefsAuthority:
    def __init__(self) -> None:
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return True


class _Writes:
    """A recorded write's home that records nothing: the step's put-back has its own tests
    (`test_shell_declared_write.py`)."""

    where = "the session's worktree"

    def git_dirs(self) -> tuple[Path, ...]:
        return ()

    def run(
        self,
        request: ExecRequest,
        operation_id: str,
        cancel: threading.Event | None,
        spawn: Spawn,
        *,
        git: GitLine,
    ) -> tuple[ExecResult | Yielded, tuple[str, ...]]:
        del git
        return spawn(request, operation_id, cancel), ()


def prefs_of(approval: dict[str, Any], level: Any = "user") -> Preferences:
    return merge([parse_level({"approval": approval}, level)])


def _rig(
    tmp_path: Path, prefs: Callable[[], Preferences]
) -> tuple[Shell, _PrefsAuthority, Scripted]:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    asked, supervisor = _PrefsAuthority(), Scripted(b"ok\n")
    own = Scratch()
    reader = Reader(own, HandleResolver(root, own.blob), vision=Vision("z"))
    shell = Shell(
        Executor(Enforcing(), supervisor=supervisor),
        root,
        lambda data: reader.store("output", data),
        authority=asked,
        writes=_Writes(),
        prefs=prefs,
    )
    return shell, asked, supervisor


def _run(shell: Shell, **args: Any) -> object:
    return shell.run(checked("shell", dict(args)), operation_id="op-1")


def test_a_class_set_to_ask_is_asked_and_the_prompt_names_the_rule(tmp_path: Path) -> None:
    shell, asked, _ = _rig(tmp_path, lambda: prefs_of({"commands": {"tests": "ask"}}))
    done = _run(shell, command="pytest -q")
    assert isinstance(done, ToolResult) and " · t1 · op-1" in done.header
    [claim] = asked.claims
    assert claim.tier is Tier.T1
    assert (
        claim.classification
        == 'tests ([approval.commands] tests = "ask" in your approval.toml)'
    )


def test_an_allowed_install_runs_unasked_and_a_network_still_asks(tmp_path: Path) -> None:
    shell, asked, _ = _rig(
        tmp_path, lambda: prefs_of({"commands": {"lockfile_install": "allow"}})
    )
    done = _run(shell, command="npm ci")
    assert isinstance(done, ToolResult) and " · t0 · op-1" in done.header
    assert asked.claims == []
    # `network: true` is T2, which no rule lowers.
    _run(shell, command="npm ci", network=True)
    [claim] = asked.claims
    assert claim.tier is Tier.T2


@pytest.mark.parametrize("line", ["git push origin main", "ls && git push origin main"])
def test_a_denied_class_is_refused_before_its_tier_with_nothing_asked(
    tmp_path: Path, line: str
) -> None:
    shell, asked, supervisor = _rig(
        tmp_path, lambda: prefs_of({"commands": {"push": "deny"}}, "repo")
    )
    done = _run(shell, command=line)
    assert done == PreferenceDenied(
        line, '[approval.commands] push = "deny" in the repository\'s approval.toml'
    )
    assert (asked.claims, supervisor.spawned) == ([], 0)


def test_the_preferences_are_read_per_call(tmp_path: Path) -> None:
    held = [Preferences()]
    shell, asked, _ = _rig(tmp_path, lambda: held[0])
    _run(shell, command="pytest -q")
    assert asked.claims == []
    held[0] = prefs_of({"commands": {"tests": "ask"}})
    _run(shell, command="pytest -q")
    assert [c.tier for c in asked.claims] == [Tier.T1]


# -- the box shell -----------------------------------------------------------------------------


class _RunJournal:
    """The run journal as the box shell writes it: its `exec` lines and output blobs."""

    run_id = "run-1"

    def __init__(self) -> None:
        self.kinds: list[EventKind] = []

    def append(self, kind: EventKind, payload: Any, **_: Any) -> int:
        self.kinds.append(kind)
        return len(self.kinds)

    def put_blob(self, content: bytes) -> str:
        return sha256_hex(content)


def _box(
    tmp_path: Path, prefs: Callable[[], Preferences], *, attended: bool
) -> tuple[BoxShell, _PrefsAuthority, Scripted]:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    asked, supervisor = _PrefsAuthority(), Scripted(b"ok\n")
    shell = BoxShell(
        Executor(Enforcing(), supervisor=supervisor),
        root,
        cast(JournalPort, _RunJournal()),
        asked if attended else None,
        asker="worker",
        task=lambda: "T2",
        prefs=prefs,
    )
    return shell, asked, supervisor


@pytest.mark.parametrize("attended", [True, False])
def test_the_box_shell_refuses_a_denied_class_attended_or_not(
    tmp_path: Path, attended: bool
) -> None:
    shell, asked, supervisor = _box(
        tmp_path, lambda: prefs_of({"commands": {"push": "deny"}}), attended=attended
    )
    done = shell.run(
        checked("shell", {"command": "git push origin main"}),
        operation_id="op-1",
        cancel=None,
    )
    assert done == PreferenceDenied(
        "git push origin main", '[approval.commands] push = "deny" in your approval.toml'
    )
    assert (asked.claims, supervisor.spawned) == ([], 0)


def test_the_box_shell_asks_a_class_set_to_ask_naming_who_and_which_task(
    tmp_path: Path,
) -> None:
    shell, asked, _ = _box(
        tmp_path, lambda: prefs_of({"commands": {"tests": "ask"}}), attended=True
    )
    shell.run(
        checked("shell", {"command": "pytest -q"}),
        operation_id="op-1",
        cancel=None,
    )
    [claim] = asked.claims
    assert (claim.tier, claim.asker, claim.task_id) == (Tier.T1, "worker", "T2")
