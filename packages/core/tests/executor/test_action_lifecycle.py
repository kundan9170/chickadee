"""Action lint, risk, authorization validity and the three-way check."""

from __future__ import annotations

from pathlib import Path

from chickadee.executor.actions import lint, to_request
from chickadee.executor.approval import (
    Authorization,
    Risk,
    classify_risk,
    concrete_hash,
    still_valid,
)
from chickadee.executor.executor import Executor
from chickadee.executor.types import ActionSpec, NetworkPolicy, PolicyClass

INSTALL = ActionSpec(
    command='uv add "pyjwt>=2.8"',
    effects=("pyproject.toml", "uv.lock"),
    network=NetworkPolicy(open=True),
    idempotent=True,
    why="packet t2 needs JWT encode/decode",
    done_when="import_smoke: jwt",
)


def test_a_complete_action_lints_clean() -> None:
    assert lint(INSTALL) == ()


def test_an_action_must_declare_its_effect_surface() -> None:
    """Without one, the second check compares against nothing."""
    issues = lint(ActionSpec(command="echo hi", why="w", done_when="d"))
    assert any(i.field == "effects" for i in issues)


def test_an_action_must_declare_done_when() -> None:
    """Exit 0 proves the command ran, not that it achieved its purpose."""
    issues = lint(ActionSpec(command="echo hi", effects=("a",), why="w"))
    assert any(i.field == "done_when" for i in issues)


def test_an_effect_glob_may_not_escape() -> None:
    issues = lint(ActionSpec(command="x", effects=("../outside/**",), why="w", done_when="d"))
    assert any("escapes" in i.message for i in issues)


def test_network_access_makes_an_action_high_risk() -> None:
    risk, reasons = classify_risk(INSTALL)
    assert risk is Risk.HIGH
    assert any("network" in r for r in reasons)


def test_a_local_action_is_low_risk() -> None:
    spec = ActionSpec(command=("mkdir", "build"), effects=("build/**",), why="w", done_when="d")
    assert classify_risk(spec)[0] is Risk.LOW


def test_a_destructive_command_is_high_risk() -> None:
    spec = ActionSpec(command="rm -rf build", effects=("build/**",), why="w", done_when="d")
    risk, reasons = classify_risk(spec)
    assert risk is Risk.HIGH
    assert "destructive command" in reasons


def test_the_hash_covers_the_effect_surface_not_just_the_command() -> None:
    """A digest over the command alone would let `effects` widen from one file to `**`
    without invalidating the approval anyone gave."""
    widened = ActionSpec(
        command=INSTALL.command,
        effects=("**",),
        network=INSTALL.network,
        idempotent=True,
        why=INSTALL.why,
        done_when=INSTALL.done_when,
    )
    assert concrete_hash(widened) != concrete_hash(INSTALL)


def test_a_changed_spec_invalidates_authorization() -> None:
    auth = Authorization(approved_hash=concrete_hash(INSTALL), risk=Risk.HIGH, confirmed=True)
    changed = ActionSpec(
        command='uv add "requests"',
        effects=INSTALL.effects,
        network=INSTALL.network,
        idempotent=True,
        why=INSTALL.why,
        done_when=INSTALL.done_when,
    )
    ok, why = still_valid(auth, changed)
    assert not ok
    assert "changed since approval" in why


def test_a_high_risk_action_needs_a_second_confirmation() -> None:
    auth = Authorization(approved_hash=concrete_hash(INSTALL), risk=Risk.HIGH, confirmed=False)
    ok, why = still_valid(auth, INSTALL)
    assert not ok
    assert "second confirmation" in why


def test_an_unchanged_confirmed_action_stays_authorized() -> None:
    auth = Authorization(approved_hash=concrete_hash(INSTALL), risk=Risk.HIGH, confirmed=True)
    assert still_valid(auth, INSTALL)[0]


def test_the_three_way_check_fails_on_an_undeclared_effect(
    unsandboxed: Executor, work: Path
) -> None:
    spec = ActionSpec(
        command=("sh", "-c", "echo x > declared.txt; echo y > sneaky.txt"),
        effects=("declared.txt",),
        why="w",
        done_when="d",
    )
    result = unsandboxed.run(to_request(spec, work), operation_id="op-1")
    assert result.exit_code == 0, "the command itself succeeded"
    assert [v.path for v in result.violations] == ["sneaky.txt"]
    assert not result.ok, "exit 0 is not enough when the effect surface was breached"


def test_an_action_request_carries_its_policy(work: Path) -> None:
    assert to_request(INSTALL, work).policy is PolicyClass.ACTION


def test_argv_boundaries_survive_the_approval_hash() -> None:
    """Verified colliding before the fix: these join to the same text and execute
    differently, so a text digest let an approval cover a command nobody approved."""
    one = ActionSpec(command=("printf", "%s", "a b"), effects=("a",), why="w", done_when="d")
    two = ActionSpec(command=("printf", "%s a", "b"), effects=("a",), why="w", done_when="d")
    assert concrete_hash(one) != concrete_hash(two)


def test_a_shell_string_and_an_argv_tuple_are_distinct() -> None:
    shell = ActionSpec(command="echo hi", effects=("a",), why="w", done_when="d")
    argv = ActionSpec(command=("echo", "hi"), effects=("a",), why="w", done_when="d")
    assert concrete_hash(shell) != concrete_hash(argv)


def test_the_hash_covers_the_network_policy() -> None:
    local = ActionSpec(command=("x",), effects=("a",), why="w", done_when="d")
    networked = ActionSpec(
        command=("x",),
        effects=("a",),
        network=NetworkPolicy(open=True),
        why="w",
        done_when="d",
    )
    assert concrete_hash(local) != concrete_hash(networked)
