"""`chickadee setup`: the person's engine config, written when absent,
and the sandbox probe, reported but never acted on."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.engine import Installed, Probe, ServerNeed
from chickadee.schema.execution import SandboxLevel
from chickadee_cli import setup

_BLOCKED = Probe(
    SandboxLevel.NATIVE,
    ran=False,
    reason="bwrap cannot create a user namespace: setting up uid map: Permission denied",
    fix="AppArmor restricts unprivileged user namespaces",
)


def test_setup_writes_the_engine_config_and_prints_its_cap(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    code, out = setup.run(path, probe=lambda: Probe(SandboxLevel.NATIVE, ran=True))
    assert code == 0 and path.is_file()
    assert f"wrote       {path}" in out
    assert "  hard cap    $5.00" in out
    assert "  sandbox     native, runs" in out


def test_run_again_it_changes_nothing_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    setup.run(path, probe=lambda: Probe(SandboxLevel.NATIVE, ran=True))
    before = path.read_bytes()
    code, out = setup.run(path, probe=lambda: Probe(SandboxLevel.NATIVE, ran=True))
    assert code == 0 and path.read_bytes() == before
    assert f"unchanged   {path} already exists; setup never edits it" in out


def test_a_sandbox_that_cannot_run_is_reported_with_its_reason_and_fix(tmp_path: Path) -> None:
    code, out = setup.run(tmp_path / "config.toml", probe=lambda: _BLOCKED)
    assert code == 0, "a missing sandbox is reported, not a failed setup"
    assert "  sandbox     native, does not run" in out
    assert _BLOCKED.reason in out and _BLOCKED.fix in out


def test_a_config_that_cannot_be_written_exits_1(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    code, out = setup.run(
        blocker / "config.toml", probe=lambda: Probe(SandboxLevel.NATIVE, ran=True)
    )
    assert code == 1
    assert "could not write" in out


# -- the language servers ----------------------------------------------------------

_FOUND = ServerNeed(("python",), "ruff", Path("/v/bin/ruff"), None, None)
_GO = ServerNeed(("go",), "gopls", None, ("go", "install", "gopls"), None)
_CS = ServerNeed(("csharp",), "csharp-ls", None, ("dotnet", "tool"), "dotnet")
_JAVA = ServerNeed(("java",), "jdtls", None, None, None)
_ALL = (_FOUND, _GO, _CS, _JAVA)


def test_setup_lists_each_server_and_asks_before_any_install(tmp_path: Path) -> None:
    asked: list[str] = []
    code, out = setup.run(
        tmp_path / "config.toml",
        probe=lambda: Probe(SandboxLevel.NATIVE, ran=True),
        servers=lambda: _ALL,
        ask=lambda need: asked.append(need.server) is not None,
        installer=lambda _need: pytest.fail("nothing was said yes to"),
    )
    assert code == 0
    assert asked == ["gopls"], "only an installable one is offered"
    assert "found at /v/bin/ruff" in out
    assert "gopls" in out and "missing; setup installs it with: go install gopls" in out
    assert "missing; installing it needs dotnet" in out
    assert "jdtls" in out and "missing; install it yourself" in out


def test_a_yes_installs_and_a_failure_is_reported_not_fatal(tmp_path: Path) -> None:
    code, out = setup.run(
        tmp_path / "config.toml",
        probe=lambda: Probe(SandboxLevel.NATIVE, ran=True),
        servers=lambda: (_GO,),
        ask=lambda _need: True,
        installer=lambda _need: Installed(False, "exit 1: network down"),
    )
    assert code == 0
    assert "not installed: exit 1: network down" in out
