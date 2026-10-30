"""Spawn monopoly and wrap/policy split."""

from __future__ import annotations

import ast
from pathlib import Path

from chickadee.executor.executor import Executor
from chickadee.executor.policy import isolation_for
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation, SandboxLevel

CORE = Path(__file__).resolve().parents[2] / "src" / "chickadee"

# Call sites frozen at G2. `handoff.py` documents that workspace/applier still use
# `subprocess.run` for git; shrinking this set is allowed, growing it is not.
_SUBPROCESS_RUN = frozenset(
    {
        "applier/applier.py",
        "executor/reconcile.py",
        "init/detect.py",
        "init/toolchain.py",
        "workspace.py",
    }
)
_POPEN_CALL = frozenset({"process/supervisor.py"})


def _attr_calls(tree: ast.AST, owner: str, name: str) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
        ):
            return True
    return False


def _rel(path: Path) -> str:
    return path.relative_to(CORE).as_posix()


def test_popen_constructor_lives_only_in_process_supervisor() -> None:
    hits = {
        _rel(path)
        for path in CORE.rglob("*.py")
        if _attr_calls(ast.parse(path.read_text(encoding="utf-8")), "subprocess", "Popen")
    }
    assert hits == _POPEN_CALL, f"Popen call sites drifted: {sorted(hits)}"


def test_subprocess_run_inventory_has_not_grown() -> None:
    hits = {
        _rel(path)
        for path in CORE.rglob("*.py")
        if _attr_calls(ast.parse(path.read_text(encoding="utf-8")), "subprocess", "run")
    }
    extra = hits - _SUBPROCESS_RUN
    assert not extra, (
        f"new subprocess.run call sites (must go through ProcessSupervisor): {extra}"
    )


def test_isolation_and_process_do_not_import_executor() -> None:
    banned = ("chickadee.executor",)
    offenders: list[str] = []
    for pkg in ("isolation", "process"):
        root = CORE / pkg
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == banned[0] or n.startswith(banned[0] + ".") for n in names):
                    offenders.append(_rel(path))
    assert offenders == []


def test_harness_spawn_does_not_construct_executor() -> None:
    tree = ast.parse((CORE / "box" / "spawn.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "chickadee.executor.executor"
            assert node.module != "chickadee.executor"
            if node.module.startswith("chickadee.executor"):
                imported = {a.name for a in node.names}
                assert "Executor" not in imported
                assert "ExecRequest" not in imported
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "Executor"


def test_isolation_wrap_rewrites_argv_and_does_not_spawn(tmp_path: Path) -> None:
    iso = Isolation(argv=("true",), cwd=tmp_path, network_off=True, repo_writable=False)
    assert Passthrough().wrap(iso) == ("true",)
    assert not hasattr(iso, "policy")


def test_executor_run_is_the_proposed_command_front(tmp_path: Path) -> None:
    request = ExecRequest(
        command=("true",),
        cwd=tmp_path,
        policy=PolicyClass.WORKTREE,
        timeout_s=15,
    )
    wrapped = isolation_for(request)
    assert wrapped.argv == ("true",)
    result = Executor(backend=Passthrough(), require_sandbox=False).run(
        request, operation_id="op-true"
    )
    assert result.exit_code == 0
    assert result.sandbox is SandboxLevel.NONE
    assert result.timed_out is False
