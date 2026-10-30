"""Harness argv: scrubbed env, extra_env overlay, no subprocess.run in core src."""

from __future__ import annotations

import ast
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.errors import InfraError
from chickadee.process import run_git, run_harness
from chickadee.process.supervisor import Completed, ProcessSupervisor

CORE = Path(__file__).resolve().parents[2] / "src" / "chickadee"


def _subprocess_bindings(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """Module aliases, plus names bound to `Popen` / `run` via `from subprocess import`."""
    modules: set[str] = set()
    popen: set[str] = set()
    run: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    modules.add(alias.asname or "subprocess")
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name == "Popen":
                    popen.add(bound)
                elif alias.name == "run":
                    run.add(bound)
    return modules, popen, run


def _has_spawn(tree: ast.AST, name: str) -> bool:
    """`subprocess.Popen` / `sp.Popen` / `from subprocess import Popen; Popen(...)`."""
    modules, popen, run = _subprocess_bindings(tree)
    imported = popen if name == "Popen" else run
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == name
            and isinstance(func.value, ast.Name)
            and func.value.id in modules
        ):
            return True
        if isinstance(func, ast.Name) and func.id in imported:
            return True
    return False


def _rel(path: Path) -> str:
    return path.relative_to(CORE).as_posix()


def test_spawn_scan_sees_aliases() -> None:
    assert _has_spawn(ast.parse("import subprocess as sp\nsp.Popen([])\n"), "Popen")
    assert _has_spawn(ast.parse("from subprocess import Popen\nPopen([])\n"), "Popen")
    assert _has_spawn(ast.parse("from subprocess import run as r\nr(['true'])\n"), "run")
    assert not _has_spawn(ast.parse("print(1)\n"), "Popen")
    assert not _has_spawn(ast.parse("import subprocess\nsubprocess.run([])\n"), "Popen")


def test_popen_constructor_lives_only_in_process_supervisor() -> None:
    hits = {
        _rel(path)
        for path in CORE.rglob("*.py")
        if _has_spawn(ast.parse(path.read_text(encoding="utf-8")), "Popen")
    }
    assert hits == {"process/supervisor.py"}, f"Popen call sites drifted: {sorted(hits)}"


def test_core_src_has_no_subprocess_run() -> None:
    hits = {
        _rel(path)
        for path in CORE.rglob("*.py")
        if _has_spawn(ast.parse(path.read_text(encoding="utf-8")), "run")
    }
    assert hits == set(), f"subprocess.run must go through ProcessSupervisor: {sorted(hits)}"


def test_run_harness_scrubs_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "leak")
    done = run_harness(
        (sys.executable, "-c", "import os; print('SECRET_TOKEN' in os.environ)"),
        cwd=tmp_path,
        timeout_s=5,
    )
    assert not done.timed_out
    assert done.stdout.strip() == b"False"


def test_extra_env_overlays_scrub(tmp_path: Path) -> None:
    done = run_harness(
        (sys.executable, "-c", "import os; print(os.environ.get('GIT_INDEX_FILE', ''))"),
        cwd=tmp_path,
        timeout_s=5,
        extra_env={"GIT_INDEX_FILE": "/tmp/chickadee-index"},
    )
    assert b"/tmp/chickadee-index" in done.stdout


def test_timed_out_is_returned_not_raised(tmp_path: Path) -> None:
    done = run_harness(("sleep", "30"), cwd=tmp_path, timeout_s=1)
    assert done.timed_out


def test_injected_supervisor_is_the_one_that_runs(tmp_path: Path) -> None:
    class Probe(ProcessSupervisor):
        def __init__(self) -> None:
            self.called = False

        def run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path,
            env: Mapping[str, str],
            timeout_s: int,
            stdin: bytes | None = None,
            cancel: threading.Event | None = None,
        ) -> Completed:
            self.called = True
            return super().run(argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin)

    probe = Probe()
    done = run_harness(("echo", "ok"), cwd=tmp_path, timeout_s=5, supervisor=probe)
    assert probe.called
    assert b"ok" in done.stdout


def test_run_harness_maps_oserror_to_infraerror(tmp_path: Path) -> None:
    class Boom(ProcessSupervisor):
        def run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path,
            env: Mapping[str, str],
            timeout_s: int,
            stdin: bytes | None = None,
            cancel: threading.Event | None = None,
        ) -> Completed:
            del cwd, env, timeout_s, stdin
            raise FileNotFoundError(argv[0])

    with pytest.raises(InfraError, match="could not run"):
        run_harness(("git", "--version"), cwd=tmp_path, timeout_s=5, supervisor=Boom())


def test_run_git_omits_dash_c(tmp_path: Path) -> None:
    class Probe(ProcessSupervisor):
        def __init__(self) -> None:
            self.argv: tuple[str, ...] | None = None

        def run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path,
            env: Mapping[str, str],
            timeout_s: int,
            stdin: bytes | None = None,
            cancel: threading.Event | None = None,
        ) -> Completed:
            self.argv = argv
            return super().run(argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin)

    probe = Probe()
    run_git(tmp_path, "rev-parse", "--is-inside-work-tree", supervisor=probe)
    assert probe.argv is not None
    assert probe.argv[0] == "git"
    assert "-C" not in probe.argv


def test_relative_dotdot_resolves_the_intended_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git applies `-C` relative to the child's cwd. Pairing cwd with `-C` of the same
    relative path joins it onto itself; `Path("..")` would then run in a parent of the
    repo. cwd-only (no `-C`) is the indexer pattern."""
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo, commit=None)
    nested = repo / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    from chickadee.init.detect import resolve

    identity = resolve(Path(".."))
    assert identity.root.resolve() == repo.resolve()
