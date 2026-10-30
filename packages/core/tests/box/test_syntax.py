"""The syntax of a written file: `box/syntax.SyntaxChecker`, gate A1's
check moved, which the edit port judges every write by.

What these pin: Python is compiled in process, never spawning; JavaScript is its profile's
command (`node --check`) through the spawn, over the file as it is on disk, a failure its
digest and a timeout said; a file no profile owns, or a command-mode language with no spawn,
is judged by nothing, and the edit tool names it `syntax unchecked`.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.box.syntax import SyntaxChecker
from chickadee.edits.draft import SyntaxCheck
from chickadee.executor.types import ExecResult
from chickadee.profiles.loader import load_profile
from chickadee.schema.execution import SandboxLevel

PYTHON = load_profile("python")
JAVASCRIPT = load_profile("javascript")


class _Spawn:
    """A spawn that records each command and answers `exit_code`."""

    def __init__(self, exit_code: int = 0, stderr: str = "", timed_out: bool = False) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, int]] = []
        self._result = ExecResult(
            exit_code=exit_code,
            stdout_sha=None,
            stderr_sha=None,
            wall_ms=1,
            timed_out=timed_out,
            sandbox=SandboxLevel.NONE,
            stderr=stderr,
        )

    def __call__(self, argv: tuple[str, ...], cwd: Path, timeout_s: int) -> ExecResult:
        self.calls.append((argv, cwd, timeout_s))
        return self._result


def test_python_is_compiled_in_process_and_never_spawns(tmp_path: Path) -> None:
    spawn = _Spawn()
    checker = SyntaxChecker(tmp_path, (PYTHON, JAVASCRIPT), spawn)
    assert checker.check("a.py", b"x = 1\n") == SyntaxCheck(True, None, "compiled")
    bad = checker.check("a.py", b"def f(:\n")
    assert bad is not None and not bad.ok and bad.detail is not None
    assert bad.detail.startswith("a.py:")
    assert spawn.calls == []


def test_javascript_is_its_profiles_command_through_the_spawn(tmp_path: Path) -> None:
    spawn = _Spawn()
    checker = SyntaxChecker(tmp_path, (PYTHON, JAVASCRIPT), spawn, timeout_s=7)
    assert checker.check("web/a.js", b"let x = 1;\n") == SyntaxCheck(True, None, "command")
    assert spawn.calls == [(("node", "--check", "web/a.js"), tmp_path, 7)]


def test_a_failing_command_is_its_digest_and_a_timeout_is_said(tmp_path: Path) -> None:
    failed = SyntaxChecker(
        tmp_path, (JAVASCRIPT,), _Spawn(1, "a.js:1\nSyntaxError: Unexpected token")
    ).check("a.js", b"let = ;\n")
    assert failed is not None and not failed.ok and "SyntaxError" in (failed.detail or "")
    late = SyntaxChecker(tmp_path, (JAVASCRIPT,), _Spawn(timed_out=True)).check("a.js", b"")
    assert late == SyntaxCheck(False, "timeout", "command")


def test_what_no_profile_or_no_spawn_judges_is_unchecked(tmp_path: Path) -> None:
    checker = SyntaxChecker(tmp_path, (PYTHON, JAVASCRIPT))
    assert checker.check("README.md", b"# hi\n") is None
    assert checker.check("a.js", b"let x = 1;\n") is None
    assert checker.profile_for("README.md") is None


def test_a_verdict_is_judged_once_per_path_and_bytes(tmp_path: Path) -> None:
    """Review: the facts judged every changed file again each round, a process per file for a
    `cmd` profile. The same bytes at the same path are judged once."""
    from chickadee.box.syntax import SyntaxChecker
    from chickadee.edits.draft import SyntaxCheck

    checker = SyntaxChecker(tmp_path, ())
    seen: list[str] = []

    def judged(path: str, content: bytes) -> SyntaxCheck | None:
        seen.append(path)
        return SyntaxCheck(True, None)

    checker._check = judged  # type: ignore[method-assign]
    for _ in range(3):
        checker.check("a.py", b"x = 1\n")
    checker.check("a.py", b"x = 2\n")
    assert seen == ["a.py", "a.py"]
