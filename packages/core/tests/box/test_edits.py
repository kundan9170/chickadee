"""The box's edit port: `WorktreeEdits` over a real git worktree.

What these pin: every refusal of §10.2 changes nothing on disk; a write lands with its mode
kept; a move keeps the source's mode; `delete` and `move` refuse what undo could not bring back;
a shell's change since the last call is read, since every call opens a fresh draft; a
syntax check that is a command judges the call's text, put in place for it and taken back.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.approval.prefs import merge, parse_level
from chickadee.box.edits import WorktreeEdits
from chickadee.box.syntax import SyntaxChecker
from chickadee.edits.draft import DraftRefused
from chickadee.edits.match import Region
from chickadee.executor.types import ExecResult
from chickadee.profiles.loader import load_profile
from chickadee.schema.execution import SandboxLevel
from chickadee.secrets.denylist import Denylist

FILES = {
    "app.py": "def go() -> int:\n    return 1\n",
    "run.sh": "#!/bin/sh\necho hi\n",
    "pkg/util.py": "U = 1\n",
    ".gitignore": "build/\n",
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    repo = init_repo(tmp_path / "repo", FILES)
    (repo / "run.sh").chmod(0o755)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "mode")
    return repo


def _port(root: Path, *, cap: int = 1024) -> WorktreeEdits:
    checker = SyntaxChecker(root, (load_profile("python"),))
    return WorktreeEdits(root, Denylist(), checker, untracked_max_bytes=cap)


def _whole(port: WorktreeEdits, path: str, content: str) -> str | DraftRefused:
    with port.edit() as call:
        refused = call.apply(path, None, content)
        return refused if refused is not None else call.answer()


def _refused(said: str | DraftRefused | None, failure: str) -> str:
    """The refusal as the model reads it: its reason, then what to do."""
    assert isinstance(said, DraftRefused) and said.failure == failure, said
    return said.render()


# -- §10.2: refusals, nothing written ---------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "words"),
    [
        ("../outside.py", "escapes with '..'"),
        ("/etc/passwd", "absolute path"),
        (".git/hooks/pre-commit", "git internals"),
        (".env", "secret path"),
    ],
)
def test_a_path_it_may_not_write_is_refused(root: Path, path: str, words: str) -> None:
    assert words in _refused(_whole(_port(root), path, "x = 1\n"), "DENIED")


def test_a_path_the_persons_deny_globs_name_is_refused(root: Path) -> None:
    """`approval.toml`'s `deny` globs refuse a write before anything is written."""
    prefs = merge([parse_level({"approval": {"deny": ["pkg/**"]}}, "repo")])
    checker = SyntaxChecker(root, (load_profile("python"),))
    port = WorktreeEdits(
        root, Denylist(), checker, untracked_max_bytes=1024, prefs=lambda: prefs
    )
    said = _refused(_whole(port, "pkg/util.py", "U = 2\n"), "DENIED")
    assert said == (
        'DENIED: pkg/util.py is denied by deny "pkg/**" in the repository\'s approval.toml. '
        "Do: leave this file alone; if the task needs it changed, say so in your report"
    )
    assert (root / "pkg" / "util.py").read_text(encoding="utf-8") == "U = 1\n"


@pytest.mark.writes_worktree(".env", "env_link.txt")
@pytest.mark.parametrize("asked", ["text", "write"])
def test_a_link_to_a_secret_is_neither_shown_nor_written_through(
    root: Path, asked: str
) -> None:
    """A link's name says nothing of where it lands: the landing is judged too, so the
    secret is not shown in an answer, and a write does not land on it."""
    (root / ".env").write_text("T=sk-live\n", encoding="utf-8")
    (root / "env_link.txt").symlink_to(root / ".env")
    port = _port(root)
    if asked == "text":
        with port.edit() as call:
            said = call.text("env_link.txt")
    else:
        said = _whole(port, "env_link.txt", "T=0\n")
    assert "leads to a secret path" in _refused(said, "DENIED")
    assert (root / ".env").read_text(encoding="utf-8") == "T=sk-live\n"


@pytest.mark.writes_worktree("vendor")
def test_a_link_out_of_the_worktree_is_refused(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "vendor").symlink_to(outside)
    reason = _refused(_whole(_port(root), "vendor/cfg.py", "x = 1\n"), "DENIED")
    assert "outside the worktree" in reason
    assert not (outside / "cfg.py").exists()


@pytest.mark.writes_worktree("lib*")
def test_a_nested_repository_is_refused(root: Path) -> None:
    init_repo(root / "lib" / "vendored", {"x.py": "x = 1\n"})
    reason = _refused(_whole(_port(root), "lib/vendored/x.py", "x = 2\n"), "DENIED")
    assert "nested repository lib/vendored" in reason


def test_editing_a_missing_file_says_it_does_not_exist(root: Path) -> None:
    with _port(root).edit() as call:
        missing = call.text("utils.py")
    assert "does not exist" in _refused(missing, "FILE_ABSENT")


def test_empty_content_over_a_file_is_refused_and_a_new_file_may_be_empty(root: Path) -> None:
    port = _port(root)
    _refused(_whole(port, "app.py", "\n"), "DENIED")
    assert (root / "app.py").read_text(encoding="utf-8") == FILES["app.py"]
    said = _whole(port, "pkg/__init__.py", "")
    assert isinstance(said, str) and said.startswith("ok: pkg/__init__.py written")
    assert (root / "pkg" / "__init__.py").read_text(encoding="utf-8") == ""


# -- writes ------------------------------------------------------------------------------------


def test_a_write_lands_on_disk_and_keeps_its_mode(root: Path) -> None:
    said = _whole(_port(root), "run.sh", "#!/bin/sh\necho bye\n")
    assert isinstance(said, str) and said.startswith("ok: run.sh written")
    assert (root / "run.sh").read_text(encoding="utf-8") == "#!/bin/sh\necho bye\n"
    assert stat.S_IMODE((root / "run.sh").stat().st_mode) == 0o755


def test_an_edit_reports_its_syntax_and_stands_under_report(root: Path) -> None:
    port = _port(root)
    with port.edit() as call:
        text = call.text("app.py")
        assert isinstance(text, str)
        start = text.index("return 1")
        assert call.apply("app.py", Region(start, start + 8), "return (") is None
        said = call.answer()
    assert isinstance(said, str) and "syntax error" in said
    assert "return (" in (root / "app.py").read_text(encoding="utf-8")


class _DiskCheck:
    """`node --check` as the box runs it: it reads the file on disk, and fails a file that
    holds `BROKEN` or does not exist. Records what it read."""

    def __init__(self) -> None:
        self.read: list[str | None] = []

    def __call__(self, argv: tuple[str, ...], cwd: Path, timeout_s: int) -> ExecResult:
        target = cwd / argv[-1]
        text = target.read_text(encoding="utf-8") if target.is_file() else None
        self.read.append(text)
        bad = text is None or "BROKEN" in text
        return ExecResult(
            exit_code=1 if bad else 0,
            stdout_sha=None,
            stderr_sha=None,
            wall_ms=1,
            timed_out=False,
            sandbox=SandboxLevel.NONE,
            stderr="SyntaxError: BROKEN" if bad else "",
        )


def _js_port(root: Path, check: _DiskCheck, edit_check: str) -> WorktreeEdits:
    checker = SyntaxChecker(root, (load_profile("javascript"),), check)
    return WorktreeEdits(
        root,
        Denylist(),
        checker,
        edit_check=edit_check,  # type: ignore[arg-type]
        untracked_max_bytes=1024,
    )


def test_a_command_check_judges_the_calls_text_not_the_disk(root: Path) -> None:
    check = _DiskCheck()
    said = _whole(_js_port(root, check, "reject"), "web/new.js", "let x = 1;\n")
    assert isinstance(said, str) and "syntax ok: web/new.js" in said
    assert check.read == ["let x = 1;\n"]
    assert (root / "web" / "new.js").read_text(encoding="utf-8") == "let x = 1;\n"


def test_a_command_check_that_fails_under_reject_leaves_the_disk_as_it_was(root: Path) -> None:
    check = _DiskCheck()
    said = _whole(_js_port(root, check, "reject"), "web/deep/new.js", "BROKEN\n")
    assert isinstance(said, DraftRefused) and said.failure == "SYNTAX_FAIL"
    assert not (root / "web").exists()


@pytest.mark.writes_worktree("a.js")
def test_a_command_check_of_an_edit_reads_the_new_text_and_puts_the_old_back(
    root: Path,
) -> None:
    (root / "a.js").write_text("let a = 1;\n", encoding="utf-8")
    check = _DiskCheck()
    said = _whole(_js_port(root, check, "reject"), "a.js", "BROKEN\n")
    assert isinstance(said, DraftRefused) and said.failure == "SYNTAX_FAIL"
    assert check.read[0] == "BROKEN\n"
    assert (root / "a.js").read_text(encoding="utf-8") == "let a = 1;\n"


@pytest.mark.writes_worktree("app.py")
def test_each_call_reads_the_file_as_it_is_now(root: Path) -> None:
    port = _port(root)
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")  # a shell command's change
    with port.edit() as call:
        assert call.text("app.py") == "x = 1\n"


# -- §10.3: delete and move --------------------------------------------------------------------


def test_a_tracked_file_is_deleted(root: Path) -> None:
    with _port(root).edit() as call:
        assert call.delete("pkg/util.py") is None
        assert call.answer() == "ok: pkg/util.py deleted"
    assert not (root / "pkg" / "util.py").exists()


def test_a_move_keeps_the_text_and_the_mode(root: Path) -> None:
    with _port(root).edit() as call:
        assert call.move("run.sh", "bin/run.sh") is None
        assert call.answer() == "ok: run.sh moved to bin/run.sh\nsyntax unchecked: bin/run.sh"
    assert not (root / "run.sh").exists()
    moved = root / "bin" / "run.sh"
    assert moved.read_text(encoding="utf-8") == FILES["run.sh"]
    assert stat.S_IMODE(moved.stat().st_mode) == 0o755


@pytest.mark.writes_worktree("win.py")
def test_a_move_keeps_a_crlf_files_newlines(root: Path) -> None:
    (root / "win.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    git(root, "add", "win.py")
    with _port(root).edit() as call:
        assert call.move("win.py", "pkg/win.py") is None
        call.answer()
    assert (root / "pkg" / "win.py").read_bytes() == b"a = 1\r\nb = 2\r\n"


@pytest.mark.writes_worktree("broken.py")
def test_a_move_of_a_file_already_broken_is_not_refused_under_reject(root: Path) -> None:
    (root / "broken.py").write_text("def f(:\n", encoding="utf-8")
    git(root, "add", "broken.py")
    checker = SyntaxChecker(root, (load_profile("python"),))
    port = WorktreeEdits(
        root, Denylist(), checker, edit_check="reject", untracked_max_bytes=1024
    )
    with port.edit() as call:
        assert call.move("broken.py", "pkg/broken.py") is None
        said = call.answer()
    assert isinstance(said, str) and "as before this call" in said
    assert (root / "pkg" / "broken.py").exists()


@pytest.mark.writes_worktree("notes.txt")
def test_an_untracked_file_within_the_cap_may_go(root: Path) -> None:
    (root / "notes.txt").write_text("n\n", encoding="utf-8")
    with _port(root).edit() as call:
        assert call.delete("notes.txt") is None
        call.answer()
    assert not (root / "notes.txt").exists()


@pytest.mark.writes_worktree("build*", "big.bin")
@pytest.mark.parametrize(
    ("make", "path", "words"),
    [
        ("ignored", "build/out.txt", "git-ignored"),
        ("big", "big.bin", "over the snapshot"),
        ("dir", "pkg", "directory"),
    ],
)
def test_what_undo_could_not_bring_back_is_refused_and_kept(
    root: Path, make: str, path: str, words: str
) -> None:
    if make == "ignored":
        (root / "build").mkdir()
        (root / "build" / "out.txt").write_text("o\n", encoding="utf-8")
    if make == "big":
        (root / "big.bin").write_text("x" * 64, encoding="utf-8")
    port = _port(root, cap=16)
    if make == "dir":
        assert words in _refused(port.removable(path), "DENIED")
        return
    with port.edit() as call:
        refused = call.delete(path)
    assert words in _refused(refused, "DENIED")
    assert "delete it with the shell" in _refused(refused, "DENIED")
    assert (root / path).exists()


def test_a_move_never_replaces_an_existing_file(root: Path) -> None:
    with _port(root).edit() as call:
        refused = call.move("app.py", "pkg/util.py")
    assert "never replaces" in _refused(refused, "DENIED")
    assert (root / "app.py").is_file()


@pytest.mark.writes_worktree(".env")
def test_a_secret_path_is_never_removable(root: Path) -> None:
    (root / ".env").write_text("T=1\n", encoding="utf-8")
    assert "secret path" in _refused(_port(root).removable(".env"), "DENIED")
