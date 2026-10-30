"""The classifier against the real `/bin/sh` (T-3.158 D).

The command table decides from a line's text what the line will start, and a grant covers the
line on that reading. Whether the reading is what the shell does is otherwise an assumption.
This test makes it a fact: it runs each pinned line as the executor runs it, `/bin/sh -c
<line>` under the sealed profile (network off, the workspace read-only), with every program
the corpus names replaced by a stub that reports its argv on stderr and does nothing else. The
argvs the shell actually started are then held to the classifier's reading.

- **Property 1, the reader.** Every command the shell started is one the reader found: by its
  whole argv when every word in the line is literal, by its program otherwise, since a word the
  shell computes (`$x`, a glob) cannot be read from the text. The reader may find more (a
  wrapper's command, a `sh -c` body, both never started by a stub): over-reading only raises.
- **Property 2, the grant.** For a line a grant could cover, every command the shell started
  binds, read alone, to that same grant: no command a granted line starts does more than the
  grant was given for.

`KNOWN_HOLES` listed the lines that failed on the reader `main` had when T-3.158 began,
measured, not guessed; it could only shrink, and T-3.158 emptied it. A new failure is a
defect, never a line to add.

Builtins (`cd`, `eval`, `true`, …) run as the shell's own and start nothing to see. A line that
names a program by its absolute path would run the real program, so the corpus leaves those
out, and the test names each one it leaves out.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from test_classifier_pins import corpus  # pyright: ignore[reportMissingImports]
from tool_args import checked

from chickadee.approval.commands import binding_of, shipped
from chickadee.approval.grants import grantable
from chickadee.context.handle import Handle
from chickadee.executor.executor import Executor
from chickadee.isolation.base import has
from chickadee.isolation.linux import BINARY
from chickadee.shell import Shell, ShellCommand
from chickadee.shell_line import read_argv, read_line, split

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")

_MARK: Final = "\x1eT3158"
_SEP: Final = "\x1f"
_END: Final = "\x1d"

# The stub: report the program's name and every argument on stderr, separated by bytes no
# corpus line holds, as one write so the stubs of a pipeline never interleave, then exit 0.
# Stderr, because a pipeline's stdout feeds the next stub.
_STUB: Final = (
    "#!/bin/sh\n"
    f"r='{_MARK}{_SEP}'\"${{0##*/}}\"\n"
    f'for a in "$@"; do r="$r"\'{_SEP}\'"$a"; done\n'
    f"printf '%s{_END}' \"$r\" >&2\n"
)

_NAME: Final = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")

# Lines pinned before T-3.158 changed anything, whose reading the shell contradicts. Each fails
# property 1 or 2 on `main`'s reader, and is fixed by the part of T-3.158 named beside it.
KNOWN_HOLES: Final[frozenset[str]] = frozenset()
"""Empty since T-3.158 step 3. On `main`'s reader it held seven lines: the force push inside
`{ }`, an `if` body, a function body and an `eval` string (A.4), and a command no row names
riding the grant beside it, `git push origin main &&./deploy.sh` and two `curl … | sh` (C)."""


def _lines() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The lines to run, and the lines left out because they name a program by path."""
    run: list[str] = []
    left_out: list[str] = []
    for line in corpus():
        if any(word.startswith("/") and Path(word).exists() for word in split(line)):
            left_out.append(line)
        else:
            run.append(line)
    return tuple(run), tuple(left_out)


def _stub_names(lines: tuple[str, ...]) -> set[str]:
    """Every word that could be a program in the corpus: over-stubbing costs nothing."""
    names = {row.prefix[0] for row in shipped().rows}
    for line in lines:
        names |= {w for w in re.split(r"[\s;&|()<>`$'\"]+", line) if _NAME.match(w)}
    return names


def _started(stderr: str) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for chunk in stderr.split(_END):
        at = chunk.find(_MARK)
        if at >= 0:
            fields = chunk[at + len(_MARK) :].split(_SEP)
            out.append(tuple(fields[1:]))
    return tuple(out)


def _base(argv: tuple[str, ...]) -> tuple[str, ...]:
    return (argv[0].rsplit("/", 1)[-1], *argv[1:]) if argv else argv


def _never_stored(data: bytes) -> Handle:
    raise AssertionError("the test runs requests itself; the shell stores nothing")


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, tuple[tuple[str, ...], ...]]]:
    """`{line: the argvs the shell started}` for every line the corpus runs."""
    lines, _ = _lines()
    # Outside `/tmp`: the sealed wrap mounts its own `/tmp`, so a workspace under it would be
    # replaced inside the sandbox (as `zeus/test_no_write.py` notes).
    base = Path("/var/tmp") if Path("/var/tmp").is_dir() else Path(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="t3158-sh-", dir=base))
    stubs, workspace = root / "stubs", root / "ws"
    stubs.mkdir()
    workspace.mkdir()
    # A program named by a path (`./deploy.sh` in the corpus, the table's `vendor/bin/phpunit`)
    # runs from the workspace, never from `PATH`, so its stub is there; every other on `PATH`.
    for name in (*_stub_names(lines), "./deploy.sh"):
        stub = workspace / name if "/" in name else stubs / name
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(_STUB)
        stub.chmod(0o755)
    saved = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{stubs}:/usr/bin:/bin"
    try:
        shell = Shell(Executor(), workspace, _never_stored)
        out: dict[str, tuple[tuple[str, ...], ...]] = {}
        for line in lines:
            parsed = shell.parse(checked("shell", {"command": line}))
            if not isinstance(parsed, ShellCommand):
                out[line] = ()
                continue
            result = Executor().run(shell.request(parsed), operation_id="op-t3158")
            out[line] = _started(result.stderr)
        yield out
    finally:
        os.environ["PATH"] = saved
        shutil.rmtree(root, ignore_errors=True)


def _literal(line: str) -> bool:
    return not re.search(r"[$`*?\[\]{}~\\]", line)


def _read(line: str) -> set[tuple[str, ...]]:
    """What the reader says `line` starts, by program name as a stub reports it."""
    try:
        return {_base(argv) for argv in read_line(line).commands}
    except ValueError:
        return set()


def _reader_holes(runs: dict[str, tuple[tuple[str, ...], ...]]) -> dict[str, str]:
    holes: dict[str, str] = {}
    for line, started in runs.items():
        read = _read(line)
        heads = {argv[0] for argv in read if argv}
        for argv in started:
            # A wrapper's argv is read as the wrapper and the command it starts.
            as_read = {_base(c) for c in read_argv(argv).commands}
            if _literal(line) and argv not in read and not as_read <= read:
                holes[line] = f"started {argv!r}, never read"
            elif not _literal(line) and argv[0] not in heads:
                holes[line] = f"started {argv[0]!r}, never read"
    return holes


def _grant_holes(runs: dict[str, tuple[tuple[str, ...], ...]]) -> dict[str, str]:
    table = shipped()
    holes: dict[str, str] = {}
    for line, started in runs.items():
        granted = grantable(table.classify_line(line))
        if granted is None:
            continue
        for argv in started:
            alone = table.classify(argv)
            bound = binding_of(alone) if alone is not None else None
            if bound != granted:
                holes[line] = f"started {argv!r}, which binds {bound!r}, not the grant"
    return holes


@bwrap_only
def test_the_shell_starts_only_what_the_reader_found(
    runs: dict[str, tuple[tuple[str, ...], ...]],
) -> None:
    holes = _reader_holes(runs)
    unexpected = {k: v for k, v in holes.items() if k not in KNOWN_HOLES}
    assert not unexpected, (
        f"{len(unexpected)} new reader holes: {dict(list(unexpected.items())[:8])}"
    )


@bwrap_only
def test_a_granted_line_starts_nothing_the_grant_does_not_cover(
    runs: dict[str, tuple[tuple[str, ...], ...]],
) -> None:
    holes = _grant_holes(runs)
    unexpected = {k: v for k, v in holes.items() if k not in KNOWN_HOLES}
    assert not unexpected, (
        f"{len(unexpected)} new grant holes: {dict(list(unexpected.items())[:8])}"
    )


@bwrap_only
def test_known_holes_only_shrink(runs: dict[str, tuple[tuple[str, ...], ...]]) -> None:
    """A line fixed leaves `KNOWN_HOLES` in the same change; the list never holds a line that
    passes."""
    failing = set(_reader_holes(runs)) | set(_grant_holes(runs))
    fixed = KNOWN_HOLES - failing
    assert not fixed, f"no longer failing, so remove from KNOWN_HOLES: {sorted(fixed)}"


@bwrap_only
def test_the_stubs_see_the_shell_start_commands(
    runs: dict[str, tuple[tuple[str, ...], ...]],
) -> None:
    """The rig works: a plain line's command is seen, and a pipeline's two."""
    assert runs["git status"] == (("git", "status"),)
    assert sorted(runs["ls; rm x"]) == [("ls",), ("rm", "x")]


def test_lines_naming_a_program_by_path_are_left_out_and_named() -> None:
    _, left_out = _lines()
    for line in left_out:
        assert any(w.startswith("/") for w in split(line)), line
