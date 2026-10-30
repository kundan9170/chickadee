"""What `read` refuses, what it does not, and what it never sends twice.

`read` refuses exactly one class of path, the deny list, by the name asked for and by where a
link lands. A path outside the workspace is an ordinary read with its honest consequences: no
outline, diagnostics unavailable, and a name that says where it is. Every other refusal is
typed and names its field, and a refused call is recorded as a call that did nothing. Lines
already on screen are pointed at rather than sent again, but only while the view that holds
them is current and still expanded.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from context_helpers import (
    Caller,
    DiagSet,
    FakeDiagnostics,
    FakeIndex,
    Sym,
    cut,
    new_session,
    numbered,
    png,
    python_module,
    reader_for,
    reopen,
    shown,
    work_unit,
    write_files,
)
from refusals import bare, instruction, stated
from tool_args import checked, zeus_answer

import chickadee.context.opener as opener_module
from chickadee.context.files import open_nonblocking
from chickadee.context.freshness import CHANGED_NOTE
from chickadee.context.homes import ReadHome, Scratch, SessionHome
from chickadee.context.pattern import GREP_TIMEOUT_S
from chickadee.context.results import (
    HandleExpired,
    IndexUnavailable,
    InvalidArgument,
    NotSupported,
    SecretPath,
    ToolResult,
)
from chickadee.context.session import Session
from chickadee.context.spans import Span
from chickadee.context.store import SessionCorrupt, session_dir
from chickadee.context.terms import Caps, Vision
from chickadee.journal.errors import BlobMissing
from chickadee.schema import ToolCall
from chickadee.secrets.denylist import Denylist
from chickadee.tools.validate import Arguments

LIB, LIB_SYMBOLS = python_module(12)


@pytest.fixture
def repo(tmp_path: Path, ws: Path) -> Path:
    """A workspace holding secrets and links to them, beside a sibling checkout and a home
    directory that holds one more."""
    write_files(
        ws,
        {
            "a.py": numbered(100),
            "lib.py": LIB,
            "pkg/mod.py": "x = 1\n",
            ".env": "TOKEN=hunter2\n",
            "keys/server.pem": "-----BEGIN-----\n",
        },
    )
    write_files(tmp_path, {"sibling/lib.py": LIB, "home/.ssh/config": "Host *\n"})
    os.symlink(ws / ".env", ws / "notes.txt")
    os.symlink(tmp_path / "home" / ".ssh" / "config", ws / "ssh.cfg")
    return ws


def read_ok(result: object) -> ToolResult:
    assert isinstance(result, ToolResult), result
    return result


# -- the one class of path read refuses ---------------------------------------------


@pytest.mark.parametrize(
    ("target", "pattern"),
    [
        pytest.param(".env", ".env", id="by-name"),
        pytest.param("keys/server.pem", "*.pem", id="by-format"),
        pytest.param("notes.txt", ".env", id="a-link-to-one"),
        pytest.param("ssh.cfg", ".ssh/**", id="a-link-out-of-the-workspace-to-one"),
        pytest.param("{outside}/home/.ssh/config", ".ssh/**", id="outside-the-workspace"),
    ],
)
def test_a_secret_is_refused_by_its_name_and_by_where_a_link_lands(
    tmp_path: Path, session: Session, repo: Path, target: str, pattern: str
) -> None:
    asked = target.format(outside=tmp_path)
    assert reader_for(session, repo).read(asked) == SecretPath(asked, pattern)
    assert len(session.state.handles) == 0, "nothing was opened, so nothing was minted"


@pytest.mark.parametrize("where", ["{home}/.docker/config.json", "sub/.docker/config.json"])
def test_a_secret_path_is_refused_however_deep_it_sits(
    tmp_path: Path, session: Session, repo: Path, where: str
) -> None:
    """A directory-shaped pattern names the file wherever it sits: by absolute path outside
    the workspace, and nested inside it."""
    home = tmp_path / "home"
    write_files(home, {".docker/config.json": '{"auths": {"r": {"auth": "hunter2"}}}\n'})
    write_files(repo, {"sub/.docker/config.json": '{"auths": {}}\n'})
    asked = where.format(home=home)
    assert reader_for(session, repo).read(asked, lines="1-1") == SecretPath(
        asked, ".docker/config.json"
    )


def test_a_handle_cannot_launder_a_secret_through_a_later_link(
    session: Session, repo: Path
) -> None:
    """The deny list is checked on every open: a file swapped for a link to a secret after
    its handle was minted is refused through that handle, by read and by grep_in."""
    reader = reader_for(session, repo)
    write_files(repo, {"plain.txt": "A=0\n"})
    minted = read_ok(reader.read("plain.txt")).handle.id
    (repo / "plain.txt").unlink()
    os.symlink(repo / ".env", repo / "plain.txt")
    assert reader.read(minted, lines="1-1") == SecretPath("plain.txt", ".env")
    assert reader.grep_in(minted, "TOKEN") == SecretPath("plain.txt", ".env")


def test_a_fifo_or_a_device_is_refused_without_reading_a_byte(
    session: Session, repo: Path
) -> None:
    """Reading one could block for a writer or stream forever, under the session's lock."""
    os.mkfifo(repo / "pipe")
    reader = reader_for(session, repo)
    assert bare(reader.read("pipe")) == InvalidArgument(
        "path", "pipe is not a regular file; read opens files"
    )
    assert bare(reader.read("/dev/zero")) == InvalidArgument(
        "path", "/dev/zero is not a regular file; read opens files"
    )


def test_a_fifo_swapped_in_just_before_the_open_cannot_hold_it(
    session: Session, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is judged by the descriptor it opens, not by a stat taken before: a regular
    file replaced by a FIFO at the last moment is refused, never waited on."""
    write_files(repo, {"notes.log": "a\n"})
    target = repo / "notes.log"
    opening = open_nonblocking

    def swap_then_open(path: Path) -> int:
        if path == target and not stat.S_ISFIFO(os.lstat(target).st_mode):
            target.unlink()
            os.mkfifo(target)
        return opening(path)

    monkeypatch.setattr(opener_module, "open_nonblocking", swap_then_open)
    result: list[object] = []
    reading = threading.Thread(
        target=lambda: result.append(reader_for(session, repo).read("notes.log")), daemon=True
    )
    reading.start()
    reading.join(5)
    assert not reading.is_alive(), "the open waited for a writer"
    assert [bare(r) for r in result] == [
        InvalidArgument("path", "notes.log is not a regular file; read opens files")
    ]


@pytest.mark.parametrize(
    ("pattern", "why"),
    [
        pytest.param("(" * 100 + "a" + ")" * 100, "groups nested deeper than", id="nesting"),
        pytest.param("(" * 500 + "a" + ")" * 500, "refused", id="the-reviewers-500"),
        pytest.param("a" * 5000, "longer than", id="length"),
    ],
)
def test_a_pattern_too_complex_to_compile_is_refused_not_raised(
    session: Session, repo: Path, pattern: str, why: str
) -> None:
    """The search timeout covers matching only; compiling recurses on nesting, so a pattern
    past the bounds is refused before either compiler sees it."""
    reader = reader_for(session, repo)
    opened = read_ok(reader.read("a.py"))
    found = reader.grep_in(opened.handle.id, pattern)
    assert isinstance(found, InvalidArgument)
    assert found.field == "pattern"
    assert why in found.why


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="fd count is read from /proc")
def test_a_read_that_opens_nothing_readable_leaks_no_descriptor(
    session: Session, repo: Path
) -> None:
    """A directory is opened before it is judged, and `open` does not close a descriptor it
    could not wrap. Unclosed, a model reading directory paths exhausts the process's
    descriptors."""
    reader = reader_for(session, repo)
    assert bare(reader.read("pkg")) == InvalidArgument(
        "path", "pkg is a directory; read opens files"
    )
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        reader.read("pkg")
    assert len(os.listdir("/proc/self/fd")) == before


def test_a_file_past_the_file_bound_is_refused_whole(session: Session, repo: Path) -> None:
    write_files(repo, {"huge.log": "x" * 5000})
    reader = reader_for(session, repo, caps=Caps(file_bytes=4096))
    assert bare(reader.read("huge.log")) == InvalidArgument(
        "path", "huge.log is over 4,096 bytes; read opens files up to that"
    )


def test_a_pattern_that_backtracks_without_end_is_stopped_and_named(
    session: Session, repo: Path
) -> None:
    write_files(repo, {"xs.txt": "x" * 5000 + "\n"})
    reader = reader_for(session, repo)
    opened = read_ok(reader.read("xs.txt"))
    started = time.monotonic()
    found = reader.grep_in(opened.handle.id, "(x+x+)+y")
    assert time.monotonic() - started < GREP_TIMEOUT_S + 2
    assert isinstance(found, InvalidArgument)
    assert found.field == "pattern"
    assert "did not finish" in found.why


def test_a_refused_read_is_recorded_as_a_call_that_did_nothing(
    session: Session, repo: Path
) -> None:
    """The model reads the typed refusal, and the secret's bytes are nowhere in the session:
    not in its log, and not in its blob store."""
    call = Caller(session, reader_for(session, repo))
    session.record_user("what is the token?")
    line = call("read", path="notes.txt", lines="1-1")
    assert (line.ran, line.handle, line.view) == (False, None, ())
    assert stated(line.text) == "SECRET_PATH{path: notes.txt, pattern: '.env'}"
    assert instruction(line.text)
    directory = session_dir(session.root, session.session_id)
    written = [p.read_bytes() for p in directory.rglob("*") if p.is_file()]
    assert written, "the scan found the session's files"
    assert not [data for data in written if b"hunter2" in data]


def test_a_handle_whose_file_became_a_link_to_a_secret_is_refused_on_reuse(
    session: Session, repo: Path
) -> None:
    """A handle names a path, and a reuse reads what the path reaches now. A link swapped in
    after the handle was minted is refused as the secret it reaches, by `read` and `grep_in`
    alike, and the secret gets no handle of its own."""
    reader = reader_for(session, repo)
    minted = read_ok(reader.read("pkg/mod.py")).handle
    target = repo / "pkg" / "mod.py"
    target.unlink()
    os.symlink(repo / ".env", target)
    refused = SecretPath("pkg/mod.py", ".env")
    assert reader.read(minted.id) == refused
    assert reader.read(minted, lines="1-1") == refused
    assert reader.grep_in(minted.id, "TOKEN") == refused
    assert reader.read("pkg/mod.py") == refused
    assert [h.id for h in session.state.handles] == [minted.id]


def swap_on_open(
    monkeypatch: pytest.MonkeyPatch, target: Path, secret: Path, *, before: bool
) -> None:
    """Replace `target` with a link to `secret` at the moment it is opened: just before the
    open, or just after it."""
    opening = open_nonblocking

    def swap() -> None:
        if not target.is_symlink():
            target.unlink()
            os.symlink(secret, target)

    def open_(path: Path) -> int:
        if path == target and before:
            swap()
        fd = opening(path)
        if path == target and not before:
            swap()
        return fd

    monkeypatch.setattr(opener_module, "open_nonblocking", open_)


@pytest.mark.parametrize(
    ("before", "refused"),
    [
        pytest.param(True, SecretPath("pkg/mod.py", ".env"), id="just-before-the-open"),
        pytest.param(
            False,
            InvalidArgument(
                "path",
                "pkg/mod.py cannot be read",
                do="do not read it again; go on without it, or ask the person about it",
            ),
            id="just-after-the-open",
        ),
    ],
)
def test_a_link_swapped_in_as_a_file_is_opened_is_never_read(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
    repo: Path,
    before: bool,
    refused: SecretPath | InvalidArgument,
) -> None:
    """The list is asked about the file that was opened, not only about the name before it.
    Swapped in just before the open, the link reaches the secret and is refused as one.
    Swapped in just after, the name no longer reaches the file that is open, so nothing can
    say what was opened, and nothing is read."""
    swap_on_open(monkeypatch, repo / "pkg" / "mod.py", repo / ".env", before=before)
    assert reader_for(session, repo).read("pkg/mod.py") == refused
    assert len(session.state.handles) == 0


# -- outside the workspace is an ordinary read ---------------------------------------


def test_a_file_outside_the_workspace_is_read_with_its_honest_consequences(
    tmp_path: Path, session: Session, repo: Path
) -> None:
    """The sibling holds the same bytes as the workspace's indexed `lib.py`, and borrows
    none of that file's outline or diagnostics: nothing indexed or analyzed it, and the read
    says so rather than going quiet."""
    reader = reader_for(
        session,
        repo,
        index=FakeIndex({"lib.py": LIB_SYMBOLS}, repo),
        diagnostics=FakeDiagnostics({"lib.py": DiagSet(coverage="full")}),
    )
    inside = read_ok(reader.read("lib.py"))
    assert inside.handle.shape.symbols == len(LIB_SYMBOLS)
    assert inside.diagnostics == ("[diagnostics: none]",)

    where = (tmp_path / "sibling" / "lib.py").as_posix()
    outside = read_ok(reader.read("../sibling/lib.py"))
    h = outside.handle
    assert (h.path, h.shape.symbols, h.shape.lines) == (where, None, len(LIB.splitlines()))
    assert h.id != inside.handle.id, "the same bytes at another path are another file"
    assert outside.content is None, "a bare read is the preview, never the file"
    assert "no index for this file, so no outline" in outside.render()
    assert outside.diagnostics == (
        "[diagnostics: unavailable — outside the workspace, nothing analyzes it]",
    )
    assert shown(read_ok(reader.read(where, lines="3-4")).render()) == {3, 4}
    unindexed = IndexUnavailable(where, "none: outside the workspace, nothing indexes it")
    assert reader.read(where, depth="signatures") == unindexed
    assert reader.read(where, at="f2") == unindexed


def test_one_file_is_one_handle_however_its_path_is_spelled(
    tmp_path: Path, session: Session, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside the workspace a file is named relative to it; outside, absolutely."""
    monkeypatch.setenv("HOME", str(tmp_path))
    reader = reader_for(session, repo)

    def named(*spellings: str) -> set[tuple[str, str | None]]:
        return {(h.id, h.path) for h in (read_ok(reader.read(s)).handle for s in spellings)}

    ((_, inside),) = named("a.py", "./a.py", "pkg/../a.py", str(repo / "a.py"))
    assert inside == "a.py"
    sibling = tmp_path / "sibling" / "lib.py"
    ((_, outside),) = named("../sibling/lib.py", "~/sibling/lib.py", str(sibling))
    assert outside == sibling.as_posix()


# -- handles live and die with their session ----------------------------------------


def test_a_handle_is_known_only_to_the_session_that_minted_it(
    tmp_path: Path, session: Session, repo: Path
) -> None:
    minted = read_ok(reader_for(session, repo).read("a.py")).handle.id
    other = new_session(tmp_path, "S-other")
    try:
        stranger = reader_for(other, repo)
        expired = HandleExpired(minted, "not_in_session")
        assert stranger.read(minted) == expired
        assert stranger.grep_in(minted, "line") == expired
        write_files(repo, {minted: "a file that shares the name\n"})
        assert read_ok(stranger.read(minted)).handle.path == minted, "a file is still a file"
    finally:
        other.close()


# -- lines already on screen are not sent again -------------------------------------


def test_lines_already_on_screen_are_pointed_at_and_not_sent_again(
    session: Session, repo: Path
) -> None:
    """A preview that shows lines is a view like any other, so asking for lines inside it,
    or for the preview again, sends nothing new."""
    call = Caller(session, reader_for(session, repo))
    session.record_user("look at a.py")
    first = call("read", path="a.py")
    assert first.handle is not None
    assert first.view == (Span(1, 20), Span(81, 100))
    for again in (call("read", path="a.py", lines="3-5"), call("read", path="a.py")):
        assert (shown(again.text), again.view) == (set(), ())
        assert again.text.endswith(f"[you already hold view 1-20, 81-100 ({first.handle.id})]")


# -- a reader of its own, for a conversation that is not Zeus's -----------------------


SCOUT = Vision("scout")


def test_both_homes_are_read_homes(session: Session, repo: Path) -> None:
    """The reader asks its home, in its own terms, and never which home it has. Zeus's
    session and a scratch both satisfy `ReadHome` (pyright checks the list's type). The
    session answers from Zeus's context; a scratch keeps none, so nothing is held or merges,
    and a blob it never kept is `BlobMissing`, which the resolver types as expired."""
    homes: list[ReadHome] = [SessionHome(session), Scratch()]
    zeus, scratch = homes
    assert zeus.handles() is session.state.handles
    call = Caller(session, reader_for(session, repo))
    session.record_user("look at a.py")
    first = call("read", path="a.py", lines="1-20")
    assert first.handle is not None
    assert zeus.held(first.handle.id) == (Span(1, 20),)
    assert zeus.merge_target(first.handle.id) is not None
    assert zeus.reads_in(first.call_id) == 1

    assert (scratch.held(first.handle.id), scratch.merge_target(first.handle.id)) == ((), None)
    sha = scratch.put_blob(b"bytes")
    assert scratch.blob(sha) == b"bytes"
    with pytest.raises(BlobMissing):
        scratch.blob("0" * 64)


def test_a_scratch_reader_sends_the_lines_zeus_already_holds(
    session: Session, repo: Path
) -> None:
    """D140, now by D146: the held check answers from Zeus's context. Zeus, holding 1-20, is
    pointed at them; the scout's own reader has no context but its own, and gets the lines."""
    reader = reader_for(session, repo)
    call = Caller(session, reader)
    session.record_user("look at a.py")
    first = call("read", path="a.py", lines="1-20")
    assert first.handle is not None

    held = call("read", path="a.py", lines="1-20")
    assert shown(held.text) == set()
    assert "already hold" in held.text

    scout = read_ok(reader.scratch(vision=SCOUT).read("a.py", lines="1-20"))
    assert shown(scout.render()) == set(range(1, 21))
    assert "already hold" not in scout.render()
    assert scout.view == (Span(1, 20),)


def test_a_scratch_reader_never_merges_into_zeus_s_view(session: Session, repo: Path) -> None:
    """Zeus's newest view of a.py is near the tail, so Zeus's next read would merge into it.
    The scout's read is its own: the lines it asked for, not Zeus's re-rendered."""
    reader = reader_for(session, repo)
    call = Caller(session, reader)
    session.record_user("look at a.py")
    call("read", path="a.py", lines="1-20")

    scout = read_ok(reader.scratch(vision=SCOUT).read("a.py", lines="30-35"))
    assert shown(scout.render()) == set(range(30, 36))
    assert scout.view == (Span(30, 35),)


@pytest.mark.parametrize("target", [".env", "notes.txt", "keys/server.pem"])
def test_a_scratch_reader_refuses_what_zeus_s_reader_refuses(
    session: Session, repo: Path, target: str
) -> None:
    """One deny list: the scout can read nothing Zeus cannot."""
    reader = reader_for(session, repo)
    assert reader.scratch(vision=SCOUT).read(target) == reader.read(target)
    assert isinstance(reader.scratch(vision=SCOUT).read(target), SecretPath)


def test_a_scratch_reader_shares_the_deny_list_itself(session: Session, repo: Path) -> None:
    """The session's own list, not a default one: a pattern the person added to it refuses
    for the scout too, while the rest of the workspace still reads."""
    denylist = Denylist(patterns=(*Denylist().patterns, "pkg/**"))
    reader = reader_for(session, repo, denylist=denylist)
    scout = reader.scratch(vision=SCOUT)
    assert isinstance(scout.read("pkg/mod.py"), SecretPath)
    assert read_ok(scout.read("a.py", lines="1-2")).handle.path == "a.py"


def test_a_scratch_reader_s_handles_are_its_own(session: Session, repo: Path) -> None:
    """The scout's handles live in its scratch and die with it. Zeus's session never
    sees them, and one scratch does not see another's."""
    reader = reader_for(session, repo)
    scout = reader.scratch(vision=SCOUT)
    minted = read_ok(scout.read("lib.py", lines="1-5")).handle

    assert len(session.state.handles) == 0, "nothing was minted into Zeus's session"
    assert read_ok(scout.read(minted.id, lines="1-5")).handle.id == minted.id
    assert reader.read(minted.id) == HandleExpired(minted.id, "not_in_session")
    other = reader.scratch(vision=SCOUT)
    assert other.read(minted.id) == HandleExpired(minted.id, "not_in_conversation")
    assert other.grep_in(minted.id, "def") == HandleExpired(minted.id, "not_in_conversation")


def test_a_scratch_reader_writes_nothing_into_the_session(session: Session, repo: Path) -> None:
    """An image's original is kept so its handle still resolves. The scout's is kept in
    its scratch, never in Zeus's session directory."""
    write_files(repo, {"shot.png": png(64, 48)})
    directory = session_dir(session.root, session.session_id)
    before = sorted(p for p in directory.rglob("*") if p.is_file())

    scout = reader_for(session, repo).scratch(vision=Vision("scout", supports_images=True))
    image = read_ok(scout.read("shot.png", depth="full"))
    assert image.image is not None
    assert read_ok(scout.read(image.handle.id, depth="full")).image is not None

    assert sorted(p for p in directory.rglob("*") if p.is_file()) == before


def test_a_scratch_reader_reads_back_what_it_stores(session: Session, repo: Path) -> None:
    """Bytes stored through a scratch reader are its own blob: they read back through it, and
    Zeus's session neither holds the handle nor the bytes."""
    scout = reader_for(session, repo).scratch(vision=SCOUT)
    stored = scout.store("output", b"line one\nline two\n")

    back = read_ok(scout.read(stored.id, lines="1-2"))
    assert shown(back.render()) == {1, 2}
    assert stored.id not in session.state.handles
    with pytest.raises(SessionCorrupt):
        session.blob(stored.content_sha)


def test_a_scratch_reader_sees_as_its_own_rung(session: Session, repo: Path) -> None:
    """Its vision is the rung's it was made for, not Zeus's: the refusal names
    the scout's rung, and a scout that can see gets the image though Zeus's reader cannot."""
    write_files(repo, {"shot.png": png(64, 48)})
    reader = reader_for(session, repo, vision=Vision("zeus"))

    blind = reader.scratch(vision=Vision("scout")).read("shot.png", depth="full")
    assert blind == NotSupported("images", "scout")
    seeing = reader.scratch(vision=Vision("scout", supports_images=True))
    assert read_ok(seeing.read("shot.png", depth="full")).image is not None
    assert reader.read("shot.png", depth="full") == NotSupported("images", "zeus")


def test_a_reader_records_nothing(session: Session, repo: Path) -> None:
    """The reader reads and mints; the tool answering a call records its result (Zeus's
    `ReadTools`). A read through Zeus's reader adds no line to the log, and one through
    a scratch reader leaves even the handle table alone."""
    before = session.state.last_seq
    read_ok(reader_for(session, repo).read("a.py", lines="1-3"))
    read_ok(reader_for(session, repo).scratch(vision=SCOUT).read("lib.py", lines="1-3"))
    assert session.state.last_seq == before
    assert [h.path for h in session.state.handles] == ["a.py"]


def test_a_view_holds_its_lines_only_while_it_is_current_and_on_screen(
    session: Session, repo: Path
) -> None:
    """§2.1 looks for an unexpired, NON-STALE view. A cut leaves a receipt, which holds no
    lines; an edit leaves a view of bytes the file no longer has. Either way the lines are
    sent again."""
    call = Caller(session, reader_for(session, repo))
    session.record_user("look at a.py")
    first = call("read", path="a.py", lines="1-10")
    assert first.handle is not None
    session.record_assistant("read it", findings={first.call_id: "a.py counts lines"})
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("thanks")
    session.record_assistant("welcome")
    done = cut(session, repo)
    assert done.cut is not None
    assert done.cut.boundary > first.seq

    assert shown(call("read", path="a.py", lines="3-5").text) == {3, 4, 5}
    assert call("read", path="a.py", lines="4-4").view == (), "on screen again, so held"
    write_files(repo, {"a.py": numbered(100).replace("line 50\n", "line fifty\n")})
    moved = call("read", path="a.py", lines="3-5")
    assert shown(moved.text) == {3, 4, 5}
    assert moved.handle is not None
    assert moved.handle.id != first.handle.id


# -- a read the reader cannot do is refused by name, and did nothing -----------------------
#
# These are the reader's own checks, made by `read` and `grep_in` themselves: they need the file
# or the handle to judge, so no tool schema can make them. Each goes through Zeus's `ReadTools`,
# the path that answers and records a call, and lands as a call that did nothing.


READER_REFUSALS = [
    pytest.param(
        "read",
        {"path": "a.py", "lines": "3"},
        "lines",
        "'3' is not a span of the form start-end",
        id="lines-not-a-span",
    ),
    pytest.param(
        "read",
        {"path": "a.py", "lines": "5-3"},
        "lines",
        "span end 3 precedes its start 5",
        id="lines-backwards",
    ),
    pytest.param(
        "read",
        {"path": "a.py", "lines": "500-510"},
        "lines",
        "end of file: 100 lines; you asked from 500",
        id="lines-past-the-end",
    ),
    pytest.param(
        "read",
        {"path": "a.py", "lines": "1-2", "at": "f1"},
        "at",
        "lines= and at= both name a region",
        id="lines-and-at",
    ),
    pytest.param("read", {"path": "a.py", "at": " "}, "at", "at= is empty", id="blank-at"),
    pytest.param(
        "read",
        {"path": "a.py", "depth": "deep"},
        "depth",
        "'deep' is not outline, signatures or full",
        id="unknown-depth",
    ),
    pytest.param(
        "read",
        {"path": "a.py", "max_bytes": 0},
        "max_bytes",
        "a cap is at least one byte",
        id="zero-cap",
    ),
    pytest.param(
        "read", {"path": ""}, "path", "a path or a handle is required", id="empty-target"
    ),
    pytest.param(
        "read", {"path": "  "}, "path", "a path or a handle is required", id="blank-target"
    ),
    pytest.param(
        "read",
        {"path": "pkg"},
        "path",
        "pkg is a directory; read opens files",
        id="a-directory",
    ),
    pytest.param(
        "read", {"path": "missing.py"}, "path", "missing.py does not exist", id="missing"
    ),
    pytest.param(
        "grep_in",
        {"handle": "a.py", "pattern": "line"},
        "handle",
        "grep_in searches a handle read returned",
        id="grep-a-path",
    ),
    pytest.param(
        "grep_in",
        {"handle": "{h}", "pattern": "("},
        "pattern",
        "not a regular expression: missing ), unterminated subpattern at position 0",
        id="grep-not-a-regex",
    ),
    pytest.param(
        "grep_in",
        {"handle": "{h}", "pattern": ""},
        "pattern",
        "the pattern is empty, which matches every line",
        id="grep-empty-pattern",
    ),
    pytest.param(
        "grep_in",
        {"handle": "{h}", "pattern": "line", "context": -1},
        "context",
        "context is zero or more lines",
        id="grep-negative-context",
    ),
]


@pytest.mark.parametrize(("tool", "args", "field", "why"), READER_REFUSALS)
def test_a_read_the_reader_cannot_do_is_refused_by_name_and_did_nothing(
    session: Session,
    repo: Path,
    tool: str,
    args: dict[str, object],
    field: str,
    why: str,
) -> None:
    reader = reader_for(session, repo)
    session.record_user("look at a.py")
    opened = Caller(session, reader)("read", path="a.py")
    assert opened.handle is not None
    sent = {k: opened.handle.id if v == "{h}" else v for k, v in args.items()}
    session.record_assistant("", [ToolCall(id="bad", name=tool, arguments=sent)])
    # Built directly, not through `validate`: several of these the schema refuses first, and
    # this is the reader's own refusal behind it.
    line = zeus_answer(session, reader, "bad", tool, Arguments(tool, sent))
    assert (line.ran, line.handle, line.view) == (False, None, ())
    assert stated(line.text) == InvalidArgument(field, why).render()
    assert instruction(line.text)


# -- store ---------------------------------------------------------------------------------


def test_store_keeps_only_bytes_that_exist_nowhere_else(session: Session, repo: Path) -> None:
    """A file handle is an address, read back from disk; `store` is for an output, a diff or a
    log, which only the blob holds."""
    with pytest.raises(ValueError, match="a file handle is an address"):
        reader_for(session, repo).store("file", b"x = 1\n")


# -- the read path is parallel-safe -------------------------------------------------


def test_reads_answered_at_once_each_land_once_and_replay_as_they_were(
    session: Session, repo: Path
) -> None:
    """Zeus's sink holds the session's lock from the read's merge decision to its
    recorded line, so reads of one file answered together never absorb the same earlier view,
    and the log replays to exactly what was live. The reader records nothing; the tool does."""
    reader = reader_for(session, repo)
    session.record_user("look all through a.py")
    calls = [
        ToolCall(id=f"p{i}", name="read", arguments={"path": "a.py", "lines": f"{s}-{s + 2}"})
        for i, s in enumerate(range(1, 100, 12))
    ]
    session.record_assistant("", calls)
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        answered = list(
            pool.map(
                lambda c: zeus_answer(
                    session, reader, c.id, c.name, checked(c.name, c.arguments)
                ),
                calls,
            )
        )
    assert sorted(line.seq for line in answered) == list(range(3, 3 + len(calls)))
    assert all(line.ran for line in answered)
    assert session.state.pending_calls() == ()
    absorbed = [cid for line in answered for cid in line.supersedes]
    assert absorbed, "the reads merged"
    assert len(absorbed) == len(set(absorbed)), "no view was absorbed twice"
    live = (list(session.state.lines()), session.state.views())
    with reopen(session) as again:
        assert (list(again.state.lines()), again.state.views()) == live


class _Aids:
    """A graph connection of an agent's own, structurally: an index and its diagnostics."""

    def __init__(self, index: FakeIndex, diagnostics: FakeDiagnostics) -> None:
        self.index = index
        self.diagnostics = diagnostics

    def file_known(self, path: str) -> bool:
        return self.index.file_known(path)

    def symbols_in(self, path: str) -> Sequence[Sym]:
        return self.index.symbols_in(path)

    def indexed_sha(self, path: str) -> str | None:
        return self.index.indexed_sha(path)

    def diagnostics_in(self, path: str, lines: range | None = None) -> DiagSet:
        return self.diagnostics.diagnostics_in(path, lines)


def test_a_scratch_reader_on_another_thread_reads_through_aids_of_its_own(
    session: Session, repo: Path
) -> None:
    """An agent's reader runs on its own thread, so it reads the
    index and diagnostics through a connection it opened there, never through Zeus's. Shared
    is the scout's default; None is a reader with no index."""
    zeus_index = FakeIndex({"lib.py": LIB_SYMBOLS}, repo)
    reader = reader_for(
        session,
        repo,
        index=zeus_index,
        diagnostics=FakeDiagnostics({"lib.py": DiagSet(coverage="full")}),
    )
    own = _Aids(FakeIndex({"lib.py": LIB_SYMBOLS[:3]}, repo), FakeDiagnostics())
    agent = reader.scratch(vision=SCOUT, aids=own)
    got = read_ok(agent.read("lib.py"))
    assert got.handle.shape.symbols == 3, "its own index, not Zeus's twelve"
    assert got.diagnostics != ("[diagnostics: none]",), "its own diagnostics: none analyzed"
    assert read_ok(reader.scratch(vision=SCOUT).read("lib.py")).handle.shape.symbols == 12
    bare = read_ok(reader.scratch(vision=SCOUT, aids=None).read("lib.py"))
    assert bare.handle.shape.symbols is None


# -- a handle as text, for an agent --------------------------------------


def test_text_of_a_handle_is_its_whole_text_opened_as_read_opens_it(
    session: Session, repo: Path
) -> None:
    """The second review of #640: an agent was given a handle's bytes off the disk, unchecked.
    It is given them as `read` opens them now: the whole text, minting nothing."""
    reader = reader_for(session, repo)
    held = read_ok(reader.read("lib.py")).handle
    minted = len(session.state.handles)
    assert reader.text_of(held.id) == (repo / "lib.py").read_text(encoding="utf-8")
    assert len(session.state.handles) == minted
    assert isinstance(reader.text_of("h_nope00"), HandleExpired)


def test_text_of_a_changed_file_says_so_and_mints_nothing(session: Session, repo: Path) -> None:
    reader = reader_for(session, repo)
    held = read_ok(reader.read("lib.py")).handle
    minted = len(session.state.handles)
    (repo / "lib.py").write_text("x = 2\n", encoding="utf-8")
    got = reader.text_of(held.id)
    assert isinstance(got, str) and got.endswith("x = 2\n") and CHANGED_NOTE in got
    assert len(session.state.handles) == minted


def test_text_of_a_handle_whose_file_is_now_denied_is_refused(
    session: Session, repo: Path
) -> None:
    held = read_ok(reader_for(session, repo).read("lib.py")).handle
    denied = Denylist(patterns=(*Denylist().patterns, "lib.py"))
    assert isinstance(reader_for(session, repo, denylist=denied).text_of(held.id), SecretPath)
