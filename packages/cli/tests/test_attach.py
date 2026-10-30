"""`chickadee_cli.attach`: staging asks the server what a
file is and keeps it until a turn takes it, a dropped path is read as a terminal pastes one,
and the clipboard is read by the platform's reader on the person's machine."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from chickadee_cli.attach import (
    CLIPBOARD_READERS,
    ClipboardReader,
    NotStaged,
    Staged,
    Staging,
    attachment_line,
    chip_lines,
    clipboard_reader,
    dropped_path,
    read_clipboard,
    ready,
)
from chickadee_serve.frames import AttachLimits, Hello
from chickadee_wire.models import Attachment, AttachmentPreview, Handle, Shape

LIMITS = AttachLimits(
    image_max_edge=1568,
    attachments_max=2,
    attachment_bytes_max=10,
    attachment_file_max_bytes=8,
    pdf_file_max_bytes=8,
    pdf_pages_max=100,
    pdf_text_min_chars=32,
    text_file_max_bytes=8,
    text_inline_bytes=8,
)


class FakeServer:
    def __init__(
        self,
        limits: AttachLimits | None = LIMITS,
        *,
        secret: bool = False,
        sent_bytes: int | None = None,
    ) -> None:
        self.server: Hello | None = Hello(
            wire="4.18", engine="0", capabilities=(), attach=limits
        )
        self.asked: list[str] = []
        self.secret = secret
        self.sent_bytes = sent_bytes
        """What the server says each file is as sent: an image downscaled is smaller."""

    def preview_attachment(self, sid: str, name: str, data: bytes) -> AttachmentPreview:
        self.asked.append(name)
        kind = "image" if name.endswith(".png") else "text"
        return AttachmentPreview(
            name=name,
            kind=kind,
            media_type="image/png" if kind == "image" else "text/plain",
            bytes=len(data) if self.sent_bytes is None else self.sent_bytes,
            size="4x4" if kind == "image" else "1 line",
            est_tokens=12,
            delivery="none" if kind == "image" else "text",
            secret=self.secret,
        )


def staged(staging: Staging, name: str, data: bytes = b"x") -> Staged:
    result = staging.stage(name, data)
    assert isinstance(result, Staged), result
    return result


def test_a_file_is_asked_of_the_server_and_kept_until_a_turn_takes_it() -> None:
    server = FakeServer()
    changes: list[int] = []
    staging = Staging(server, "S-1", changed=lambda: changes.append(1))
    staged(staging, "a.png")
    staged(staging, "b.txt")
    assert server.asked == ["a.png", "b.txt"]
    assert [s.name for s in staging.items()] == ["a.png", "b.txt"]
    taken = staging.take()
    assert Staging.files(taken) == (("a.png", b"x"), ("b.txt", b"x"))
    assert staging.items() == ()
    staging.put_back(taken)
    assert len(staging.items()) == 2
    assert len(changes) == 4


def test_a_chip_says_how_zeus_gets_it_and_the_total() -> None:
    staging = Staging(FakeServer(), "S-1")
    staged(staging, "a.png")
    staged(staging, "b.txt")
    rows = chip_lines(staging.items(), staging.over())
    assert rows == (
        "[1] a.png · 4x4 · ~12 tok · Zeus cannot see this",
        "[2] b.txt · 1 line · ~12 tok",
        "2 attached · ~24 tok",
    )


def test_the_limits_refuse_before_and_after_staging() -> None:
    staging = Staging(FakeServer(), "S-1")
    assert staging.stage("big.txt", b"123456789") == NotStaged("big.txt is over 8 bytes")
    for name in ("1.txt", "2.txt", "3.txt"):
        staged(staging, name)
    assert staging.over() == "3 attached; at most 2 a turn"
    not_sent = ready(staging, lambda _p: "y")
    assert isinstance(not_sent, str)
    assert not_sent.startswith("not sent: 3 attached")
    assert len(staging.items()) == 3, "a refusal leaves the draft as it was"


def test_the_total_is_of_the_files_as_sent_as_the_server_counts_it() -> None:
    # Two 6-byte images the server sends at 2 bytes each: 4 bytes a turn, under the cap of 10,
    # though the bytes as read come to 12.
    staging = Staging(FakeServer(sent_bytes=2), "S-1")
    staged(staging, "a.png", b"123456")
    staged(staging, "b.png", b"123456")
    assert staging.over() is None


def test_a_secret_is_asked_about_once() -> None:
    staging = Staging(FakeServer(secret=True), "S-1")
    staged(staging, "env.txt")
    asked: list[str] = []
    assert isinstance(ready(staging, lambda p: asked.append(p) or "n"), str)
    assert len(staging.items()) == 1
    sent = ready(staging, lambda p: "yes")
    assert isinstance(sent, tuple)
    assert asked == ["env.txt holds a secret value; send it? [y/N] "]


def test_an_older_server_offers_no_attaching() -> None:
    staging = Staging(FakeServer(None), "S-1")
    assert not staging.offered
    assert staging.stage("a.txt", b"x") == NotStaged("this server takes no attachments")


def test_a_file_on_disk_is_read_and_a_missing_one_said(tmp_path: Path) -> None:
    staging = Staging(FakeServer(), "S-1")
    (tmp_path / "a.txt").write_bytes(b"abc")
    assert isinstance(staging.stage_file(tmp_path / "a.txt"), Staged)
    missing = staging.stage_file(tmp_path / "gone.txt")
    assert isinstance(missing, NotStaged)
    assert missing.why.endswith("no such file")


@pytest.mark.parametrize(
    "paste",
    [
        "{p}",
        "'{p}'",
        '"{p}"',
        "file://{p}",
        "  {p}  ",
    ],
)
def test_a_dropped_path_in_each_form_a_terminal_pastes(tmp_path: Path, paste: str) -> None:
    target = tmp_path / "my shot.png"
    target.write_bytes(b"\x89PNG")
    text = paste.format(p=target)
    if paste.startswith("file://"):
        text = text.replace(" ", "%20")
    assert dropped_path(text) == target


def test_a_backslash_escaped_path(tmp_path: Path) -> None:
    target = tmp_path / "my shot.png"
    target.write_bytes(b"\x89PNG")
    assert dropped_path(str(target).replace(" ", "\\ ")) == target


def test_anything_else_is_pasted_as_text(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    assert dropped_path("just words") is None
    assert dropped_path(f"{tmp_path}/a.txt\n{tmp_path}/a.txt") is None
    assert dropped_path(str(tmp_path)) is None, "a directory is not a file"
    assert dropped_path("a.txt") is None, "a relative path is a mention, not a drop"


def test_the_reader_is_the_first_that_applies_and_is_installed() -> None:
    def reader(name: str, applies: bool) -> ClipboardReader:
        return ClipboardReader(name, (name,), lambda: applies, f"install {name}")

    readers = (reader("a", False), reader("b", True), reader("c", True))
    found = clipboard_reader(readers, which=lambda p: f"/bin/{p}" if p == "c" else None)
    assert isinstance(found, ClipboardReader)
    assert found.program == "c"
    assert clipboard_reader(readers, which=lambda _p: None) == NotStaged(
        "no clipboard reader: install b"
    )
    assert [r.program for r in CLIPBOARD_READERS] == [
        "osascript",
        "wl-paste",
        "xclip",
        "powershell.exe",
    ]


def _ran(code: int, out: bytes = b"", err: bytes = b"") -> object:
    def run(argv: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, code, out, err)

    return run


def test_the_clipboards_image_or_why_there_is_none() -> None:
    wl = CLIPBOARD_READERS[1]
    png = b"\x89PNG\r\n\x1a\nrest"
    assert read_clipboard(wl, run=_ran(0, png)) == png  # type: ignore[arg-type]
    assert read_clipboard(wl, run=_ran(0, b"some text")) == NotStaged(  # type: ignore[arg-type]
        "the clipboard holds no image"
    )
    failed = read_clipboard(wl, run=_ran(1, err=b"No selection\nmore"))  # type: ignore[arg-type]
    assert failed == NotStaged("the clipboard could not be read: No selection")


def test_a_reader_that_hangs_is_given_up_on() -> None:
    def hangs(argv: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(argv, 5)

    assert read_clipboard(CLIPBOARD_READERS[2], run=hangs) == NotStaged(
        "the clipboard could not be read: it did not answer in 5 s"
    )


def test_a_sent_attachment_is_named_as_its_record_says() -> None:
    handle = Handle(
        id="h_ab12",
        kind="attachment",
        shape=Shape(bytes=3, kind="application/pdf"),
        preview="",
        content_sha="c" * 64,
    )
    pdf = Attachment(
        id="A-2", kind="pdf", name="spec.pdf", media_type="application/pdf", bytes=3,
        est_tokens=10, delivery="text", handle=handle, pages=12,
    )  # fmt: skip
    assert attachment_line(pdf) == "A-2 spec.pdf · pdf · 12 pages"
