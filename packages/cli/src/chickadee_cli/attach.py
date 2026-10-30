"""What the person attaches before a turn is sent: one home
for the TUI's composer, the REPL's `/attach` and `--file`.

**The client speaks the wire alone**. It reads a file or the
clipboard and asks the server what the bytes are (`POST /session/{sid}/attachment`): the one
check a turn runs, how Zeus's rung would be sent it, and whether its text holds a secret. The
turn then sends the same bytes, and the server checks them again before anything is recorded.

**The clipboard is the person's machine's.** Its reader is the person's own process, run here as
`$EDITOR` is, with a fixed argv and a timeout: a server attached over TCP never runs on the
machine whose clipboard the person means.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import unquote, urlparse

from chickadee_serve.frames import AttachLimits, Hello
from chickadee_wire.errors import WireError
from chickadee_wire.models import Attachment, AttachmentPreview

__all__ = [
    "CLIPBOARD_READERS",
    "CLIPBOARD_TIMEOUT_S",
    "NotStaged",
    "Previewer",
    "Staged",
    "Staging",
    "attachment_line",
    "chip_lines",
    "clipboard_reader",
    "dropped_path",
    "read_clipboard",
    "ready",
]

CLIPBOARD_TIMEOUT_S: Final = 5.0
CLIPBOARD_NAME: Final = "clipboard.png"
NO_IMAGE: Final = "the clipboard holds no image"


@dataclass(frozen=True)
class NotStaged:
    why: str


@dataclass(frozen=True)
class Staged:
    """A file on the draft, not yet sent: its bytes as read, and what the server said."""

    data: bytes
    preview: AttachmentPreview

    @property
    def name(self) -> str:
        return self.preview.name

    def marks(self) -> tuple[str, ...]:
        """What a chip says beside its size: how Zeus would get it, when that is less than
        the file, and a secret."""
        p = self.preview
        out: list[str] = []
        if p.delivery == "none":
            out.append("Zeus cannot see this" if p.kind == "image" else "Zeus cannot read this")
        elif p.kind == "pdf" and p.delivery in ("text", "outline"):
            out.append("Zeus gets its text only")
        if p.secret:
            out.append("holds a secret")
        return tuple(out)

    def chip(self, n: int) -> str:
        """`[1] mockup.png · 1440x900 · ~1,730 tok`, and its marks."""
        p = self.preview
        return " · ".join((f"[{n}] {p.name}", p.size, f"~{p.est_tokens:,} tok", *self.marks()))


class Previewer(Protocol):
    """`chickadee_wire.Client`, structurally: its hello, and the one route staging asks."""

    @property
    def server(self) -> Hello | None: ...

    def preview_attachment(self, sid: str, name: str, data: bytes) -> AttachmentPreview: ...


class Staging:
    """The files on one session's draft, in order. Read and changed from the screen's loop and
    from the line thread (`/attach` runs there), so every access holds the lock;
    `changed` runs after each change, outside it."""

    def __init__(
        self,
        client: Previewer,
        session_id: str,
        *,
        changed: Callable[[], None] = lambda: None,
    ) -> None:
        self._client = client
        self._sid = session_id
        self._items: list[Staged] = []
        self._lock = threading.Lock()
        self.changed = changed

    @property
    def limits(self) -> AttachLimits | None:
        """The server's (`Hello.attach`), as it said when the client connected; None from one
        that takes no attachments. Read when asked, since a client reconnects."""
        server = self._client.server
        return server.attach if server is not None else None

    @property
    def offered(self) -> bool:
        return self.limits is not None

    def items(self) -> tuple[Staged, ...]:
        with self._lock:
            return tuple(self._items)

    def check(self, name: str, data: bytes) -> Staged | NotStaged:
        """What `data` would be on the next turn, asked of the server; nothing is staged."""
        big = self._too_big(name, len(data))
        if big is not None:
            return big
        try:
            return Staged(data, self._client.preview_attachment(self._sid, name, data))
        except WireError as exc:
            return NotStaged(exc.message)

    def check_file(self, path: Path) -> Staged | NotStaged:
        """A file read and checked; its size asked of the disk first, so a huge one is
        refused unread."""
        try:
            if not path.is_file():
                return NotStaged(f"{path}: no such file")
            big = self._too_big(path.name, path.stat().st_size)
            if big is not None:
                return big
            data = path.read_bytes()
        except OSError as exc:
            return NotStaged(f"{path}: {exc.strerror or exc}")
        return self.check(path.name, data)

    def add(self, staged: Staged) -> Staged:
        with self._lock:
            self._items.append(staged)
        self.changed()
        return staged

    def stage(self, name: str, data: bytes) -> Staged | NotStaged:
        checked = self.check(name, data)
        return self.add(checked) if isinstance(checked, Staged) else checked

    def stage_file(self, path: Path) -> Staged | NotStaged:
        checked = self.check_file(path)
        return self.add(checked) if isinstance(checked, Staged) else checked

    def _too_big(self, name: str, size: int) -> NotStaged | None:
        if self.limits is None:
            return NotStaged("this server takes no attachments")
        cap = self.limits.attachment_file_max_bytes
        return NotStaged(f"{name} is over {_mb(cap)}") if size > cap else None

    def stage_clipboard(self) -> Staged | NotStaged:
        read = read_clipboard()
        if isinstance(read, NotStaged):
            return read
        return self.stage(CLIPBOARD_NAME, read)

    def drop(self, n: int) -> Staged | None:
        """The `n`th staged file (from 1) removed, or None when there is none."""
        with self._lock:
            if not 1 <= n <= len(self._items):
                return None
            gone = self._items.pop(n - 1)
        self.changed()
        return gone

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
        self.changed()

    def over(self) -> str | None:
        """Why the draft cannot be sent as it is: past the count or the total. None when it
        can."""
        items = self.items()
        limits = self.limits if items else None
        if limits is None:
            return None
        if len(items) > limits.attachments_max:
            return f"{len(items)} attached; at most {limits.attachments_max} a turn"
        # What the server totals: an image as it is sent, downscaled, which its preview
        # reports; the bytes as read would refuse a turn the server takes.
        total = sum(s.preview.bytes for s in items)
        if total > limits.attachment_bytes_max:
            return f"{_mb(total)} attached; at most {_mb(limits.attachment_bytes_max)} a turn"
        return None

    def secrets(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.items() if s.preview.secret)

    def take(self) -> tuple[Staged, ...]:
        """Everything staged, to send with a turn, and the draft emptied."""
        with self._lock:
            taken = tuple(self._items)
            self._items.clear()
        if taken:
            self.changed()
        return taken

    def put_back(self, staged: Sequence[Staged]) -> None:
        """A turn that was not sent: its files staged again, ahead of anything staged since."""
        if not staged:
            return
        with self._lock:
            self._items[:0] = staged
        self.changed()

    @staticmethod
    def files(staged: Sequence[Staged]) -> tuple[tuple[str, bytes], ...]:
        """As a turn carries them (`Client.turn(attachments=)`)."""
        return tuple((s.name, s.data) for s in staged)

    def lines(self) -> tuple[str, ...]:
        """`/attach` with no arguments: the chips, or that nothing is attached."""
        return chip_lines(self.items(), self.over()) or ("nothing attached",)


def attachment_line(a: Attachment) -> str:
    """`A-1 shot.png · image · 1568x523`: a sent attachment, as its turn's record names it."""
    if a.kind == "image":
        size = f"{a.width}x{a.height}"
    else:
        n, noun = (a.pages, "page") if a.kind == "pdf" else (a.lines, "line")
        size = f"{n} {noun}{'' if n == 1 else 's'}"
    return f"{a.id} {a.name} · {a.kind} · {size}"


def chip_lines(staged: Sequence[Staged], over: str | None) -> tuple[str, ...]:
    """Each chip, then the total and the limit the draft breaks, when it breaks one: what the
    TUI draws above the composer and `/attach` lists."""
    if not staged:
        return ()
    rows = [s.chip(n) for n, s in enumerate(staged, 1)]
    tokens = sum(s.preview.est_tokens for s in staged)
    total = f"{len(staged)} attached · ~{tokens:,} tok"
    rows.append(f"{total} · over the limit: {over}" if over is not None else total)
    return tuple(rows)


def ready(staging: Staging, ask: Callable[[str], str]) -> tuple[Staged, ...] | str:
    """What a turn about to be sent takes from the draft, or why the turn is not sent: the
    draft breaks a limit, or it holds a secret and the person did not say yes. Asked once,
    whichever surface the person types into. A refusal leaves the draft as
    it was."""
    over = staging.over()
    if over is not None:
        return f"not sent: {over}; /attach drop <n> removes one"
    secret = staging.secrets()
    if secret:
        names = ", ".join(secret)
        answer = ask(f"{names} holds a secret value; send it? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return "not sent: /attach drop <n> removes it"
    return staging.take()


# -- a dropped file ---------------------------------------------------------------------------

_QUOTED: Final = re.compile(r"""^(['"])(.*)\1$""", re.DOTALL)


def dropped_path(pasted: str) -> Path | None:
    """The one existing file a paste names, as a terminal pastes a dragged file: plain, quoted
    (`'…'`, `"…"`), backslash-escaped, or a `file://` URL. None for anything else, which is
    pasted as text."""
    text = pasted.strip()
    if not text or "\n" in text:
        return None
    quoted = _QUOTED.match(text)
    if quoted is not None:
        text = quoted.group(2)
    elif text.startswith("file://"):
        text = unquote(urlparse(text).path)
    else:
        text = re.sub(r"\\(.)", r"\1", text)
    path = Path(text).expanduser()
    try:
        return path if path.is_absolute() and path.is_file() else None
    except OSError:
        return None


# -- the clipboard -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipboardReader:
    """One platform's way to put the clipboard's image in a file: a fixed argv, `{out}` the
    file it writes, or its stdout when the argv names none."""

    program: str
    argv: tuple[str, ...]
    when: Callable[[], bool]
    install: str

    @property
    def to_stdout(self) -> bool:
        return not any("{out}" in a for a in self.argv)


def _mac() -> bool:
    return os.uname().sysname == "Darwin" if hasattr(os, "uname") else False


def _wsl_or_windows() -> bool:
    if os.name == "nt":
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


_POWERSHELL_SAVE: Final = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$i = Get-Clipboard -Format Image; if ($i -eq $null) { exit 3 }; "
    "$i.Save([Console]::OpenStandardOutput(), [System.Drawing.Imaging.ImageFormat]::Png)"
)

CLIPBOARD_READERS: Final[tuple[ClipboardReader, ...]] = (
    ClipboardReader(
        "osascript",
        (
            "osascript",
            "-e",
            'set f to open for access POSIX file "{out}" with write permission',
            "-e",
            "write (the clipboard as «class PNGf») to f",
            "-e",
            "close access f",
        ),
        _mac,
        "osascript ships with macOS",
    ),
    ClipboardReader(
        "wl-paste",
        ("wl-paste", "--no-newline", "--type", "image/png"),
        lambda: bool(os.environ.get("WAYLAND_DISPLAY")),
        "install wl-clipboard",
    ),
    ClipboardReader(
        "xclip",
        ("xclip", "-selection", "clipboard", "-t", "image/png", "-o"),
        lambda: bool(os.environ.get("DISPLAY")),
        "install xclip",
    ),
    ClipboardReader(
        "powershell.exe",
        ("powershell.exe", "-NoProfile", "-Command", _POWERSHELL_SAVE),
        _wsl_or_windows,
        "PowerShell ships with Windows",
    ),
)
"""In the order they are tried: the first whose platform applies and whose program is on
`PATH`."""


def clipboard_reader(
    readers: Sequence[ClipboardReader] = CLIPBOARD_READERS,
    which: Callable[[str], str | None] = shutil.which,
) -> ClipboardReader | NotStaged:
    applies = [r for r in readers if r.when()]
    for reader in applies:
        if which(reader.program) is not None:
            return reader
    if not applies:
        return NotStaged("no clipboard reader for this terminal: use /attach <path>")
    return NotStaged(f"no clipboard reader: {applies[0].install}")


def read_clipboard(
    reader: ClipboardReader | None = None,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes | NotStaged:
    """The clipboard's image, as PNG bytes, or why there is none."""
    chosen = reader if reader is not None else clipboard_reader()
    if isinstance(chosen, NotStaged):
        return chosen
    with tempfile.TemporaryDirectory(prefix="chickadee-clip-") as tmp:
        out = Path(tmp) / CLIPBOARD_NAME
        argv = [a.replace("{out}", str(out)) for a in chosen.argv]
        try:
            done = run(argv, capture_output=True, timeout=CLIPBOARD_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            return NotStaged("the clipboard could not be read: it did not answer in 5 s")
        except OSError as exc:
            return NotStaged(f"the clipboard could not be read: {exc.strerror or exc}")
        if done.returncode != 0:
            said = (done.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[0]
            why = f"the clipboard could not be read: {said}" if said else NO_IMAGE
            return NotStaged(why)
        data = done.stdout if chosen.to_stdout else (out.read_bytes() if out.exists() else b"")
    if not data.startswith(b"\x89PNG"):
        return NotStaged(NO_IMAGE)
    return data


def _mb(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.0f} MB"
    return f"{n // 1024} KB" if n >= 1024 else f"{n} bytes"
