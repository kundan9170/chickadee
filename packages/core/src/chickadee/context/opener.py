"""A target, a path or a handle, opened under the read rule, and its handle minted.

The read rule is `secrets.read_denied`, asked before the open and again of the file opened;
`files.open_nonblocking` is the open. A stale handle does not error: it reports
the staleness and offers what the file says now.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path

from chickadee.context.files import open_nonblocking, read_bounded, same_file
from chickadee.context.freshness import CHANGED_NOTE
from chickadee.context.handle import (
    HANDLE_ID_RE,
    Handle,
    Shape,
    content_sha,
)
from chickadee.context.homes import ReadHome
from chickadee.context.outline import (
    Sniffed,
    sniff,
    stored_sniff,
)
from chickadee.context.parts import Parts
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import (
    HandleExpired,
    InvalidArgument,
    SecretPath,
    ToolFailure,
)
from chickadee.context.terms import Caps
from chickadee.graph.ignore import listed
from chickadee.paths import ROOT_RULE, Place, name_under, suggest, suggest_under
from chickadee.profiles.loader import ProfileSet
from chickadee.secrets.denylist import Denial, Denylist, read_denied

__all__ = [
    "Opened",
    "Opener",
    "stored_sniff",
]


@dataclass(frozen=True)
class Opened:
    handle: Handle
    data: bytes
    sniffed: Sniffed
    key: str | None
    """The index's name for the file — its workspace-relative path — or None when nothing
    indexes it: a stored handle, or a file outside the workspace."""
    notes: tuple[str, ...] = ()
    """What opening it had to say: the handle was stale, or its file is gone."""


class Opener:
    """A target, a path or a handle, opened under the read rule, and its handle minted. The one
    place a read opens a file.

    A path and a handle both open through `_read_allowed`, on every use: a handle names a
    path, and what the path reaches now is what gets read."""

    def __init__(
        self,
        home: ReadHome,
        resolver: HandleResolver,
        parts: Parts,
        *,
        denylist: Denylist,
        caps: Caps,
        languages: ProfileSet | None = None,
    ) -> None:
        self._home = home
        self._resolver = resolver
        self._parts = parts
        self._denylist = denylist
        self._caps = caps
        self._languages = languages
        """Which files are code, and in which language (`outline.code_language`); None is the
        shipped set."""
        self._workspace = Path(os.path.normpath(os.path.abspath(resolver.workspace)))
        self._real_workspace = Path(os.path.realpath(self._workspace))
        """The workspace's real path, resolved once: what the read rule spells a landing
        from, and a second root `_locate` names a path under, so a workspace reached through
        a link names its files one way however they are spelled."""

    def open(self, target: str | Path | Handle, *, mint: bool = True) -> Opened | ToolFailure:
        """`target` opened under the read rule. `mint` False only looks: a file the table
        holds no handle for is answered under an unminted one (no id, no preview), so asking
        what a read would ask (`Reader.repeat_key`) neither mints nor builds a preview."""
        if isinstance(target, Handle):
            return self._open_handle(target, mint=mint)
        text = str(target)
        if isinstance(target, str) and HANDLE_ID_RE.match(text):
            held = self._home.handles().get(text)
            if held is not None:
                return self._open_handle(held, mint=mint)
            if not self._resolver.file_path(text).exists():
                return self._home.never_minted(text)
        return self._open_path(text, mint=mint)

    def open_existing(self, handle: str | Handle) -> Opened | ToolFailure:
        if isinstance(handle, Handle):
            return self._open_handle(handle)
        if not HANDLE_ID_RE.match(handle):
            return InvalidArgument(
                "handle",
                "grep_in searches a handle read returned",
                do="read the file first, then grep_in the handle its answer names",
            )
        held = self._home.handles().get(handle)
        if held is None:
            return self._home.never_minted(handle)
        return self._open_handle(held)

    def _open_path(self, text: str, *, mint: bool = True) -> Opened | ToolFailure:
        if not text.strip():
            return InvalidArgument(
                "path",
                "a path or a handle is required",
                do="send path, or a handle read returned",
            )
        if "\x00" in text:
            # No file is named with a NUL byte, and the path layer raises on one rather than
            # answering: refused here, as an empty target is.
            return InvalidArgument(
                "path", "a path holds no NUL byte", do="send the path again without it"
            )
        path, key, inside = self._locate(text)
        data = self._read_allowed(path, key)
        if isinstance(data, Denial):
            return SecretPath(key, data.pattern)
        if isinstance(data, InvalidArgument):
            return data
        if data is None:
            if path.is_dir():
                return InvalidArgument(
                    "path",
                    f"{key} is a directory; read opens files",
                    do=f'list it with glob(pattern="{key}/*"), then read a file in it',
                )
            if path.exists():
                return InvalidArgument(
                    "path",
                    f"{key} cannot be read",
                    do="do not read it again; go on without it, or ask the person about it",
                )
            missing = InvalidArgument(
                "path",
                f"{key} does not exist",
                do="find the file with glob, then read it by the path glob shows (from the "
                "repository root)",
            )
            # The file it most likely meant, from what `glob` lists.
            meant = suggest(text, self._listed())
            if meant is None or meant == key:
                return missing
            return replace(
                missing,
                did_you_mean=meant,
                rule=ROOT_RULE,
                do=f'read {meant} if that is the file you meant: read(path="{meant}")',
            )
        return self._opened(key, data, key if inside else None, mint=mint)

    def place(self, text: str) -> Place | None:
        """Where a path a model sent lands in the workspace (`FileView.place`): as sent,
        when it names a file or a directory there the read rule allows; else the file it most
        likely meant, the directory, or for a file not there yet its directory with its name
        (`paths.suggest`, `suggest_under`); else None."""
        if not text.strip() or "\x00" in text:
            return None
        path, key, inside = self._locate(text)
        if inside and path.exists():
            denied = read_denied(key, path, self._denylist, workspace=self._real_workspace)
            return None if denied is not None else Place(key, path.is_dir())
        listed = self._listed()
        meant = suggest(text, listed)
        if meant is not None:
            return Place(meant)
        directories = {f.rsplit("/", 1)[0] for f in listed if "/" in f}
        folder = suggest(text, directories)
        if folder is not None:
            return Place(folder, directory=True)
        under = suggest_under(text, listed)
        return Place(under) if under is not None else None

    def _listed(self) -> tuple[str, ...]:
        """What `glob` lists in the workspace: no ignored file and no deny-listed one, so a
        suggestion never says a secret exists. Walked only when a path did not resolve."""
        return listed(self._workspace, self._denylist)

    def _opened(
        self, key: str, data: bytes, index_key: str | None, *, mint: bool = True
    ) -> Opened:
        sniffed = sniff(key, data, self._languages)
        sha = content_sha(data)
        table = self._home.handles()
        handle = table.find("file", path=key, run_id=None, sha=sha)
        if handle is None and not mint:
            lines = len(sniffed.lines) if sniffed.category == "text" else None
            shape = Shape(len(data), lines, None, sniffed.kind)
            return Opened(
                Handle("", "file", shape, "", (), sha, path=key), data, sniffed, index_key
            )
        if handle is None:
            if sniffed.category == "image":
                # The blob holds the original, so the handle still resolves after
                # whatever took the screenshot overwrites or deletes it.
                self._home.put_blob(data)
            shape, preview, ops = self._parts.of(
                sniffed,
                data,
                index_key,
                self._caps.whole_file,
                any_whole=self._caps.any_whole,
            )
            handle = table.mint(
                "file",
                path=key,
                run_id=None,
                sha=sha,
                make=lambda hid: Handle(hid, "file", shape, preview, ops, sha, path=key),
            )
        return Opened(handle, data, sniffed, index_key)

    def _open_handle(self, h: Handle, *, mint: bool = True) -> Opened | ToolFailure:
        if h.path is None:
            data = self._resolver.stored_bytes(h)
            if isinstance(data, HandleExpired):
                return data
            # An attachment is outlined by its name, as `Reader.attach` minted it; stored
            # output as plain text, whatever its first lines say.
            sniffed = (
                sniff(h.name, data, self._languages)
                if h.name is not None
                else stored_sniff(data)
            )
            return Opened(h, data, sniffed, None)
        index_key = None if Path(h.path).is_absolute() else h.path
        now = self._read_allowed(self._resolver.file_path(h.path), h.path)
        if isinstance(now, Denial):
            return SecretPath(h.path, now.pattern)
        if isinstance(now, InvalidArgument):
            return now
        if now is None:
            return self._gone(h, index_key)
        if content_sha(now) == h.content_sha:
            return Opened(h, now, sniff(h.path, now, self._languages), index_key)
        # §6: a stale handle does not error. It reports the staleness and offers what the
        # file says now, under the handle those bytes have.
        fresh = self._opened(h.path, now, index_key, mint=mint)
        note = (
            f"[{h.label} {CHANGED_NOTE} — {h.id} named {h.content_sha[:6]}; "
            f"this is {fresh.handle.id}, at {fresh.handle.content_sha[:6]}]"
        )
        return replace(fresh, notes=(note,))

    def _gone(self, h: Handle, index_key: str | None) -> Opened | HandleExpired:
        """A file handle whose file is gone. An image's original is in the home's blob store,
        so it still reads while the home lives; a text file stored nothing, so it is
        absent."""
        if h.shape.is_image and h.path is not None:
            original = self._resolver.stored_bytes(h)
            if isinstance(original, bytes):
                note = f"[{h.label} is no longer on disk: this is the image {h.id} named]"
                sniffed = sniff(h.path, original, self._languages)
                return Opened(h, original, sniffed, index_key, (note,))
        return HandleExpired(h.id, "absent")

    def _locate(self, text: str) -> tuple[Path, str, bool]:
        """The file, the name its handle carries, and whether it is inside the workspace. A
        workspace path is named relative to it, so one file is one handle however the path
        was spelled; anything outside is named absolutely and is still read."""
        raw = Path(os.path.expanduser(text))
        path = Path(os.path.normpath(raw if raw.is_absolute() else self._workspace / raw))
        key = name_under(self._workspace, path)
        if key is None:
            key = name_under(self._real_workspace, path)
        if key is not None:
            return path, key, True
        return path, path.as_posix(), False

    def _read_allowed(self, path: Path, key: str) -> bytes | Denial | InvalidArgument | None:
        """What `path` holds now, or the deny list's refusal, or a refusal of what is
        not a regular file or is past `file_bytes`, or None when there is no file there this
        process can read.

        A path and a handle both open through here, on every use: a handle names a path,
        and what the path reaches now is what gets read, so a link swapped in after the
        handle was minted is refused like one asked for by name. The list is asked twice.
        Before the open, so a link to a secret is refused even when nothing is there to
        read. And after it, about a name that reaches the very file that was opened, so a
        path that changes between the check and the open is judged as what was opened; one
        that changes while it is being judged is not read at all, since nothing can then
        say which file is open. The rule is `secrets.read_denied`."""
        denial = read_denied(key, path, self._denylist, workspace=self._real_workspace)
        if denial is not None:
            return denial
        try:
            # Opened non-blocking, so a FIFO — even one swapped in just now — cannot hold
            # the open, and judged by the descriptor, never by a stat of the name taken
            # before it: nothing is read until the file that is open is known to be a
            # regular one. A FIFO or a device can block or stream forever, under the
            # session's lock.
            fd = open_nonblocking(path)
            try:
                mode = os.fstat(fd).st_mode
                if stat.S_ISDIR(mode):
                    return None
                real = Path(os.path.realpath(path))
                if not same_file(real, fd):
                    return None
                denial = read_denied(
                    key, path, self._denylist, workspace=self._real_workspace, real=real
                )
                if denial is not None:
                    return denial
                if not stat.S_ISREG(mode):
                    why = f"{key} is not a regular file; read opens files"
                    return InvalidArgument(
                        "path", why, do="do not read it; read a file instead"
                    )
                data = read_bounded(fd, self._caps.file_bytes)
                if data is None:
                    cap = self._caps.file_bytes
                    why = f"{key} is over {cap:,} bytes; read opens files up to that"
                    return InvalidArgument(
                        "path",
                        why,
                        do=f'search it instead: grep(pattern=..., paths=["{key}"]) for '
                        "what you need",
                    )
                return data
            finally:
                os.close(fd)
        except OSError:
            return None
