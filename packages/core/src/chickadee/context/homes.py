"""Where a reader keeps what it reads, and the context it answers from.

Zeus's session is one home (`SessionHome`), a box role's journaled conversation another
(`ConversationHome`), and a conversation that keeps no context of its reads,
the scout's, the third (`Scratch`). What a home holds in context, how old each view of it is,
and which earlier view a read may merge into, are the home's to say. The locality rule,
`MERGE_MAX_BETWEEN`, is applied here.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

from chickadee.client.request import Message, TextPart
from chickadee.context.handle import (
    Handle,
    HandleKind,
    HandleTable,
    Shape,
    content_sha,
)
from chickadee.context.lines import CutLine
from chickadee.context.results import (
    HandleExpired,
)
from chickadee.context.session import ToolLine, Turn
from chickadee.context.spans import (
    Span,
    coalesce,
)
from chickadee.context.terms import READ
from chickadee.journal.errors import BlobMissing
from chickadee.limits import tokens_est

if TYPE_CHECKING:
    from chickadee.context.session import Session, View

__all__ = [
    "MERGE_MAX_BETWEEN",
    "ConversationHome",
    "HeldView",
    "Now",
    "ReadHome",
    "Scratch",
    "SessionHome",
    "mint_stored",
]


def mint_stored(
    home: ReadHome,
    kind: HandleKind,
    data: bytes,
    parts: tuple[Shape, str, tuple[str, ...]],
    *,
    name: str | None = None,
) -> Handle:
    """Bytes put in `home`'s store and a stored handle minted for them, its shape, preview and
    ops as given: the one way a reader keeps bytes that came from nowhere on disk (a command's
    output, an attachment, a PDF's text layer). `name` is an attachment's file name. The
    caller holds the home's lock."""
    sha = home.put_blob(data)
    shape, preview, ops = parts
    return home.handles().mint(
        kind,
        path=None,
        run_id=None,
        sha=sha,
        make=lambda hid: Handle(hid, kind, shape, preview, ops, sha, name=name),
    )


MERGE_MAX_BETWEEN: Final = 1
"""The locality rule: merge in place only while at most this many other results stand
between the earlier view and the tail, and no user turn does. Past that, a merge would
rewrite the conversation, so a fresh view is appended instead."""


@dataclass(frozen=True)
class HeldView:
    """One earlier view still in the conversation, and how old it is: what
    a re-read of its lines is answered by."""

    spans: tuple[Span, ...]
    named: str
    """How the pointer names the call that made it: `call 6`."""
    calls_ago: int
    turns_ago: int
    """Model turns since the view was shown."""
    tokens_ago: int
    """Conversation since it, estimated as P-2 estimates (`limits.tokens_est`)."""


class ReadHome(Protocol):
    """Where a `Reader` keeps what it reads, and the context it answers from.

    The reader asks this, in its own terms, and never which home it has: Zeus's session is one
    (`SessionHome`), another conversation's own memory the other (`Scratch`). What a home holds
    in context, and which earlier view a read may merge into, are the home's to say; a home
    that keeps no context says nothing is held and nothing merges."""

    @property
    def lock(self) -> AbstractContextManager[bool]:
        """Held while a read decides and mints, so parallel reads cannot both claim one id,
        nor both absorb one earlier view."""
        ...

    def handles(self) -> HandleTable:
        """The table reads mint into and resolve from."""
        ...

    def put_blob(self, data: bytes) -> str:
        """Keep bytes a handle names; returns their sha."""
        ...

    def blob(self, sha: str) -> bytes:
        """Bytes this home kept. A miss raises `BlobMissing`, which the resolver reports as
        `HANDLE_EXPIRED`."""
        ...

    def held(self, handle_id: str) -> tuple[Span, ...]:
        """The lines of this handle already expanded in this conversation's context."""
        ...

    def held_views(self, handle_id: str) -> tuple[HeldView, ...]:
        """Each view of this handle still in context, oldest first, with its age."""
        ...

    def changed_since(self, path: str, handle_id: str) -> str | None:
        """How the newest view of `path` still in context names its call, when it is of other
        bytes than `handle_id`'s: the file changed since the model saw it."""
        ...

    def served(self, call_id: str, handle: Handle, spans: tuple[Span, ...]) -> None:
        """A result of call `call_id` showed `spans` of `handle`. A home whose conversation
        records results itself (Zeus's session) ignores it."""
        ...

    def conversation(self) -> str | None:
        """Which conversation the reads land in: a reader's memory of them is per
        conversation, so a new one (the next task) starts from nothing."""
        ...

    def room(self) -> int | None:
        """Tokens still free in the conversation, for a raised read's share; None where the
        home cannot say, and nothing is raised."""
        ...

    def merge_target(self, handle_id: str) -> View | None:
        """The earlier view of this handle a read may merge into, if any (architecture
        §7.4.2)."""
        ...

    def reads_in(self, call_id: str) -> int:
        """How many reads the view this call made has absorbed, itself included."""
        ...

    def never_minted(self, handle_id: str) -> HandleExpired:
        """Why an id this home's table does not hold reads as expired, in the home's terms."""
        ...


class SessionHome:
    """Zeus's session as a reader's home: its handle table and blob store, and its views. What
    a read keeps is the session's, and lasts as long as it does; what is held, and what may
    merge, is read from the session's context."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def lock(self) -> AbstractContextManager[bool]:
        return self._session.lock

    def handles(self) -> HandleTable:
        return self._session.state.handles

    def put_blob(self, data: bytes) -> str:
        return self._session.put_blob(data)

    def blob(self, sha: str) -> bytes:
        return self._session.blob(sha)

    def held(self, handle_id: str) -> tuple[Span, ...]:
        """Shown by a result in the verbatim tail or a pinned body. A receipt holds no lines."""
        return coalesce(
            s
            for v in self._session.state.expanded_views()
            if v.handle_id == handle_id
            for s in v.spans
        )

    def held_views(self, handle_id: str) -> tuple[HeldView, ...]:
        return tuple(
            self._aged(v)
            for v in self._session.state.expanded_views()
            if v.handle_id == handle_id
        )

    def changed_since(self, path: str, handle_id: str) -> str | None:
        table = self._session.state.handles
        for view in reversed(self._session.state.expanded_views()):
            held = table.get(view.handle_id)
            if held is not None and held.path == path and view.handle_id != handle_id:
                return f"your call {view.call_id}"
        return None

    def served(self, call_id: str, handle: Handle, spans: tuple[Span, ...]) -> None:
        del call_id, handle, spans  # the session's `tool` line records the view

    def conversation(self) -> str | None:
        return self._session.state.session_id

    def room(self) -> int | None:
        return None

    def _aged(self, view: View) -> HeldView:
        """`view` with what came after it in the session: model turns, calls and tokens."""
        turns = calls = tokens = 0
        for _, record in self._session.state.lines(view.seq + 1):
            if isinstance(record, Turn):
                turns += record.role == "assistant"
                tokens += tokens_est(record.text)
            elif isinstance(record, ToolLine):
                calls += 1
                tokens += tokens_est(record.text)
        return HeldView(view.spans, f"your call {view.call_id}", calls, turns, tokens)

    def merge_target(self, handle_id: str) -> View | None:
        """The newest live view of this handle, if a read may merge into it."""
        for view in reversed(self._session.state.views()):
            if view.handle_id == handle_id:
                return view if self._mergeable(view) else None
        return None

    def never_minted(self, handle_id: str) -> HandleExpired:
        return HandleExpired(handle_id, "not_in_session")

    def reads_in(self, call_id: str) -> int:
        line = self._session.state.result(call_id)
        if line is None:
            return 0
        return 1 + sum(self.reads_in(old) for old in line.supersedes)

    def _mergeable(self, view: View) -> bool:
        """A view merges only if a full-text `read` made it, it is past the last cut, and it
        is near the tail: at most `MERGE_MAX_BETWEEN` other results after it and no user
        words. Past that a merge rewrites the conversation."""
        state = self._session.state
        call = state.call(view.call_id)
        if call is None or call.tool != READ or call.args.get("depth") == "signatures":
            return False
        if view.seq < state.boundary:
            return False
        between = 0
        for _, record in state.lines(view.seq + 1):
            if isinstance(record, CutLine) or (
                isinstance(record, Turn) and record.role == "user"
            ):
                return False
            if isinstance(record, ToolLine):
                if state.is_user_words(record.seq):
                    return False
                between += 1
                if between > MERGE_MAX_BETWEEN:
                    return False
        return True


class Scratch:
    """A conversation's own handles and bytes, held in memory and dropped with it.

    The scout's, for one `locate`: what it opens is minted here, an image it reads is kept
    here, and none of it reaches Zeus's session, in memory or on disk. It keeps no context of
    its own reads, so nothing is held and nothing merges."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handles = HandleTable()
        self._blobs: dict[str, bytes] = {}

    @property
    def lock(self) -> AbstractContextManager[bool]:
        return self._lock

    def handles(self) -> HandleTable:
        return self._handles

    def put_blob(self, data: bytes) -> str:
        sha = content_sha(data)
        self._blobs[sha] = data
        return sha

    def blob(self, sha: str) -> bytes:
        found = self._blobs.get(sha)
        if found is None:
            raise BlobMissing(sha)
        return found

    def held(self, handle_id: str) -> tuple[Span, ...]:
        return coalesce(s for v in self.held_views(handle_id) for s in v.spans)

    def held_views(self, handle_id: str) -> tuple[HeldView, ...]:
        return ()

    def changed_since(self, path: str, handle_id: str) -> str | None:
        return None

    def served(self, call_id: str, handle: Handle, spans: tuple[Span, ...]) -> None:
        return None

    def conversation(self) -> str | None:
        return None

    def room(self) -> int | None:
        return None

    def merge_target(self, handle_id: str) -> View | None:
        return None

    def reads_in(self, call_id: str) -> int:
        return 0

    def never_minted(self, handle_id: str) -> HandleExpired:
        return HandleExpired(handle_id, "not_in_conversation")


@dataclass(frozen=True)
class Now:
    """A box role's conversation as it stands, for its reader's home: which one it is, its
    messages (the current call's request among them), the calls whose results it evicted, and
    the tokens still free in its request."""

    key: str
    messages: Sequence[Message]
    evicted: frozenset[str] = frozenset()
    room: Callable[[], int] | None = None
    """Tokens still free in its request, measured when asked: only a raise asks."""


@dataclass(frozen=True)
class _Served:
    call_id: str
    handle_id: str
    path: str | None
    spans: tuple[Span, ...]


@dataclass
class _Shown:
    key: str | None = None
    views: list[_Served] = field(default_factory=lambda: list[_Served]())


class ConversationHome(Scratch):
    """A box role's reads, held in memory as the scout's are, and answered from the context of
    its journaled conversation: `now` says what the conversation holds.

    A view is held while the call that made it is answered in the conversation and its result
    not evicted; its age is the model turns and tokens after it there. What this home records
    is only which call showed which lines, for the conversation it was shown in: a new
    conversation, or a resumed process, starts from none, so its first re-read is resent, which
    is never wrong."""

    def __init__(self, now: Callable[[], Now | None]) -> None:
        super().__init__()
        self._now = now
        self._shown = _Shown()
        self._ages: tuple[tuple[str, int, frozenset[str]], _Ages] | None = None
        """The last conversation's ages, by what they were measured on: one read asks several
        times, and the conversation does not move between."""

    def served(self, call_id: str, handle: Handle, spans: tuple[Span, ...]) -> None:
        now = self._now()
        if now is None or not spans:
            return
        shown = self._current(now)
        shown.views.append(_Served(call_id, handle.id, handle.path, spans))

    def held_views(self, handle_id: str) -> tuple[HeldView, ...]:
        now = self._now()
        if now is None:
            return ()
        shown = self._current(now)
        ages = self._aged(now)
        return tuple(
            aged
            for v in shown.views
            if v.handle_id == handle_id and (aged := ages.of(v)) is not None
        )

    def changed_since(self, path: str, handle_id: str) -> str | None:
        now = self._now()
        if now is None:
            return None
        ages = self._aged(now)
        for v in reversed(self._current(now).views):
            if v.path == path and v.handle_id != handle_id and (aged := ages.of(v)) is not None:
                return aged.named
        return None

    def conversation(self) -> str | None:
        now = self._now()
        return None if now is None else now.key

    def room(self) -> int | None:
        now = self._now()
        return None if now is None or now.room is None else now.room()

    def _current(self, now: Now) -> _Shown:
        if self._shown.key != now.key:
            self._shown = _Shown(now.key)
        return self._shown

    def _aged(self, now: Now) -> _Ages:
        """The conversation measured once per state of it. An evicted call never returns to
        it, so its views are dropped then: a long conversation's list stays the size of what
        it holds."""
        stamp = (now.key, len(now.messages), now.evicted)
        if self._ages is not None and self._ages[0] == stamp:
            return self._ages[1]
        ages = _Ages(now)
        shown = self._current(now)
        shown.views = [v for v in shown.views if v.call_id not in now.evicted]
        self._ages = (stamp, ages)
        return ages


class _Ages:
    """Where each call stands in one conversation: its number, and what came after it, each
    counted once, so a view's age is a lookup."""

    def __init__(self, now: Now) -> None:
        self._evicted = now.evicted
        self._number: dict[str, int] = {}
        self._asked_at: dict[str, int] = {}
        self._answered: set[str] = set()
        n = len(now.messages)
        # What follows message i: model turns and tokens, summed from the end.
        self._turns_after = [0] * (n + 1)
        self._tokens_after = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            m = now.messages[i]
            self._turns_after[i] = self._turns_after[i + 1] + (m.role == "assistant")
            self._tokens_after[i] = self._tokens_after[i + 1] + tokens_est(_text(m))
        for i, message in enumerate(now.messages):
            for call in message.tool_calls or ():
                self._number[call.id] = len(self._number) + 1
                self._asked_at[call.id] = i
            if message.role == "tool" and message.tool_call_id is not None:
                self._answered.add(message.tool_call_id)

    def of(self, view: _Served) -> HeldView | None:
        """`view`'s age, or None once it is not in the conversation: never answered there,
        or evicted to a receipt."""
        cid = view.call_id
        if cid not in self._answered or cid in self._evicted or cid not in self._number:
            return None
        after = self._asked_at[cid] + 1
        calls = len(self._number) - self._number[cid]
        return HeldView(
            view.spans,
            f"call {self._number[cid]}",
            calls,
            self._turns_after[after],
            self._tokens_after[after],
        )


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(p.text for p in message.content if isinstance(p, TextPart))
