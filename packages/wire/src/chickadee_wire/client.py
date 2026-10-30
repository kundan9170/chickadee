"""The client every frontend uses: one typed method per route.

`Client` blocks, `AsyncClient` awaits; both are the same table. Each method hands a
`calls.RouteCall` to its own `_send`, the one place either touches HTTP: the bearer token on
every request, and an answer decoded into the route's type or raised as a `WireError`. A
route is therefore written once, in `calls.py`, and `test_client_mirrors_routes.py` holds
both classes to `routes.ROUTES`: a method for every row, and none besides.

**`connect` first.** It asks `GET /` before anything else and refuses a server whose wire major
differs (`WireMismatch`); `server` keeps its `Hello`, whose `capabilities` say which
routes answer there.

The four stream routes return iterators (`events.py`): `session_events` and `run_events` stay
attached, or poll the journal with `poll=True`; `session_journal` and `run_journal` read the
records once, to the end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Final, Literal, Self

import httpx
from pydantic import ValidationError

from chickadee_serve.bind import Tcp
from chickadee_serve.frames import (
    KEEPALIVE_S,
    WIRE_VERSION,
    AnswerTaken,
    CoderStarted,
    ErrorBody,
    ErrorCode,
    Hello,
    Hidden,
    RecordFrame,
    Resumed,
    TokenClass,
    TurnAccepted,
    TurnWorkspace,
    major,
)
from chickadee_wire import calls
from chickadee_wire.calls import RouteCall, Stream
from chickadee_wire.errors import Unanswered, WireError, WireMismatch
from chickadee_wire.events import AsyncEvents, Events, Reconnect, Watch
from chickadee_wire.models import (
    AttachmentPreview,
    CoderRequest,
    FileText,
    Findings,
    Forked,
    HandoffResult,
    HandoffSummary,
    JobView,
    Liveness,
    ModelSwitch,
    ModelWiring,
    OpenWorkInfo,
    Restored,
    RunBlob,
    RunDiff,
    RunHeader,
    RunMetrics,
    SessionBudget,
    SessionInfo,
    Undoable,
    UndoReport,
    Unresolved,
    WorkProgress,
)
from chickadee_wire.transport import (
    Address,
    TokenAuth,
    TokenSource,
    async_transport,
    base_url,
    discover,
    sync_transport,
    token_in,
)

__all__ = ["AsyncClient", "Client"]

TIMEOUT_S: Final = 30.0
"""A route's answer. Work is never waited on here: a turn or an episode answers at once."""

STREAM_READ_S: Final = 3 * KEEPALIVE_S
"""A stream silent this long is gone: the server sends a keep-alive every `KEEPALIVE_S`."""

_JSON: Final = {"content-type": "application/json"}


def _unanswered(
    method: str, path: str, exc: httpx.TransportError, bounds: httpx.Timeout
) -> Unanswered:
    """What the transport said, as `Unanswered`: which bound or failure it was, and
    whether the request may have reached the server. Only a request that never left (no
    connection made, none free) is known not to have."""

    def within(limit: float | None) -> str:
        return f"within {limit:g} s" if limit is not None else "in time"

    match exc:
        case httpx.ConnectTimeout():
            return Unanswered(method, path, f"no connection was made {within(bounds.connect)}")
        case httpx.PoolTimeout():
            return Unanswered(method, path, f"no connection was free {within(bounds.pool)}")
        case httpx.ConnectError():
            return Unanswered(method, path, f"the connection failed ({exc})")
        case httpx.WriteTimeout():
            why = f"the request was not sent {within(bounds.write)}"
        case httpx.TimeoutException():
            why = f"no answer came {within(bounds.read)}"
        case _:
            why = f"the connection broke ({exc})"
    return Unanswered(method, path, why, maybe_delivered=True)


def _answer[R](call: RouteCall[R], status: int, raw: bytes) -> R:
    if status >= 400:
        raise _refusal(status, raw)
    return call.decode(raw)


def _refusal(status: int, raw: bytes) -> WireError:
    """A route's `ErrorBody`; a body that is not one (a proxy, a crash) is still a typed
    error, `INTERNAL`, carrying what came back."""
    try:
        error = ErrorBody.model_validate_json(raw)
    except ValidationError:
        error = ErrorBody(code=ErrorCode.INTERNAL, message=raw.decode(errors="replace"))
    return WireError(status, error)


def _checked(hello: Hello) -> Hello:
    if major(hello.wire) != major(WIRE_VERSION):
        raise WireMismatch(WIRE_VERSION, hello.wire)
    return hello


def _records(lines: Iterator[str]) -> tuple[RecordFrame, ...]:
    """A `/journal` read to its end: records only, the stream carries nothing else."""
    return tuple(RecordFrame.model_validate_json(line) for line in lines if line.strip())


class Client:
    """A blocking client of one server. `connect` it; close it, or use it as a context."""

    server: Hello | None = None

    def __init__(
        self,
        address: Address,
        token: TokenSource,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            transport=transport if transport is not None else sync_transport(address),
            base_url=base_url(address),
            timeout=timeout_s,
            auth=TokenAuth(token),
        )

    @classmethod
    def connect(
        cls,
        address: Address,
        token: TokenSource,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> Self:
        client = cls(address, token, timeout_s=timeout_s, transport=transport)
        try:
            client.server = _checked(client.hello())
        except BaseException:
            client.close()
            raise
        return client

    @classmethod
    def attach(
        cls,
        state_dir: Path,
        token_class: TokenClass = TokenClass.INTERACTIVE,
        *,
        tcp: Tcp | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> Self:
        """A connected client of the server serving `state_dir` (a repository's repo-local
        state), wherever it listens (`discover`), its token read from the server's file and
        read again whenever the server restarts. `NotServing` when nothing answers."""
        address = discover(state_dir, tcp)
        return cls.connect(
            address, lambda: token_in(state_dir, token_class), timeout_s=timeout_s
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send[R](self, call: RouteCall[R]) -> R:
        content = call.content()
        try:
            response = self._http.request(
                call.method, call.path, content=content, headers=_JSON if content else None
            )
        except httpx.TransportError as exc:
            raise _unanswered(call.method, call.path, exc, self._http.timeout) from exc
        return _answer(call, response.status_code, response.content)

    @contextmanager
    def _lines(self, stream: Stream, after: int) -> Iterator[Iterator[str]]:
        timeout = httpx.Timeout(self._http.timeout.connect, read=STREAM_READ_S)
        with self._http.stream(
            "GET", stream.path, params={"from": after}, timeout=timeout
        ) as response:
            if response.status_code >= 400:
                raise _refusal(response.status_code, response.read())
            yield response.iter_lines()

    def _read(self, stream: Stream, after: int) -> tuple[RecordFrame, ...]:
        with self._lines(stream, after) as lines:
            return _records(lines)

    # -- the wire itself ---------------------------------------------------------------------

    def hello(self) -> Hello:
        return self._send(calls.hello())

    # -- sessions ----------------------------------------------------------------------------

    def create_session(
        self, session_id: str | None = None, *, adopt: str | None = None
    ) -> SessionInfo:
        return self._send(calls.create_session(session_id, adopt=adopt))

    def open_works(self) -> tuple[OpenWorkInfo, ...]:
        return self._send(calls.open_works())

    def sessions(self) -> tuple[SessionInfo, ...]:
        return self._send(calls.sessions())

    def session_info(self, sid: str) -> SessionInfo:
        return self._send(calls.session_info(sid))

    def rename(self, sid: str, text: str) -> SessionInfo:
        return self._send(calls.rename(sid, text))

    def close_session(self, sid: str) -> None:
        return self._send(calls.close_session(sid))

    def liveness(self, sid: str) -> Liveness:
        return self._send(calls.liveness(sid))

    def header(self, sid: str) -> RunHeader:
        return self._send(calls.header(sid))

    def work(self, sid: str) -> WorkProgress | None:
        """Where the session's open work stands, each task across its runs."""
        return self._send(calls.work(sid))

    def hidden(self, sid: str) -> Hidden:
        return self._send(calls.hidden(sid))

    # -- the streams -------------------------------------------------------------------------

    def session_events(
        self,
        sid: str,
        after: int = 0,
        *,
        poll: bool = False,
        interval_s: float = 0.5,
        reconnect: Reconnect | None = None,
        watch: Watch | None = None,
    ) -> Events:
        stream = calls.session_journal(sid) if poll else calls.session_events(sid)
        return Events(
            self._lines,
            stream,
            after=after,
            poll=poll,
            interval_s=interval_s,
            reconnect=reconnect,
            watch=watch,
        )

    def session_journal(self, sid: str, after: int = 0) -> tuple[RecordFrame, ...]:
        return self._read(calls.session_journal(sid), after)

    def run_events(
        self,
        run_id: str,
        after: int = 0,
        *,
        poll: bool = False,
        interval_s: float = 0.5,
        reconnect: Reconnect | None = None,
        watch: Watch | None = None,
    ) -> Events:
        stream = calls.run_journal(run_id) if poll else calls.run_events(run_id)
        return Events(
            self._lines,
            stream,
            after=after,
            poll=poll,
            interval_s=interval_s,
            reconnect=reconnect,
            watch=watch,
        )

    def run_journal(self, run_id: str, after: int = 0) -> tuple[RecordFrame, ...]:
        return self._read(calls.run_journal(run_id), after)

    # -- the conversation and answering ------------------------------------------------------

    def turn(
        self,
        sid: str,
        text: str,
        *,
        workspace: TurnWorkspace | None = None,
        attachments: Sequence[tuple[str, bytes]] = (),
    ) -> TurnAccepted:
        return self._send(calls.turn(sid, text, workspace=workspace, attachments=attachments))

    def preview_attachment(self, sid: str, name: str, data: bytes) -> AttachmentPreview:
        return self._send(calls.preview_attachment(sid, name, data))

    def interrupt(self, sid: str, *, hard: bool = False) -> None:
        return self._send(calls.interrupt(sid, hard=hard))

    def answer(
        self,
        sid: str,
        qid: str,
        choice: str,
        note: str | None = None,
        *,
        text: str | None = None,
    ) -> AnswerTaken:
        return self._send(calls.answer(sid, qid, choice, note, text=text))

    # -- the cap -----------------------------------------------------------------------------

    def budget(self, sid: str) -> SessionBudget:
        return self._send(calls.budget(sid))

    def set_cap(
        self, sid: str, hard_cost_usd: float | None, rung: str | None = None
    ) -> SessionBudget:
        return self._send(calls.set_cap(sid, hard_cost_usd, rung))

    # -- doing work --------------------------------------------------------------------------

    def coder(self, sid: str, request: CoderRequest) -> CoderStarted:
        return self._send(calls.coder(sid, request))

    def resume(self, sid: str) -> Resumed:
        return self._send(calls.resume(sid))

    # -- models ------------------------------------------------------------------------------

    def models(self, sid: str) -> ModelWiring:
        return self._send(calls.models(sid))

    def switch_model(self, sid: str, rung: str, model: str) -> ModelSwitch:
        return self._send(calls.switch_model(sid, rung, model))

    # -- background jobs ---------------------------------------------------------------------

    def jobs(self, sid: str) -> tuple[JobView, ...]:
        return self._send(calls.jobs(sid))

    def kill_job(self, sid: str, job_id: str) -> None:
        self._send(calls.kill_job(sid, job_id))

    def pause_run(self, sid: str) -> str:
        return self._send(calls.pause_run(sid))

    def drop_run(self, sid: str) -> str:
        return self._send(calls.drop_run(sid))

    # -- undo and fork -----------------------------------------------------------------------

    def undoable(self, sid: str) -> tuple[Undoable, ...]:
        return self._send(calls.undoable(sid))

    def undo(
        self,
        sid: str,
        to_turn: int,
        *,
        code: bool,
        conversation: bool,
        commit: bool | None = None,
    ) -> UndoReport:
        call = calls.undo(sid, to_turn, code=code, conversation=conversation, commit=commit)
        return self._send(call)

    def redo(self, sid: str) -> UndoReport:
        return self._send(calls.redo(sid))

    def undo_task(self, sid: str, run_id: str, task_id: str) -> Restored:
        return self._send(calls.undo_task(sid, run_id, task_id))

    def undo_operation(
        self, sid: str, operation_id: str, run_id: str | None = None
    ) -> Restored:
        return self._send(calls.undo_operation(sid, operation_id, run_id))

    def fork(
        self, sid: str, at_seq: int, reason: str, code: Literal["restore", "keep"] | None = None
    ) -> Forked:
        return self._send(calls.fork(sid, at_seq, reason, code))

    # -- reading -----------------------------------------------------------------------------

    def handoff(self, sid: str) -> HandoffSummary:
        return self._send(calls.handoff(sid))

    def session_metrics(self, sid: str) -> RunMetrics:
        return self._send(calls.session_metrics(sid))

    def run_metrics(self, run_id: str) -> RunMetrics:
        return self._send(calls.run_metrics(run_id))

    def run_blob(self, run_id: str, sha: str) -> RunBlob:
        return self._send(calls.run_blob(run_id, sha))

    def task_diff(self, run_id: str, task_id: str) -> RunDiff:
        return self._send(calls.task_diff(run_id, task_id))

    def step_diff(self, run_id: str, operation_id: str) -> RunDiff:
        return self._send(calls.step_diff(run_id, operation_id))

    def finalize(self, sid: str, choice: str) -> HandoffResult:
        return self._send(calls.finalize(sid, choice))

    def locate(self, question: str) -> Findings | Unresolved:
        return self._send(calls.locate(question))

    def files(self, sid: str) -> tuple[str, ...]:
        return self._send(calls.files(sid))

    def file_text(self, sid: str, path: str) -> FileText:
        return self._send(calls.file_text(sid, path))

    def holds_secret(self, text: str) -> bool:
        return self._send(calls.holds_secret(text))


class AsyncClient:
    """`Client`, awaited. `await AsyncClient.connect(...)`; `aclose` it, or `async with`."""

    server: Hello | None = None

    def __init__(
        self,
        address: Address,
        token: TokenSource,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            transport=transport if transport is not None else async_transport(address),
            base_url=base_url(address),
            timeout=timeout_s,
            auth=TokenAuth(token),
        )

    @classmethod
    async def connect(
        cls,
        address: Address,
        token: TokenSource,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> Self:
        client = cls(address, token, timeout_s=timeout_s, transport=transport)
        try:
            client.server = _checked(await client.hello())
        except BaseException:
            await client.aclose()
            raise
        return client

    @classmethod
    async def attach(
        cls,
        state_dir: Path,
        token_class: TokenClass = TokenClass.INTERACTIVE,
        *,
        tcp: Tcp | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> Self:
        """`Client.attach`, awaited. The probe that finds the server runs on a worker thread,
        so it never blocks the event loop."""
        address = await asyncio.to_thread(discover, state_dir, tcp)
        return await cls.connect(
            address, lambda: token_in(state_dir, token_class), timeout_s=timeout_s
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _send[R](self, call: RouteCall[R]) -> R:
        content = call.content()
        try:
            response = await self._http.request(
                call.method, call.path, content=content, headers=_JSON if content else None
            )
        except httpx.TransportError as exc:
            raise _unanswered(call.method, call.path, exc, self._http.timeout) from exc
        return _answer(call, response.status_code, response.content)

    @asynccontextmanager
    async def _lines(self, stream: Stream, after: int) -> AsyncIterator[AsyncIterator[str]]:
        timeout = httpx.Timeout(self._http.timeout.connect, read=STREAM_READ_S)
        async with self._http.stream(
            "GET", stream.path, params={"from": after}, timeout=timeout
        ) as response:
            if response.status_code >= 400:
                raise _refusal(response.status_code, await response.aread())
            yield response.aiter_lines()

    async def _read(self, stream: Stream, after: int) -> tuple[RecordFrame, ...]:
        async with self._lines(stream, after) as lines:
            return _records(iter([line async for line in lines]))

    # -- the wire itself ---------------------------------------------------------------------

    async def hello(self) -> Hello:
        return await self._send(calls.hello())

    # -- sessions ----------------------------------------------------------------------------

    async def create_session(
        self, session_id: str | None = None, *, adopt: str | None = None
    ) -> SessionInfo:
        return await self._send(calls.create_session(session_id, adopt=adopt))

    async def open_works(self) -> tuple[OpenWorkInfo, ...]:
        return await self._send(calls.open_works())

    async def sessions(self) -> tuple[SessionInfo, ...]:
        return await self._send(calls.sessions())

    async def session_info(self, sid: str) -> SessionInfo:
        return await self._send(calls.session_info(sid))

    async def rename(self, sid: str, text: str) -> SessionInfo:
        return await self._send(calls.rename(sid, text))

    async def close_session(self, sid: str) -> None:
        return await self._send(calls.close_session(sid))

    async def liveness(self, sid: str) -> Liveness:
        return await self._send(calls.liveness(sid))

    async def header(self, sid: str) -> RunHeader:
        return await self._send(calls.header(sid))

    async def work(self, sid: str) -> WorkProgress | None:
        """Where the session's open work stands, each task across its runs."""
        return await self._send(calls.work(sid))

    async def hidden(self, sid: str) -> Hidden:
        return await self._send(calls.hidden(sid))

    # -- the streams -------------------------------------------------------------------------

    def session_events(
        self,
        sid: str,
        after: int = 0,
        *,
        poll: bool = False,
        interval_s: float = 0.5,
        reconnect: Reconnect | None = None,
        watch: Watch | None = None,
    ) -> AsyncEvents:
        stream = calls.session_journal(sid) if poll else calls.session_events(sid)
        return AsyncEvents(
            self._lines,
            stream,
            after=after,
            poll=poll,
            interval_s=interval_s,
            reconnect=reconnect,
            watch=watch,
        )

    async def session_journal(self, sid: str, after: int = 0) -> tuple[RecordFrame, ...]:
        return await self._read(calls.session_journal(sid), after)

    def run_events(
        self,
        run_id: str,
        after: int = 0,
        *,
        poll: bool = False,
        interval_s: float = 0.5,
        reconnect: Reconnect | None = None,
        watch: Watch | None = None,
    ) -> AsyncEvents:
        stream = calls.run_journal(run_id) if poll else calls.run_events(run_id)
        return AsyncEvents(
            self._lines,
            stream,
            after=after,
            poll=poll,
            interval_s=interval_s,
            reconnect=reconnect,
            watch=watch,
        )

    async def run_journal(self, run_id: str, after: int = 0) -> tuple[RecordFrame, ...]:
        return await self._read(calls.run_journal(run_id), after)

    # -- the conversation and answering ------------------------------------------------------

    async def turn(
        self,
        sid: str,
        text: str,
        *,
        workspace: TurnWorkspace | None = None,
        attachments: Sequence[tuple[str, bytes]] = (),
    ) -> TurnAccepted:
        return await self._send(
            calls.turn(sid, text, workspace=workspace, attachments=attachments)
        )

    async def preview_attachment(self, sid: str, name: str, data: bytes) -> AttachmentPreview:
        return await self._send(calls.preview_attachment(sid, name, data))

    async def interrupt(self, sid: str, *, hard: bool = False) -> None:
        return await self._send(calls.interrupt(sid, hard=hard))

    async def answer(
        self,
        sid: str,
        qid: str,
        choice: str,
        note: str | None = None,
        *,
        text: str | None = None,
    ) -> AnswerTaken:
        return await self._send(calls.answer(sid, qid, choice, note, text=text))

    # -- the cap -----------------------------------------------------------------------------

    async def budget(self, sid: str) -> SessionBudget:
        return await self._send(calls.budget(sid))

    async def set_cap(
        self, sid: str, hard_cost_usd: float | None, rung: str | None = None
    ) -> SessionBudget:
        return await self._send(calls.set_cap(sid, hard_cost_usd, rung))

    # -- doing work --------------------------------------------------------------------------

    async def coder(self, sid: str, request: CoderRequest) -> CoderStarted:
        return await self._send(calls.coder(sid, request))

    async def resume(self, sid: str) -> Resumed:
        return await self._send(calls.resume(sid))

    # -- models ------------------------------------------------------------------------------

    async def models(self, sid: str) -> ModelWiring:
        return await self._send(calls.models(sid))

    async def switch_model(self, sid: str, rung: str, model: str) -> ModelSwitch:
        return await self._send(calls.switch_model(sid, rung, model))

    # -- background jobs ---------------------------------------------------------------------

    async def jobs(self, sid: str) -> tuple[JobView, ...]:
        return await self._send(calls.jobs(sid))

    async def kill_job(self, sid: str, job_id: str) -> None:
        await self._send(calls.kill_job(sid, job_id))

    async def pause_run(self, sid: str) -> str:
        return await self._send(calls.pause_run(sid))

    async def drop_run(self, sid: str) -> str:
        return await self._send(calls.drop_run(sid))

    # -- undo and fork -----------------------------------------------------------------------

    async def undoable(self, sid: str) -> tuple[Undoable, ...]:
        return await self._send(calls.undoable(sid))

    async def undo(
        self,
        sid: str,
        to_turn: int,
        *,
        code: bool,
        conversation: bool,
        commit: bool | None = None,
    ) -> UndoReport:
        call = calls.undo(sid, to_turn, code=code, conversation=conversation, commit=commit)
        return await self._send(call)

    async def redo(self, sid: str) -> UndoReport:
        return await self._send(calls.redo(sid))

    async def undo_task(self, sid: str, run_id: str, task_id: str) -> Restored:
        return await self._send(calls.undo_task(sid, run_id, task_id))

    async def undo_operation(
        self, sid: str, operation_id: str, run_id: str | None = None
    ) -> Restored:
        return await self._send(calls.undo_operation(sid, operation_id, run_id))

    async def fork(
        self, sid: str, at_seq: int, reason: str, code: Literal["restore", "keep"] | None = None
    ) -> Forked:
        return await self._send(calls.fork(sid, at_seq, reason, code))

    # -- reading -----------------------------------------------------------------------------

    async def handoff(self, sid: str) -> HandoffSummary:
        return await self._send(calls.handoff(sid))

    async def session_metrics(self, sid: str) -> RunMetrics:
        return await self._send(calls.session_metrics(sid))

    async def run_metrics(self, run_id: str) -> RunMetrics:
        return await self._send(calls.run_metrics(run_id))

    async def run_blob(self, run_id: str, sha: str) -> RunBlob:
        return await self._send(calls.run_blob(run_id, sha))

    async def task_diff(self, run_id: str, task_id: str) -> RunDiff:
        return await self._send(calls.task_diff(run_id, task_id))

    async def step_diff(self, run_id: str, operation_id: str) -> RunDiff:
        return await self._send(calls.step_diff(run_id, operation_id))

    async def finalize(self, sid: str, choice: str) -> HandoffResult:
        return await self._send(calls.finalize(sid, choice))

    async def locate(self, question: str) -> Findings | Unresolved:
        return await self._send(calls.locate(question))

    async def files(self, sid: str) -> tuple[str, ...]:
        return await self._send(calls.files(sid))

    async def file_text(self, sid: str, path: str) -> FileText:
        return await self._send(calls.file_text(sid, path))

    async def holds_secret(self, text: str) -> bool:
        return await self._send(calls.holds_secret(text))
