"""The ASGI app: a handler per row of `routes.ROUTES`, each one facade call.

**Nothing here decides anything**. A handler validates its body, calls the facade method
its row names, and renders the result; the stops are the engine's, a command's tier is the
shell's, the lease is the workspace's. A handler that needed a rule would be a route that
should not exist, and a capability not on the facade is a reason to extend the facade.

**The facade is whatever has the methods.** A live server is given `PacketEngine`; a replay
server is given `replay.ReplayFacade`. A row's method that the facade lacks, or marks
`@not_built`, answers `501 NOT_BUILT` naming the task that builds it, and `GET /` leaves its
capability out, so a client sees each route switch on when the engine gains its method.

**Facade calls run on worker threads** (`anyio.to_thread`): they read files, take locks and
start work, and the event loop serves every stream meanwhile.

**Errors are typed on the wire** (`frames.ErrorBody`): the engine's public exceptions map to
codes in `_failure`, and anything else is a 500 that says so, never a hang or a bare trace.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, cast

import anyio.to_thread
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route as StarletteRoute

from chickadee.context.store import SessionMissing
from chickadee.engine import CoderRequest, NotListed, NotNamed, NotReadable, TurnHandle
from chickadee.errors import (
    AttachmentRefused,
    ConfigError,
    EngineBusy,
    HandoffError,
    PlannerError,
    RepoLeaseBusy,
    RunNotStarted,
    SessionError,
)
from chickadee.scopes import Cursor, Page
from chickadee.unbuilt import owner_of
from chickadee_serve.auth import TokenClass, TokenStore, bearer, source_for
from chickadee_serve.frames import (
    WIRE_VERSION,
    AnswerBody,
    AnswerTaken,
    AttachLimits,
    CapBody,
    CoderBody,
    CoderStarted,
    CreateSessionBody,
    ErrorBody,
    ErrorCode,
    FileBody,
    FinalizeBody,
    ForkBody,
    Hello,
    Hidden,
    InterruptBody,
    LocateBody,
    RenameBody,
    Resumed,
    RungBody,
    SecretBody,
    TurnAccepted,
    TurnAttachment,
    TurnBody,
    UndoBody,
    UndoOperationBody,
    UndoTaskBody,
    encode_frame,
)
from chickadee_serve.interaction import (
    ChoiceRefused,
    Given,
    PendingQuestions,
    QuestionAnswered,
    QuestionUnknown,
)
from chickadee_serve.progress import ProgressHub
from chickadee_serve.records import RECORD_PAGE, stream
from chickadee_serve.routes import ROUTES, Route

__all__ = ["NDJSON", "Served", "build_app", "capabilities"]

log = logging.getLogger(__name__)

NDJSON: Final = "application/x-ndjson"

TurnStarted = Callable[[str, TurnHandle], None]
"""Told of every turn this server starts, with its handle: where its progress is pumped from
(`progress.pump_turn`). A replay server has no turns."""


@dataclass
class Served:
    """What the app serves, and the parts of the server every handler shares."""

    facade: object
    tokens: TokenStore
    hub: ProgressHub = field(default_factory=ProgressHub)
    pending: PendingQuestions = field(default_factory=PendingQuestions)
    engine_version: str = "0.0.0"
    replay: bool = False
    question_timeout_s: float | None = None
    """How long a question waits for a client's answer here (`Hello`)."""
    on_turn: TurnStarted | None = None
    stopping: threading.Event = field(default_factory=threading.Event)
    """Set when the server begins to stop (`server._Server.handle_exit`): every attached
    stream ends at its next idle turn, so a graceful shutdown never waits on a client to
    leave first."""


# -- failures --------------------------------------------------------------------------------


class _Failure(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _failure(exc: BaseException) -> _Failure:
    """An exception from a handler or the facade as the wire reports it."""
    match exc:
        case _Failure():
            return exc
        case ValidationError():
            return _Failure(422, ErrorCode.BAD_REQUEST, _say(exc))
        case AttachmentRefused():
            # A file that is not an image, a PDF or a text file, or over a cap; the
            # message names it and its place, and nothing was recorded.
            return _Failure(422, ErrorCode.BAD_REQUEST, str(exc))
        case ChoiceRefused() | NotReadable():
            # NotReadable: a diff asked of a task or step the run does not hold.
            return _Failure(422, ErrorCode.BAD_REQUEST, str(exc))
        case QuestionUnknown():
            return _Failure(404, ErrorCode.NOT_FOUND, str(exc))
        case QuestionAnswered():
            return _Failure(409, ErrorCode.ANSWERED, str(exc))
        case RunNotStarted() | NotNamed() | NotListed():
            # A run asked for before its journal exists: not found yet, never a
            # session's conflict. A follower asks again until its episode settles. NotNamed: a
            # sha no record of the run names, so nothing of the run's to serve.
            # NotListed: a file `/files` does not list, so not one to read.
            return _Failure(404, ErrorCode.NOT_FOUND, str(exc))
        case SessionError() if isinstance(exc.__cause__, SessionMissing):
            return _Failure(404, ErrorCode.NOT_FOUND, str(exc))
        case SessionError():
            return _Failure(409, ErrorCode.CONFLICT, str(exc))
        case EngineBusy() | RepoLeaseBusy():
            return _Failure(409, ErrorCode.BUSY, str(exc))
        case ConfigError():
            return _Failure(422, ErrorCode.CONFIG, str(exc))
        case HandoffError():
            # Refused with nothing changed: the source moved, or there is no worktree.
            return _Failure(409, ErrorCode.CONFLICT, str(exc))
        case PlannerError():
            # Athena gave no plan to act on: the design, as it stands, cannot
            # be previewed or run. The reason travels in the message, as the box's BLOCKED does.
            return _Failure(409, ErrorCode.CONFLICT, f"BLOCKED{{{exc.blocked}}}: {exc}")
        case NotImplementedError():
            return _Failure(501, ErrorCode.NOT_BUILT, str(exc) or "not built yet")
        case _:
            log.exception("a served call failed", exc_info=exc)
            return _Failure(500, ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")


def _say(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first["loc"]) or "body"
    return f"{where}: {first['msg']}"


def _error(failure: _Failure) -> JSONResponse:
    body = ErrorBody(code=failure.code, message=failure.message)
    return JSONResponse(body.model_dump(mode="json"), status_code=failure.status)


# -- the call --------------------------------------------------------------------------------


def _method(facade: object, name: str) -> Callable[..., Any]:
    """The facade's method `name`, or `NOT_BUILT` naming why it cannot be called."""
    method = getattr(facade, name, None)
    if not callable(method):
        raise _Failure(501, ErrorCode.NOT_BUILT, f"the facade has no {name} yet")
    owner = owner_of(method)
    if owner is not None:
        raise _Failure(
            501, ErrorCode.NOT_BUILT, f"{name} is not built yet; it arrives with {owner}"
        )
    return cast("Callable[..., Any]", method)


def capabilities(served: Served) -> tuple[str, ...]:
    """The capability of every row whose facade member answers here. The answer row
    needs the pending table, which every server has; `GET /` needs nothing."""
    found: set[str] = set()
    for row in ROUTES:
        if row.facade is None or row.facade == "interaction":
            found.add(row.capability)
            continue
        method = getattr(served.facade, row.facade, None)
        if callable(method) and owner_of(method) is None:
            found.add(row.capability)
    if served.replay:
        found.add("replay")
    return tuple(sorted(found))


@dataclass(frozen=True)
class _RouteCall:
    """One request, as a handler reads it."""

    served: Served
    row: Route
    request: Request
    token_class: TokenClass | None

    def param(self, name: str) -> str:
        return str(self.request.path_params[name])

    def cursor(self) -> Cursor:
        raw = self.request.query_params.get("from", "0")
        if not (raw.isascii() and raw.isdigit()):  # `isdigit` alone admits "²", `int` not
            raise _Failure(
                422, ErrorCode.BAD_REQUEST, f"from: a seq is a whole number, not {raw!r}"
            )
        return Cursor(int(raw))

    async def body[M: BaseModel](self, model: type[M]) -> M:
        raw = await self.request.body()
        return model.model_validate_json(raw or b"{}")

    async def facade(self, *args: Any, **kwargs: Any) -> Any:
        """The row's facade method, called on a worker thread."""
        assert self.row.facade is not None
        method = _method(self.served.facade, self.row.facade)
        return await anyio.to_thread.run_sync(lambda: method(*args, **kwargs))


Handler = Callable[[_RouteCall], Awaitable[Response]]


def _json(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(_wire(value), status_code=status)


def _wire(value: object) -> Any:
    """A facade result as JSON: a model as it dumps, a dataclass by its type, a sequence item
    by item. Nothing is re-rendered from richer state."""
    match value:
        case None:
            return {}
        case BaseModel():
            return value.model_dump(mode="json")
        case tuple() | list():
            return [_wire(v) for v in cast("tuple[object, ...]", value)]
        case _:
            return TypeAdapter(type(value)).dump_python(value, mode="json")


# -- handlers ---------------------------------------------------------------------------------


async def _hello(call: _RouteCall) -> Response:
    served = call.served
    hello = Hello(
        wire=WIRE_VERSION,
        engine=served.engine_version,
        capabilities=capabilities(served),
        replay=served.replay,
        question_timeout_s=served.question_timeout_s,
        attach=None if served.replay else _attach_limits(served.facade),
    )
    return _json(hello)


def _attach_limits(facade: object) -> AttachLimits | None:
    """The engine's `attach_caps()` on the wire, or None from a facade that has none: a client
    then does not offer attaching."""
    caps = getattr(facade, "attach_caps", None)
    if not callable(caps):
        return None
    return AttachLimits.model_validate(caps(), from_attributes=True)


async def _create_session(call: _RouteCall) -> Response:
    body = await call.body(CreateSessionBody)
    return _json(await call.facade(body.session_id, adopt=body.adopt), status=201)


async def _no_args(call: _RouteCall) -> Response:
    return _json(await call.facade())


async def _by_session(call: _RouteCall) -> Response:
    return _json(await call.facade(call.param("sid")))


async def _work(call: _RouteCall) -> Response:
    """Where the session's open work stands, or JSON `null` with none: an answer that
    may be absent, never `{}`, which says an action had nothing to report."""
    progress = await call.facade(call.param("sid"))
    return JSONResponse(None if progress is None else _wire(progress))


async def _by_run(call: _RouteCall) -> Response:
    return _json(await call.facade(call.param("run_id")))


async def _hidden(call: _RouteCall) -> Response:
    seqs = await call.facade(call.param("sid"))
    return _json(Hidden(seqs=tuple(seqs)))


def _records(*, scope_param: str, follow: bool) -> Handler:
    """`/events` (live: records, then progress and keep-alives) and `/journal` (records to the
    end, then closed). The first page is read before the response starts, so a missing scope
    is an ordinary 404 rather than a stream that breaks."""

    async def handler(call: _RouteCall) -> Response:
        scope = call.param(scope_param)
        assert call.row.facade is not None
        method = _method(call.served.facade, call.row.facade)

        def read(cursor: Cursor) -> Page:
            return cast("Page", method(scope, cursor, RECORD_PAGE))

        first = await anyio.to_thread.run_sync(read, call.cursor())
        hub = call.served.hub

        async def body() -> AsyncIterator[bytes]:
            async with contextlib.AsyncExitStack() as stack:
                sub = await stack.enter_async_context(hub.subscribe(scope)) if follow else None
                frames = stream(
                    read,
                    first,
                    scope=scope,
                    follow=follow,
                    progress=sub,
                    gone=call.request.is_disconnected,
                    stopping=call.served.stopping.is_set,
                )
                async for frame in frames:
                    yield encode_frame(frame)

        return StreamingResponse(body(), media_type=NDJSON)

    return handler


async def _close_session(call: _RouteCall) -> Response:
    await call.facade(call.param("sid"))
    return _json(None)


async def _turn(call: _RouteCall) -> Response:
    body = await call.body(TurnBody)
    sid = call.param("sid")
    files = tuple((a.name, _decoded(i, a)) for i, a in enumerate(body.attachments))
    if files:
        handle = await call.facade(sid, body.text, workspace=body.workspace, attachments=files)
    else:
        # A turn with nothing attached calls the facade as it always did, so a host whose
        # `turn` predates attachments still answers one.
        handle = await call.facade(sid, body.text, workspace=body.workspace)
    if call.served.on_turn is not None:
        call.served.on_turn(sid, handle)
    return _json(TurnAccepted(queued=bool(handle.queued)), status=202)


def _decoded(index: int, attachment: TurnAttachment) -> bytes:
    """An attachment's bytes, or `BAD_REQUEST` naming it: a body whose base64 does not decode
    is refused before the engine sees any of it."""
    try:
        return base64.b64decode(attachment.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        why = f"attachment {index + 1} ({attachment.name}): its data is not base64: {exc}"
        raise _Failure(422, ErrorCode.BAD_REQUEST, why) from exc


async def _preview_attachment(call: _RouteCall) -> Response:
    """What one file would be on the session's next turn; nothing is
    recorded."""
    body = await call.body(TurnAttachment)
    data = _decoded(0, body)
    return _json(await call.facade(call.param("sid"), body.name, data))


async def _interrupt(call: _RouteCall) -> Response:
    body = await call.body(InterruptBody)
    await call.facade(call.param("sid"), hard=body.hard)
    return _json(None, status=202)


async def _answer(call: _RouteCall) -> Response:
    """The other half of a `question_asked`. Who answered is the token's class:
    the body carries a choice and a note, never a source."""
    body = await call.body(AnswerBody)
    if call.token_class is None:  # the auth gate let no tokenless request this far
        raise _Failure(401, ErrorCode.UNAUTHORIZED, "an answer needs a token")
    qid = call.param("qid")
    given = Given(body.choice, call.token_class, body.note, body.text)
    await anyio.to_thread.run_sync(call.served.pending.answer, call.param("sid"), qid, given)
    return _json(AnswerTaken(question_id=qid, source=source_for(call.token_class).value))


async def _cap(call: _RouteCall) -> Response:
    body = await call.body(CapBody)
    return _json(await call.facade(call.param("sid"), body.hard_cost_usd, body.rung))


async def _coder(call: _RouteCall) -> Response:
    body = await call.body(CoderBody)
    request = CoderRequest.model_validate(body.request)
    handle = await call.facade(request, session_id=call.param("sid"))
    started = CoderStarted(run_id=handle.run_id)
    return _json(started, status=202)


async def _resume(call: _RouteCall) -> Response:
    handle = await call.facade(call.param("sid"))
    in_flight = handle.in_flight
    resumed = Resumed(
        session_id=handle.session_id,
        reconciled=tuple(handle.reconciled),
        in_flight=in_flight.run_id if in_flight is not None else None,
    )
    return _json(resumed)


async def _switch_model(call: _RouteCall) -> Response:
    body = await call.body(RungBody)
    return _json(await call.facade(call.param("sid"), body.rung, body.model))


async def _undo(call: _RouteCall) -> Response:
    body = await call.body(UndoBody)
    report = await call.facade(
        call.param("sid"),
        body.to_turn,
        code=body.code,
        conversation=body.conversation,
        commit=body.commit,
    )
    return _json(report)


async def _undo_task(call: _RouteCall) -> Response:
    body = await call.body(UndoTaskBody)
    return _json(await call.facade(call.param("sid"), body.run_id, body.task_id))


async def _undo_operation(call: _RouteCall) -> Response:
    body = await call.body(UndoOperationBody)
    return _json(await call.facade(call.param("sid"), body.operation_id, body.run_id))


async def _run_blob(call: _RouteCall) -> Response:
    return _json(await call.facade(call.param("run_id"), call.param("sha")))


async def _task_diff(call: _RouteCall) -> Response:
    return _json(await call.facade(call.param("run_id"), task_id=call.param("task_id")))


async def _step_diff(call: _RouteCall) -> Response:
    run_id, operation_id = call.param("run_id"), call.param("operation_id")
    return _json(await call.facade(run_id, operation_id=operation_id))


async def _kill_job(call: _RouteCall) -> Response:
    await call.facade(call.param("sid"), call.param("job_id"))
    return _json(None, status=202)


async def _fork(call: _RouteCall) -> Response:
    body = await call.body(ForkBody)
    forked = await call.facade(call.param("sid"), body.at_seq, body.reason, body.code)
    return _json(forked, status=201)


async def _finalize(call: _RouteCall) -> Response:
    body = await call.body(FinalizeBody)
    return _json(await call.facade(call.param("sid"), body.choice))


async def _rename(call: _RouteCall) -> Response:
    body = await call.body(RenameBody)
    return _json(await call.facade(call.param("sid"), body.text))


async def _file_text(call: _RouteCall) -> Response:
    body = await call.body(FileBody)
    return _json(await call.facade(call.param("sid"), body.path))


async def _holds_secret(call: _RouteCall) -> Response:
    body = await call.body(SecretBody)
    return _json(await call.facade(body.text))


async def _locate(call: _RouteCall) -> Response:
    body = await call.body(LocateBody)
    return _json(await call.facade(body.question))


HANDLERS: Final[Mapping[tuple[str, str], Handler]] = {
    ("GET", "/"): _hello,
    ("POST", "/session"): _create_session,
    ("GET", "/works"): _no_args,
    ("GET", "/session"): _no_args,
    ("GET", "/session/{sid}"): _by_session,
    ("DELETE", "/session/{sid}"): _close_session,
    ("GET", "/session/{sid}/liveness"): _by_session,
    ("GET", "/session/{sid}/header"): _by_session,
    ("GET", "/session/{sid}/events"): _records(scope_param="sid", follow=True),
    ("GET", "/session/{sid}/journal"): _records(scope_param="sid", follow=False),
    ("GET", "/run/{run_id}/events"): _records(scope_param="run_id", follow=True),
    ("GET", "/run/{run_id}/journal"): _records(scope_param="run_id", follow=False),
    ("GET", "/session/{sid}/hidden"): _hidden,
    ("POST", "/session/{sid}/turn"): _turn,
    ("POST", "/session/{sid}/interrupt"): _interrupt,
    ("POST", "/session/{sid}/answer/{qid}"): _answer,
    ("GET", "/session/{sid}/budget"): _by_session,
    ("POST", "/session/{sid}/cap"): _cap,
    ("POST", "/session/{sid}/coder"): _coder,
    ("POST", "/session/{sid}/resume"): _resume,
    ("POST", "/session/{sid}/pause"): _by_session,
    ("POST", "/session/{sid}/drop"): _by_session,
    ("GET", "/session/{sid}/rung"): _by_session,
    ("POST", "/session/{sid}/rung"): _switch_model,
    ("GET", "/session/{sid}/jobs"): _by_session,
    ("GET", "/session/{sid}/work"): _work,
    ("POST", "/session/{sid}/jobs/{job_id}/kill"): _kill_job,
    ("GET", "/session/{sid}/undo"): _by_session,
    ("POST", "/session/{sid}/undo"): _undo,
    ("POST", "/session/{sid}/redo"): _by_session,
    ("POST", "/session/{sid}/undo/task"): _undo_task,
    ("POST", "/session/{sid}/undo/operation"): _undo_operation,
    ("POST", "/session/{sid}/fork"): _fork,
    ("GET", "/session/{sid}/handoff"): _by_session,
    ("GET", "/session/{sid}/metrics"): _by_session,
    ("GET", "/run/{run_id}/metrics"): _by_run,
    ("GET", "/run/{run_id}/blob/{sha}"): _run_blob,
    ("GET", "/run/{run_id}/diff/task/{task_id}"): _task_diff,
    ("GET", "/run/{run_id}/diff/step/{operation_id}"): _step_diff,
    ("POST", "/session/{sid}/finalize"): _finalize,
    ("POST", "/locate"): _locate,
    ("GET", "/session/{sid}/files"): _by_session,
    ("POST", "/session/{sid}/attachment"): _preview_attachment,
    ("POST", "/session/{sid}/file"): _file_text,
    ("POST", "/session/{sid}/title"): _rename,
    ("POST", "/secret"): _holds_secret,
}
"""Every row's handler. `test_no_new_capability.py` holds this and `ROUTES` to one set."""


# -- the app ----------------------------------------------------------------------------------


def _endpoint(
    served: Served, row: Route, handler: Handler
) -> Callable[[Request], Awaitable[Response]]:
    """The gate every request passes: a token (but for `GET /`), the replay refusal of a
    mutating row, then the handler, with every failure rendered as an `ErrorBody`."""

    async def endpoint(request: Request) -> Response:
        try:
            token_class = served.tokens.verify(bearer(request.headers.get("authorization")))
            if row.facade is not None and token_class is None:
                raise _Failure(
                    401, ErrorCode.UNAUTHORIZED, "send `Authorization: Bearer <token>`"
                )
            if served.replay and row.mutating:
                raise _Failure(
                    405,
                    ErrorCode.REPLAY_READ_ONLY,
                    f"{row.method} {row.path} writes; a replay does not",
                )
            return await handler(_RouteCall(served, row, request, token_class))
        except Exception as exc:  # every failure is answered, typed
            return _error(_failure(exc))

    return endpoint


def build_app(served: Served) -> Starlette:
    """The app over `served`: one Starlette route per row of `ROUTES`."""
    missing = {(r.method, r.path) for r in ROUTES} ^ set(HANDLERS)
    if missing:
        raise RuntimeError(f"routes and handlers disagree: {sorted(missing)}")
    routes = [
        StarletteRoute(
            row.path,
            _endpoint(served, row, HANDLERS[(row.method, row.path)]),
            methods=[row.method],
            name=f"{row.method} {row.path}",
        )
        for row in ROUTES
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        served.hub.bind(asyncio.get_running_loop())
        try:
            yield
        finally:
            served.hub.unbind()

    return Starlette(routes=routes, lifespan=lifespan)
