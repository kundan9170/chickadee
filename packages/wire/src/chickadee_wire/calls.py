"""Every route as a typed call: the one place a route's meaning lives.

A builder takes a route's arguments and returns a `RouteCall[R]`: its row of `routes.ROUTES`,
the path with its parameters filled, the body the server reads, and how its answer decodes
into `R`. `Client` and `AsyncClient` each hand a `RouteCall` to their own `_send`, so a route
is written once whichever client sends it. The two record streams of each scope are
`Stream`s, read by `events.py`.

A route's path, method and body come from the server's own `routes` and `frames`; its answer's
type from `models.py`, generated from the golden. Every answer is typed: a route whose facade is
not built yet still declares the type it will answer with, and answers `NOT_BUILT` until then.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Final, Literal
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, TypeAdapter

from chickadee_serve.frames import (
    AnswerBody,
    AnswerTaken,
    CapBody,
    CoderBody,
    CoderStarted,
    CreateSessionBody,
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
    TurnWorkspace,
    UndoBody,
    UndoOperationBody,
    UndoTaskBody,
)
from chickadee_serve.routes import Method, Route, route
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

__all__ = ["RouteCall", "Stream"]

_PARAM: Final = re.compile(r"\{(\w+)\}")


def _fill(row: Route, params: Mapping[str, str]) -> str:
    """`row.path` with each `{name}` replaced by its value, percent-quoted whole: a session id
    holding a `/` names one segment, never two."""
    path = _PARAM.sub(lambda m: quote(params[m.group(1)], safe=""), row.path)
    unused = set(params) - set(_PARAM.findall(row.path))
    if unused:
        raise ValueError(f"{row.method} {row.path} takes no {sorted(unused)}")
    return path


@cache
def _adapter(of: Any) -> TypeAdapter[Any]:
    return TypeAdapter(of)


def _as[R](of: type[R] | Any) -> Callable[[bytes], R]:
    """How an answer of type `of` decodes."""
    return lambda raw: _adapter(of).validate_json(raw)


def _nothing(_raw: bytes) -> None:
    """An answer that is `{}`: the route did what it said and has nothing to report."""


@dataclass(frozen=True)
class RouteCall[R]:
    row: Route
    params: Mapping[str, str]
    body: BaseModel | None
    decode: Callable[[bytes], R]
    path: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _fill(self.row, self.params))

    @property
    def method(self) -> Method:
        return self.row.method

    def content(self) -> bytes | None:
        """The body as sent: exactly the fields the server's model declares."""
        return None if self.body is None else self.body.model_dump_json(by_alias=True).encode()


@dataclass(frozen=True)
class Stream:
    """One scope's record stream: `/events` (live) or `/journal` (to its end, then closed)."""

    row: Route
    params: Mapping[str, str]
    path: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _fill(self.row, self.params))


def _row(method: Method, path: str) -> Route:
    return route(method, path)


# -- the wire itself -------------------------------------------------------------------


def hello() -> RouteCall[Hello]:
    return RouteCall(_row("GET", "/"), {}, None, _as(Hello))


# -- sessions ---------------------------------------------------------------------------------


def create_session(
    session_id: str | None = None, *, adopt: str | None = None
) -> RouteCall[SessionInfo]:
    body = CreateSessionBody(session_id=session_id, adopt=adopt)
    return RouteCall(_row("POST", "/session"), {}, body, _as(SessionInfo))


def open_works() -> RouteCall[tuple[OpenWorkInfo, ...]]:
    return RouteCall(_row("GET", "/works"), {}, None, _as(tuple[OpenWorkInfo, ...]))


def sessions() -> RouteCall[tuple[SessionInfo, ...]]:
    return RouteCall(_row("GET", "/session"), {}, None, _as(tuple[SessionInfo, ...]))


def session_info(sid: str) -> RouteCall[SessionInfo]:
    return RouteCall(_row("GET", "/session/{sid}"), {"sid": sid}, None, _as(SessionInfo))


def rename(sid: str, text: str) -> RouteCall[SessionInfo]:
    """The person's name for the session: its info, titled so."""
    row = _row("POST", "/session/{sid}/title")
    return RouteCall(row, {"sid": sid}, RenameBody(text=text), _as(SessionInfo))


def close_session(sid: str) -> RouteCall[None]:
    return RouteCall(_row("DELETE", "/session/{sid}"), {"sid": sid}, None, _nothing)


def liveness(sid: str) -> RouteCall[Liveness]:
    return RouteCall(_row("GET", "/session/{sid}/liveness"), {"sid": sid}, None, _as(Liveness))


def work(sid: str) -> RouteCall[WorkProgress | None]:
    """Where the session's open work stands: each task across its runs; None
    with no open work."""
    row = _row("GET", "/session/{sid}/work")
    return RouteCall(row, {"sid": sid}, None, _as(WorkProgress | None))


def header(sid: str) -> RouteCall[RunHeader]:
    return RouteCall(_row("GET", "/session/{sid}/header"), {"sid": sid}, None, _as(RunHeader))


def hidden(sid: str) -> RouteCall[Hidden]:
    return RouteCall(_row("GET", "/session/{sid}/hidden"), {"sid": sid}, None, _as(Hidden))


# -- the streams ---------------------------------------------------------------------


def session_events(sid: str) -> Stream:
    return Stream(_row("GET", "/session/{sid}/events"), {"sid": sid})


def session_journal(sid: str) -> Stream:
    return Stream(_row("GET", "/session/{sid}/journal"), {"sid": sid})


def run_events(run_id: str) -> Stream:
    return Stream(_row("GET", "/run/{run_id}/events"), {"run_id": run_id})


def run_journal(run_id: str) -> Stream:
    return Stream(_row("GET", "/run/{run_id}/journal"), {"run_id": run_id})


# -- the conversation and answering -----------------------------------------------


def turn(
    sid: str,
    text: str,
    *,
    workspace: TurnWorkspace | None = None,
    attachments: Sequence[tuple[str, bytes]] = (),
) -> RouteCall[TurnAccepted]:
    """The person's words, and the files sent with them as `(name, bytes)`, base64 on
    the wire."""
    row = _row("POST", "/session/{sid}/turn")
    files = [
        TurnAttachment(name=name, data=base64.b64encode(data).decode("ascii"))
        for name, data in attachments
    ]
    body = TurnBody(text=text, workspace=workspace, attachments=files)
    return RouteCall(row, {"sid": sid}, body, _as(TurnAccepted))


def preview_attachment(sid: str, name: str, data: bytes) -> RouteCall[AttachmentPreview]:
    """What a file would be on the session's next turn, before it is sent: the server's one
    check, and how Zeus's rung would be sent it."""
    row = _row("POST", "/session/{sid}/attachment")
    body = TurnAttachment(name=name, data=base64.b64encode(data).decode("ascii"))
    return RouteCall(row, {"sid": sid}, body, _as(AttachmentPreview))


def interrupt(sid: str, *, hard: bool = False) -> RouteCall[None]:
    row = _row("POST", "/session/{sid}/interrupt")
    return RouteCall(row, {"sid": sid}, InterruptBody(hard=hard), _nothing)


def answer(
    sid: str,
    qid: str,
    choice: str,
    note: str | None = None,
    *,
    text: str | None = None,
) -> RouteCall[AnswerTaken]:
    """No source: who answered is the token's class, fixed when it was issued. At a
    stop, `edit` carries the person's markdown as `text`."""
    row = _row("POST", "/session/{sid}/answer/{qid}")
    body = AnswerBody(choice=choice, note=note, text=text)
    return RouteCall(row, {"sid": sid, "qid": qid}, body, _as(AnswerTaken))


# -- the cap ---------------------------------------------------------------------


def budget(sid: str) -> RouteCall[SessionBudget]:
    return RouteCall(
        _row("GET", "/session/{sid}/budget"), {"sid": sid}, None, _as(SessionBudget)
    )


def set_cap(
    sid: str, hard_cost_usd: float | None, rung: str | None = None
) -> RouteCall[SessionBudget]:
    row = _row("POST", "/session/{sid}/cap")
    body = CapBody(hard_cost_usd=hard_cost_usd, rung=rung)
    return RouteCall(row, {"sid": sid}, body, _as(SessionBudget))


# -- doing work ------------------------------------------------------------------------


def _sent(model: BaseModel) -> dict[str, JsonValue]:
    """A request type as its body carries it: JSON, which the server validates against the
    facade's own type (`frames.CoderBody` stays core-free)."""
    return model.model_dump(mode="json", by_alias=True)


def coder(sid: str, request: CoderRequest) -> RouteCall[CoderStarted]:
    row = _row("POST", "/session/{sid}/coder")
    return RouteCall(row, {"sid": sid}, CoderBody(request=_sent(request)), _as(CoderStarted))


def resume(sid: str) -> RouteCall[Resumed]:
    return RouteCall(_row("POST", "/session/{sid}/resume"), {"sid": sid}, None, _as(Resumed))


def pause_run(sid: str) -> RouteCall[str]:
    """The session's run, paused now; what the engine said."""
    return RouteCall(_row("POST", "/session/{sid}/pause"), {"sid": sid}, None, _as(str))


def drop_run(sid: str) -> RouteCall[str]:
    """The session's run, ended for good; what the engine said."""
    return RouteCall(_row("POST", "/session/{sid}/drop"), {"sid": sid}, None, _as(str))


# -- models ----------------------------------------------------------------------------


def models(sid: str) -> RouteCall[ModelWiring]:
    return RouteCall(_row("GET", "/session/{sid}/rung"), {"sid": sid}, None, _as(ModelWiring))


def switch_model(sid: str, rung: str, model: str) -> RouteCall[ModelSwitch]:
    row = _row("POST", "/session/{sid}/rung")
    return RouteCall(row, {"sid": sid}, RungBody(rung=rung, model=model), _as(ModelSwitch))


# -- background jobs -----------------------------------------------------


def jobs(sid: str) -> RouteCall[tuple[JobView, ...]]:
    row = _row("GET", "/session/{sid}/jobs")
    return RouteCall(row, {"sid": sid}, None, _as(tuple[JobView, ...]))


def kill_job(sid: str, job_id: str) -> RouteCall[None]:
    row = _row("POST", "/session/{sid}/jobs/{job_id}/kill")
    return RouteCall(row, {"sid": sid, "job_id": job_id}, None, _nothing)


# -- undo and fork ---------------------------------------------------------------------


def undoable(sid: str) -> RouteCall[tuple[Undoable, ...]]:
    row = _row("GET", "/session/{sid}/undo")
    return RouteCall(row, {"sid": sid}, None, _as(tuple[Undoable, ...]))


def undo(
    sid: str, to_turn: int, *, code: bool, conversation: bool, commit: bool | None = None
) -> RouteCall[UndoReport]:
    row = _row("POST", "/session/{sid}/undo")
    body = UndoBody(to_turn=to_turn, code=code, conversation=conversation, commit=commit)
    return RouteCall(row, {"sid": sid}, body, _as(UndoReport))


def redo(sid: str) -> RouteCall[UndoReport]:
    return RouteCall(_row("POST", "/session/{sid}/redo"), {"sid": sid}, None, _as(UndoReport))


def undo_task(sid: str, run_id: str, task_id: str) -> RouteCall[Restored]:
    row = _row("POST", "/session/{sid}/undo/task")
    body = UndoTaskBody(run_id=run_id, task_id=task_id)
    return RouteCall(row, {"sid": sid}, body, _as(Restored))


def undo_operation(
    sid: str, operation_id: str, run_id: str | None = None
) -> RouteCall[Restored]:
    row = _row("POST", "/session/{sid}/undo/operation")
    body = UndoOperationBody(operation_id=operation_id, run_id=run_id)
    return RouteCall(row, {"sid": sid}, body, _as(Restored))


def fork(
    sid: str, at_seq: int, reason: str, code: Literal["restore", "keep"] | None = None
) -> RouteCall[Forked]:
    row = _row("POST", "/session/{sid}/fork")
    body = ForkBody(at_seq=at_seq, reason=reason, code=code)
    return RouteCall(row, {"sid": sid}, body, _as(Forked))


# -- reading ----------------------------------------------------------------------------------


def handoff(sid: str) -> RouteCall[HandoffSummary]:
    row = _row("GET", "/session/{sid}/handoff")
    return RouteCall(row, {"sid": sid}, None, _as(HandoffSummary))


def session_metrics(sid: str) -> RouteCall[RunMetrics]:
    return RouteCall(_row("GET", "/session/{sid}/metrics"), {"sid": sid}, None, _as(RunMetrics))


def run_metrics(run_id: str) -> RouteCall[RunMetrics]:
    row = _row("GET", "/run/{run_id}/metrics")
    return RouteCall(row, {"run_id": run_id}, None, _as(RunMetrics))


def run_blob(run_id: str, sha: str) -> RouteCall[RunBlob]:
    """A blob a record of the run names: its text, or why there is none."""
    row = _row("GET", "/run/{run_id}/blob/{sha}")
    return RouteCall(row, {"run_id": run_id, "sha": sha}, None, _as(RunBlob))


def task_diff(run_id: str, task_id: str) -> RouteCall[RunDiff]:
    """What a task of the run changed, between trees its journal names."""
    row = _row("GET", "/run/{run_id}/diff/task/{task_id}")
    return RouteCall(row, {"run_id": run_id, "task_id": task_id}, None, _as(RunDiff))


def step_diff(run_id: str, operation_id: str) -> RouteCall[RunDiff]:
    """What one step of the run changed."""
    row = _row("GET", "/run/{run_id}/diff/step/{operation_id}")
    params = {"run_id": run_id, "operation_id": operation_id}
    return RouteCall(row, params, None, _as(RunDiff))


def finalize(sid: str, choice: str) -> RouteCall[HandoffResult]:
    row = _row("POST", "/session/{sid}/finalize")
    # Checked here as the server checks it: a word that is not a choice fails before sending.
    body = FinalizeBody.model_validate({"choice": choice})
    return RouteCall(row, {"sid": sid}, body, _as(HandoffResult))


# -- the REPL's input --------------------------------------------------------


def files(sid: str) -> RouteCall[tuple[str, ...]]:
    """The paths `glob` would list in the session's view: what `@` completes, never a path the
    read rule refuses."""
    row = _row("GET", "/session/{sid}/files")
    return RouteCall(row, {"sid": sid}, None, _as(tuple[str, ...]))


def file_text(sid: str, path: str) -> RouteCall[FileText]:
    """One file `files` lists, as text, or why not: the CODE pane's view of a file no
    task changed."""
    row = _row("POST", "/session/{sid}/file")
    return RouteCall(row, {"sid": sid}, FileBody(path=path), _as(FileText))


def holds_secret(text: str) -> RouteCall[bool]:
    """Whether `text` holds a secret value: a line the REPL keeps out of its history."""
    return RouteCall(_row("POST", "/secret"), {}, SecretBody(text=text), _as(bool))


def locate(question: str) -> RouteCall[Findings | Unresolved]:
    """`Findings`, or `Unresolved` when the question names nothing the scout can settle.
    Told apart by shape: only `Unresolved` requires `question`, and it is preferred when
    both fit (pydantic's smart union counts the fields each one sets)."""
    body = LocateBody(question=question)
    return RouteCall(_row("POST", "/locate"), {}, body, _as(Findings | Unresolved))
