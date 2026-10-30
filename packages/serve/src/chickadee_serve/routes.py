"""The route table, as data: one row per route, each naming the one facade
method it calls.

**A table, so three readers agree.** `app.py` binds a handler to each row;
`test_no_new_capability.py` checks every row against `PacketEngine`; and `chickadee_wire`'s
`test_client_mirrors_routes.py` checks the client has a method for each and none besides.
A route that is not a row does not exist.

**Core-free**, as `frames.py` is: a client reads this table without importing the engine.

`facade` is the `PacketEngine` attribute the route reaches, and nothing else. Two rows are not
plain methods and say why: `GET /` reaches no facade at all, and the answer route
resolves a question pending on `PacketEngine.interaction`, the engine's own injected port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["ROUTES", "Method", "Route", "route"]

Method = Literal["GET", "POST", "DELETE"]


@dataclass(frozen=True)
class Route:
    method: Method
    path: str
    """Starlette's syntax: `{sid}`, `{run_id}`, `{qid}`."""
    facade: str | None
    """The one `PacketEngine` attribute the route reaches. None only for `GET /`."""
    capability: str
    """What `GET /` reports when the route answers here. Routes sharing one share a feature."""
    mutating: bool
    """Writes something durable: a `--replay` server refuses it with `REPLAY_READ_ONLY`. An
    answer is not mutating here: a replayed question waits for one as a live one does, and it
    is written to the replay's scratch copy, never the recording."""
    stream: bool = False
    """Answers NDJSON frames (`frames.StreamFrame`), not one JSON document."""


ROUTES: Final[tuple[Route, ...]] = (
    # the wire itself
    Route("GET", "/", None, "wire", mutating=False),
    # sessions
    Route("POST", "/session", "create_session", "sessions", mutating=True),
    Route("GET", "/session", "sessions", "sessions", mutating=False),
    Route("GET", "/session/{sid}", "session_info", "sessions", mutating=False),
    # the repository's open work, which a new session may continue
    Route("GET", "/works", "open_works", "sessions", mutating=False),
    Route("DELETE", "/session/{sid}", "close_session", "sessions", mutating=True),
    Route("GET", "/session/{sid}/liveness", "liveness", "liveness", mutating=False),
    Route("GET", "/session/{sid}/header", "header", "header", mutating=False),
    # where the session's open work stands, each task across its runs
    Route("GET", "/session/{sid}/work", "work", "sessions", mutating=False),
    # the person's name for the session
    Route("POST", "/session/{sid}/title", "rename", "sessions", mutating=True),
    # the two streams of a session, and of a run (a run is a scope too)
    Route("GET", "/session/{sid}/events", "journal", "events", mutating=False, stream=True),
    Route("GET", "/session/{sid}/journal", "journal", "journal", mutating=False, stream=True),
    Route("GET", "/run/{run_id}/events", "journal", "runs", mutating=False, stream=True),
    Route("GET", "/run/{run_id}/journal", "journal", "runs", mutating=False, stream=True),
    Route("GET", "/session/{sid}/hidden", "hidden", "undo", mutating=False),
    # the conversation
    Route("POST", "/session/{sid}/turn", "turn", "turns", mutating=True),
    # what a file would be on the next turn, before it is sent
    Route("POST", "/session/{sid}/attachment", "preview_attachment", "turns", mutating=False),
    Route("POST", "/session/{sid}/interrupt", "interrupt", "turns", mutating=True),
    # answering
    Route("POST", "/session/{sid}/answer/{qid}", "interaction", "questions", mutating=False),
    # the cap
    Route("GET", "/session/{sid}/budget", "budget", "budget", mutating=False),
    Route("POST", "/session/{sid}/cap", "set_cap", "budget", mutating=True),
    # doing work: the work is approved at its stop before `coder`
    Route("POST", "/session/{sid}/coder", "coder", "coder", mutating=True),
    Route("POST", "/session/{sid}/resume", "resume", "resume", mutating=True),
    # the team in the background: paused now, or ended for good
    Route("POST", "/session/{sid}/pause", "pause_run", "coder", mutating=True),
    Route("POST", "/session/{sid}/drop", "drop_run", "coder", mutating=True),
    # models
    Route("GET", "/session/{sid}/rung", "models", "models", mutating=False),
    Route("POST", "/session/{sid}/rung", "switch_model", "models", mutating=True),
    # undo and fork
    # background jobs
    Route("GET", "/session/{sid}/jobs", "jobs", "jobs", mutating=False),
    Route("POST", "/session/{sid}/jobs/{job_id}/kill", "kill_job", "jobs", mutating=True),
    Route("GET", "/session/{sid}/undo", "undoable", "undo", mutating=False),
    Route("POST", "/session/{sid}/undo", "undo", "undo", mutating=True),
    Route("POST", "/session/{sid}/redo", "redo", "undo", mutating=True),
    Route("POST", "/session/{sid}/undo/task", "undo_task", "undo", mutating=True),
    Route("POST", "/session/{sid}/undo/operation", "undo_operation", "undo", mutating=True),
    Route("POST", "/session/{sid}/fork", "fork", "fork", mutating=True),
    # reading
    Route("GET", "/session/{sid}/handoff", "handoff", "handoff", mutating=False),
    Route("GET", "/session/{sid}/metrics", "metrics", "metrics", mutating=False),
    Route("GET", "/run/{run_id}/metrics", "metrics", "metrics", mutating=False),
    # what a run's records name: a blob, and a task's or step's diff
    Route("GET", "/run/{run_id}/blob/{sha}", "run_blob", "blobs", mutating=False),
    Route("GET", "/run/{run_id}/diff/task/{task_id}", "run_diff", "diffs", mutating=False),
    Route("GET", "/run/{run_id}/diff/step/{operation_id}", "run_diff", "diffs", mutating=False),
    Route("POST", "/session/{sid}/finalize", "finalize", "finalize", mutating=True),
    Route("POST", "/locate", "locate", "locate", mutating=False),
    # the REPL's input
    Route("GET", "/session/{sid}/files", "files", "input", mutating=False),
    # the CODE pane's view of a file no task changed
    Route("POST", "/session/{sid}/file", "file_text", "input", mutating=False),
    Route("POST", "/secret", "holds_secret", "input", mutating=False),
)


def route(method: Method, path: str) -> Route:
    """The row for `method path`; `KeyError` when there is none."""
    for r in ROUTES:
        if r.method == method and r.path == path:
            return r
    raise KeyError(f"{method} {path}")
