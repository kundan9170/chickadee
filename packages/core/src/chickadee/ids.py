"""Every id that carries meaning, formatted and parsed in one place.

An id that something reads back is built here and read back here, so its format has one
home. Operation ids name an effect: a box step's and a task commit's, which resume reads
back (`box/resume.py`, `box/desk.py`), a handoff's, a bootstrap's and a scout escape's. The
packet loop's operation and candidate ids retired with it.

The other minted ids live here too: a session's and a stop's. The run id's generator stays in
the frozen `schema/ids.py` and is re-exported, never copied.

Not here: the ledgers' counters (`C-n`, `R-n`, `K-n`), handle ids (content digests) and lease
ids. Each is minted and read in one module, and nothing else parses them.
"""

from __future__ import annotations

import re
import secrets
import string
from typing import Final

from chickadee.errors import InfraError
from chickadee.schema.ids import new_run_id

__all__ = [
    "ATHENA_ROUND",
    "bootstrap_operation",
    "escape_id",
    "handoff_operation",
    "in_task",
    "job_of",
    "job_operation",
    "job_precondition",
    "new_run_id",
    "new_session_id",
    "operation_component",
    "parent_of",
    "split_task_key",
    "step_id",
    "step_prefix",
    "stop_id",
    "task_commit_id",
    "task_depth",
    "task_key",
    "task_of_commit",
]

# -- operations ---------------------------------------------------------------------


def handoff_operation(session_id: str, n: int) -> str:
    """The session's `n`th handoff: the trailer a `squash` commit carries,
    and the id its `handoff_intended` and `handoff_completed` lines share."""
    return f"handoff:{session_id}:{n}"


_COMPONENT_SAFE: Final = frozenset(string.ascii_letters + string.digits + "-_")
"""The characters an operation id carries into a path or ref name unencoded. Deliberately
narrow: `:` is refused by git refs and by Windows file names, and `.` is excluded so no encoded
id can start with a dot, contain `..`, or end in `.lock`, closed by the alphabet rather than by
remembering each rule."""


def operation_component(operation_id: str) -> str:
    """One name legal as a git ref component and as a file name on every platform, for
    `operation_id`, percent-encoded. The ref and the manifest directory
    (`operations/<this>/manifest.json`) are both named by it.

    Operation ids are `<kind>:<id>` by convention across this codebase, and `:` is one of the
    characters `git check-ref-format` and Windows refuse; interpolating the id raw failed on
    every real caller.

    Percent-encoding over sanitising: replacing the unsafe characters with `-` would map
    `a:b` and `a-b` onto one name, and an undo unit that silently aliases another operation's
    snapshot is worse than the crash it replaces. This is reversible and injective, so a name
    belongs to exactly one operation. UTF-8 first, so a non-ASCII id encodes per byte.
    """
    if not operation_id:
        raise InfraError("an operation id is required; the empty string cannot name a ref")
    encoded = operation_id.encode("utf-8")
    return "".join(chr(b) if chr(b) in _COMPONENT_SAFE else f"%{b:02X}" for b in encoded)


def job_operation(step_operation_id: str) -> str:
    """The job a shell call's command became: `job:<the call's operation>`,
    its own lifecycle beside the step that ran until the yield."""
    return f"job:{step_operation_id}"


_JOB: Final = "job="


def job_precondition(job_id: str) -> str:
    """How a job's intent names its id (`j2`): read back on resume and by the session's fold,
    which knows its running jobs from the log alone."""
    return f"{_JOB}{job_id}"


def job_of(preconditions: tuple[str, ...]) -> str | None:
    """The job id a `JOB` intent names, or `None`."""
    return next((p[len(_JOB) :] for p in preconditions if p.startswith(_JOB)), None)


def bootstrap_operation(run_id: str) -> str:
    """The run's toolchain bootstrap, run-scoped: `bootstrap:<run_id>`."""
    return f"bootstrap:{run_id}"


# -- minted ids ----------------------------------------------------------------------------


def new_session_id() -> str:
    """`S-` and eight hex digits (`28-context.md` §5.1.1 writes them `S-8a31`)."""
    return f"S-{secrets.token_hex(4)}"


def escape_id() -> str:
    """One scout escape's operation."""
    return f"scout-escape:{secrets.token_hex(8)}"


ATHENA_ROUND: Final = 0
"""The round Athena's own steps under a task name: never a worker's, whose rounds
start at 1, so resume never takes her step for the worker's."""


def step_id(task: str | None, round_: int | None, call_id: str) -> str:
    """One box step's operation: a writing call of the worker in `task`'s
    `round_`, or of Athena (`task` None). Fresh each call, as an escape's is: a call served
    again after a crash is a second lifecycle."""
    return f"{step_prefix(task, round_, call_id)}{secrets.token_hex(4)}"


def step_prefix(task: str | None, round_: int | None, call_id: str) -> str:
    """What every step of one call's operation id starts with: how resume finds the step a
    cut-off call made, whichever lifecycle it was."""
    where = f"{task}:{round_}" if task is not None else "athena:0"
    return f"step:{where}:{operation_component(call_id)}:"


def stop_id(kind: str, sha: str) -> str:
    """One showing of a stop: unique per showing, since each is answered once, and
    naming what it shows. `kind` is the stop's (`design`, `plan`)."""
    return f"{kind}-{sha[:12]}-{secrets.token_hex(4)}"


# -- task ids ------------------------------------------------------------------------


def task_key(task: str, attempt: int = 1) -> str:
    """How the journal keys `task`'s `attempt`th: the id for the first, then
    `T3@2`. Each attempt has its own conversation, step chain and commit lifecycle, and the
    model only ever reads the id."""
    if attempt < 1:
        raise ValueError(f"an attempt is counted from 1, not {attempt}")
    return task if attempt == 1 else f"{task}@{attempt}"


def split_task_key(key: str) -> tuple[str, int]:
    """`task_key`'s inverse: `"T3@2"` is `("T3", 2)`, and an id written in older versions is its
    own
    first attempt."""
    task, at, attempt = key.partition("@")
    return (task, int(attempt)) if at else (task, 1)


def parent_of(task: str) -> str | None:
    """The task a re-cut piece belongs to: `T3.1` is `T3`'s, `T3` nobody's."""
    head, dot, _ = task.rpartition(".")
    return head if dot else None


def in_task(name: str, task: str) -> bool:
    """Whether `name` is `task` or one of its pieces, at any depth: `T3.1.2`
    is in `T3`, `T31` is not."""
    return name == task or name.startswith(task + ".")


def task_depth(task: str) -> int:
    """A task's re-cut depth: its number of dots."""
    return task.count(".")


def task_commit_id(task: str, attempt: int = 1) -> str:
    """An accepted task's commit, `task` its key, its `attempt`th:
    a commit whose git raised keeps its lifecycle (`ambiguous`), so another accept is another
    attempt. Resume finds a commit by these ids in its trailer rather than committing
    twice."""
    return f"task-commit:{task}:{attempt}"


_TASK_COMMIT: Final = re.compile(r"^task-commit:([^:]+):(\d+)$")


def task_of_commit(op: str) -> str | None:
    """The task an accepted task's commit operation names, or None for any other operation."""
    match = _TASK_COMMIT.match(op)
    return match.group(1) if match is not None else None
