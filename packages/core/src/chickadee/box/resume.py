"""Resume: the worker's rows.

A worker round the crash left open is continued from its conversation as journaled
(`boxctx/conversation.rebuilt`), never re-rendered. What the crash cut off is reconciled
first, and every call of the last turn left without an answer is answered, so the
conversation the model is shown again is whole (a call and its result one pair):

- **a step started, not completed**: the paths changed since its pre-image tree are put back
  to it (`workspace.restore_tree`), the step completed `failed`, and the call answered
  `interrupted; the files are as after the previous call`;
- **such a step whose command declared the network**: the same, and it is never run again:
  what it did beyond the worktree is unknown, and its answer says so;
- **a step completed whose answer was not journaled**: its files stand; its answer is lost,
  and it says so;
- **a call that wrote nothing** (a read), or never started: not run; the model may call it
  again.

A round's `task_reported` line is written before its reply joins the conversation, so a round
with a reply is never open: a crash between the two leaves the reply owed to the next round
(`owed_reply`), which `WorkerRun.cont` adds before the feedback.
Then the round continues: a model call the crash cut off is asked again, its spend before the
crash already in the journal.

**A permission question left waiting**: a cut-off step whose command was still being
put to the person (`asking`, which reads the session log the question is on) never ran. It is
put back like any cut-off step, then its call is dispatched again, so the person is asked
again, never answered for them.

**The episode's rows** (`athena_owed`): Athena's own cut-off calls are owed the same
answers, from her steps; a cut-off `plan`, `delegate` or `review` is dispatched again, and the
desk continues it from its own line for that call (`box/desk.py`).

Its callers are the worker, the desk, the loop, Athena and the episode's resume
(`box/episode.Box.resume`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chickadee.box.commit import Reindexer
from chickadee.box.steps import networked, put_back, reached, task_steps
from chickadee.box.tasks import TaskBook
from chickadee.boxctx.conversation import BoxConversation
from chickadee.client.request import Message
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.stop import stopped_at
from chickadee.ids import ATHENA_ROUND, step_prefix
from chickadee.journal.effects import EffectLog, on_journal, reopen
from chickadee.journal.port import JournalPort
from chickadee.journal.projection import Operation, RunProjection
from chickadee.schema.adapter import ToolCall
from chickadee.schema.journal import BoxMessageP, TaskReportedP

__all__ = [
    "INTERRUPTED",
    "LOST",
    "NOT_RUN",
    "OUTCOME_UNKNOWN",
    "Asking",
    "OpenRound",
    "Owed",
    "athena_owed",
    "never_asking",
    "open_round",
    "owed_reply",
    "task_trees",
    "unanswered",
    "unanswered_calls",
]

INTERRUPTED = "interrupted; the files are as after the previous call"
OUTCOME_UNKNOWN = (
    "outcome unknown: the command declared the network and was cut off; the files are as "
    "after the previous call, but what it did beyond them is not known. Check before running "
    "it again"
)
LOST = (
    "this call completed before an interruption, but its answer was lost; read the files to "
    "see them as they are now"
)
NOT_RUN = "not run: the round was interrupted; call it again if you still need it"

Owed = Message | ToolCall
"""What a cut-off call is owed on resume: its answer, or, as the call itself, dispatched again
(a permission question left waiting; a desk call of Athena's)."""

Asking = Callable[[str], bool]
"""Whether a permission question for this operation is waiting, unanswered, on the session log:
the step's command never ran."""


def never_asking(operation_id: str) -> bool:
    """No question is waiting: what a run with no session log reads (unattended, or a test)."""
    del operation_id
    return False


@dataclass(frozen=True)
class OpenRound:
    """A worker round to continue: its task and round, the trees its steps run between, and
    the answers its last turn is owed, in the order of its calls."""

    task: str
    round: int
    start_tree: str
    last_tree: str
    """The tree the last completed step reached, or the start: the next step's before."""
    answers: tuple[Owed, ...]
    """In call order; a `ToolCall` is dispatched again rather than answered."""


def open_round(
    run: RunProjection,
    task: str,
    conversation: BoxConversation,
    root: Path,
    journal: JournalPort,
    indexer: Reindexer | None,
    *,
    untracked_max_bytes: int,
    asking: Asking = never_asking,
) -> OpenRound | None:
    """`task`'s open round, its cut-off step reconciled; None when no round is open (none
    begun, or the last reported), which is the desk's to start or answer."""
    entry = TaskBook(run).get(task)
    if entry is None:
        raise ValueError(f"{task} was never delegated; there is nothing to resume")
    round_ = _last_round(run, task)
    if round_ is None or round_ <= entry.rounds:
        return None
    steps = task_steps(run, task)
    log = on_journal(journal, task_id=task, round=round_)
    owed = _reconciled(steps, root, log, indexer, untracked_max_bytes, asking)
    answers = tuple(
        _owed(call, [op for op in steps if op.operation_id.startswith(prefix)], owed)
        for call in unanswered_calls(conversation)
        if (prefix := step_prefix(task, round_, call.id))
    )
    last = reached(task_steps(journal.projection(), task), entry.start_tree)
    return OpenRound(task, round_, entry.start_tree, last, answers)


def athena_owed(
    run: RunProjection,
    conversation: BoxConversation,
    root: Path,
    journal: JournalPort,
    indexer: Reindexer | None,
    *,
    untracked_max_bytes: int,
    desk_tools: frozenset[str],
    asking: Asking = never_asking,
) -> tuple[Owed, ...]:
    """What each call of Athena's last turn left unanswered is owed: a desk call
    (`desk_tools`, by the name she sent) dispatched again; any other, from the steps it made,
    under no task or under the task she held, as a worker's call is."""
    calls = unanswered_calls(conversation)
    tasks: tuple[str | None, ...] = (None, *run.tasks)
    out: list[Owed] = []
    for call in calls:
        if call.name in desk_tools:
            out.append(call)
            continue
        prefixes = tuple(
            step_prefix(t, ATHENA_ROUND if t is not None else None, call.id) for t in tasks
        )
        made = [op for op in run.operations.values() if op.operation_id.startswith(prefixes)]
        held = next((op.task_id for op in made), None)
        log = on_journal(journal, task_id=held)
        owed = _reconciled(made, root, log, indexer, untracked_max_bytes, asking)
        out.append(_owed(call, made, owed))
    return tuple(out)


def _reconciled(
    steps: tuple[Operation, ...] | list[Operation],
    root: Path,
    log: EffectLog,
    indexer: Reindexer | None,
    untracked_max_bytes: int,
    asking: Asking,
) -> dict[str, str | None]:
    """Each cut-off step put back and completed: its answer, or None when its call is to be
    dispatched again (its permission question still waiting)."""
    return {
        op.operation_id: _reconcile(op, root, log, indexer, untracked_max_bytes, asking)
        for op in steps
        if op.completed is None
    }


def _owed(call: ToolCall, made: list[Operation], owed: dict[str, str | None]) -> Owed:
    cut = [owed[op.operation_id] for op in made if op.operation_id in owed]
    if cut and cut[0] is None:
        return call
    text = cut[0] if cut else (LOST if made else NOT_RUN)
    assert text is not None
    return Message(role="tool", content=text, tool_call_id=call.id)


def task_trees(run: RunProjection, task: str) -> tuple[str, str]:
    """`task`'s start tree and the tree its last completed step reached: what a worker that
    did not run it in this process continues from."""
    entry = TaskBook(run).get(task)
    if entry is None:
        raise ValueError(f"{task} was never delegated; there is nothing to continue")
    return entry.start_tree, reached(task_steps(run, task), entry.start_tree)


def owed_reply(
    run: RunProjection, task: str, conversation: BoxConversation, blob: Callable[[str], bytes]
) -> Message | None:
    """The worker's last reply when its round's report was written and the reply was not:
    the report is written first, so a crash between the two lines leaves the conversation
    without it, and the next round must follow it. None when it is there, or there was none."""
    reports = [
        p.payload for p in run.tasks.get(task, ()) if isinstance(p.payload, TaskReportedP)
    ]
    if not reports or reports[-1].report_sha is None:
        return None
    messages = conversation.messages
    if messages and messages[-1].role == "assistant" and not messages[-1].tool_calls:
        return None
    return Message(role="assistant", content=blob(reports[-1].report_sha).decode("utf-8"))


def owed_continuation(
    run: RunProjection,
    task: str,
    round_: int,
    conversation: BoxConversation,
    hooks: RoleHooks,
) -> str | None:
    """The words a hook's `Continue` gave an open round, when a crash cut it off between the
    worker's reply and them: the round's conversation ends with the reply,
    and the stop's lines, written after the message before it and before it, hold them. The
    words are the ones the live round sends (`hooks/stop.StopSaid.words`). None otherwise."""
    messages = conversation.messages
    if not messages or messages[-1].role != "assistant" or messages[-1].tool_calls:
        return None
    said = [
        p.seq
        for p in run.tasks.get(task, ())
        if isinstance(p.payload, BoxMessageP) and p.round == round_
    ]
    if not said:
        return None
    after = said[-2] if len(said) > 1 else 0
    return stopped_at(hooks, task=task, round_=round_, between=(after, said[-1])).words


def _last_round(run: RunProjection, task: str) -> int | None:
    rounds = [p.round for p in run.tasks.get(task, ()) if isinstance(p.payload, BoxMessageP)]
    found = [r for r in rounds if r is not None]
    return max(found) if found else None


def unanswered(conversation: BoxConversation) -> tuple[str, ...]:
    """The ids of `unanswered_calls`."""
    return tuple(c.id for c in unanswered_calls(conversation))


def unanswered_calls(conversation: BoxConversation) -> tuple[ToolCall, ...]:
    """The calls of the last turn that asked for tools with no answer after it: a crash's, or
    a round that ended mid-turn (a call raised, or the model kept calling after its end)."""
    messages = conversation.messages
    asked = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "assistant"), None
    )
    if asked is None:
        return ()
    answered = {m.tool_call_id for m in messages[asked + 1 :] if m.role == "tool"}
    return tuple(c for c in messages[asked].tool_calls if c.id not in answered)


def _reconcile(
    op: Operation,
    root: Path,
    log: EffectLog,
    indexer: Reindexer | None,
    untracked_max_bytes: int,
    asking: Asking,
) -> str | None:
    """A step cut off: the paths it changed put back to its pre-image tree, completed failed;
    the answer the call is owed, or None when its permission question is still waiting, so
    its command never ran and the call is dispatched again."""
    intended = op.intended
    if intended is None or intended.preimages_sha is None:
        raise ValueError(f"{op.operation_id} has no pre-image tree; it cannot be put back")
    before = intended.preimages_sha
    restored = put_back(root, before, untracked_max_bytes, indexer)
    reopen(log, op).complete("failed", observed=restored, post_state=before)
    if asking(op.operation_id):
        return None
    return OUTCOME_UNKNOWN if networked(intended.preconditions) else INTERRUPTED
