"""`coder` — delegation to the box.

Zeus delegates one of two kinds of work, exactly one:

- **a design** (`coder(work)`): the session's open work by name, its `design/`
  written with `doc`. At dispatch the harness inserts two sections into its README and changes
  nothing else: the Request, the person's turn word for word, and the Constraints, every
  constraint that binds; checks the folder's shape (Amendment 1); and stop 1 shows the
  README with the folder, whose tree the approval pins. The README's sha names the design
  everywhere after: the episode line carries the sha the person approved;
- **a one-task plan** (`coder(title=…, brief=…)`): a brief written to the brief template, for
  work one worker can do in one task. It is asked at stop 2, with no stop 1.

The stop is the engine's, asked at dispatch before anything is leased (`engine._EngineBox`).
The digest a repeated call is keyed by is the work as Zeus wrote it (`work_digest`), so
delegating one design again against an unchanged workspace is the same call.

The episode line is written at dispatch, before the box runs, so a crash mid-run leaves a
session that knows which run its `coder` call opened. That order is the port's contract, not a
hope: `BoxPort.open` names the run and runs nothing, and the box's work starts only under
`CoderRun.report()`, which the loop calls after the line is durable.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeGuard, get_args

from chickadee.box.episode import Complete, Paused
from chickadee.box.report import CoderReport, ReportedTask
from chickadee.box.request import AttachedFile
from chickadee.context.ledger import Constraint, binding_constraints
from chickadee.context.lines import Attachment, EpisodeLine
from chickadee.context.results import CoderOutcome, InvalidArgument, ToolResult
from chickadee.context.standing import constraint_row
from chickadee.context.work_line import WorkLine
from chickadee.schema import sha256_hex
from chickadee.schema.isolation import WorkspaceMode
from chickadee.tools.tool import Said
from chickadee.work import open_work, sections

if TYPE_CHECKING:
    from chickadee.context.session import Session, SessionState, Turn
    from chickadee.tools.validate import Arguments

__all__ = [
    "CONSTRAINTS",
    "REQUEST",
    "BoxPort",
    "CoderRun",
    "Delegation",
    "OpenedEpisode",
    "amended_request",
    "asked_workspace",
    "attached_files",
    "given_back_on_failure",
    "inserted",
    "open_episode",
    "parse",
    "render_outcome",
    "render_report",
    "request_attachments",
    "request_of",
    "request_section",
    "requested",
    "task_lines",
    "with_attached",
    "work_digest",
]

REQUEST: Final = "## Request"
"""The section the harness fills with the person's turn, word for word."""

CONSTRAINTS: Final = "## Constraints from our conversation"
"""The section the harness fills with the constraints that bind the work."""

_NONE_BIND: Final = "None."

_SHORT_COMMIT: Final = 7


@dataclass(frozen=True)
class Delegation:
    """What Zeus hands the box. `request` is the call's own arguments, as
    sent; the rest is read from them."""

    request: Mapping[str, object]
    kind: Literal["design", "task"]
    text: str
    """What its stop shows: the design with the Request and Constraints sections inserted, or
    the one-task plan's brief as Zeus wrote it."""
    title: str | None
    """The one-task plan's title; None for a design."""
    constraints: tuple[str, ...]
    workspace: WorkspaceMode | None = None
    """Zeus's `workspace` as sent, or None: the engine chooses at dispatch
    (`delegation.chosen_workspace`)."""
    work: WorkLine | None = None
    """The open work a design is in: its README is read, its sections
    inserted and its shape checked at dispatch, where the folder is (`engine._EngineBox`), so
    `text` is empty until then. None for a one-task plan."""


class CoderRun(Protocol):
    """One box run, named and not yet running. `report()` runs it to its end, blocks until it
    ends, and returns the `CoderReport` as Zeus reads it: the outcome, each task, Athena's
    report, and handles, never the diff (invariant 14)."""

    @property
    def run_id(self) -> str: ...

    @property
    def design_sha(self) -> str | None:
        """The work the person approved at its stop, after any edit there (invariant 15):
        the design's sha, or the one-task brief's. None for a call nothing was approved for."""
        ...

    def report(self) -> ToolResult | Said: ...

    def abandon(self) -> None:
        """Release what `open` took, the repository lease, for a run whose `report()` will
        never be asked for: the turn's stream ended between the two. The run did no work, and
        the lease is released on every exit path."""
        ...


class BoxPort(Protocol):
    """The box. The engine asks the work's stop, builds the `CoderRequest` from the approved
    text, takes the repository lease, and names the run; the session's cancellation token
    reaches it as CL-10's hard interrupt."""

    def open(
        self, delegation: Delegation, *, session_id: str, cancel: threading.Event | None
    ) -> CoderRun:
        """Ask the stop, then lease and name the run, and perform nothing. The session's
        `episode` line must be durable before anything the run does: a run that had
        started when the process died, with no line naming it, is work the session could never
        reconcile."""
        ...


@dataclass(frozen=True)
class OpenedEpisode:
    delegation: Delegation
    run: CoderRun
    line: EpisodeLine


def work_digest(sent: object) -> str | None:
    """`coder`'s repetition digest: the work as Zeus named or wrote it, the work
    folder's name or the one-task plan's title and brief, so the same work however the call
    around it was worded is one key; the folder's files are the workspace's generation. None
    for a call that names no work."""
    if not isinstance(sent, Mapping):
        return None
    args: Mapping[str, object] = sent  # pyright: ignore[reportUnknownVariableType]
    work, brief = args.get("work"), args.get("brief")
    if isinstance(work, str) and work.strip():
        return sha256_hex(f"work:{work}")
    if isinstance(brief, str) and brief.strip():
        return sha256_hex(f"{args.get('title')}\n{brief}")
    return None


def parse(args: Arguments, session: Session) -> Delegation | InvalidArgument:
    """The delegation a `coder` call makes, or the argument that is wrong with it: exactly one
    of `work`, the session's open work by name, or `title` with `brief`. A
    malformed call is the model's to correct, never a harness bug. `design`, the text a design
    was handed over as, is retired, and refused naming `work` by the schema
    (`ToolSpec.retyped`)."""
    sent = args.sent
    work, title, brief = sent.get("work"), sent.get("title"), sent.get("brief")
    has_work = work not in (None, "")
    has_task = title not in (None, "") or brief not in (None, "")
    if has_work == has_task:
        return InvalidArgument(
            "work",
            "a delegation is a design (work) or one task (title with brief), and this sent "
            "both or neither",
            do='send work="<the open work\'s name>" for a design, or title and brief for one '
            "task",
        )
    sent_mode = sent.get("workspace")
    workspace: WorkspaceMode | None = None
    if _is_mode(sent_mode):
        workspace = sent_mode
    elif sent_mode is not None:
        return InvalidArgument(
            "workspace",
            f"{sent_mode!r} is not a workspace",
            do='send workspace as "worktree" or "in_place", or leave it out',
        )
    bound = binding_constraints(session.state, ())
    if isinstance(bound, InvalidArgument):
        return bound
    ids = tuple(c.id for c in bound)
    if has_work:
        opened = open_work(session.state)
        if opened is None:
            return InvalidArgument(
                "work",
                "no work is open",
                do="write the design with doc first, then delegate it",
            )
        if work != opened.name:
            return InvalidArgument(
                "work",
                f"{work!r} is not the open work; the open work is {opened.name!r}",
                do=f'send work="{opened.name}"',
            )
        return Delegation(
            request=sent,
            kind="design",
            text="",
            title=None,
            constraints=ids,
            workspace=workspace,
            work=opened,
        )
    if not isinstance(title, str) or not title.strip():
        return InvalidArgument(
            "title",
            "the title is missing or empty",
            do="send title: the task's title, one line",
        )
    if not isinstance(brief, str) or not brief.strip():
        return InvalidArgument(
            "brief",
            "the brief is missing or empty",
            do="send brief: the task's brief, written to the brief template",
        )
    return Delegation(
        request=sent,
        kind="task",
        text=brief,
        title=" ".join(title.split()),
        constraints=ids,
        workspace=workspace,
    )


def inserted(design: str, request: str, bound: Sequence[Constraint]) -> str:
    """`design` with the harness's two sections in place:
    the Request after its title, the person's words exactly; the Constraints at its end, every
    constraint that binds, one row each. A section of either name Zeus wrote is replaced, since
    the template tells it to leave both as they are: what the person reads there is always the
    harness's. The rest of the text is unchanged, line for line."""
    kept = _without((REQUEST, CONSTRAINTS), design.strip("\n").splitlines())
    rows = [constraint_row(c) for c in bound] or [_NONE_BIND]
    title = 1 if kept and kept[0].startswith("# ") else 0
    head, body = kept[:title], kept[title:]
    while body and not body[0].strip():
        body = body[1:]
    while body and not body[-1].strip():
        body = body[:-1]
    parts = [*head, "", REQUEST, request.strip("\n"), ""]
    if body:
        parts += [*body, ""]
    parts += [CONSTRAINTS, *rows]
    return "\n".join(parts if head else parts[1:]) + "\n"


def request_section(design: str) -> str | None:
    """The Request section of an approved design (`inserted`'s), its text alone; None when it
    has none. An amended design keeps it and adds the person's new turn after it."""
    # `work.sections` reads a `##` heading inside a fenced block as the block's, so a request
    # quoting code keeps all of it.
    return sections(design).get(REQUEST.removeprefix("## ").lower()) or None


def amended_request(earlier: str, latest: str) -> str:
    """The Request section of an amended design: what the person asked first, then what they
    asked since, each word for word."""
    return f"{earlier}\n\nThen:\n{latest}"


def _without(headings: tuple[str, ...], lines: list[str]) -> list[str]:
    """`lines` less each section headed by one of `headings`, through to the next heading of
    the same level or higher."""
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if line.rstrip() in headings:
                skipping = True
                continue
            if skipping and level <= 2:
                skipping = False
        if not skipping:
            out.append(line)
    return out


def request_of(state: SessionState) -> str:
    """The person's turn the call answers: the session's latest user turn, verbatim."""
    turn = _latest_user(state)
    return turn.text if turn is not None else ""


def requested(state: SessionState) -> str:
    """What the Request section says: `request_of`, and a line for each file sent with it,
    so the files are approved with the work."""
    return with_attached(request_of(state), request_attachments(state))


def request_attachments(state: SessionState) -> tuple[Attachment, ...]:
    """The files sent with the turn the call answers: the ones `request_of` lists."""
    turn = _latest_user(state)
    return turn.attachments if turn is not None else ()


def with_attached(text: str, attachments: Sequence[Attachment]) -> str:
    """`text`, then `Attached: A-1 mockup.png (image, 1440x900)` for each file."""
    if not attachments:
        return text
    lines = [f"Attached: {a.id} {a.name} ({a.kind}, {a.size})" for a in attachments]
    return "\n".join((text, "", *lines))


def attached_files(attachments: Sequence[Attachment]) -> tuple[AttachedFile, ...]:
    """The request's record of `attachments` (`CoderRequest.attachments`)."""
    return tuple(
        AttachedFile(id=a.id, name=a.name, sha=a.handle.content_sha) for a in attachments
    )


def _is_mode(value: object) -> TypeGuard[WorkspaceMode]:
    return value in get_args(WorkspaceMode)


def asked_workspace(state: SessionState) -> WorkspaceMode | None:
    """Where the person said the work of the turn a call answers runs (`--worktree`,
    `--in-place`), or None."""
    turn = _latest_user(state)
    return turn.workspace if turn is not None else None


def _latest_user(state: SessionState) -> Turn | None:
    return next((t for t in reversed(state.turns()) if t.role == "user"), None)


def open_episode(
    session: Session,
    port: BoxPort,
    delegation: Delegation,
    *,
    opened_at_seq: int,
    cancel: threading.Event | None,
) -> OpenedEpisode:
    """Name the run and write its `episode` line — at dispatch, while the call still
    waits for its result, which is when the fold accepts it. The run does nothing until its
    `report()` is asked for, so the line is synced to disk before any of its work."""
    run = port.open(delegation, session_id=session.session_id, cancel=cancel)
    with given_back_on_failure(run.run_id, run.abandon):
        line = session.record_episode(
            EpisodeLine(
                run_id=run.run_id,
                path="planned",
                tier=None,
                design_sha=run.design_sha,
                constraints=delegation.constraints,
                opened_at_seq=opened_at_seq,
            )
        )
        # Flushed is enough to survive this process; the run's journal fsyncs its effects,
        # and the line that names the run must outlive a power cut as well as they do, or
        # the repair finds a waiting call with no run to reconcile it from.
        session.sync()
    return OpenedEpisode(delegation, run, line)


@contextmanager
def given_back_on_failure(run_id: str, abandon: Callable[[], None]) -> Generator[None]:
    """The bracket around naming a run in its session: `open` took the repository
    lease, so whatever fails before the `episode` line is durable — a check, the line on a full
    disk, its sync — gives it back before it propagates. A run that will never report would
    otherwise hold the lease until the process exits. Zeus's dispatch and a host's
    `PacketEngine.coder` both open a run through it.

    The failure that broke the open is the one raised. An abandon that fails too (its own write
    on the same full disk) is noted on it, never raised in its place."""
    try:
        yield
    except BaseException as exc:
        try:
            abandon()
        except Exception as also:
            exc.add_note(f"abandoning run {run_id} also failed: {also!r}")
        raise


_MEANING: Final = {
    "COMPLETED": "every task was reviewed and committed; whether the change does what was "
    "asked is still yours to check",
    "BLOCKED": "the work could not go on; nothing more runs unless you call coder again",
    "ASK": "the user sent a note on what you sent: revise it by the note and call coder again",
    "BUDGET_CAP": "the spend cap stopped the work",
    "ABORTED": "the work was stopped; nothing more runs",
    "OUTGROWN": "the one-task plan outgrew its task: what is committed stays on the session's "
    "branch, and the team's report says what is missing, what is done and what it proposes. "
    "Do not call the work done: ask the user about what is missing, then call coder with a new "
    "one-task plan that names the done work, or write a design",
}
"""What each outcome means for the reply, in the report's words: Zeus told the user
"successfully implemented" on a `PARTIAL` run that committed nothing."""


def _meaning(outcome: str, committed: int, tasks: int) -> str:
    if outcome == "PARTIAL":
        return (
            f"not finished: {committed} of {tasks} task(s) committed; what was not committed "
            "is still on disk, uncommitted. Tell the user so; do not call the work done"
        )
    return _MEANING.get(outcome, "see why")


def _why(report: CoderReport) -> tuple[list[str], list[str]]:
    """The report's why and growth lines, and the why lines its typed outcome carries."""
    lines: list[str] = []
    why: list[str] = []
    if report.why:
        lines.append(f"reason: {report.why}")
        why.append(report.why)
    if report.growth:
        lines.append(f"growth: {report.growth}")
    return lines, why


def render_outcome(outcome: CoderReport | Paused | Complete) -> ToolResult | Said:
    """A run's outcome as Zeus reads it: its report (`render_report`), that it paused, or that
    its work has nothing left. A run handed over in the background is read so at its
    end, as its trigger-4 notice (`engine._EngineBox._tell`)."""
    if isinstance(outcome, Paused):
        return Said(
            f"PAUSED · run {outcome.run_id}: a stop paused it. Call coder with the same work "
            "to continue it: nothing done is redone, and no design or plan is asked again"
        )
    if isinstance(outcome, Complete):
        return Said(
            f"COMPLETE · work {outcome.work}: every task of its plan is done, and its last run "
            "ended COMPLETED; nothing ran. To add to it, change its design with doc and call "
            "coder again",
            CoderOutcome("COMPLETE"),
        )
    return render_report(outcome)


def task_lines(task: ReportedTask) -> list[str]:
    """One task as Zeus reads it: its state, Athena's last review, its commit, and each command
    its facts measured failing. The report's rows, and a live run's board (`job tail`)."""
    parts = [f"{task.id} {task.title!r}: {task.state}", f"{task.rounds} round(s)"]
    if task.answer is not None:
        parts.append(f"reviewed {task.answer}")
    if task.commit is not None:
        parts.append(f"commit {task.commit[:_SHORT_COMMIT]}")
    if task.stopped is not None and task.stopped != "replied":
        parts.append(f"its last round ended {task.stopped}")
    lines = [" · ".join(parts)]
    facts = task.facts
    if facts is not None:
        for command in facts.commands:
            if command.exit_code not in (None, 0) or command.timed_out:
                ran = "timed out" if command.timed_out else f"exit {command.exit_code}"
                lines.append(f"  ran {command.command!r}: {ran}")
    return lines


def render_report(report: CoderReport) -> ToolResult | Said:
    """A `CoderReport` as Zeus reads it: how the episode ended and why, each task
    with its state, Athena's last review of it, its commit and what its facts measured, then her
    report verbatim, and what changed, counted — never the diff (invariant 14). The result
    carries one handle, the diff's, or the journal's when nothing changed, so Zeus can read it;
    an episode that ran nothing has no handle. Its outcome and the lines that say why travel
    typed too (`CoderOutcome`), for a reader that shows the person why without parsing the
    text."""
    cost = report.cost.cost_usd
    head = report.outcome
    lines = [f"{head} · run {report.run_id or 'not started'} · ${cost:.4f}"]
    committed = sum(1 for t in report.tasks if t.commit is not None)
    lines.append(f"meaning: {_meaning(head, committed, len(report.tasks))}")
    said, why = _why(report)
    lines.extend(said)
    for task in report.tasks:
        lines.extend(task_lines(task))
    if report.athena:
        # Her own account, which can claim more than the outcome ("Completed" on a
        # run that committed nothing). Said in plain words, never by her name.
        lines.extend(
            ("the team's report (its own account; the outcome above holds):", report.athena)
        )
    change = report.change_summary
    if change.files:
        lines.append(
            f"changed {len(change.files)} file(s), +{change.lines_added} "
            f"-{change.lines_removed}: {', '.join(change.files)}"
        )
    if change.symbols_added:
        lines.append(f"symbols added: {', '.join(change.symbols_added)}")
    if change.symbols_removed:
        lines.append(f"symbols removed: {', '.join(change.symbols_removed)}")
    if not change.symbols_known:
        lines.append("symbols: not measured for every task (the run was resumed)")
    if report.unanswered:
        # Never silently. The person's words he passed on were not answered.
        lines.append(f"the team did not answer: {', '.join(report.unanswered)}")
    if report.harness:
        # A hook's words, never the team's: said apart, under the harness's name.
        lines.extend(("the harness's hooks said:", *report.harness))
    told = CoderOutcome(head, tuple(why))
    handles = report.handles
    if handles is None:
        return Said("\n".join(lines), told)
    handle = handles.diff or handles.journal
    lines.append(f"{handle.kind}: {handle.id}")
    return ToolResult(handle, content="\n".join(lines), outcome=told)
