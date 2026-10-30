"""The slash commands: a registry, each command one call over the wire.

**The registry is the table.** Its names equal core's reserved set, which
`test_repl_turns.py` asserts, so a skill can never shadow one. A command whose facade is not
built calls its route anyway and prints the `NOT_BUILT` answer naming the task that builds it:
when that facade lands, the command works with no change here. A command with no route at all
says so, rather than being left out.

**The person's commands stay the person's**. `/budget` and `/rung` are typed here and
nowhere a model reaches; `/budget` below what is already spent asks first.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from chickadee_cli.approval import stop_rate, stop_rate_line
from chickadee_cli.attach import NotStaged, Staging
from chickadee_cli.render.board import RunView, board_lines, fold_run
from chickadee_cli.render.diff import (
    CHOICES,
    asked_on_leaving,
    diff_lines,
    finalize_lines,
    handoff_lines,
    offered,
)
from chickadee_cli.render.stream import Verbosity
from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.models import Restored, SessionBudget, Undoable, UndoReport

__all__ = [
    "COMMANDS",
    "LEAVE_PENDING",
    "Command",
    "Context",
    "Exit",
    "NoAnswer",
    "dispatch",
    "resume",
    "review",
]


class Exit(Exception):
    """`/exit`: leave the REPL."""


class NoAnswer(Exception):
    """`Context.ask` got no answer: Ctrl-C or Ctrl-D at the REPL's question, or the TUI closing
    while it asks. Never an empty line, which is an answer: Enter takes a question's default
    (`/undo`'s halves, `both`, among them). The command stops there and sends nothing more; a
    question the server holds stays open, to be answered at the prompt."""


class Context(Protocol):
    """What a command may do: reach the wire, print, ask the person, and steer the REPL."""

    client: Client
    style: Style
    staging: Staging
    """The files on the draft: what `/attach` stages and the next turn sends."""

    @property
    def session_id(self) -> str: ...

    def say(self, text: str, tone: Tone = Tone.PLAIN, *, indent: str = "") -> None: ...

    def show(self, lines: Sequence[str]) -> None:
        """Lines a renderer already styled with `style`, printed as they are."""
        ...

    def ask(self, prompt: str) -> str:
        """`prompt`, and the person's line. `NoAnswer` when they give none."""
        ...

    def answering[T](self, call: Callable[[], T]) -> T:
        """`call()`, while the person answers each question it raises (`interaction.py`)."""
        ...

    def put_back(self, text: str) -> None:
        """Words to edit at the next prompt."""
        ...

    def tell(self, text: str) -> None: ...

    def set_verbosity(self, verbosity: Verbosity) -> None: ...

    @property
    def verbosity(self) -> Verbosity: ...

    def clear(self) -> None: ...

    def follow_run(self, run_id: str, after: int) -> None:
        """Stream run `run_id`'s records past `after` beside the session's."""
        ...


Handler = Callable[[Context, str], None]


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    help: str
    run: Handler

    @property
    def words(self) -> tuple[str, ...]:
        """The fixed words its first argument may be, read from `usage`'s first bracket: a
        choice (`/verbose [q|normal|v]`), or a word with arguments after it (`/jobs [kill
        <id>]`). A lone word (`/journal [run]`) or a `<placeholder>` names a value, not a word
        to type. What the prompt completes after the name."""
        _, opened, rest = self.usage.partition("[")
        if not opened:
            return ()
        alternatives = [a.split() for a in rest.partition("]")[0].split("|") if a.split()]
        return tuple(
            tokens[0]
            for tokens in alternatives
            if (len(alternatives) > 1 or len(tokens) > 1) and not tokens[0].startswith("<")
        )

    @property
    def on_the_wire(self) -> bool:
        """Whether a route serves it: one that does not prints `NOT_ON_THE_WIRE` (a screen
        draws it dimmed)."""
        return self.run is not _not_on_the_wire


NOT_ON_THE_WIRE: Final = "not on the wire yet: no route serves it (30-serve §7)"


def is_command(line: str) -> bool:
    """Whether `line` names a command: `/` and a registered name. `/etc/app` does not, so a
    free-text answer may start with a slash."""
    return line.startswith("/") and line.removeprefix("/").partition(" ")[0] in COMMANDS


def dispatch(ctx: Context, line: str) -> None:
    """Run the command `line` names (`/name args`), or say there is none."""
    name, _, args = line.removeprefix("/").partition(" ")
    command = COMMANDS.get(name)
    if command is None:
        ctx.say(f"no command /{name}; /help lists them", Tone.BAD)
        return
    try:
        command.run(ctx, args.strip())
    except WireError as exc:
        ctx.say(f"/{name}: {exc.message}", Tone.BAD)


# -- session control ------------------------------------------------------------------------


def _help(ctx: Context, _args: str) -> None:
    width = max(len(c.usage) for c in COMMANDS.values())
    for command in sorted(COMMANDS.values(), key=lambda c: c.name):
        ctx.say(f"{command.usage:<{width}}  {command.help}")
    ctx.say(
        "anything else is said to the session; Ctrl-C stops a turn, Ctrl-D leaves", Tone.DIM
    )


def _exit(_ctx: Context, _args: str) -> None:
    raise Exit


def _clear(ctx: Context, _args: str) -> None:
    ctx.clear()


def _verbose(ctx: Context, args: str) -> None:
    if not args:
        ctx.say(f"verbosity {ctx.verbosity.value}")
        return
    try:
        ctx.set_verbosity(Verbosity(args))
    except ValueError:
        ctx.say("/verbose q|normal|v", Tone.BAD)
        return
    ctx.say(f"verbosity {args}", Tone.DIM)


def _attach(ctx: Context, args: str) -> None:
    """`/attach <path>` stages a file; alone, lists what is staged; `drop <n>` and `clear`
    remove."""
    staging = ctx.staging
    if not staging.offered:
        ctx.say(NO_ATTACHING, Tone.BAD)
        return
    word, _, rest = args.partition(" ")
    if not args:
        for line in staging.lines():
            ctx.say(line)
    elif word == "drop":
        n = int(rest) if rest.strip().isdigit() else 0
        gone = staging.drop(n)
        if gone is None:
            ctx.say(f"no attachment {rest.strip() or '?'}; /attach lists them", Tone.BAD)
        else:
            ctx.say(f"dropped {gone.name}", Tone.DIM)
    elif word == "clear" and not rest:
        staging.clear()
        ctx.say("nothing attached", Tone.DIM)
    else:
        staged = staging.stage_file(Path(args).expanduser())
        if isinstance(staged, NotStaged):
            ctx.say(f"not attached: {staged.why}", Tone.BAD)
        else:
            ctx.say(staged.chip(len(staging.items())))


NO_ATTACHING: Final = (
    "this server takes no attachments (it is older than wire 4.18, or a replay)"
)


# -- words with a route of their own --------------------------------------------------------


def _say_as_turn(ctx: Context, args: str) -> None:
    """`/run`, `/ask`, `/investigate`: the session decides how to answer, so the words
    are a turn like any other; the command only says they are meant as one."""
    if not args:
        ctx.say("say what: /run <goal>, /ask <question>, /investigate <symptom>", Tone.BAD)
        return
    ctx.tell(args)


# -- reading --------------------------------------------------------------------------------


def _status(ctx: Context, _args: str) -> None:
    sid = ctx.session_id
    info = ctx.client.session_info(sid)
    alive = ctx.client.liveness(sid)
    ctx.say(f"session {sid}" + (f" — {info.title}" if info.title else ""))
    why = f" ({alive.why})" if alive.why else ""
    ctx.say(f"  state      {alive.kind}{why}")
    if alive.outcome:
        ctx.say(f"  last run   {info.last_run_id}: {alive.outcome}")
    elif info.last_run_id:
        ctx.say(f"  last run   {info.last_run_id}")
    _show_budget(ctx, ctx.client.budget(sid))
    # Shown where the person looks, so a floor that asks too often is seen.
    text, tone = stop_rate_line(stop_rate(ctx.client.session_journal(sid)))
    ctx.say(f"  approvals  {text}", tone)
    for line in ctx.client.header(sid).lines:
        ctx.say(line.text, Tone.DIM, indent="  ")  # the engine's text names its own kind
    if info.last_run_id:
        ctx.show(board_lines(_run_view(ctx, info.last_run_id), ctx.style))


def _diff(ctx: Context, args: str) -> None:
    run_id = args.strip() or ctx.client.session_info(ctx.session_id).last_run_id
    if not run_id:
        ctx.say("no run yet: a run starts when the session first writes code", Tone.DIM)
        return
    ctx.show(diff_lines(_run_view(ctx, run_id), ctx.style))


def _run_view(ctx: Context, run_id: str) -> RunView:
    return fold_run(run_id, ctx.client.run_journal(run_id))


def _workspace(ctx: Context, _args: str) -> None:
    shown = 0
    for line in ctx.client.header(ctx.session_id).lines:
        if line.kind in ("workspace", "isolation", "sandbox"):
            ctx.say(line.text)
            shown += 1
    if not shown:
        ctx.say("no workspace yet: one is taken when the session first writes", Tone.DIM)


def _journal(ctx: Context, args: str) -> None:
    records = (
        ctx.client.run_journal(args) if args else ctx.client.session_journal(ctx.session_id)
    )
    for record in records:
        ctx.say(_raw(record), Tone.DIM)


def _raw(record: RecordFrame) -> str:
    body = json.dumps(record.payload, sort_keys=True)
    return f"{record.seq:>5} {record.kind:<18} {body[:160]}"


# -- the cap -------------------------------------------------------------


def _rename(ctx: Context, args: str) -> None:
    """The person's name for the session, which the title role never overwrites."""
    if not args.strip():
        ctx.say("/rename <name>", Tone.BAD)
        return
    info = ctx.client.rename(ctx.session_id, args)
    ctx.say(f"named {info.title}", Tone.DIM)


def _budget(ctx: Context, args: str) -> None:
    sid = ctx.session_id
    words = args.split()
    if not words:
        _show_budget(ctx, ctx.client.budget(sid))
        return
    rung = words[0] if len(words) == 2 else None
    amount = words[-1]
    if len(words) > 2:
        ctx.say("/budget [usd|off] or /budget <rung> <usd|off>", Tone.BAD)
        return
    if amount == "off":
        cap: float | None = None
    else:
        try:
            cap = float(amount.removeprefix("$"))
        except ValueError:
            ctx.say(f"not an amount: {amount}", Tone.BAD)
            return
    now = ctx.client.budget(sid)
    if cap is not None and cap < now.spent_usd:
        answer = ctx.ask(
            f"${cap} is below the ${now.spent_usd:.2f} already spent; set it? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            ctx.say("cap unchanged", Tone.DIM)
            return
    _show_budget(ctx, ctx.client.set_cap(sid, cap, rung))


def _show_budget(ctx: Context, budget: SessionBudget) -> None:
    cap = "none" if budget.cap_usd is None else f"${budget.cap_usd:.2f}"
    ctx.say(f"  spend      ${budget.spent_usd:.4f} of {cap} (cap from {budget.cap_source})")
    for name, rung in sorted((budget.rungs or {}).items()):
        # A rung reads as the session line does: what it spent, and its own cap if it has one.
        spent = f"${rung.spent_usd:.4f}"
        if rung.cap_usd is None:
            line = f"{spent}, no cap of its own"
        else:
            line = f"{spent} of ${rung.cap_usd:.2f} (cap from {rung.cap_source})"
        ctx.say(f"    {name:<12} {line}", Tone.DIM)


# -- undo and redo -------------------------------------------------------------------

_MODES: Final = {"both": (True, True), "conversation": (False, True), "code": (True, False)}


def _undo(ctx: Context, args: str) -> None:
    """Which turn, then which halves, then the engine's questions (a
    conflict per path), then what was restored, kept and not undone. `/undo task <run>
    <task>` and `/undo <operation> [<run>]` restore one task or one operation instead;
    the run names which one's when operation ids repeat across runs. A turn given
    with its halves (`/undo 3 code`) is not asked them again, so a screen that chose them
    already (the TUI's undo screen) sends one line."""
    sid = ctx.session_id
    words = args.split()
    if words and words[0] == "task":
        if len(words) != 3:
            ctx.say("/undo task <run> <task>", Tone.BAD)
            return
        run_id, task_id = words[1], words[2]
        _show_restored(ctx, ctx.answering(lambda: ctx.client.undo_task(sid, run_id, task_id)))
        return
    if words and words[0] not in _FLAGS and not words[0].isdigit():
        if len(words) > 2:
            ctx.say("/undo <operation> [<run>]", Tone.BAD)
            return
        operation_id = words[0]
        run = words[1] if len(words) == 2 else None
        _show_restored(
            ctx, ctx.answering(lambda: ctx.client.undo_operation(sid, operation_id, run))
        )
        return
    commit = True if "--commit" in words else False if "--no-commit" in words else None
    numbers = [w for w in words if w.isdigit()]
    halves = [w for w in words if w in _MODES]
    candidates = ctx.client.undoable(sid)
    if not candidates:
        ctx.say("nothing to undo", Tone.DIM)
        return
    chosen = _pick_turn(ctx, candidates, int(numbers[0]) if numbers else None)
    if chosen is None:
        return
    mode = _pick_mode(ctx, chosen, halves[0] if halves else None)
    if mode is None:
        return
    code, conversation = _MODES[mode]
    report = ctx.answering(
        lambda: ctx.client.undo(
            sid, chosen.turn, code=code, conversation=conversation, commit=commit
        )
    )
    _show_report(ctx, "undone", report)
    if report.prompt:
        # The turn's words, to edit and send again. The prompt edits one
        # line, so a turn of several is joined, and the person is told.
        lines = report.prompt.splitlines()
        if len(lines) > 1:
            ctx.say(f"(its {len(lines)} lines are joined into one to edit)", Tone.DIM)
        ctx.put_back(" ".join(line.strip() for line in lines))


_FLAGS: Final = ("--commit", "--no-commit")


def _pick_turn(
    ctx: Context, candidates: Sequence[Undoable], wanted: int | None
) -> Undoable | None:
    by_turn = {c.turn: c for c in candidates}
    if wanted is None:
        for c in candidates:
            ctx.say(f"  turn {c.turn}: {c.prompt}")
        answer = ctx.ask("take back from which turn? ").strip()
        if not answer.isdigit():
            ctx.say("undo cancelled", Tone.DIM)
            return None
        wanted = int(answer)
    chosen = by_turn.get(wanted)
    if chosen is None:
        ctx.say(f"turn {wanted} cannot be taken back; /undo lists those that can", Tone.BAD)
    return chosen


def _pick_mode(ctx: Context, chosen: Undoable, given: str | None) -> str | None:
    """Only the halves that apply; with one, or `given`, no question. Halves
    the turn does not offer are refused by name, never swapped for others."""
    options = list(chosen.options)
    if chosen.no_code:
        ctx.say(chosen.no_code, Tone.DIM)
    if given is not None:
        if given in options:
            return given
        ctx.say(f"turn {chosen.turn} offers {' / '.join(options)}, not {given}", Tone.BAD)
        return None
    if len(options) == 1:
        return options[0]
    answer = ctx.ask(f"undo {' / '.join(options)}? [{options[0]}] ").strip() or options[0]
    if answer not in options:
        ctx.say("undo cancelled", Tone.DIM)
        return None
    return answer


def _show_restored(ctx: Context, restored: Restored) -> None:
    """What a task or operation undo put back, in which workspace."""
    ctx.say(f"restored in {restored.root}", Tone.GOOD)
    for label, paths in (
        ("restored", restored.restored),
        ("kept", restored.kept),
        ("kept by default", restored.kept_by_default),
    ):
        if paths:
            ctx.say(f"{label}: {', '.join(paths)}", Tone.DIM, indent="  ")
    if restored.beside:
        # What it put back may hold a job's writes, which no step recorded.
        jobs = ", ".join(restored.beside)
        ctx.say(f"it ran while {jobs} ran: it may have put back {jobs}'s writes", Tone.ASK)


def _jobs(ctx: Context, args: str) -> None:
    """The session's jobs, or `kill <id>`: the kill is recorded by the
    job's waiter, and tells nobody."""
    words = args.split()
    if words[:1] == ["kill"] and len(words) == 2:
        ctx.answering(lambda: ctx.client.kill_job(ctx.session_id, words[1]))
        ctx.say(f"{words[1]} killed", Tone.GOOD)
        return
    if words:
        ctx.say("usage: /jobs [kill <id>]", Tone.BAD)
        return
    jobs = ctx.answering(lambda: ctx.client.jobs(ctx.session_id))
    if not jobs:
        ctx.say("no jobs in this session", Tone.DIM)
    for job in jobs:
        # The engine's own line for it, the one `job list` gives Zeus.
        ctx.say(job.row, Tone.PLAIN)


def _redo(ctx: Context, _args: str) -> None:
    _show_report(ctx, "redone", ctx.answering(lambda: ctx.client.redo(ctx.session_id)))


def _show_report(ctx: Context, what: str, report: UndoReport) -> None:
    """What an undo or a redo did, path by path, and what it could not do (decision 3)."""
    halves = (
        "code and conversation"
        if report.code and report.conversation
        else ("code" if report.code else "conversation")
    )
    ctx.say(f"{what} from turn {report.to_turn}: {halves}", Tone.GOOD)
    for outcome in report.workspaces:
        ctx.say(f"{outcome.where} {outcome.root}", Tone.DIM, indent="  ")
        for label, paths in (
            ("restored", outcome.restored),
            ("kept", outcome.kept),
            ("kept by default", outcome.kept_by_default),
            ("skipped", outcome.skipped),
        ):
            if paths:
                ctx.say(f"{label}: {', '.join(paths)}", Tone.DIM, indent="    ")
        if outcome.committed:
            ctx.say(f"committed {outcome.committed}", Tone.DIM, indent="    ")
        if outcome.note:
            ctx.say(outcome.note, Tone.DIM, indent="    ")
    for missed in report.not_undone:
        ctx.say(
            f"not undone: {missed.kind} at seq {missed.seq}: {missed.detail}",
            Tone.BAD,
            indent="  ",
        )
    for why in report.withheld:
        ctx.say(f"not redone: {why}", Tone.BAD, indent="  ")


# -- doing and ending work ------------------------------------------------------------------


def _pause(ctx: Context, _args: str) -> None:
    """The team's run paused now; `coder` with its work, or `/resume`, continues
    it."""
    ctx.say(ctx.answering(lambda: ctx.client.pause_run(ctx.session_id)), Tone.GOOD)


def _drop(ctx: Context, _args: str) -> None:
    """The team's run, live or paused, ended for good; what it committed stays."""
    ctx.say(ctx.answering(lambda: ctx.client.drop_run(ctx.session_id)), Tone.GOOD)


def _resume(ctx: Context, _args: str) -> None:
    resume(ctx, attaching=False)


def resume(ctx: Context, *, attaching: bool) -> None:
    """Continue the session's unfinished run, and follow it. `attaching`: as the REPL
    opens an existing session, only a run that was paused, and not one still going; and when
    there is none, nothing is said."""
    sid = ctx.session_id
    live = ctx.client.liveness(sid)
    paused = live.paused_run
    if attaching and (paused is None or live.kind in ("working", "awaiting")):
        return
    # Read before the run goes on: what it wrote so far was shown when it ran, and it is
    # followed from the last of that, so nothing it writes once resumed is missed.
    seen = ctx.client.run_journal(paused) if paused is not None else ()
    resumed = ctx.client.resume(sid)
    run_id = resumed.in_flight
    if run_id is None:
        if not attaching:
            ctx.say("nothing in flight to continue", Tone.DIM)
        return
    for earlier in resumed.reconciled[:-1]:
        ctx.say(f"reconciled {earlier}", Tone.DIM)
    ctx.say(f"continuing run {run_id}, paused when it was left", Tone.GOOD)
    ctx.follow_run(run_id, seen[-1].seq if seen and run_id == paused else 0)


def _handoff(ctx: Context, args: str) -> None:
    """With a choice, carry it out; with none, review the
    session's worktree and ask which."""
    if args:
        if args not in CHOICES:
            ctx.say(f"/handoff [{'|'.join(CHOICES)}]", Tone.BAD)
            return
        _finalize(ctx, args)
        return
    review(ctx, leaving=False)


LEAVE_PENDING: Final = "left pending; /handoff, or --continue, comes back to it"


def review(ctx: Context, *, leaving: bool) -> None:
    """Show what the session leaves behind and ask for one of the choices the engine accepts.
    Enter (or Ctrl-D) leaves the worktree pending. `leaving`: asked as the REPL closes, and
    only where `render.diff.asked_on_leaving` says there is something to choose over."""
    summary = ctx.client.handoff(ctx.session_id)
    if leaving and not asked_on_leaving(summary):
        return
    ctx.show(handoff_lines(summary, ctx.style))
    choices = offered(summary)
    if not choices:
        return
    try:
        answer = ctx.ask(f"{', '.join(choices)}? (Enter leaves it pending) ").strip()
    except NoAnswer:
        answer = ""  # Ctrl-D, as the REPL closes: pending, as Enter leaves it
    if not answer:
        ctx.say(LEAVE_PENDING, Tone.DIM)
    elif answer not in choices:
        ctx.say(f"{answer!r} is not one of {', '.join(choices)}; {LEAVE_PENDING}", Tone.BAD)
    else:
        _finalize(ctx, answer)


RUNNING_HINT: Final = (
    "a handoff waits for the work to end: let it finish, or stop it for good (Ctrl-C twice "
    "while it runs), then /handoff"
)
"""Said when the engine is busy: one Ctrl-C only pauses a run, which no handoff lands from
under.
"""


def _finalize(ctx: Context, choice: str) -> None:
    try:
        result = ctx.client.finalize(ctx.session_id, choice)
    except WireError as exc:
        if exc.code != "BUSY":
            raise
        ctx.say(f"/handoff: {exc.message}", Tone.BAD)
        ctx.say(RUNNING_HINT, Tone.DIM, indent="  ")
        return
    ctx.show(finalize_lines(result, ctx.style))


def _rung(ctx: Context, args: str) -> None:
    """With no rung, each rung's model and the catalog; with a rung,
    its model and the catalog's others; with both, switch that rung from its next call."""
    words = args.split()
    if len(words) > 2:
        ctx.say("/rung [rung] [model]", Tone.BAD)
        return
    if len(words) == 2:
        moved = ctx.client.switch_model(ctx.session_id, words[0], words[1])
        ctx.say(
            f"{moved.rung}: {moved.from_model} → {moved.to_model}, from its next call",
            Tone.GOOD,
        )
        return
    wiring = ctx.client.models(ctx.session_id)
    switched = {s.rung for s in wiring.switches}
    if not words:
        for rung, model in sorted(wiring.rungs.items()):
            note = "  (switched this session)" if rung in switched else ""
            ctx.say(f"  {rung:<12} {model}{note}")
        ctx.say(f"  catalog: {', '.join(wiring.catalog)}", Tone.DIM)
        return
    rung = words[0]
    model = wiring.rungs.get(rung)
    if model is None:
        ctx.say(f"no rung {rung}; the rungs are {', '.join(sorted(wiring.rungs))}", Tone.BAD)
        return
    others = [name for name in wiring.catalog if name != model]
    ctx.say(f"  {rung:<12} {model}")
    ctx.say(f"  others: {', '.join(others) or 'none'}", Tone.DIM)


def _not_on_the_wire(ctx: Context, _args: str) -> None:
    ctx.say(NOT_ON_THE_WIRE, Tone.DIM)


def _command(name: str, usage: str, help_text: str, run: Handler) -> tuple[str, Command]:
    return name, Command(name, usage, help_text, run)


COMMANDS: Final[dict[str, Command]] = dict(
    [
        _command("help", "/help", "these commands", _help),
        _command("exit", "/exit", "leave (Ctrl-D too)", _exit),
        _command("clear", "/clear", "clear the screen", _clear),
        _command(
            "attach",
            "/attach [<path>|drop <n>|clear]",
            "attach an image, a PDF or a text file to the next turn",
            _attach,
        ),
        _command("verbose", "/verbose [q|normal|v]", "how much of a turn is shown", _verbose),
        _command("run", "/run <goal>", "say a goal to the session", _say_as_turn),
        _command("ask", "/ask <question>", "ask the session", _say_as_turn),
        _command("investigate", "/investigate <symptom>", "ask why", _say_as_turn),
        _command("plan", "/plan <goal>", "plan only, stop at the gate", _not_on_the_wire),
        _command("status", "/status", "state, last run, spend, header", _status),
        _command("workspace", "/workspace", "where the session writes", _workspace),
        _command("rename", "/rename <name>", "name the session", _rename),
        _command("journal", "/journal [run]", "the session's records, or a run's", _journal),
        _command("budget", "/budget [rung] [usd|off]", "show or set the cap (D143)", _budget),
        _command(
            "undo",
            "/undo [turn [both|conversation|code]|task <run> <task>|<operation>]"
            " [--commit|--no-commit]",
            "take turns back, or one task or operation",
            _undo,
        ),
        _command("redo", "/redo", "take the last undo back", _redo),
        _command("jobs", "/jobs [kill <id>]", "the session's jobs, or end one", _jobs),
        _command("resume", "/resume", "continue a run left unfinished", _resume),
        _command("pause", "/pause", "pause the team's run now", _pause),
        _command("drop", "/drop", "end the team's run for good", _drop),
        _command(
            "handoff",
            "/handoff [bring|apply|squash|keep|discard]",
            "review the session's worktree and land it",
            _handoff,
        ),
        _command("rung", "/rung [rung] [model]", "show or switch a rung's model", _rung),
        _command("diff", "/diff [run]", "the files each task changed", _diff),
        _command("replay", "/replay <run>", "replay assertion", _not_on_the_wire),
        _command("context", "/context [expand <n>]", "what is in context", _not_on_the_wire),
        _command("notify", "/notify [on|off|test]", "completion notice", _not_on_the_wire),
        _command("gc", "/gc", "prune blobs", _not_on_the_wire),
        _command("skills", "/skills", "the skill index", _not_on_the_wire),
        _command("unload", "/unload <skill>", "release a skill", _not_on_the_wire),
        _command("config", "/config", "this session's configuration", _not_on_the_wire),
    ]
)
"""The table, in its order."""
