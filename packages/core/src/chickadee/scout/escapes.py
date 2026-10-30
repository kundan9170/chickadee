"""`request_exec`: the scout's escape hatch, requested and never chosen.

The scout cannot choose to run a command. It *requests* one with a typed tool call, and a
deterministic policy grants or refuses it. The request is read from
the completion's `tool_calls` and validated against its schema — never scanned out of prose
— so a model that mentions a command, or quotes a file containing one, has requested
nothing. The command runs exactly as written or not at all: nothing here rewrites it.

Policy, in the order, no model involved:

1. **Already known?** A command the session already ran, or one promoted into the snapshot,
   is answered from memory. No budget is spent and nothing runs.
2. **Budget?** `ESCAPES_PER_LOCATE` granted runs per `locate` call.
3. **A mutation in disguise?** The command table answers: a line any of whose commands is
   not a read (installing, fetching, writing files, changing the repository's history) is
   an action, and actions are Zeus's, with approval.
4. **The Executor's pre-filter**: a deny-listed path named in the argv, or an obviously
   mutating DISCOVERY command. A pass proves nothing; the sandbox is the boundary.

A granted request runs through `Executor.run` under `DISCOVERY` — repository read-only,
network off, the deny list masked — with its `operation_id` and D40 phases recorded through
the caller's sink *before* dispatch.

**Only where that holds.** On a machine with no sandbox backend nothing enforces it, and the
command is the model's, chosen after reading the repository — so no escape runs there unless
the person allowed it (`allow_unsandboxed`, default False), and the loop does not even offer
`request_exec`. The worker's shell is refused the same way.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from chickadee.approval.commands import CLASS_WORDS, CommandTable, shipped
from chickadee.errors import HarnessBug, InfraError
from chickadee.executor.executor import Executor
from chickadee.executor.policy import SECRET_PATH, prefilter
from chickadee.executor.reconcile import Landed, Reconciliation
from chickadee.executor.types import ExecRequest, ExecResult, PolicyClass
from chickadee.ids import escape_id
from chickadee.isolation.base import level_of
from chickadee.isolation.types import SandboxLevel
from chickadee.journal.effects import EffectKind, Intent, perform
from chickadee.limits import tokens_for_chars
from chickadee.schema import sha256_hex
from chickadee.schema.journal import (
    EffectIntendedP,
    EventKind,
    ExecP,
    ScoutEscapeP,
)
from chickadee.scout.sink import ScoutSink
from chickadee.tools.builtin.request_exec import SPEC as REQUEST_EXEC_SPEC
from chickadee.tools.spec import (
    INLINE_BYTES_DEFAULT,
)
from chickadee.tools.validate import ArgumentViolation, InvalidArguments

__all__ = [
    "ESCAPES_PER_LOCATE",
    "EXEC_TIMEOUT_S",
    "REQUEST_EXEC",
    "REQUEST_EXEC_SPEC",
    "EscapeOutcome",
    "EscapeRequest",
    "Escapes",
    "Refusal",
    "action_in_disguise",
    "reconcile",
]

REQUEST_EXEC = REQUEST_EXEC_SPEC.name

ESCAPES_PER_LOCATE = 3
"""SC-5's default: granted runs per `locate` call. An answer from memory spends none."""

EXEC_TIMEOUT_S = 30
"""A discovery command answers a question about the repository; one that runs longer than
this is doing something else."""


Refusal = Literal["no_sandbox", "budget", "mutation", "prefilter", "deny_list", "unavailable"]


@dataclass(frozen=True)
class EscapeRequest:
    command: str
    reason: str

    @staticmethod
    def of(command: str, reason: str) -> EscapeRequest | InvalidArguments:
        """A request from arguments the schema already passed: what the schema cannot say, a
        command of whitespace alone, is `INVALID_ARGUMENTS` too."""
        if not command.strip():
            return InvalidArguments(
                REQUEST_EXEC,
                (ArgumentViolation("command", "a command is more than whitespace"),),
                REQUEST_EXEC_SPEC.schema,
            )
        return EscapeRequest(command=command, reason=reason)


@dataclass(frozen=True)
class EscapeOutcome:
    request: EscapeRequest
    granted: bool
    known: bool = False
    """Answered from what the session already knew; nothing ran and no budget was spent."""
    refusal: Refusal | None = None
    why: str = ""
    output: str = ""
    output_sha: str | None = None
    operation_id: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    cancelled: bool = False

    def render(self) -> str:
        """What the scout's model reads. A refusal says the command did not run, so it cannot
        be mistaken for an empty answer; what to do then is the tool's description's to say."""
        cmd = self.request.command
        if self.refusal is not None:
            refused = f"REFUSED{{reason: {self.refusal}, detail: {self.why!r}}}"
            return f"{refused} — `{cmd}` did not run"
        if self.known:
            return f"request_exec `{cmd}`: known this session, nothing re-run\n{self.output}"
        status = "timed out" if self.timed_out else f"exit {self.exit_code}"
        return f"request_exec `{cmd}` · {status}\n{self.output}"


# -- step 3: a mutation in disguise -----------------------------------------------------


def action_in_disguise(command: str, commands: CommandTable | None = None) -> str | None:
    """Why `command` is an action rather than a question, or None. The command table answers:
    a line any of whose commands is not a read (`READS`) is an action. A
    recogniser, not the boundary: what it misses the sandbox refuses at the syscall.
    `commands` is the run's `[[commands]]` table; `None` is the shipped table."""
    table = commands if commands is not None else shipped()
    found = table.classify_line(command)
    action = found.action() if found is not None else None
    if action is None:
        return None
    return f"{action.text!r} {CLASS_WORDS[action.cls]}; actions are Zeus's, with approval"


# -- the escapes of one `locate` call ---------------------------------------------------


def _cap(text: str, limit: int) -> str:
    data = text.encode("utf-8", "replace")
    if len(data) <= limit:
        return text
    kept = data[:limit].decode("utf-8", "ignore")
    return f"{kept}\n[truncated: {len(data) - limit:,} of {len(data):,} bytes not shown]"


@dataclass
class Escapes:
    """The escapes of one `locate` call: the budget is per call, `memo` per session —
    the caller hands the same dict to every call, seeded with the snapshot's promoted
    outputs, so a command asked twice in a session runs once.

    The deny list is the executor's — the one the run resolved — never a second copy:
    two lists would let the policy grant what the executor then refuses, and journal the
    refusal as a granted escape."""

    executor: Executor
    cwd: Path
    memo: dict[str, str] = field(default_factory=dict[str, str])
    budget: int = ESCAPES_PER_LOCATE
    timeout_s: int = EXEC_TIMEOUT_S
    output_bytes: int = INLINE_BYTES_DEFAULT
    new_id: Callable[[], str] = escape_id
    used: int = 0
    history: list[EscapeOutcome] = field(default_factory=list[EscapeOutcome])
    allow_unsandboxed: bool = False
    """The person's opt-in to run escapes where no sandbox backend exists. Never a
    repository's to set: a repository that could would choose to run its own model-chosen
    commands unprotected on every clone."""
    commands: CommandTable | None = None
    """The `[[commands]]` table step 3 asks whether a command is a read; `None` is its
    executor's, the engine's one table."""
    promoted: dict[str, str | None] = field(default_factory=dict[str, str | None])
    """Commands `run_promoted` put in this session's snapshot, with their output's sha. They
    count toward promotion on every call while they stay promoted: the model never asks for
    what the snapshot already holds, and counting only its asks would demote them."""

    def sealed(self) -> bool:
        """Whether a sandbox backend will enforce DISCOVERY's read-only, networkless run."""
        backend = self.executor.backend
        return level_of(backend) is not SandboxLevel.NONE

    def runnable(self) -> bool:
        """Whether an escape may run here at all: sealed, or unsealed with the person's
        opt-in. When not, `request_exec` is not offered."""
        return self.allow_unsandboxed or self.sealed()

    def for_call(self) -> Escapes:
        """The escapes of one `locate` call: its own budget and history, and the
        session's memo and promoted commands, shared by reference so what one call learns the
        next can recall. Each call holds its own, so no call can reset another's budget."""
        return replace(self, used=0, history=[])

    def begin(self) -> None:
        """A new `locate` call: a fresh budget and history; the session's memo stays."""
        self.used = 0
        self.history = []

    def request(
        self,
        req: EscapeRequest,
        sink: ScoutSink,
        cancel: threading.Event | None = None,
    ) -> EscapeOutcome:
        """The policy, then the dispatch. Every outcome is recorded as `scout_escape`."""
        known = self.memo.get(req.command)
        if known is not None:
            outcome = EscapeOutcome(req, granted=True, known=True, output=known)
            return self._recorded(outcome, sink, exec_seq=0)
        refusal = self._policy(req)
        if refusal is not None:
            return self._recorded(refusal, sink, exec_seq=0)
        self.used += 1
        outcome, exec_seq = self._dispatch(req, sink, cancel)
        if outcome.refusal is None and not (outcome.timed_out or outcome.cancelled):
            # A non-zero exit is an answer (`grep` finding nothing exits 1) and re-running it
            # would give the same one; a run cut short is not, so it may be asked again.
            self.memo[req.command] = outcome.output
        return self._recorded(outcome, sink, exec_seq=exec_seq)

    def run_promoted(
        self,
        commands: Iterable[str],
        sink: ScoutSink,
        cancel: threading.Event | None = None,
    ) -> dict[str, str] | None:
        """Re-run the promoted commands as part of the deterministic snapshot: under
        the same DISCOVERY policy and D40 phases, outside the budget, because no model asked.
        A promoted command that is now refused, or fails, is left out rather than stale — and
        stops counting toward its own promotion, so it can demote.

        The promoted set is shared by every call's escapes (`for_call`), so it is replaced in
        place once the new set is known, never emptied while the commands run. The caller
        serialises a rebuild against its readers (`ScoutLocator`).

        The session's stop ends the rebuild between commands, and reaps the one running.
        A stopped rebuild is `None` and changes nothing: a command cut short did not fail, so
        the set stays as it was rather than demote it, and the caller keeps no snapshot."""
        outputs: dict[str, str] = {}
        promoted: dict[str, str | None] = {}
        for command in commands:
            if cancel is not None and cancel.is_set():
                return None
            req = EscapeRequest(command=command, reason="promoted into the snapshot (SC-8)")
            if self._policy(req, spend=False) is not None:
                continue
            outcome, exec_seq = self._dispatch(req, sink, cancel)
            self._recorded(outcome, sink, exec_seq=exec_seq)
            if outcome.cancelled or (cancel is not None and cancel.is_set()):
                return None
            if outcome.refusal is None and outcome.exit_code == 0 and not outcome.timed_out:
                outputs[command] = outcome.output
                self.memo[command] = outcome.output
                promoted[command] = outcome.output_sha or sha256_hex(outcome.output)
        self._swap_promoted(promoted)
        return outputs

    def _swap_promoted(self, promoted: dict[str, str | None]) -> None:
        # Built first, swapped last: the set is never left empty while commands run. The two
        # dict operations are not one atomic step; what makes a concurrent reader safe is the
        # locator, which rebuilds the snapshot and reads `asked()` under one lock.
        self.promoted.clear()
        self.promoted.update(promoted)

    def room(self, command: object) -> int:
        """The most this request's answer can add to what a `locate` call has read, in
        tokens: an escape is an effect, so it is refused before it runs when this could not
        fit. A command the session already knows runs nothing and adds no output."""
        if isinstance(command, str) and command in self.memo:
            return 0
        return tokens_for_chars(self.output_bytes)

    def asked(self) -> tuple[tuple[str, str | None], ...]:
        """The commands this call ran or recalled, and those the snapshot holds promoted,
        with their output's sha — what the hint store counts toward promotion."""
        seen: dict[str, str | None] = dict(self.promoted)
        for o in self.history:
            if o.granted:
                seen.setdefault(o.request.command, o.output_sha or sha256_hex(o.output))
        return tuple(seen.items())

    # -- internals ----------------------------------------------------------------------

    def _policy(self, req: EscapeRequest, *, spend: bool = True) -> EscapeOutcome | None:
        if not self.runnable():
            return EscapeOutcome(
                req, granted=False, refusal="no_sandbox", why="no sandbox on this machine"
            )
        if spend and self.used >= self.budget:
            return EscapeOutcome(
                req,
                granted=False,
                refusal="budget",
                why=f"{self.budget} escape(s) per locate call, all spent",
            )
        table = self.commands if self.commands is not None else self.executor.commands
        disguised = action_in_disguise(req.command, table)
        if disguised is not None:
            return EscapeOutcome(req, granted=False, refusal="mutation", why=disguised)
        probe = ExecRequest(
            command=req.command, cwd=self.cwd, policy=PolicyClass.DISCOVERY, timeout_s=1
        )
        denial = prefilter(probe, self.executor.denylist, table)
        if denial is not None:
            kind: Refusal = "deny_list" if denial.startswith(SECRET_PATH) else "prefilter"
            return EscapeOutcome(req, granted=False, refusal=kind, why=denial)
        return None

    def _dispatch(
        self, req: EscapeRequest, sink: ScoutSink, cancel: threading.Event | None
    ) -> tuple[EscapeOutcome, int]:
        backend = self.executor.backend
        level = level_of(backend)
        request = ExecRequest(
            command=req.command,
            cwd=self.cwd,
            policy=PolicyClass.DISCOVERY,
            timeout_s=self.timeout_s,
            label=f"scout: {req.reason}",
        )
        op = self.new_id()
        intent = Intent(
            op,
            EffectKind.SCOUT_ESCAPE,
            # No person approved this; the policy's grant stands in, and hashing the command
            # lets a resume tell this escape from any other.
            approved_sha=sha256_hex(req.command),
            preconditions=(
                "policy=discovery",
                "repository read-only",
                "network off",
                f"sandbox={level.value}",
            ),
            idempotent=True,
            surface=(),
        )
        # Anything but the executor's refusal may have come after the command ran, so it is
        # closed `ambiguous` (the default). A kill writes nothing.
        with perform(sink, intent) as fx:
            try:
                result = self.executor.run(request, operation_id=op, cancel=cancel)
            except InfraError as exc:
                # Refused before anything spawned: `failed`, and answered as a refusal.
                fx.complete("failed")
                refused = EscapeOutcome(
                    req, granted=False, refusal="unavailable", why=str(exc), operation_id=op
                )
                return refused, 0
            exec_seq = fx.record(
                EventKind.EXEC,
                ExecP.model_validate(Executor.exec_event(request, result, op)),
            )
            fx.complete(
                "succeeded" if result.ok else "failed",
                observed=tuple(v.path for v in result.violations),
                result=result.stdout_sha,
            )
        return self._granted(req, result, op), exec_seq

    def _granted(self, req: EscapeRequest, result: ExecResult, op: str) -> EscapeOutcome:
        text = result.stdout
        if result.stderr.strip():
            text = f"{text}\n--- stderr ---\n{result.stderr}" if text else result.stderr
        output = _cap(text, self.output_bytes)
        return EscapeOutcome(
            req,
            granted=True,
            output=output,
            output_sha=result.stdout_sha,
            operation_id=op,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
        )

    def _recorded(
        self, outcome: EscapeOutcome, sink: ScoutSink, *, exec_seq: int
    ) -> EscapeOutcome:
        """`scout_escape` for every outcome. `exec_seq` is the `exec` record's sequence
        number, and 0 when nothing ran — a refusal, or an answer from memory."""
        sink.record(
            EventKind.SCOUT_ESCAPE,
            ScoutEscapeP(
                command=outcome.request.command,
                granted=outcome.granted,
                reason=outcome.request.reason,
                exec_seq=exec_seq,
            ),
        )
        self.history.append(outcome)
        return outcome


# -- recovery ---------------------------------------------------------------------------


def reconcile(intent: EffectIntendedP) -> Reconciliation:
    """What recovery may conclude about an escape that reached `effect_started` and never
    `effect_completed`.

    A DISCOVERY run is sealed — repository read-only, network off — so it cannot have landed
    an effect, and its only product was output nobody recorded: `NO`, which with
    `idempotent=True` is what permits a re-run. Except at `sandbox=none`, where nothing
    enforced the seal: whether it wrote cannot be observed from here, so `UNKNOWN`,
    which never permits one."""
    if intent.effect_kind != EffectKind.SCOUT_ESCAPE:
        raise HarnessBug(f"{intent.operation_id} is a {intent.effect_kind!r}, not an escape")
    if f"sandbox={SandboxLevel.NONE.value}" in intent.preconditions:
        return Reconciliation(
            Landed.UNKNOWN,
            "the escape ran at sandbox=none, where its read-only policy was not enforced",
        )
    return Reconciliation(
        Landed.NO,
        "a sealed DISCOVERY run writes nothing and reaches nothing; its only product was "
        "output, which was never recorded",
    )
