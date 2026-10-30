"""The run header: where a session runs, and every guarantee it runs without.

At least nine specs put a fact "in the run header" — reduced isolation, `sandbox=none`, a
removed deny-list entry, `allow_secret_read`, a bootstrap that ran nothing, a suspected
committed secret — and until `18-engine.md` §2 none said what the header was, so every one of
those guarantees was stated to nobody. This module is that header.

**A projection, never a record.** Every line comes from a fact already written down: the
resolved config, the session log's `workspace` and `episode` lines, and the episodes' run
journals (`exec`, the bootstrap's effect, `secret_suspected`). So `header()` adds no journal
kind and no event, and a line that arises mid-session is on the next call. It is computed per
call like `liveness()`, because a stored header would be stale the moment a fact changed.

**A kind with nothing to state contributes no line.** The header is a list of reductions and
locations, not a status panel: a line that says "all fine" is one a reader learns to skip,
and then skips the one that was not. `workspace` and `sandbox` are the exceptions the table
makes — they say *where*, which is always worth one line.

The kinds whose features are not built (`mcp`, `discovery`, `storage`) are in the type so the
vocabulary is §2's, and join with the tasks that build them. `models` joined with `T-5.10`:
each rung's model, from the config and the session's `model_switched` lines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from chickadee.approval.commands import CommandTable
from chickadee.budget.family import SessionBudget
from chickadee.config import SHELL_ROLES, EngineConfig, ShellRole
from chickadee.context.project import JournalRecords
from chickadee.context.replay import ReplayCache
from chickadee.context.store import SessionStoreError
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.errors import InfraError, SessionError
from chickadee.isolation.base import configured, level_of, reads_masked
from chickadee.journal.effects import EffectKind
from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.projection import RunProjection
from chickadee.journal.runs import RunProjections
from chickadee.schema.execution import SandboxLevel
from chickadee.schema.isolation import WorkspaceMode
from chickadee.schema.journal import ExecP, RunStartP, SecretSuspectedP
from chickadee.secrets.denylist import Denylist
from chickadee.state.resolve import PROJECT_LAYER
from chickadee.state.roots import Roots
from chickadee.wiring import switches_of, wired
from chickadee.workspace import dirty_paths

__all__ = [
    "BOOTSTRAP_FAILED",
    "IN_PLACE_NO_BOOTSTRAP",
    "NO_BOOTSTRAP",
    "Episode",
    "HeaderFacts",
    "HeaderKind",
    "HeaderLine",
    "RunHeader",
    "detected_level",
    "gather",
    "header",
    "project",
    "spend_facts",
]

HeaderKind = Literal[
    "workspace",
    "isolation",
    "sandbox",
    "deny_list",
    "commands",
    "bootstrap",
    "mcp",
    "discovery",
    "secrets",
    "storage",
    "models",
    "spend",
]

NO_BOOTSTRAP = "no bootstrap — the worktree has the checkout and nothing else"
IN_PLACE_NO_BOOTSTRAP = "no bootstrap — in-place mode runs in the checkout's own environment"
BOOTSTRAP_FAILED = "bootstrap failed — the worktree was not prepared, and no episode ran in it"
"""The two ways a bootstrap runs nothing. Defined here, and re-exported by `engine` for
`BootstrapReport.header`, so the episode's report and the session's header say it in one
wording rather than two that drift."""


@dataclass(frozen=True)
class HeaderLine:
    kind: HeaderKind
    text: str
    """One line, a fact about this session: `sandbox: none — reads unmasked, …`."""
    source: str
    """The rule that requires it: `EX-2`, `SE-8`, `D118 item 5`, …"""


@dataclass(frozen=True)
class RunHeader:
    session_id: str
    lines: tuple[HeaderLine, ...]
    sandbox: str = ""
    """The level a command runs at here now (`isolation.level_of` of the configured backend):
    what the `sandbox` line words, as a value a frontend can draw. An
    approved command runs outside it whatever it is."""
    uncommitted: tuple[str, ...] = ()
    """The checkout's changed paths, untracked included (`workspace.dirty_paths`): what an
    in-place run snapshots first, and a worktree run does not take."""

    def unverified(self) -> tuple[str, ...]:
        """The lines as the handoff's `unverified` carries them, so nothing said at the start
        is missing at the end. `handoff()` prepends these to its own."""
        return tuple(line.text for line in self.lines)


# -- facts ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """What one `coder` episode's run journal says about the header's kinds."""

    run_id: str
    started: bool
    """The journal holds something past `run_start`. The bootstrap is an episode's first
    operation, so once anything else is written its absence is a fact rather than a race."""
    bootstrap: Literal["none", "pending", "succeeded", "failed"] = "none"
    """Read from the bootstrap's effect lifecycle, not its intent alone: a bootstrap that was
    refused or failed prepared nothing, and one still running has not yet."""
    execs: tuple[ExecP, ...] = ()
    suspected: tuple[SecretSuspectedP, ...] = ()


@dataclass(frozen=True)
class HeaderFacts:
    """Everything the header reads, gathered once, so `project` is a pure function of it."""

    mode: WorkspaceMode
    """The configured mode — what the first `coder` call will take. A recorded workspace
    overrides it: the session's own fact beats the config's intention."""
    level: SandboxLevel
    """The sandbox the session's commands will run under, detected now."""
    asked: str = "auto"
    """`[sandbox] backend`: what the person asked for. When it does not run here and
    another backend does instead, the sandbox line names both."""
    require_sandbox: bool = True
    """At `none`, commands refuse to run rather than run unwrapped."""
    workspace: WorkspaceRef | None = None
    removed: tuple[str, ...] = ()
    """Built-in deny-list entries the run's list removes — in force."""
    inert_removed: tuple[str, ...] = ()
    """Removals the config asks for that are not in force: the clone's owner has not
    accepted the trust digest. Named anyway, marked inert."""
    allow_secret_read: bool = False
    """The run's list is off whole — in force."""
    allow_secret_read_inert: bool = False
    """The config sets `allow_secret_read` and it is not in force, for the same reason."""
    inert_commands: tuple[tuple[str, ...], ...] = ()
    """The prefixes of the repository's `[[commands]]` rows not in force, because the clone's
    owner has not accepted them. Named, with the command that accepts them."""
    allow_unsandboxed_escapes: bool = False
    """The person let the scout's `request_exec` run where no sandbox backend exists.
    Engine config only, so it is the person's, never a clone's."""
    python_unsandboxed: bool = False
    """The person let the worker's Python blocks run where no sandbox backend
    exists (`sandbox.allow_unsandboxed_python`). Engine config only, as above."""
    lifted: tuple[ShellRole, ...] = ()
    """The roles whose shell the person let run commands unasked where no
    sandbox backend exists, in `SHELL_ROLES` order. Engine config only, as above."""
    never_ask: bool = False
    """D199 item 8b: the person's `sandbox.never_ask`: no shell command is asked."""
    episodes: tuple[Episode, ...] = field(default=())
    uncapped: tuple[str, int, float] | None = None
    """The person lifted the family's cap — the member whose log holds the `cap` line,
    that line's seq, and the family's spend. `None` while any cap is in force."""
    uncapped_rungs: tuple[tuple[str, str, int, float], ...] = ()
    """Each rung whose cap the person lifted — the rung, the member whose log holds
    the line, its seq, and the family's spend on that rung. A rung that never had a cap of
    its own is not here: nothing was lifted."""
    rungs: tuple[tuple[str, str], ...] = ()
    """Each declared rung and the catalog model it runs on now, in `[rungs]` order."""
    switched: tuple[tuple[str, str, int], ...] = ()
    """Each rung the person moved — the rung, the model its latest switch left, and
    that switch's seq."""
    uncommitted: tuple[str, ...] = ()
    """The checkout's changed paths (`RunHeader.uncommitted`)."""


def detected_level(backend: str = "auto") -> SandboxLevel:
    """The level every engine spawn site gets: the backend `[sandbox] backend` names, through
    `isolation.base.configured` as the engine chooses it, at the level it runs at
    here. The reading `Executor.run` makes before it spawns."""
    return level_of(configured(backend))


def _refused(e: ExecP) -> bool:
    """An `exec` the pre-filter denied before anything spawned (`Executor.run`: exit 126,
    no wall time, the denial as its violation). It ran under no sandbox and opened no
    network, so it says nothing about either."""
    return e.exit_code == 126 and e.wall_ms == 0 and bool(e.violations)


def _episode(run_id: str, run: RunProjection) -> Episode:
    started = any(model is not RunStartP for model in run.by_type)
    bootstrap: Literal["none", "pending", "succeeded", "failed"] = "none"
    # The last bootstrap the run intended, and how it ended.
    booted = [op for op in run.operations.values() if op.kind == EffectKind.BOOTSTRAP]
    if booted:
        outcome = booted[-1].outcome
        bootstrap = (
            "pending"
            if outcome is None
            else ("succeeded" if outcome == "succeeded" else "failed")
        )
    execs = tuple(line.payload for line in run.lines(ExecP))
    suspected = tuple(line.payload for line in run.lines(SecretSuspectedP))
    return Episode(run_id, started, bootstrap, execs, suspected)


def gather(
    roots: Roots,
    config: EngineConfig,
    session_id: str,
    *,
    level: SandboxLevel,
    denylist: Denylist,
    folds: ReplayCache,
    runs: RunProjections,
    uncapped: tuple[str, int, float] | None = None,
    uncapped_rungs: tuple[tuple[str, str, int, float], ...] = (),
    commands: CommandTable | None = None,
) -> HeaderFacts:
    """Read the session log and each episode's run journal, without a lock and without
    writing: a torn tail is skipped in memory, never truncated, because reading the header
    must not be an effect on the thing it describes — and an episode may be mid-write.

    `denylist` is the run's own list, as the engine resolved it: the header says
    what is in force by reading it, and what the config asked for beyond it is inert. It
    never asks the trust question itself, so it cannot answer it differently from the run.
    `commands` is the run's `[[commands]]` table, read the same way: a repository row it does
    not hold is inert. `None` is the shipped table, which holds none.

    `folds` and `runs` are the engine's one cache of session folds and one of run journals,
    so a header asked every turn reads only what changed since.

    `uncapped` is the engine's reading of the family's cap: the family spans other
    sessions' logs, which the engine already folds, so the header takes the answer rather
    than reading the family a second way."""
    try:
        log, state = folds.get(session_id)
    except (SessionStoreError, OSError, ValueError) as exc:
        raise SessionError(f"session {session_id} has no readable header: {exc}") from exc

    episodes: list[Episode] = []
    records = JournalRecords(roots, runs)
    for line in state.episodes():
        try:
            run = records.projection(line.run_id)
        except FileNotFoundError:  # an episode line is written at dispatch, before its journal
            continue
        except (JournalCorrupt, OSError, ValueError) as exc:
            # ValueError covers a torn tail cut inside a multi-byte character, which
            # `read_events` decodes before it can drop the partial line.
            raise SessionError(
                f"session {session_id} has no readable header: episode {line.run_id}'s "
                f"journal could not be read ({exc})"
            ) from exc
        episodes.append(_episode(line.run_id, run))

    asked = config.secrets
    switches = switches_of(state, log.payload)
    latest = {move.rung: (move.from_model, seq) for seq, move in switches}
    return HeaderFacts(
        mode=config.workspace.mode,
        level=level,
        asked=config.sandbox.backend,
        require_sandbox=config.sandbox.require_sandbox,
        workspace=state.workspace_at(state.last_seq),
        removed=denylist.removed,
        inert_removed=tuple(
            dict.fromkeys(r for r in asked.deny_remove if r not in denylist.removed)
        ),
        allow_secret_read=denylist.allow_secret_read,
        allow_secret_read_inert=asked.allow_secret_read and not denylist.allow_secret_read,
        inert_commands=_inert_commands(config, commands),
        allow_unsandboxed_escapes=config.sandbox.allow_unsandboxed_escapes,
        python_unsandboxed=config.sandbox.allow_unsandboxed_python,
        lifted=tuple(r for r in SHELL_ROLES if config.sandbox.allow_unsandboxed_shell.of(r)),
        never_ask=config.sandbox.never_ask,
        episodes=tuple(episodes),
        uncapped=uncapped,
        uncapped_rungs=uncapped_rungs,
        rungs=tuple(wired(config.rungs, (move for _, move in switches)).items()),
        switched=tuple((rung, left, seq) for rung, (left, seq) in latest.items()),
        uncommitted=_uncommitted(roots),
    )


def _uncommitted(roots: Roots) -> tuple[str, ...]:
    """The checkout's changed paths, read without writing; none when git cannot say, as a
    header line that cannot be read is said rather than refused."""
    try:
        # `project` is the checkout's `.chickadee/`; its parent is the checkout.
        return dirty_paths(roots.project.parent)
    except (InfraError, OSError):
        return ()


def _inert_commands(
    config: EngineConfig, commands: CommandTable | None
) -> tuple[tuple[str, ...], ...]:
    """The repository's rows the run's table does not hold as the repository's."""
    out: list[tuple[str, ...]] = []
    for row in config.commands:
        if not row.from_repository:
            continue
        held = commands.row(row.prefix) if commands is not None else None
        if held is None or held.source != PROJECT_LAYER:
            out.append(row.prefix)
    return tuple(dict.fromkeys(out))


# -- the projection -------------------------------------------------------------------------


def project(session_id: str, facts: HeaderFacts) -> RunHeader:
    """The header, as a pure function of `facts`. Order is the table order."""
    lines = [
        *_workspace(facts),
        *_isolation(facts),
        *_never_ask(facts),
        *_sandbox(facts),
        *_deny_list(facts),
        *_commands(facts),
        *_bootstrap(facts),
        *_secrets(facts),
        *_models(facts),
        *_spend(facts, session_id),
    ]
    return RunHeader(
        session_id, tuple(lines), sandbox=facts.level.value, uncommitted=facts.uncommitted
    )


def _models(facts: HeaderFacts) -> Sequence[HeaderLine]:
    """Always, as the table says: each rung's model, and for a rung the person moved, the
    model it left and when."""
    if not facts.rungs:
        return []
    left = {rung: (model, seq) for rung, model, seq in facts.switched}
    parts: list[str] = []
    for rung, model in facts.rungs:
        part = f"{rung} {model}"
        if rung in left:
            was, seq = left[rung]
            part += f" (switched from {was} at seq {seq})"
        parts.append(part)
    return [HeaderLine("models", "models: " + " · ".join(parts), "D128")]


def _mode(facts: HeaderFacts) -> WorkspaceMode:
    return facts.workspace.mode if facts.workspace is not None else facts.mode


def _workspace(facts: HeaderFacts) -> Sequence[HeaderLine]:
    ref = facts.workspace
    source = "01-orchestrator.md §3.5.1"
    if ref is None:
        taken = (
            "a dedicated worktree" if facts.mode == "worktree" else "your checkout, in place"
        )
        return [
            HeaderLine(
                "workspace",
                f"workspace: none taken yet — the first `coder` call takes {taken} under the "
                "repository WRITE lease",
                source,
            )
        ]
    base = f"base {ref.source_head[:12]} ({ref.source_ref})"
    if ref.mode == "in_place":
        where = "in place, in your checkout"
    elif ref.worktree_path is None:
        where = f"worktree gone after handoff ({ref.handoff}), branch {ref.branch}"
    else:
        where = f"worktree {ref.worktree_path}, branch {ref.branch}"
    return [
        HeaderLine("workspace", f"workspace: {where} · lease {ref.lease_id} · {base}", source)
    ]


def _isolation(facts: HeaderFacts) -> Sequence[HeaderLine]:
    if _mode(facts) != "in_place":
        return []
    return [
        HeaderLine(
            "isolation",
            "isolation: reduced — in-place, so writes land in your checkout and packet commits "
            "and rollbacks touch the branch you are on",
            "01-orchestrator.md §3.5.2",
        )
    ]


def _sandbox(facts: HeaderFacts) -> Sequence[HeaderLine]:
    ran = [e for ep in facts.episodes for e in ep.execs if not _refused(e)]
    # What commands actually ran under wins over what is detected now: a command that ran
    # unmasked, or at a different level, did so whatever the machine offers today.
    observed = sorted({e.sandbox for e in ran})
    if SandboxLevel.NONE.value in observed:
        return [_unwrapped(facts)]
    if not observed and facts.level is SandboxLevel.NONE:
        if facts.require_sandbox:
            return [
                HeaderLine(
                    "sandbox",
                    "sandbox: none — require_sandbox is set, so every command refuses to run "
                    "rather than run unwrapped" + _asked(facts),
                    "EX-2",
                )
            ]
        return [_unwrapped(facts)]
    levels = "/".join(observed) if observed else facts.level.value
    # Every level here is an enforcing one: `none` returned above.
    shown = SandboxLevel(observed[-1]) if observed else facts.level
    masked = (
        "deny-listed paths masked"
        if reads_masked(shown, allow_secret_read=facts.allow_secret_read)
        else "nothing masked (allow_secret_read)"
    )
    lines = [
        HeaderLine(
            "sandbox",
            f"sandbox: {levels} — {masked}, network off unless a command declares it"
            + _asked(facts),
            "EX-2",
        )
    ]
    # A command that asks for the network gets the whole network, once approved. At
    # `none` every command already does, which `_unwrapped` says.
    opened = sum(1 for e in ran if e.network)
    if opened:
        lines.append(
            HeaderLine(
                "sandbox",
                f"sandbox: {opened} command(s) ran with the network on, each approved",
                "D199, 22-secrets.md §3",
            )
        )
    return lines


def _unwrapped(facts: HeaderFacts) -> HeaderLine:
    # When the person opted in, the line says the escapes run unsandboxed; the
    # model is not told, and the person always is.
    # Only where there is no sandbox now and nothing refuses to run without one: an earlier
    # command that ran at `none` on a machine that has a sandbox today, or a config that
    # now sets `require_sandbox`, does not make the escapes run unsandboxed.
    unwrapped_now = facts.level is SandboxLevel.NONE and not facts.require_sandbox
    escapes = (
        "the scout's escapes run unsandboxed (sandbox.allow_unsandboxed_escapes)"
        if facts.allow_unsandboxed_escapes and unwrapped_now
        else "the scout's escapes are off"
    )
    blocks = (
        "; the worker's Python blocks run unsandboxed (sandbox.allow_unsandboxed_python)"
        if facts.python_unsandboxed and unwrapped_now
        else ""
    )
    return HeaderLine(
        "sandbox",
        f"sandbox: none — reads unmasked and the network open for every command; {escapes}"
        + blocks
        + _shells(facts, unwrapped_now)
        + _asked(facts),
        "EX-2, EX-16, D134, D137, D226",
    )


def _shells(facts: HeaderFacts, unwrapped_now: bool) -> str:
    """D137 decision 5, per role since item 8: which holds for each shell at `none`.
    Before anything ran, `require_sandbox` has its own line; here a command already ran
    unwrapped, so the line says what happens next. "Run none nobody confirms" because a
    command nobody can confirm is refused, so "ask" alone would overstate what runs. Where a
    sandbox wraps the next command, nothing is added."""
    if facts.level is not SandboxLevel.NONE:
        return ""
    if not unwrapped_now:
        return "; every command now refuses to run (require_sandbox)"
    if facts.never_ask:
        # `_never_ask`'s line, first in the section, says what still refuses.
        return "; every shell runs commands unprotected and unasked (sandbox.never_ask)"
    lifted: list[ShellRole] = [r for r in SHELL_ROLES if r in facts.lifted]
    asking: list[ShellRole] = [r for r in SHELL_ROLES if r not in facts.lifted]
    out = ""
    if lifted:
        keys = ", ".join(f"sandbox.allow_unsandboxed_shell.{r}" for r in lifted)
        runs = "runs" if len(lifted) == 1 else "run"
        out += f"; {_whose(lifted)} {runs} commands unprotected ({keys})"
    if asking:
        asks, runs = ("asks", "runs") if len(asking) == 1 else ("ask", "run")
        out += (
            f"; {_whose(asking)} {asks} before every command, and {runs} none nobody confirms"
        )
    return out


_SHELL_NAMES: Final[Mapping[ShellRole, str]] = {
    "zeus": "Zeus's",
    "worker": "the worker's",
    "athena": "Athena's",
}


def _whose(roles: Sequence[ShellRole]) -> str:
    """`Zeus's shell`, or `the worker's and Athena's shells`."""
    names = [_SHELL_NAMES[r] for r in roles]
    if len(names) == 1:
        return f"{names[0]} shell"
    return ", ".join(names[:-1]) + " and " + names[-1] + " shells"


def _never_ask(facts: HeaderFacts) -> Sequence[HeaderLine]:
    """D199 item 8b: said first, whatever the sandbox, because it changes what every other
    line about asking means."""
    if not facts.never_ask:
        return []
    return [
        HeaderLine(
            "sandbox",
            "sandbox: no shell command is asked (sandbox.never_ask) — what would be asked "
            "runs, the network included; deny rules and secret paths still refuse",
            "D199",
        )
    ]


def _asked(facts: HeaderFacts) -> str:
    """A backend the person asked for that does not run here falls through to another,
    and the line says so rather than naming only what runs. Nothing to add for `auto`, which
    asks for no backend in particular, or when what was asked for runs."""
    if facts.asked == "auto" or facts.asked == facts.level.value:
        return ""
    return f" (asked for {facts.asked}; {facts.level.value} runs)"


def _deny_list(facts: HeaderFacts) -> Sequence[HeaderLine]:
    # What shrinks the list is inert until the clone's owner accepts its
    # digest. An inert removal is still named, with that status, rather than left out: the
    # person who wrote it should learn why it did nothing.
    inert = " — inert until the config's trust digest is accepted, so still denied"
    lines: list[HeaderLine] = []
    if facts.removed:
        lines.append(
            HeaderLine(
                "deny_list",
                f"deny_list: built-in entries removed — {', '.join(facts.removed)}; the rest "
                "stay in force",
                "D118 item 5",
            )
        )
    if facts.inert_removed:
        lines.append(
            HeaderLine(
                "deny_list",
                f"deny_list: removal of {', '.join(facts.inert_removed)} requested{inert}",
                "D118 items 3 and 5",
            )
        )
    if facts.allow_secret_read:
        lines.append(
            HeaderLine(
                "deny_list",
                "deny_list: off — allow_secret_read is set, so credential files are neither "
                "refused nor masked",
                "SE-8",
            )
        )
    if facts.allow_secret_read_inert:
        lines.append(
            HeaderLine(
                "deny_list",
                f"deny_list: allow_secret_read requested{inert}",
                "SE-8, D118 item 3",
            )
        )
    return lines


TRUST_COMMAND = "chickadee trust"
"""The command that shows the repository's executable config and records its acceptance."""


def _commands(facts: HeaderFacts) -> Sequence[HeaderLine]:
    # A repository's rows are executable config, inert until accepted; named, so the
    # person learns why a row did nothing and how to accept it.
    if not facts.inert_commands:
        return []
    rows = ", ".join(" ".join(p) for p in facts.inert_commands)
    return [
        HeaderLine(
            "commands",
            f"commands: the repository's [[commands]] rows for {rows} are inert until this "
            f"clone accepts them — run `{TRUST_COMMAND}` to read and accept them",
            "RB-27, 31-state.md §6",
        )
    ]


def _bootstrap(facts: HeaderFacts) -> Sequence[HeaderLine]:
    if _mode(facts) == "in_place":
        return [HeaderLine("bootstrap", IN_PLACE_NO_BOOTSTRAP, "D109")]
    # One worktree per session: it was prepared if any episode's bootstrap succeeded, is
    # being prepared while one runs, and was not if one failed or none ran past its start.
    states = {ep.bootstrap for ep in facts.episodes}
    if states & {"succeeded", "pending"}:
        return []
    if "failed" in states:
        return [HeaderLine("bootstrap", BOOTSTRAP_FAILED, "D109")]
    if any(ep.started for ep in facts.episodes):
        return [HeaderLine("bootstrap", NO_BOOTSTRAP, "D109")]
    return []


def _secrets(facts: HeaderFacts) -> Sequence[HeaderLine]:
    seen: dict[tuple[str, int, str], None] = {}
    for ep in facts.episodes:
        for s in ep.suspected:
            seen.setdefault((s.path, s.line, s.detector), None)
    return [
        HeaderLine(
            "secrets",
            f"secrets: suspected committed secret at {path}:{line} ({detector}) was in an "
            "assembled context — advisory, nothing was withheld",
            "SE-11",
        )
        for path, line, detector in seen
    ]


def _spend(facts: HeaderFacts, session_id: str) -> Sequence[HeaderLine]:
    # A lifted cap is never hidden. A capped session says nothing here; its
    # caps are the config's or `cap` lines, and either stops it.
    lines: list[HeaderLine] = []
    if facts.uncapped is not None:
        member, seq, spent = facts.uncapped
        where = "" if member == session_id else f" in {member}"
        text = f"spend: uncapped (since seq {seq}{where}) — ${spent:.2f} spent"
        lines.append(HeaderLine("spend", text, "D144"))
    for rung, member, seq, spent in facts.uncapped_rungs:
        where = "" if member == session_id else f" in {member}"
        text = f"spend: {rung} uncapped (since seq {seq}{where}) — ${spent:.2f} spent"
        lines.append(HeaderLine("spend", text, "D145"))
    return lines


def header(
    roots: Roots,
    config: EngineConfig,
    session_id: str,
    *,
    denylist: Denylist,
    folds: ReplayCache,
    runs: RunProjections,
    level: SandboxLevel | None = None,
    uncapped: tuple[str, int, float] | None = None,
    uncapped_rungs: tuple[tuple[str, str, int, float], ...] = (),
    commands: CommandTable | None = None,
) -> RunHeader:
    """Gather and project. `denylist` and `commands` are the run's, resolved by the engine;
    `level` defaults to what this machine's backend gives now."""
    detected = level if level is not None else detected_level(config.sandbox.backend)
    facts = gather(
        roots,
        config,
        session_id,
        level=detected,
        denylist=denylist,
        folds=folds,
        runs=runs,
        uncapped=uncapped,
        uncapped_rungs=uncapped_rungs,
        commands=commands,
    )
    return project(session_id, facts)


def spend_facts(
    budget: SessionBudget | None, members: Mapping[str | None, str], session_id: str
) -> tuple[tuple[str, int, float] | None, tuple[tuple[str, str, int, float], ...]]:
    """The header's reading of the family's lifted caps, from `budget()`'s value
    and the member whose log holds each `cap` line in force, keyed by rung (`None` for the
    total): `uncapped` for the total, then each lifted rung. A rung the person lifted, never
    one that simply has no cap. `budget` is `None` when the session cannot be read, and the
    header then says why instead."""
    if budget is None:
        return None, ()
    uncapped: tuple[str, int, float] | None = None
    if budget.cap_usd is None and budget.cap_seq is not None:
        uncapped = (members.get(None, session_id), budget.cap_seq, budget.spent_usd)
    lifted: list[tuple[str, str, int, float]] = []
    for rung, share in budget.rungs.items():
        if share.cap_usd is None and share.cap_seq is not None:
            lifted.append((rung, members.get(rung, session_id), share.cap_seq, share.spent_usd))
    return uncapped, tuple(lifted)
