"""The task commit: the commit, the reindex, the head.

Moved whole from `orchestrator/projector.py`; every caller imports it from here.
Its one change: `commit` takes its subject and no packet, so each caller builds its own, the
packet loop's driver `chickadee: <packet_id> <goal-slug> (attempt <n>)` and the desk
`chickadee: T2 <title>`.

A commit is made on the run worktree's internal branch with a `Chickadee-Operation` trailer,
and exactly the changed files are reindexed, so the next task and every navigation tool see a
fresh graph. The commit is the task's rollback unit and the "last good" reference; the original
checkout is untouched until the reviewed handoff.

**The trailer is the evidence recovery reads.** `executor/reconcile.py` names it, and
`landed` asks for it: a commit found by its trailer landed even when its `effect_completed`
never became durable, so resume records it instead of committing twice.

**In place there is no commit** (never auto-commit the person's work). The projection
there is the reindex alone; the known-good state is what the effect's completion records.

`commit_effect` is the commit as an effect, in one place for both its callers, the packet
loop's driver and the desk: the intent, the commit or (in place) the head, the completion.

Every git call goes through `chickadee.process` — the supervisor is the only spawn —
and names only the operation's own paths: `git add -A -- <changed>`, never `git add -A`, so a
file the run did not write is never swept into a commit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chickadee.errors import HarnessBug, InfraError
from chickadee.executor.reconcile import TRAILER, trailer_commit
from chickadee.graph.store import graph_failures_are_infra
from chickadee.journal.effects import EffectKind, EffectLog, Intent, perform
from chickadee.paths import os_junk
from chickadee.process import ProcessSupervisor, git, run_git
from chickadee.schema.ids import sha256_hex
from chickadee.schema.journal import CommandFactP
from chickadee.work import sections
from chickadee.workspace import SNAPSHOT_IDENTITY, snapshot_operation

__all__ = [
    "Commits",
    "Projector",
    "Reindexer",
    "commit_effect",
    "commit_message",
    "ran_line",
    "task_body",
]


class Reindexer(Protocol):
    """`graph.indexer.Indexer`'s incremental half: reparse exactly these paths."""

    def reindex(self, changed: Sequence[Path] | None = None) -> object: ...


def commit_message(message: str, operation_id: str, body: str = "") -> str:
    """The subject the caller chose, its body when it has one, and the trailer recovery
    searches for, last. Blank lines separate them, so git reads the trailer as one."""
    if "\n" in message:
        raise HarnessBug(f"a commit subject is one line, got {message!r}")
    middle = f"{body.strip()}\n\n" if body.strip() else ""
    return f"{message}\n\n{middle}{TRAILER}: {operation_id}\n"


def task_body(
    brief: str,
    reviewed_round: int,
    commands: Sequence[CommandFactP],
    *,
    growth: str | None = None,
) -> str:
    """An accepted task's commit body, written by the harness, never a model:
    `Goal` and `Done when` copied from the brief's template sections, a section the brief
    lacks left out, never invented; `Reviewed` and `Ran` from the journal's facts, so it never
    says more ran than did. A command the facts hold that never ran is not listed. `growth`,
    a one-task plan's that grew (`box/tasks.growth`, decision 19), is its own line."""
    have = sections(brief)
    lines = [
        _field(label, have[key])
        for label, key in (("Goal", "goal"), ("Done when", "done when"))
        if have.get(key)
    ]
    lines.append(f"Reviewed: accepted by Athena, round {reviewed_round}")
    ran = [ran_line(c) for c in commands if c.exit_code is not None]
    lines.append("Ran: " + ("; ".join(ran) if ran else "nothing"))
    if growth is not None:
        lines.append(f"Growth: {growth}")
    return "\n".join(lines)


def _field(label: str, text: str) -> str:
    rows = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if len(rows) == 1:
        return f"{label}: {rows[0]}"
    return "\n".join((f"{label}:", *(f"  {row}" for row in rows)))


def ran_line(command: CommandFactP) -> str:
    """One command as a record says it ran: its exit, and whether it timed out or tested."""
    said = f"exit {command.exit_code}"
    if command.timed_out:
        said += ", timed out"
    if command.test_outcome is not None:
        said += f", tests {command.test_outcome}"
    return f"`{command.command}` ({said})"


@dataclass
class Projector:
    """The commit port over a git worktree and the graph index."""

    root: Path
    indexer: Reindexer
    in_place: bool = False
    supervisor: ProcessSupervisor | None = None
    run_id: str | None = None
    """The run whose operations this projects. It keys their snapshot refs, since a session's
    runs share one branch and operation ids repeat across runs (`workspace.ops_ref`)."""

    def commit(
        self, subject: str, operation_id: str, changed: Sequence[str], body: str = ""
    ) -> str:
        """Stage exactly `changed`, less any file the operating system wrote (`os_junk`), and
        commit it under `subject`, with `body` before the trailer. Returns the new HEAD.

        `--allow-empty` so work that left the bytes as they were still has its rollback unit
        and its trailer; `--no-verify` because this is the harness's internal rollback unit
        in its own worktree, and a person's hooks were written for their commits, not for
        ours."""
        if self.in_place:
            raise HarnessBug(
                f"{operation_id}: in-place runs never commit the person's work (D39)"
            )
        if changed:
            staged = self._without_junk(changed)
            if staged:
                git(self.root, "add", "-A", "--", *staged, supervisor=self.supervisor)
        git(
            self.root,
            *SNAPSHOT_IDENTITY,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--no-verify",
            "--allow-empty",
            "-q",
            "-m",
            commit_message(subject, operation_id, body),
            supervisor=self.supervisor,
        )
        head = self.head()
        if head is None:
            raise InfraError(f"{operation_id}: committed, and HEAD still resolves to nothing")
        return head

    def _without_junk(self, changed: Sequence[str]) -> list[str]:
        """`changed` less any file the operating system wrote (`os_junk`): a folder
        it names (a design approved at a stop) may hold a Finder file, which is no one's work.
        Unchanged when none is there, so every other commit stages what it always did. With
        one, the folder is staged as its files, named. Not an `:(exclude)` pathspec: with one,
        git 2.43 left untracked files out of `add` in the session's worktree (a plan approved
        at a stop, a run's `report.md`), so the commit lost the work it was for."""
        named = [p for p in changed if not os_junk(p)]
        if not named:
            return []
        listed = git(
            self.root,
            "ls-files",
            "-z",
            "--others",
            "--modified",
            "--deleted",
            "--exclude-standard",
            "--",
            *named,
            supervisor=self.supervisor,
        )
        files = {p for p in listed.split("\0") if p}
        if not any(os_junk(p) for p in files):
            return named
        return sorted(p for p in files if not os_junk(p))

    def reindex(self, changed: Sequence[str]) -> None:
        """Exactly the changed files: a full rescan would re-derive what did not move, and
        the next task's navigation reads this index.

        The index failing itself — a locked or broken database, the disk under it — is the
        graph unavailable: `GraphUnavailable`, an `InfraError`, by the
        one translation (`graph.store.graph_failures_are_infra`). A raw `sqlite3.Error` would
        pass the run's handler by."""
        with graph_failures_are_infra(f"the graph index could not take {list(changed)}"):
            self.indexer.reindex([self.root / p for p in changed])

    def head(self) -> str | None:
        done = run_git(
            self.root, "rev-parse", "--verify", "-q", "HEAD", supervisor=self.supervisor
        )
        text = done.stdout.decode("utf-8", "replace").strip()
        return text if done.exit_code == 0 and text else None

    def landed(self, operation_id: str, *, since: str | None = None) -> str | None:
        """The commit carrying this operation's trailer, or `None` if none reached HEAD.

        Only commits after `since` — the run's starting head, `run_start.repo_head` — are
        searched. Operation ids are unique within a run, not across runs: every run's first
        task is likely `T1`, so an earlier run applied to this branch can have left a commit
        under the very same id in the history this one started from, and finding it would
        record another run's commit and never make this one's.

        The lookup is `reconcile.trailer_commit`, the one `by_operation_trailer` reads, so
        what counts as the trailer is decided once: the operation's own line, as git splits
        a message, never a prefix of a longer id or words inside a sentence. A git that cannot
        answer raises `InfraError`, as does a `since` the history was rewritten under:
        "unknown" is not "did not land", and only a clean "did not land" permits a second
        commit (`reconcile.Landed.UNKNOWN`). A caller's bug raises `HarnessBug`: an operation
        id that is not one token, or a `since` that is empty or starts with `-`. Resume
        passes `repo_head or None`, never `""`."""
        return trailer_commit(self.root, operation_id, since=since, supervisor=self.supervisor)

    def snapshot(self, operation_id: str) -> str:
        """The tree parked under the operation's ref before a declared effect starts."""
        return snapshot_operation(self.root, operation_id, run_id=self.run_id)


class Commits(Protocol):
    """What `commit_effect` asks of a projector: `Projector`, or the driver's port."""

    def head(self) -> str | None: ...

    def commit(
        self, subject: str, operation_id: str, changed: Sequence[str], body: str = ""
    ) -> str: ...


def commit_effect(
    projector: Commits,
    log: EffectLog,
    operation_id: str,
    subject: str,
    changed: Sequence[str],
    *,
    kind: EffectKind,
    in_place: bool,
    landed: str | None = None,
    body: str = "",
) -> str:
    """One commit as one effect: its intent (the head before as pre-image), then the commit,
    or in place the head as it is, then its completion. `landed` is a commit already
    found by this work's trailer under an earlier lifecycle that could not say so: it is
    recorded, and nothing is committed twice. Returns the commit, or the head in place ("" when
    there is none). A git that raises leaves the effect `ambiguous`: only the trailer can say
    whether the commit was made."""
    head = projector.head()
    surface = tuple(changed)
    intent = Intent(
        operation_id,
        kind,
        approved_sha=sha256_hex(subject + "\n" + "\n".join(surface)),
        preconditions=(f"head={head or 'unborn'}",),
        idempotent=False,
        surface=surface,
        preimages=head,
    )
    with perform(log, intent) as fx:
        if landed is not None:
            sha = landed
        elif in_place:
            sha = head or ""
        else:
            sha = projector.commit(subject, operation_id, surface, body)
        fx.complete("succeeded", observed=surface, post_state=sha or None)
    return sha
