"""Reconciliation before retry.

**A missing `effect_completed` is never by itself permission to retry.** The process may
have died after the effect landed and before the event was durable, so "no completion
record" and "the effect did not happen" are different statements — and `alembic revision`
run twice yields two migrations.

So recovery *observes* rather than assumes: the worktree, HEAD and its operation trailers,
and whatever external evidence the action declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from chickadee.errors import HarnessBug, InfraError
from chickadee.process import Completed, ProcessSupervisor, run_git

TRAILER = "Chickadee-Operation"


class Landed(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"
    """Observation could not decide. Treated as `YES` for retry purposes, because retrying
    an effect that already landed is the failure mode this exists to prevent."""


@dataclass(frozen=True)
class Reconciliation:
    landed: Landed
    evidence: str

    @property
    def may_retry(self) -> bool:
        """Only a positive "it did not land" permits a retry. `UNKNOWN` does not."""
        return self.landed is Landed.NO


def trailer_commit(
    repo: Path,
    operation_id: str,
    *,
    since: str | None = None,
    until: str | None = None,
    supervisor: ProcessSupervisor | None = None,
) -> str | None:
    """The sha of the commit whose message carries this operation's trailer line, or `None`
    when none of the commits searched does. `InfraError` when git cannot say.

    A commit answers only when a line of its message, as git splits it, is exactly the
    trailer. A substring finds `commit:p1:1:5` inside `commit:p1:1:51`, and the same words
    inside a sentence. Either is a false "landed", which forbids the retry of an effect that
    never happened.

    `since` is the revision the caller knows preceded the operation: a run's starting head,
    or the source head before a squash. Only the commits after it are searched, because
    operation ids are unique within a run, not across runs, and an earlier run's commit under
    the same id is in the history a later run starts from. `since` must still be an ancestor
    of HEAD. If the history under it was rewritten, "after it" would take in the rewritten
    copies of the commits before it, the earlier run's among them.

    `until` ends the search there instead of at HEAD: where the run's commits end when later
    runs committed on the same branch, since their ids repeat this run's.

    Without `since`, all of HEAD's history is searched. A cap on how far back to look would
    turn a commit just past it into "did not land", the one answer that permits a retry.
    Git does the searching, and hands back only the commits whose message holds the
    trailer's words, so a long history costs git's walk and never a copy of every message.
    """
    if operation_id.split() != [operation_id] or "\x00" in operation_id:
        # Matched against stripped lines, an id with spaces at its ends never matches: "did
        # not land", the one verdict that permits a retry. A NUL is an argv the spawn
        # refuses: the id is git's filter.
        raise HarnessBug(f"an operation id is one token, not {operation_id!r}")
    for name, rev in (("since", since), ("until", until)):
        if rev is not None and (not rev or rev.startswith("-") or "\x00" in rev):
            # Not a revision: an empty range, an option git obeys, or an argv the spawn refuses.
            raise HarnessBug(f"{name} must name a revision, not {rev!r}")
    tip = until or "HEAD"
    if since is None:
        revisions: tuple[str, ...] = (tip,)
    else:
        _ancestor_of_head(repo, operation_id, since, supervisor)
        # Two arguments, never `f"{since}..HEAD"`: a `since` ending in `.` would make it
        # `X...HEAD`, which also takes in commits HEAD cannot reach.
        revisions = (tip, f"^{since}")
    want = f"{TRAILER}: {operation_id}"
    # `--`: a file named `HEAD` in the worktree makes a bare `HEAD` ambiguous to git, and
    # `--no-show-signature`: `log.showSignature` would print verification lines before the sha.
    # `--grep` keeps only the commits whose message holds the trailer's words anywhere, which
    # every commit that could match does, so the line check below still decides alone.
    # `--fixed-strings`: a person's `grep.patternType` would read the id as a pattern, and
    # one that does not match itself filters out the very commit searched for.
    out = run_git(
        repo,
        "log",
        "--no-show-signature",
        "--fixed-strings",
        f"--grep={want}",
        "--format=%H%n%B%x00",
        *revisions,
        "--",
        supervisor=supervisor,
    )
    if out.timed_out or out.exit_code != 0:
        raise InfraError(f"cannot tell whether {operation_id} landed: git log {_failed(out)}")
    for entry in out.stdout.decode("utf-8", "replace").split("\x00"):
        sha, _, message = entry.lstrip("\n").partition("\n")
        # `\n` alone, as git splits a message: `splitlines` also breaks at U+2028 and its
        # kin, which would make the words inside a sentence a line of their own.
        if any(line.strip(" \t\r") == want for line in message.split("\n")):
            return sha
    return None


def operations_after(
    repo: Path, since: str, *, supervisor: ProcessSupervisor | None = None
) -> tuple[str, ...]:
    """The operation each commit above `since` carries in its trailer, newest first, and a
    commit's own sha where it carries none: what sits on top of `since` on the branch, named."""
    if not since or since.startswith("-") or "\x00" in since:
        raise HarnessBug(f"since must name a revision, not {since!r}")
    fmt = f"--format=%H %(trailers:key={TRAILER},valueonly,separator=%x2C)"
    out = run_git(
        repo,
        "log",
        "--no-show-signature",
        fmt,
        "HEAD",
        f"^{since}",
        "--",
        supervisor=supervisor,
    )
    if out.timed_out or out.exit_code != 0:
        raise InfraError(f"cannot list the commits above {since}: git log {_failed(out)}")
    named: list[str] = []
    for line in out.stdout.decode("utf-8", "replace").splitlines():
        sha, _, trailer = line.partition(" ")
        named.append(trailer.strip() or sha[:8])
    return tuple(named)


def _ancestor_of_head(
    repo: Path, operation_id: str, since: str, supervisor: ProcessSupervisor | None
) -> None:
    done = run_git(repo, "merge-base", "--is-ancestor", since, "HEAD", supervisor=supervisor)
    if done.exit_code == 1 and not done.timed_out:
        raise InfraError(
            f"cannot tell whether {operation_id} landed: {since} is no longer an ancestor of "
            "HEAD, so the history it preceded was rewritten"
        )
    if done.timed_out or done.exit_code != 0:
        raise InfraError(
            f"cannot tell whether {operation_id} landed: git merge-base {_failed(done)}"
        )


def _failed(done: Completed) -> str:
    if done.timed_out:
        return "timed out"
    return done.stderr.decode("utf-8", "replace").strip() or f"exited {done.exit_code}"


def by_operation_trailer(
    repo: Path, operation_id: str, *, since: str | None = None
) -> Reconciliation:
    """Did a commit for this operation reach HEAD? (`trailer_commit`, as a verdict.)

    The trailer is written by the same commit that records the effect, so its presence is
    evidence the work landed even when the completion event never became durable.
    """
    try:
        sha = trailer_commit(repo, operation_id, since=since)
    except (OSError, InfraError) as exc:
        # `UNKNOWN`, never `NO`. Git being unable to say tells nothing about whether the
        # effect landed, and `NO` is the one answer that permits a retry.
        return Reconciliation(Landed.UNKNOWN, str(exc))
    if sha is not None:
        return Reconciliation(
            Landed.YES, f"{TRAILER}: {operation_id} is a trailer of commit {sha}"
        )
    searched = f"the commits after {since}" if since is not None else "HEAD's history"
    return Reconciliation(Landed.NO, f"no {TRAILER} line for {operation_id} in {searched}")


def by_file_evidence(repo: Path, expected: tuple[str, ...]) -> Reconciliation:
    """Did the declared artifacts appear? Absence of *all* of them is the only clear no."""
    if not expected:
        return Reconciliation(Landed.UNKNOWN, "the action declared no file evidence")
    present = [p for p in expected if (repo / p).exists()]
    if len(present) == len(expected):
        return Reconciliation(Landed.YES, f"all declared artifacts present: {present}")
    if not present:
        return Reconciliation(Landed.NO, "none of the declared artifacts exist")
    return Reconciliation(
        Landed.UNKNOWN, f"partial: {present} present, {set(expected) - set(present)} missing"
    )


def may_retry(idempotent: bool, reconciliation: Reconciliation) -> tuple[bool, str]:
    """EX-5, stated as one function so no caller has to remember the rule.

    Non-idempotent actions are never auto-retried under any taxonomy — not because a retry
    would probably be wrong, but because deciding it is *safe* requires knowledge the
    harness does not have.
    """
    if not idempotent:
        return False, "the action is not idempotent and is never auto-retried (EX-5)"
    if not reconciliation.may_retry:
        return (
            False,
            "reconciliation could not prove the effect did not land: "
            f"{reconciliation.evidence}",
        )
    return True, reconciliation.evidence
