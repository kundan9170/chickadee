"""Reviewed Git handoff: the five completion choices.

Primitives only — the session's half, which worktree and the record of the choice, is
`session_handoff.py`. What lives here is the part
that has to be right whether or not a human is watching: establishing that the source
branch is still where the run left it, and then carrying out exactly one of `apply`,
`squash`, `keep`, `discard` or `bring` without ever pushing.

**Refused by overlap, never by state** (Amendment 2 decision 3). A landing is
refused when a path the run changed also changed in the checkout since the run started
(`HandoffPreview.overlapping`), never because the checkout has some uncommitted change: git
writes only the run's paths, so the person's other edits are never in its way.

Three properties, each of which a simpler implementation loses:

- **Nothing is mutated until everything is checked.** `HandoffError` promises that the
  source branch and the worktree are untouched when it is raised, so every precondition is
  evaluated up front rather than as the operation goes. A handoff that half-applied and
  then failed is worse than one that refused.
- **The commits this module creates can be identified afterwards.**
  `executor/reconcile.py` has read `Chickadee-Operation` trailers since T-2.3 and nothing
  in the repository has ever written one, so `by_operation_trailer` answered `NO` — "it did
  not land", the one answer that permits a retry — for every operation ever performed. The
  squash commit carries the trailer. For the four choices that create no commit,
  `reconcile()` answers the same question by observing refs, which is what
  `HARNESS_DESIGN` §9 asks for: recovery observes state, it does not assume it.
- **It never pushes, structurally.** Not as a convention anyone has to remember: the argv
  this module builds is its entire effect on the repository, and `test_handoff.py` drives
  all four choices through a recording supervisor and asserts that none of it is a push.

**The choices are defined over the worktree branch, so in-place gets fewer of them.** D55
names what each choice does to "the worktree branch", and `--in-place` has none: the run's
changes are already in the checkout the user is standing on, uncommitted, because an in-place
run never commits the person's work (`box/commit.py`). `apply` and `keep` are
therefore honest no-ops there, and `squash` and `discard` are refused — both would mean
rewriting the user's own work, and `01-orchestrator` §3.5.2 already says in-place recovery
"uses those durable preimages, not `git reset --hard`". The refusal names the snapshot ref, so
the user has the same power without the harness taking it on their behalf.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chickadee.durable.files import write_new
from chickadee.errors import HandoffError, HarnessBug, InfraError
from chickadee.executor.reconcile import TRAILER, Landed, Reconciliation, by_operation_trailer
from chickadee.process import (
    DIFF_FLAGS,
    GIT_TIMEOUT_S,
    Completed,
    ProcessSupervisor,
    git,
    run_git,
)
from chickadee.schema.isolation import GitHandoffChoice, IsolationSpec, WorkspaceMode
from chickadee.work import marked
from chickadee.work_records import INDEX, REPORT, indexed, missing_rows, with_rows
from chickadee.workspace import (
    EMPTY_TREE,
    SEED_TRAILER,
    SNAPSHOT_IDENTITY,
    SNAPSHOT_REF_PREFIX,
)


@dataclass(frozen=True)
class HandoffPreview:
    """Everything `01-orchestrator` §3.5.1 requires the handoff to identify, plus whether
    the source has moved underneath the run.

    Read-only and non-raising on drift: the review surface has to *show* the user that the
    branch moved, which it cannot do if learning that fact is an exception.
    """

    mode: WorkspaceMode
    lease_id: str
    """Who held the repository WRITE lease."""
    branch: str | None
    """The run's internal branch. `None` in-place, which has no branch of its own."""
    worktree_path: str | None
    base: str
    """The commit the run started from. The empty tree for an in-place run that started
    before the repository had any commits — `workspace.snapshot` names it that."""
    tip: str | None
    """The run's final commit, or `None` when there is not one yet: an in-place run that
    began on an unborn HEAD and has committed nothing has no tip, and saying so is better
    than inventing one."""
    commits: tuple[str, ...]
    """`base..tip`, oldest first — the final commit range."""
    source_branch: str | None
    """`None` when the source checkout is on a detached HEAD."""
    source_head: str | None
    source_clean: bool
    drift: str | None
    """Why the source has moved out from under the run, in one sentence, or `None`. This
    is one input to `refusal`, not the whole answer — ask `refusal` instead."""
    uncommitted: tuple[str, ...] = ()
    """What the worktree holds that no commit does, as `git status --porcelain` lists it: a
    changed tracked file or an untracked one, ignored files aside. No choice lands it; `discard`
    deletes it with the worktree, and the review lists it first so the person knows."""
    overlapping: tuple[str, ...] = ()
    """The paths the run changed that also differ in the source checkout from where the run
    started (`start`): a commit of the person's since, an uncommitted change, or an untracked
    or ignored file there. A landing would write over them, so it is refused while there are
    any."""
    seed: str | None = None
    """The worktree's first commit when it holds the checkout's uncommitted changes
    (`seed_of`); None when it started from `base` as it was."""

    @property
    def start(self) -> str:
        """Where the run's own work starts: after the seed, or `base`. What `bring` writes and
        what an overlap is read against, so the person's own changes, still in their
        checkout, are never written twice."""
        return self.seed or self.base

    @property
    def uncommitted_paths(self) -> tuple[str, ...]:
        """`uncommitted` as the paths it names: porcelain's two status columns and the space
        after them dropped, the name kept whole."""
        return tuple(line[3:] for line in self.uncommitted)

    def refusal(self, choice: GitHandoffChoice) -> str | None:
        """Why `finalize` would refuse this choice, in one sentence, or `None`.

        **The single source of truth, because `finalize` asks this too.** A review surface
        that has to re-derive the rule will get it wrong the first time the rule changes,
        and the way it gets it wrong is by offering the user a choice that then fails. So
        the property and the enforcement are the same code, and a choice that is not
        refused here is one `finalize` will carry out.

        There is deliberately no single `ready` boolean: acceptance is per choice and per
        mode — an in-place run accepts `apply` and refuses `squash` — so any one flag would
        have to be wrong about something.
        """
        if self.mode == "in_place":
            if choice is GitHandoffChoice.BRING:
                return (
                    "an in-place run's changes are already in the checkout, so there is "
                    "nothing to bring: keep says the work is done as it is"
                )
            # `apply` and `keep` in-place move nothing at all, so the state of the source
            # checkout cannot be a reason to refuse them. That matters rather than being a
            # nicety: in-place is specified to snapshot pre-existing changes instead of
            # committing them, so a dirty tree is the *normal* state of such a run, and a
            # shared precondition would refuse the common case.
            if choice in (GitHandoffChoice.SQUASH, GitHandoffChoice.DISCARD):
                return (
                    f"{choice.value} is defined over the run's worktree branch (D55) and an "
                    "in-place run has none — its commits are on "
                    f"{self.source_branch or 'the current checkout'}, the branch you are "
                    "standing on. Rewriting it is not the harness's to do; the pre-run "
                    f"state is under {SNAPSHOT_REF_PREFIX}/ and {self._added_range()}."
                )
            return None
        if choice is GitHandoffChoice.DISCARD:
            if self.branch is not None and self.branch == self.source_branch:
                return (
                    f"the run's branch {self.branch!r} is the branch the source checkout is "
                    "on, so discarding it would delete the user's own branch"
                )
            return None
        if choice is GitHandoffChoice.KEEP:
            return None
        if self.overlapping:
            return self._overlap_refusal()
        if choice is GitHandoffChoice.BRING:
            # It moves no branch, so neither a moved branch nor a detached HEAD is in its way.
            return None
        if self.seed is not None:
            return (
                "the worktree started from the checkout's uncommitted changes, so "
                f"{choice.value} would commit them onto your branch for you. {_BRING_INSTEAD}, "
                "beside your own changes"
            )
        if self.drift is not None:
            return f"{self.drift}. {_BRING_INSTEAD}"
        if self.source_branch is None:
            return (
                "the source checkout is on a detached HEAD, so there is no source branch "
                f"to {choice.value} onto. {_BRING_INSTEAD}"
            )
        return None

    def says(self, choice: GitHandoffChoice) -> str:
        """What `choice` does, in one line said before the person picks it: to their
        branch, the worktree, and the work no commit holds. The one wording, which every
        review (`HandoffSummary.offers`) and Zeus's confirmation of a landing show."""
        onto = self.source_branch or "the checkout"
        n = len(self.commits)
        left = "; the uncommitted work stays in the worktree" if self.uncommitted else ""
        lost = " and the uncommitted work listed above" if self.uncommitted else ""
        return {
            GitHandoffChoice.BRING: (
                f"write the {n} commit(s)' changes into your checkout, uncommitted, for you "
                f"to review and commit; {onto} does not move{left}"
            ),
            GitHandoffChoice.APPLY: (
                f"fast-forward {onto} onto the {n} commit(s), each kept{left}"
            ),
            GitHandoffChoice.SQUASH: f"put the {n} commit(s) on {onto} as one commit{left}",
            GitHandoffChoice.KEEP: (
                "change nothing; the next coder call reattaches this worktree"
            ),
            GitHandoffChoice.DISCARD: (
                f"delete the worktree and branch {self.branch}: the {n} commit(s){lost} are "
                f"lost, {onto} is untouched"
            ),
        }[choice]

    def _overlap_refusal(self) -> str:
        shown = ", ".join(self.overlapping[:_OVERLAP_SHOWN])
        more = len(self.overlapping) - _OVERLAP_SHOWN
        listed = shown + (f" and {more} more" if more > 0 else "")
        return (
            f"{len(self.overlapping)} file(s) the run changed were also changed in the "
            f"checkout since it started: {listed}. Landing would write over them. Commit "
            f"those changes, then `git merge {self.branch}` brings the work in, with any "
            "conflict for the coder to resolve; or keep the worktree as it is"
        )

    def _added_range(self) -> str:
        """How to see what this run added, as a command that actually runs.

        A refusal exists to leave the user able to act, so the command it hands them has to
        work in the state that produced the refusal. `EMPTY_TREE` is a tree, not a commit,
        so `git log 4b825dc642cb..HEAD` is rejected by git — and that is exactly the state
        an in-place run beginning on an unborn HEAD is in.
        """
        if self.base != EMPTY_TREE:
            return f"`git log {self.base[:12]}..HEAD` is the range this run added"
        if self.tip is None:
            return "this run added no commits — the repository still has none"
        return "`git log HEAD` is the range this run added, which is the whole history"

    @property
    def accepted(self) -> tuple[GitHandoffChoice, ...]:
        """The choices `finalize` would carry out right now — what a review surface offers."""
        return tuple(c for c in GitHandoffChoice if self.refusal(c) is None)


_OVERLAP_SHOWN = 10

_BRING_INSTEAD = "bring writes the same changes into the checkout uncommitted, moving no branch"
"""The exit a refused `apply` or `squash` names: one the harness owns."""


@dataclass(frozen=True)
class HandoffResult:
    """What a completed handoff did. `pushed` is a field rather than a docstring claim
    because a host that has to trust prose is a host that cannot assert on it."""

    choice: GitHandoffChoice
    operation_id: str
    source_branch: str | None
    source_head: str | None
    """The source branch's commit *after* the handoff. `None` only where there was never
    one: an in-place run that began on an unborn HEAD and committed nothing."""
    commits_applied: tuple[str, ...]
    """Commits this handoff put on the source branch, oldest first. Empty for `keep`,
    `discard`, `bring`, and an `apply` that had nothing to replay."""
    branch: str | None
    """The run's branch, if it still exists afterwards."""
    worktree_path: str | None
    """The worktree, if it still exists afterwards."""
    detail: str
    pushed: Literal[False] = False


# --- git --------------------------------------------------------------------


def _run(sup: ProcessSupervisor, root: Path, *args: str) -> Completed:
    """Every git call in this module goes through `run_git`.

    Harness-owned work: the supervisor, never `Executor.run`. Isolation wrap is not
    applied; a reviewed handoff needs the real git, not a sandboxed echo of it.

    **A timeout is never an answer.** Raised here, once, rather than left to each caller:
    the exit code of a killed process is indistinguishable from a real non-zero exit, so
    `_rev` would have read a hung git as "that revision does not exist" and `_ok` would
    have read it as "no, not an ancestor" — both of them confident, both of them wrong,
    and one of them the input to a drift check that decides whether to mutate a branch.
    `InfraError` rather than `HandoffError` for the same reason: git failing to finish
    says nothing about the state of the repository.
    """
    done = run_git(root, *args, supervisor=sup)
    if done.timed_out:
        raise InfraError(f"git {args[0]} timed out after {GIT_TIMEOUT_S}s in {root}")
    return done


def _text(done: Completed) -> str:
    return done.stdout.decode("utf-8", "replace").strip()


def _git(sup: ProcessSupervisor, root: Path, *args: str, hooks: bool = False) -> str:
    return git(root, *args, supervisor=sup, on_error=HandoffError, hooks=hooks)


def _fed(sup: ProcessSupervisor, root: Path, data: bytes, *args: str) -> Completed:
    """`_run` with `data` on git's stdin: a timeout is never an answer, for the same reason."""
    done = run_git(root, *args, supervisor=sup, stdin=data)
    if done.timed_out:
        raise InfraError(f"git {args[0]} timed out after {GIT_TIMEOUT_S}s in {root}")
    return done


def _ok(sup: ProcessSupervisor, root: Path, *args: str) -> bool:
    """For questions where a non-zero exit *is* the answer, not a failure."""
    return _run(sup, root, *args).exit_code == 0


def _rev(sup: ProcessSupervisor, root: Path, rev: str) -> str | None:
    done = _run(sup, root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    return _text(done) or None if done.exit_code == 0 else None


def _range(
    sup: ProcessSupervisor, root: Path, base: str | None, tip: str | None
) -> tuple[str, ...]:
    """`base..tip`, oldest first. `base=None` means the run started from no commit at all,
    so its range is everything reachable from the tip."""
    if tip is None or base == tip:
        return ()
    out = _git(sup, root, "log", "--format=%H", tip if base is None else f"{base}..{tip}")
    return tuple(reversed([line for line in out.splitlines() if line]))


# --- reading the world ------------------------------------------------------

DIFF_CAP = 200_000
"""Characters of `diff_of`'s text: a review shows what changed, and a diff past this is cut at
a line, with a line saying how to see the rest, rather than shipped whole over the wire."""


def diff_of(
    source_root: Path,
    base: str,
    tip: str | None,
    *,
    supervisor: ProcessSupervisor | None = None,
) -> str:
    """`base..tip` as `git diff` prints it, for the person to review before they choose: empty
    when there is no tip or nothing changed. Cut at `DIFF_CAP`, at a line.

    No renames, whatever the user's `diff.renames` says: a moved file is its deletion and its
    creation, as `workspace.diff_stats` counts it, so the text and the counts beside it name
    the same paths."""
    if tip is None or tip == base:
        return ""
    sup = supervisor or ProcessSupervisor()
    text = _git(sup, source_root, "diff", *DIFF_FLAGS, "--no-renames", base, tip)
    if len(text) <= DIFF_CAP:
        return text
    cut = text.rfind("\n", 0, DIFF_CAP)
    return (
        f"{text[: cut if cut > 0 else DIFF_CAP]}\n"
        f"… cut at {DIFF_CAP} characters; `git diff {base[:12]} {tip[:12]}` shows it whole"
    )


def landed_paths(
    source_root: Path,
    before: str | None,
    after: str | None,
    *,
    supervisor: ProcessSupervisor | None = None,
) -> tuple[str, ...]:
    """The paths that differ between the source branch's commit `before` a handoff and the one
    `after` it: what a choice that landed put on the person's branch, a rename as both its
    names. Every path `after` holds when there was no commit before; none when the branch did
    not move."""
    if after is None or before == after:
        return ()
    sup = supervisor or ProcessSupervisor()
    if before is None:
        return _names(_git(sup, source_root, "ls-tree", "-r", "-z", "--name-only", after))
    return _names(
        _git(sup, source_root, "diff", "--name-only", "-z", "--no-renames", before, after)
    )


def _names(listed: str) -> tuple[str, ...]:
    """A `-z` listing of git's as its paths, in its order."""
    return tuple(p for p in listed.split("\0") if p)


def describe(
    source_root: Path,
    spec: IsolationSpec,
    *,
    base: str,
    supervisor: ProcessSupervisor | None = None,
) -> HandoffPreview:
    """The run's final state and whether the source is still writable.

    `base` is the commit the run started from — `RunStartP.repo_head` for a worktree run,
    `IsolationSpec.pre_image_sha` in-place. It is a parameter rather than something derived
    from a merge-base because a merge-base cannot tell "the source never moved" from "the
    source moved and then someone merged it back": both leave the same graph, and only one
    of them is safe to fast-forward.

    An in-place run may legitimately have started before the repository had a first commit:
    `workspace.snapshot` supports that and names `EMPTY_TREE` as the pre-image, so `acquire`
    produces such a spec. Every completion choice comes through here, so refusing that shape
    would mean such a run could never be handed off at all — including `keep`, which does
    nothing.
    """
    sup = supervisor or ProcessSupervisor()
    source_head = _rev(sup, source_root, "HEAD")
    if source_head is None and spec.mode == "worktree":
        raise HandoffError(
            f"{source_root} has no HEAD commit, so there is no source branch to hand back to"
        )

    branch_out = _run(sup, source_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    source_branch = _text(branch_out) if branch_out.exit_code == 0 else None
    source_clean = not _git(sup, source_root, "status", "--porcelain")

    # `EMPTY_TREE` is a tree, not a commit, so it cannot be one end of a revision range.
    # Compared against the named constant rather than probed with `rev-parse`, because
    # "does not resolve to a commit" would also swallow a caller passing a wrong sha.
    from_nothing = base == EMPTY_TREE
    if spec.mode == "worktree":
        tip = _rev(sup, source_root, spec.branch or "")
        if tip is None:
            raise HandoffError(
                f"the run's branch {spec.branch!r} no longer exists in {source_root} — "
                "it was deleted or the run was already handed off"
            )
    else:
        tip = source_head
    uncommitted: tuple[str, ...] = ()
    seed = (
        seed_of(source_root, base, tip, supervisor=sup)
        if spec.mode == "worktree" and tip is not None and not from_nothing
        else None
    )
    if spec.mode == "worktree" and spec.worktree_path and Path(spec.worktree_path).is_dir():
        # Not `_git`, which strips: porcelain's first column may be a space (` M path`).
        listed = _run(sup, Path(spec.worktree_path), "status", "--porcelain")
        if listed.exit_code != 0:
            raise HandoffError(f"git status failed in {spec.worktree_path}: {_text(listed)}")
        out = listed.stdout.decode("utf-8", "replace")
        uncommitted = tuple(line for line in out.splitlines() if line)
    overlapping = (
        _overlapping(sup, source_root, seed or base, tip)
        if spec.mode == "worktree" and tip is not None and not from_nothing
        else ()
    )

    return HandoffPreview(
        mode=spec.mode,
        lease_id=spec.lease_id,
        branch=spec.branch,
        worktree_path=spec.worktree_path,
        base=base,
        tip=tip,
        commits=_range(sup, source_root, None if from_nothing else base, tip),
        source_branch=source_branch,
        source_head=source_head,
        source_clean=source_clean,
        drift=_drift(
            sup,
            source_root,
            spec,
            base=None if from_nothing else base,
            head=source_head,
        ),
        uncommitted=uncommitted,
        overlapping=overlapping,
        seed=seed,
    )


def _drift(
    sup: ProcessSupervisor,
    source_root: Path,
    spec: IsolationSpec,
    *,
    base: str | None,
    head: str | None,
) -> str | None:
    """Whether the source has moved out from under the run, in the user's terms.

    Stated as sentences rather than an enum on purpose: every one of these ends up printed
    to a person who has to decide what to do next, and "DRIFT_HEAD_MOVED" tells them less
    than the two commits does.

    `base=None` means the run started from no commit, so there is nothing it could have
    drifted away from — every history descends from nothing.

    An uncommitted change in the checkout is not drift: what a landing could
    write over is `HandoffPreview.overlapping`, and only that refuses.
    """
    if base is None or head is None:
        return None
    if spec.mode == "worktree":
        if head != base:
            return (
                f"the source branch moved during the run: it was at {base[:12]} when the "
                f"run started and is at {head[:12]} now, so the run's commits no longer "
                "replay onto it as a fast-forward"
            )
        return None
    # In-place: the run committed onto the source branch itself, so the branch is *meant*
    # to have moved. What must still hold is that it moved forward from the base.
    if head != base and not _ok(sup, source_root, "merge-base", "--is-ancestor", base, head):
        return (
            f"the source branch no longer descends from {base[:12]}, where this in-place "
            "run started, so its commits are not the ones sitting on the branch now"
        )
    return None


def seed_of(
    source_root: Path,
    base: str,
    tip: str,
    *,
    supervisor: ProcessSupervisor | None = None,
) -> str | None:
    """The worktree's seed, read by observation: the first commit
    of `base..tip` when its one parent is `base` and it carries `workspace.SEED_TRAILER`, which
    `workspace._seed_commit` writes; else None."""
    if base == tip:
        return None
    sup = supervisor or ProcessSupervisor()
    first = _git(sup, source_root, "rev-list", "--reverse", "--topo-order", f"{base}..{tip}")
    commit = next(iter(first.split()), None)
    if commit is None:
        return None
    fmt = f"%P%x00%(trailers:key={SEED_TRAILER},valueonly,separator=%x2C)"
    parents, _, trailer = _git(
        sup, source_root, "log", "-1", f"--format={fmt}", commit
    ).partition("\0")
    return commit if parents.split() == [base] and trailer.strip() else None


def _literal(paths: Sequence[str]) -> tuple[str, ...]:
    """Pathspecs that mean exactly these paths, whatever characters they hold."""
    return tuple(f":(literal){p}" for p in paths)


def _overlapping(
    sup: ProcessSupervisor, source_root: Path, start: str, tip: str
) -> tuple[str, ...]:
    """The paths `start..tip` changes that also differ in the source checkout from `start`:
    committed since, changed and uncommitted, or present there untracked or ignored where
    `start` had nothing. Exactly what a landing would write over."""
    paths = landed_paths(source_root, start, tip, supervisor=sup)
    if not paths:
        return ()
    specs = _literal(paths)
    # The working tree against `start`: a commit since and an uncommitted change alike.
    moved = _git(
        sup, source_root, "diff", "--name-only", "-z", "--no-renames", start, "--", *specs
    )
    # Untracked or ignored files `git diff` does not see, where the run writes one.
    loose = _git(sup, source_root, "ls-files", "-z", "--others", "--", *specs)
    found = {*_names(moved), *_names(loose)}
    return tuple(p for p in paths if p in found)


# --- the five choices -------------------------------------------------------


def finalize(
    source_root: Path,
    spec: IsolationSpec,
    choice: GitHandoffChoice,
    *,
    operation_id: str,
    base: str,
    supervisor: ProcessSupervisor | None = None,
) -> HandoffResult:
    """Carry out one reviewed completion choice. Never pushes, under any choice."""
    sup = supervisor or ProcessSupervisor()
    preview = describe(source_root, spec, base=base, supervisor=sup)

    # One question, asked in one place. `HandoffPreview.refusal` is what a review surface
    # renders from, so enforcing anything it does not know about would let the two disagree
    # — and the way they disagree is that the user is offered a choice that then fails.
    reason = preview.refusal(choice)
    if reason is not None:
        raise HandoffError(f"handoff refused, nothing was changed: {reason}")

    if spec.mode == "in_place":
        return _in_place(preview, choice, operation_id=operation_id)

    tip = preview.tip
    if tip is None:  # pragma: no cover - `describe` refuses a worktree run without a tip
        raise HarnessBug(f"worktree run {spec.branch!r} reached finalize with no tip commit")
    if choice is GitHandoffChoice.APPLY:
        return _apply(sup, source_root, preview, tip, operation_id=operation_id)
    if choice is GitHandoffChoice.SQUASH:
        return _squash(sup, source_root, preview, tip, operation_id=operation_id)
    if choice is GitHandoffChoice.KEEP:
        return _keep(preview, operation_id=operation_id)
    if choice is GitHandoffChoice.BRING:
        return _bring(sup, source_root, preview, tip, operation_id=operation_id)
    return _discard(sup, source_root, preview, operation_id=operation_id)


def _apply(
    sup: ProcessSupervisor,
    source_root: Path,
    preview: HandoffPreview,
    tip: str,
    *,
    operation_id: str,
) -> HandoffResult:
    """Fast-forward the source branch onto the run's tip. The packet commits keep their
    own hashes, which is the strongest reading of D55's "history intact" — a cherry-pick
    would preserve the shape and lose the identity, and each packet is a rollback unit
    someone may already have referred to by sha.

    No new commit is created, so there is no trailer to write; `reconcile()` answers for
    this choice by ancestry instead.
    """
    if not preview.commits:
        return HandoffResult(
            choice=GitHandoffChoice.APPLY,
            operation_id=operation_id,
            source_branch=preview.source_branch,
            source_head=preview.source_head,
            commits_applied=(),
            branch=preview.branch,
            worktree_path=preview.worktree_path,
            detail="the run produced no commits; the source branch is unchanged",
        )
    # Onto the person's branch, for them: their hooks run, as on their own merge.
    _git(sup, source_root, "merge", "--ff-only", tip, hooks=True)
    return HandoffResult(
        choice=GitHandoffChoice.APPLY,
        operation_id=operation_id,
        source_branch=preview.source_branch,
        source_head=tip,
        commits_applied=preview.commits,
        branch=preview.branch,
        worktree_path=preview.worktree_path,
        detail=(
            f"fast-forwarded {preview.source_branch} to {tip[:12]}; "
            f"{len(preview.commits)} packet commit(s) kept. The worktree at "
            f"{preview.worktree_path} is still there — remove it when you are done with it."
        ),
    )


def _squash(
    sup: ProcessSupervisor,
    source_root: Path,
    preview: HandoffPreview,
    tip: str,
    *,
    operation_id: str,
) -> HandoffResult:
    """Collapse the run into one commit carrying the operation trailer.

    Built with plumbing — `commit-tree` against the run tip's tree, then a fast-forward
    onto the new commit — rather than `git merge --squash && git commit`. The porcelain
    pair stages the whole merge into the source index first, so a failure between the two
    leaves the user's checkout mid-squash, which is exactly the half-applied state
    `HandoffError` promises never to produce. `commit-tree` touches neither the index nor
    the working tree, so until the final fast-forward there is nothing to unwind.
    """
    if not preview.commits:
        return HandoffResult(
            choice=GitHandoffChoice.SQUASH,
            operation_id=operation_id,
            source_branch=preview.source_branch,
            source_head=preview.source_head,
            commits_applied=(),
            branch=preview.branch,
            worktree_path=preview.worktree_path,
            detail="the run produced no commits; there was nothing to squash",
        )
    tree = _git(sup, source_root, "rev-parse", f"{tip}^{{tree}}")
    message = squash_message(
        preview.branch or "the run", _logged(sup, source_root, preview.base, tip), operation_id
    )
    commit = _git(
        sup,
        source_root,
        # The identity is the harness's, for the same reason `workspace.snapshot` gives:
        # taking it from ambient config makes the commit depend on whether `user.name`
        # happens to be set, and a machine without one fails with "Author identity
        # unknown". Imported rather than restated so the two cannot drift apart.
        *SNAPSHOT_IDENTITY,
        "commit-tree",
        tree,
        "-p",
        preview.base,
        "-m",
        message,
    )
    _git(sup, source_root, "merge", "--ff-only", commit, hooks=True)
    return HandoffResult(
        choice=GitHandoffChoice.SQUASH,
        operation_id=operation_id,
        source_branch=preview.source_branch,
        source_head=commit,
        commits_applied=(commit,),
        branch=preview.branch,
        worktree_path=preview.worktree_path,
        detail=(
            f"squashed {len(preview.commits)} packet commit(s) into {commit[:12]} on "
            f"{preview.source_branch}. The individual commits are no longer separate "
            f"rollback units; the branch {preview.branch} still holds them."
        ),
    )


def _bring(
    sup: ProcessSupervisor,
    source_root: Path,
    preview: HandoffPreview,
    tip: str,
    *,
    operation_id: str,
) -> HandoffResult:
    """Write `start..tip` into the source checkout's working tree, uncommitted (`start` is after
    the seed, when there is one): no branch moves, nothing is staged,
    nothing is committed, as Codex's Handoff and Cursor's Apply leave the work.

    `git apply` is all or nothing, and is checked first with `--check`, so a refusal there
    leaves every file as it was. `refusal` has already refused an overlap, so on the paths it
    writes the checkout holds exactly the base's content and the patch applies as written.
    """
    paths = landed_paths(source_root, preview.start, tip, supervisor=sup)
    if not preview.commits or not paths:
        return HandoffResult(
            choice=GitHandoffChoice.BRING,
            operation_id=operation_id,
            source_branch=preview.source_branch,
            source_head=preview.source_head,
            commits_applied=(),
            branch=preview.branch,
            worktree_path=preview.worktree_path,
            detail="the run changed nothing; there was nothing to bring",
        )
    patch = _run(
        sup,
        source_root,
        "diff",
        "--binary",
        *DIFF_FLAGS,
        "--no-renames",
        preview.start,
        tip,
    )
    if patch.exit_code != 0:
        raise HandoffError(f"git diff failed in {source_root}: {_text(patch)}")
    for check in (("--check",), ()):
        done = _fed(sup, source_root, patch.stdout, "apply", *check)
        if done.exit_code != 0:
            why = done.stderr.decode("utf-8", "replace").strip() or f"exit {done.exit_code}"
            prefix = "handoff refused, nothing was changed: " if check else ""
            raise HandoffError(f"{prefix}git apply failed in {source_root}: {why}")
    return HandoffResult(
        choice=GitHandoffChoice.BRING,
        operation_id=operation_id,
        source_branch=preview.source_branch,
        source_head=preview.source_head,
        commits_applied=(),
        branch=preview.branch,
        worktree_path=preview.worktree_path,
        detail=(
            f"brought {len(paths)} file(s) from {preview.branch} into the checkout, "
            "uncommitted: review them there, then commit or drop them. The branch "
            f"{preview.branch} still holds the work, and the worktree at "
            f"{preview.worktree_path} is still there."
        ),
    )


def record_works(
    root: Path, work_dir: str, *, supervisor: ProcessSupervisor | None = None
) -> tuple[str, ...]:
    """Append the index's row for each work landed on the person's branch it lacks (B3), as one
    harness commit naming only the index: at a handoff that lands, after the
    landing; at a session's start, the heal. A work is landed when its `report.md` is
    committed on the branch; its row names the commit that added it. Nothing on a detached
    head, nor while the person has their own change to the index, nor when the index is the
    person's own file, unmarked: it is never written over. Returns the
    works it recorded."""
    sup = supervisor or ProcessSupervisor()
    index = f"{work_dir}/{INDEX}"
    if not _ok(sup, root, "symbolic-ref", "-q", "HEAD"):
        return ()
    if _git(sup, root, "status", "--porcelain", "--", index):
        return ()
    at = root / index
    if at.exists() and not marked(at):
        return ()
    before = at.read_text(encoding="utf-8") if at.is_file() else None

    def landed(name: str) -> str | None:
        # The commit that added it, never the latest to touch it: a later edit of the report
        # (the person's typo fix) must not move the row, which is never rewritten.
        report = f"{work_dir}/{name}/{REPORT}"
        return _git(sup, root, "log", "-1", "--diff-filter=A", "--format=%H", "--", report)

    rows = missing_rows(root / work_dir, before, landed)
    if not rows:
        return ()
    after = with_rows(before, rows)
    names = tuple(sorted(indexed(after) - indexed(before or "")))
    subject = (
        f"chickadee: record {names[0]}"
        if len(names) == 1
        else f"chickadee: record {len(names)} works"
    )
    _commit_file(sup, root, index, after, subject)
    return names


def _commit_file(
    sup: ProcessSupervisor, root: Path, path: str, text: str, subject: str
) -> None:
    """`path` set to `text` in one commit on HEAD, then fast-forwarded onto, as `_squash` does:
    the blob, tree and commit are written through a throwaway index, so the person's own index
    and working tree are untouched until the fast-forward, which changes only `path`."""
    with tempfile.TemporaryDirectory() as tmp:
        body = Path(tmp) / "blob"
        write_new(body, text.encode("utf-8"))
        blob = _git(sup, root, "hash-object", "-w", str(body))
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}

        def plumb(*args: str) -> str:
            return git(root, *args, supervisor=sup, extra_env=env, on_error=HandoffError)

        plumb("read-tree", "HEAD")
        plumb("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        tree = plumb("write-tree")
    commit = _git(
        sup, root, *SNAPSHOT_IDENTITY, "commit-tree", tree, "-p", "HEAD", "-m", subject
    )
    _git(sup, root, "merge", "--ff-only", "-q", commit, hooks=True)


_KEPT: tuple[str, ...] = ("Goal:", "Ran:")
"""The lines of a task's commit body a squash keeps: its goal and what ran."""


def squash_message(
    branch: str, logged: Sequence[tuple[str, str]], operation_id: str | None
) -> str:
    """The squash commit's message: every commit of the run, oldest first, by
    its subject, with its `Goal` and `Ran` lines where its body has them (a task's, decision
    2), so no task is reduced to a count; then the trailer. `logged` is `(subject, body)`.
    `operation_id` None is the message offered in place for the person's own commit, which
    carries no trailer of ours."""
    rows: list[str] = []
    for subject, body in logged:
        rows.append(f"- {subject}")
        keeping = False
        for line in body.splitlines():
            if line.startswith(_KEPT):
                keeping = True
            elif not line.startswith("  "):
                keeping = False
            if keeping:
                rows.append(f"  {line}")
    count = f"{len(logged)} commit{'s' if len(logged) != 1 else ''}"
    listed = "\n".join(rows)
    trailer = f"\n\n{TRAILER}: {operation_id}" if operation_id is not None else ""
    return f"chickadee: {count} from {branch}\n\n{listed}{trailer}\n"


def _logged(
    sup: ProcessSupervisor, root: Path, base: str, tip: str
) -> tuple[tuple[str, str], ...]:
    """Each commit of `base..tip`, oldest first, as `(subject, body)`."""
    out = _git(sup, root, "log", "--reverse", "--format=%s%x1f%b%x1e", f"{base}..{tip}")
    found: list[tuple[str, str]] = []
    for record in out.split("\x1e"):
        if not record.strip():
            continue
        subject, _, body = record.strip("\n").partition("\x1f")
        found.append((subject, body))
    return tuple(found)


def _keep(preview: HandoffPreview, *, operation_id: str) -> HandoffResult:
    """Change nothing and say where the work is. The one choice with no git call at all."""
    return HandoffResult(
        choice=GitHandoffChoice.KEEP,
        operation_id=operation_id,
        source_branch=preview.source_branch,
        source_head=preview.source_head,
        commits_applied=(),
        branch=preview.branch,
        worktree_path=preview.worktree_path,
        detail=(
            f"kept {preview.branch} at {(preview.tip or 'nothing')[:12]} with "
            f"{len(preview.commits)} commit(s); the source branch is untouched"
        ),
    )


def _discard(
    sup: ProcessSupervisor,
    source_root: Path,
    preview: HandoffPreview,
    *,
    operation_id: str,
) -> HandoffResult:
    """Remove the worktree and delete the run's branch. Never touches the source.

    The work no commit holds goes with the worktree: the person chose `discard` after
    the review listed it. `--force` is passed only when this preview, read under the lease just
    before, lists such work, so it deletes exactly what was listed; with none listed, git's own
    refusal of a dirty worktree still stands for a file written in between.
    """
    if preview.worktree_path is not None and Path(preview.worktree_path).exists():
        force = ("--force",) if preview.uncommitted else ()
        _git(sup, source_root, "worktree", "remove", *force, preview.worktree_path)
    # `prune` after `remove` rather than instead of it: `remove` refuses a dirty worktree
    # (which is the point), while `prune` only clears administrative entries whose
    # directory is already gone, so it can never be the thing that destroys the work.
    _git(sup, source_root, "worktree", "prune")
    if preview.branch is not None:
        _git(sup, source_root, "branch", "-D", preview.branch)
    return HandoffResult(
        choice=GitHandoffChoice.DISCARD,
        operation_id=operation_id,
        source_branch=preview.source_branch,
        source_head=preview.source_head,
        commits_applied=(),
        branch=None,
        worktree_path=None,
        detail=(
            f"discarded {preview.branch} and its worktree; "
            f"{len(preview.commits)} commit(s) are now unreferenced"
            + (
                f"; deleted work no commit held: {', '.join(preview.uncommitted_paths)}"
                if preview.uncommitted
                else ""
            )
        ),
    )


def _in_place(
    preview: HandoffPreview,
    choice: GitHandoffChoice,
    *,
    operation_id: str,
) -> HandoffResult:
    """In-place has no worktree branch, so two of the four choices have nothing to mean.

    Only `apply` and `keep` reach here — `squash` and `discard` are refused by
    `HandoffPreview.refusal`, which is where every refusal lives so a review surface and
    `finalize` cannot disagree about one. Both of these are already true: the run committed
    onto the source branch as it went, so reporting them as done is honest, and neither
    runs a single git command. That is why the source-checkout preconditions do not apply
    to them — in-place is specified to snapshot pre-existing changes rather than commit
    them, so a dirty tree is the ordinary state of such a run.
    """
    verb = "applied as it ran" if choice is GitHandoffChoice.APPLY else "kept"
    return HandoffResult(
        choice=choice,
        operation_id=operation_id,
        source_branch=preview.source_branch,
        source_head=preview.source_head,
        commits_applied=(),
        branch=None,
        worktree_path=None,
        detail=(
            f"in-place run: {len(preview.commits)} commit(s) were {verb} directly on "
            f"{preview.source_branch}; nothing was moved"
        ),
    )


# --- reconciliation ---------------------------------------------------------


def landed(
    source_root: Path,
    start: str,
    tip: str,
    *,
    supervisor: ProcessSupervisor | None = None,
) -> GitHandoffChoice | None:
    """Has a worktree run's work from `start` to `tip` reached the source branch, and how?
    Answered by observation, as `reconcile` answers, since nothing records a handoff a person
    made: `apply` if the tip is reachable from the source `HEAD`; `squash` if a commit since
    `start` carries the operation trailer and the tip's tree, which is what `_squash` writes;
    `bring` if every path the run changed holds the tip's content in the checkout, which is what
    `_bring` leaves (`_brought`); else None, the run's branch still the harness's alone. An undo
    reads it to know whether the code it takes back lives in the worktree or in the person's
    checkout."""
    sup = supervisor or ProcessSupervisor()
    # From the seed, when there is one: the person's own changes are in their checkout already,
    # so they say nothing of whether the run's reached it.
    start = seed_of(source_root, start, tip, supervisor=sup) or start
    if start == tip:
        return None  # nothing was committed, so nothing could have been handed off
    if _ok(sup, source_root, "merge-base", "--is-ancestor", tip, "HEAD"):
        return GitHandoffChoice.APPLY
    tree = _git(sup, source_root, "rev-parse", f"{tip}^{{tree}}")
    fmt = f"--format=%T %(trailers:key={TRAILER},valueonly,separator=%x2C)"
    for line in _git(sup, source_root, "log", fmt, "HEAD", "--not", start).splitlines():
        seen, _, trailer = line.partition(" ")
        if seen == tree and trailer.strip():
            return GitHandoffChoice.SQUASH
    if _brought(sup, source_root, start, tip)[0] is Landed.YES:
        return GitHandoffChoice.BRING
    return None


def _brought(
    sup: ProcessSupervisor, source_root: Path, start: str, tip: str
) -> tuple[Landed, str]:
    """Whether `start..tip` sits in the checkout's working tree, by content: `YES` when every
    path the run changed holds the tip's blob there (or is absent where the tip deleted it),
    `NO` when every one holds `start`'s, `UNKNOWN` otherwise, since a mix cannot tell a
    cut-short `bring` from the person's own edits since."""
    paths = landed_paths(source_root, start, tip, supervisor=sup)
    if not paths:
        return Landed.UNKNOWN, f"{start[:12]}..{tip[:12]} changes no path, so nothing shows it"
    at_tip = _blobs(sup, source_root, tip, paths)
    at_start = _blobs(sup, source_root, start, paths)
    present = [p for p in paths if (source_root / p).is_file()]
    here: dict[str, str | None] = dict.fromkeys(paths)
    if present:
        hashed = _fed(
            sup, source_root, "\n".join(present).encode(), "hash-object", "--stdin-paths"
        )
        if hashed.exit_code != 0:
            raise HandoffError(f"git hash-object failed in {source_root}: {_text(hashed)}")
        here.update(zip(present, _text(hashed).split(), strict=True))
    if all(here[p] == at_tip.get(p) for p in paths):
        if not at_tip and not _blobs(sup, source_root, "HEAD", paths):
            # Every path deleted, and none ever on the person's branch: a checkout that never
            # had them looks the same whether a bring ran or not.
            return Landed.UNKNOWN, "the run only deleted paths the checkout never had"
        return (
            Landed.YES,
            f"the checkout holds {tip[:12]}'s content for all {len(paths)} path(s)",
        )
    if all(here[p] == at_start.get(p) for p in paths):
        return Landed.NO, f"the checkout still holds {start[:12]}'s content for every path"
    return Landed.UNKNOWN, "the checkout holds some of the run's paths at its content, not all"


def _blobs(
    sup: ProcessSupervisor, root: Path, commit: str, paths: Sequence[str]
) -> dict[str, str]:
    """Each of `paths` in `commit` as its blob sha; a path `commit` lacks is absent."""
    out = _git(sup, root, "ls-tree", "-z", commit, "--", *_literal(paths))
    found: dict[str, str] = {}
    for entry in _names(out):
        meta, _, name = entry.partition("\t")
        found[name] = meta.split()[2]
    return found


def reconcile(
    source_root: Path,
    spec: IsolationSpec,
    choice: GitHandoffChoice,
    *,
    operation_id: str,
    tip: str | None = None,
    base: str | None = None,
    supervisor: ProcessSupervisor | None = None,
) -> Reconciliation:
    """Did this handoff land? Answered by observation, per `HARNESS_DESIGN` §9.

    A missing `effect_completed` is never by itself permission to retry, so each choice is
    reconciled against the state it would have left:

    - `squash` wrote a commit carrying `Chickadee-Operation`, so the existing trailer
      reader answers it;
    - `apply` created no commit, so the evidence is ancestry — the run's tip reachable
      from the source head;
    - `keep` and `discard` are answered by whether the branch is still there;
    - `bring` moved nothing in git, so the evidence is the checkout's content on the paths
      `base..tip` changes (`_brought`); with no `base` or tip it cannot be observed.

    `NO` — the one verdict that permits a retry — is returned only when the observation
    positively shows the operation did not happen.
    """
    sup = supervisor or ProcessSupervisor()
    # In-place first, and `squash` is why. An in-place `squash` is refused by design, so it
    # never writes a trailer — and the trailer reader's answer to a commit that was never
    # written is `NO`, the one verdict that permits a retry. Recovery would loop on an
    # operation that cannot succeed. Nothing in-place moves, so nothing in-place is
    # observable, and `UNKNOWN` is the honest verdict for all four.
    if spec.mode == "in_place":
        return Reconciliation(
            Landed.UNKNOWN,
            "an in-place handoff moves nothing, so there is no state change to observe",
        )
    if choice is GitHandoffChoice.SQUASH:
        return by_operation_trailer(source_root, operation_id)

    if choice is GitHandoffChoice.BRING:
        if base is None or tip is None:
            return Reconciliation(
                Landed.UNKNOWN, "no base and tip were supplied, so the bring cannot be observed"
            )
        start = seed_of(source_root, base, tip, supervisor=sup) or base
        landed_, evidence = _brought(sup, source_root, start, tip)
        return Reconciliation(landed_, evidence)

    branch = spec.branch or ""
    present = _rev(sup, source_root, branch)

    if choice is GitHandoffChoice.DISCARD:
        if present is None:
            return Reconciliation(Landed.YES, f"the run's branch {branch} is gone")
        return Reconciliation(Landed.NO, f"the run's branch {branch} is still present")

    if choice is GitHandoffChoice.KEEP:
        if present is None:
            # Not `NO`: something removed the branch, and "keep did not happen" is not the
            # same statement as "the branch is missing". Retrying a keep on a branch that
            # no longer exists cannot restore it.
            return Reconciliation(Landed.UNKNOWN, f"the run's branch {branch} is missing")
        return Reconciliation(Landed.YES, f"the run's branch {branch} is at {present[:12]}")

    target = tip or present
    if target is None:
        return Reconciliation(
            Landed.UNKNOWN,
            f"the run's branch {branch} is gone and no tip was supplied, so whether it "
            "was applied cannot be observed",
        )
    if _ok(sup, source_root, "merge-base", "--is-ancestor", target, "HEAD"):
        return Reconciliation(Landed.YES, f"{target[:12]} is an ancestor of the source HEAD")
    return Reconciliation(Landed.NO, f"{target[:12]} is not reachable from the source HEAD")
