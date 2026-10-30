"""Zeus's `worktree` port: the person
asks, Zeus moves the work, and the harness does every git step.

- **enter.** The session goes into the worktree it holds (`held_worktree`), or a new one made
  the one way a `coder` call makes it (`delegation.session_worktree`), and Zeus's view follows.
  Asked of the person first unless their words named a worktree: a model
  holding the tool never moves the session on its own judgment.
- **leave.** A `workspace` line naming the checkout: the view comes back, the next `coder`
  call runs in place, and the worktree stays held, for `enter`, the handoff and a later
  session (`open_works`).
- **land and discard.** The handoff, the one `finalize` takes, from inside the turn
  (`engine._land_in_turn`), and only after the person approves it at dispatch. `land` brings
  by default: the work written into the checkout uncommitted. A discard
  keeps a snapshot of the worktree under `refs/chickadee/discarded/` first, so it can be
  recovered.

A worktree made from a checkout with uncommitted changes starts with them
(`workspace._seed_commit`), and a landing writes only what came after (`handoff.seed_of`).
"""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from chickadee.context.view import held_worktree, latest_workspace, still_written
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.edits.draft import DraftRefused
from chickadee.errors import ConfigError, EngineBusy, HandoffError, RepoLeaseBusy
from chickadee.ids import operation_component
from chickadee.process import git
from chickadee.schema.authorization import Tier
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import ActionApproval, InteractionPort
from chickadee.schema.isolation import GitHandoffChoice, WorkspaceMode
from chickadee.tools.tool import Said
from chickadee.workspace import ops_ref, snapshot_operation
from chickadee.zeus.delegate import asked_workspace, request_of

if TYPE_CHECKING:
    from chickadee.context.session import Session
    from chickadee.handoff import HandoffResult
    from chickadee.handoff_summary import HandoffSummary

__all__ = ["DISCARDED_REF_PREFIX", "WorktreeMoves", "named_worktree"]

DISCARDED_REF_PREFIX: Final = "refs/chickadee/discarded"
"""Where a discarded worktree's last tree is kept: `<prefix>/<session>/<operation>`, outside
the operation refs a handoff may retire, so a discard can always be taken back by hand."""

_ASK_DO: Final = "tell the user what the reason says, and ask them how to go on"


_CLAUSE: Final = re.compile(r"[.,;:!?\n]|\bbut\b")
_NEGATION: Final = re.compile(r"\b(?:no|not|never|without|dont|do not)\b|n't\b")


def named_worktree(text: str) -> bool:
    """Whether the person's words ask for a worktree: a word match, never a model's reading,
    with a mention in a clause that holds a negation ("don't use a worktree", "a worktree
    isn't needed") not counted. A miss only costs the one question asked otherwise, so a
    clause that might refuse one never counts."""
    clauses = _CLAUSE.split(text.casefold().replace("\u2019", "'"))
    return any("worktree" in c and not _NEGATION.search(c) for c in clauses)


class WorktreeMoves:
    """The `worktree` port over one session. `open_worktree` makes and records the session's
    worktree and gives its lease back; `follow` moves Zeus's view to where the session's
    latest `workspace` line says, answering a note when it could not; `land` is the handoff
    from inside the turn; `review` is the session's handoff summary, whose words for each
    landing (`HandoffSummary.offers`) the person's confirmation shows."""

    def __init__(
        self,
        session: Session,
        interaction: InteractionPort,
        *,
        open_worktree: Callable[[], None],
        follow: Callable[[], str | None],
        land: Callable[[GitHandoffChoice], HandoffResult],
        review: Callable[[], HandoffSummary],
    ) -> None:
        self._session = session
        self._interaction = interaction
        self._open = open_worktree
        self._follow = follow
        self._land = land
        self._review = review

    # -- the four moves --------------------------------------------------------------------

    def enter(
        self, *, operation_id: str, cancel: threading.Event | None
    ) -> Said | DraftRefused:
        with self._session.lock:
            state = self._session.state
            here = still_written(latest_workspace(state))
            held = held_worktree(state)
            asked = request_of(state)
            # The person's `--worktree`: neither unset nor in place.
            flagged = asked_workspace(state) not in (None, "in_place")
        if here is not None:
            return Said(f"already in the worktree {here.branch} at {here.worktree_path}")
        if not (flagged or named_worktree(asked)):
            display = (
                "work in a separate worktree: a branch of its own for this conversation, "
                "your checkout left as it is"
            )
            if not self._approved(display, Tier.T1, operation_id, cancel, reversible=True):
                return Said(
                    "not entered: the user said no, so the work stays in their checkout"
                )
        if held is not None:
            # Back into the worktree it left: a line naming it again, so the view and the next
            # `coder` call follow it.
            with self._session.lock:
                self._session.record_workspace(held)
        else:
            try:
                self._open()
            except RepoLeaseBusy as busy:
                return DraftRefused("DENIED", str(busy), do="call worktree again once it ends")
            except ConfigError as refused:
                return DraftRefused("DENIED", str(refused), do=_ASK_DO)
        return Said(self._entered() + self._moved())

    def follow_flag(self, flag: WorkspaceMode) -> str | None:
        """The person's `--in-place` as their turn opens: a
        session in its worktree leaves it, as `leave` does, and what was said is Zeus's notice.
        `--worktree` creates nothing early: the next work goes to the worktree
        (`context/view.session_mode`). None when nothing moved."""
        if flag != "in_place":
            return None
        with self._session.lock:
            if still_written(latest_workspace(self._session.state)) is None:
                return None
        return f"the user started this turn with --in-place: {self.leave().text}"

    def leave(self) -> Said:
        with self._session.lock:
            here = still_written(latest_workspace(self._session.state))
            if here is None:
                return Said("already in the user's checkout")
            self._session.record_workspace(
                WorkspaceRef(
                    mode="in_place",
                    lease_id=here.lease_id,
                    source_ref=here.source_ref,
                    source_head=here.source_head,
                    branch=None,
                    worktree_path=None,
                    handoff="pending",
                )
            )
        return Said(
            f"back in the user's checkout; the worktree {here.branch} is kept at "
            f"{here.worktree_path}: worktree enter returns to it, worktree land brings its "
            "work here" + self._moved()
        )

    def land(
        self, how: str, *, operation_id: str, cancel: threading.Event | None
    ) -> Said | DraftRefused:
        held = self._held()
        if held is None:
            return DraftRefused(
                "DENIED",
                "this conversation holds no worktree to land",
                do="the work is already in the user's checkout; nothing to land",
            )
        # `how` is one of the spec's three, `bring` by default.
        choice = GitHandoffChoice(how)
        review = self._review()
        refused = review.refused.get(choice.value)
        offer = review.offers.get(choice.value)
        if refused is not None or offer is None:
            # Refused before the person is asked: no one approves a landing that then fails.
            # No offer and no refusal: the worktree could not be described, its notes say why.
            why = refused or "; ".join(review.notes) or "the worktree cannot be described"
            return DraftRefused("DENIED", why, do=_ASK_DO)
        display = f"land the worktree {held.branch} ({choice.value}): {offer}"
        if not self._approved(display, Tier.T2, operation_id, cancel, reversible=False):
            return Said(f"not landed: the user said no; the worktree {held.branch} is kept")
        done = self._handed_off(choice)
        if isinstance(done, DraftRefused):
            return done
        if choice is GitHandoffChoice.BRING:
            return Said(
                f"landed {held.branch} into the checkout. {done.detail}" + self._moved()
            )
        landed = len(done.commits_applied)
        return Said(
            f"landed {held.branch} into {done.source_branch or held.source_ref}: {landed} "
            f"commit(s). {done.detail}" + self._moved()
        )

    def discard(
        self, *, operation_id: str, cancel: threading.Event | None
    ) -> Said | DraftRefused:
        held = self._held()
        if held is None:
            return DraftRefused(
                "DENIED", "this conversation holds no worktree to discard", do=_ASK_DO
            )
        display = (
            f"discard the worktree {held.branch} and its work (a snapshot is kept under "
            f"{DISCARDED_REF_PREFIX}/)"
        )
        if not self._approved(display, Tier.T2, operation_id, cancel, reversible=False):
            return Said(f"not discarded: the user said no; the worktree {held.branch} is kept")
        kept = self._keep_snapshot(held, operation_id)
        done = self._handed_off(GitHandoffChoice.DISCARD)
        if isinstance(done, DraftRefused):
            return done
        where = f"; its last state is at {kept}" if kept else ""
        return Said(f"discarded {held.branch}{where}. {done.detail}" + self._moved())

    # -- the parts -------------------------------------------------------------------------

    def _held(self) -> WorkspaceRef | None:
        with self._session.lock:
            return held_worktree(self._session.state)

    def _entered(self) -> str:
        held = self._held()
        if held is None:
            return "entered a worktree"
        return f"entered the worktree {held.branch} at {held.worktree_path}"

    def _moved(self) -> str:
        note = self._follow()
        return f" ({note})" if note else ""

    def _handed_off(self, choice: GitHandoffChoice) -> HandoffResult | DraftRefused:
        try:
            return self._land(choice)
        except (HandoffError, EngineBusy, RepoLeaseBusy) as refused:
            return DraftRefused("DENIED", str(refused), do=_ASK_DO)

    def _keep_snapshot(self, held: WorkspaceRef, operation_id: str) -> str | None:
        """The worktree as it is, its uncommitted files too, kept under a ref no handoff or
        retention retires: the operation's own snapshot commit (`snapshot_operation`), named
        again there, so `git switch -c <name> <ref>` brings it back. None when its directory
        is gone, since its branch then holds all there is."""
        if held.worktree_path is None or not Path(held.worktree_path).is_dir():
            return None
        root = Path(held.worktree_path)
        snapshot_operation(root, operation_id)
        commit = git(root, "rev-parse", ops_ref(root, operation_id))
        name = operation_component(operation_id)
        ref = f"{DISCARDED_REF_PREFIX}/{self._session.session_id}/{name}"
        git(root, "update-ref", ref, commit)
        return ref

    def _approved(
        self,
        display: str,
        tier: Tier,
        operation_id: str,
        cancel: threading.Event | None,
        *,
        reversible: bool,
    ) -> bool:
        approval = ActionApproval(
            action_id=f"worktree-{uuid.uuid4().hex[:12]}",
            operation_id=operation_id,
            kind="git",
            display=display,
            spec_sha=sha256_hex(display),
            network=False,
            idempotent=False,
            reversible=reversible,
            tier=tier,
        )
        decision = self._interaction.confirm_action(approval, cancel=cancel)
        verdict: Literal["approve", "deny", "abort"] = decision.verdict
        return verdict == "approve"
