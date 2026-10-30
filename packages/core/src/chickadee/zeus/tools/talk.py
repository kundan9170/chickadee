"""Family D — talk: the person, as Zeus's tools reach them (contract 6).

Both adapters here sit over the session's `InteractionPort`, so the engine never touches a
terminal and a session with nobody attached resolves every ask to its default:
- `ZeusPerson` is the `user` port `ask_user` is given. Which answers are the person's own
  words, and which are a default nobody gave, is the built-in's (`ask_user`).
- `PersonAuthority` is the shell's `ShellAuthority`: a command above T0 is put to the person
  as an `ActionApproval` on its concrete argv before it spawns.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Final, Literal

from chickadee.approval.commands import shipped
from chickadee.approval.grants import Offer, offer
from chickadee.schema import sha256_hex
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import ActionApproval, Answer, InteractionPort, Question
from chickadee.shell import ShellClaim
from chickadee.shell_line import exact_words

__all__ = ["PersonAuthority", "ZeusPerson", "approval_of"]


class ZeusPerson:
    """`UserPort` over the session's `InteractionPort`."""

    def __init__(self, port: InteractionPort) -> None:
        self._port = port

    def ask(self, question: Question, *, cancel: threading.Event | None = None) -> Answer:
        return self._port.ask(question, cancel=cancel)

    def confirm(self, question: str, tier: Tier) -> bool:
        """No rule of Zeus's pipeline asks the person before a call (the shell's own tier is
        confirmed through `PersonAuthority`); with nobody to ask, a confirmation is declined,
        never assumed."""
        del question, tier
        return False


class PersonAuthority:
    """`ShellAuthority` over the session's `InteractionPort`.

    `approve` is authority for this call and nothing else. `deny` is no authority, so the
    call is `REFUSED_ABOVE_T0`. `abort` is the person stopping the turn: `stop` is the
    conversation's hard interrupt, and the shell answers the call `CANCELLED` because its
    token is now set."""

    def __init__(self, port: InteractionPort, stop: Callable[[], None]) -> None:
        self._port = port
        self._stop = stop

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        decision = self._port.confirm_action(approval_of(claim), cancel=cancel)
        if decision.verdict == "abort":
            self._stop()
        return decision.verdict == "approve"


def approval_of(claim: ShellClaim) -> ActionApproval:
    """The question a shell claim becomes. Every field is what the call is, never a claim about
    it: the shell declares no credentials, says where a command may write, and
    nothing it runs above T0 is known to be undoable — with no sandbox nothing seals
    it, and `network: true` opens the whole network. Its tier comes from the command
    table and D137's floor, not the router's rows, so `hits` is empty and `display` names the
    classification instead.

    Every command a person is asked about runs outside the sandbox once approved, so the
    question says so, and what that reaches (`effect_surface`, the CLI's `reaches`
    line): the approval is the person's decision to trust the command with what they can do."""
    declared = " · outside the sandbox"
    declared += " · network on" if claim.network else ""
    # A command that may write says where: the session's worktree, or the
    # person's own checkout before the session has one.
    if claim.where:
        declared += f" · writes {claim.where}"
    # What each *always* would cover, read as the shell's own table read the
    # command. Should the two still differ, the question offers *once* only: no
    # *always* whose breadth the prompt could not state truly.
    offered = offer(
        claim.found if claim.found is not None else shipped().classify_line(claim.command),
        exact_words(claim.command),
        network=claim.network,
    )
    if offered.grant != claim.grant:
        offered = Offer(None, None, offer_repo=False, repo_any_arguments=False)
    return ActionApproval(
        action_id=f"shell-{uuid.uuid4().hex[:12]}",
        operation_id=claim.operation_id,
        task_id=claim.task_id,
        asker=_asker(claim.asker),
        kind="command",
        display=(
            f"{claim.command}  (in {claim.cwd}{declared}; {claim.classification} at "
            f"{claim.tier.value})"
        ),
        spec_sha=sha256_hex(repr((claim.command, claim.cwd, claim.network, claim.where))),
        network=claim.network,
        idempotent=claim.idempotent,
        reversible=False,
        effect_surface=(REACHES,),
        tier=claim.tier,
        grant=offered.grant,
        exact=offered.exact,
        offer_repo=offered.offer_repo,
        repo_any_arguments=offered.repo_any_arguments,
        command=claim.command,
        cwd=claim.cwd,
        sandboxed=False,
    )


REACHES: Final = "anything you can: files outside the worktree, the network, your credentials"
"""What an approved command reaches, outside the sandbox."""


def _asker(role: str) -> Literal["zeus", "athena", "worker"]:
    """The claim's role as the question names it: a box role's own, else Zeus's."""
    if role == "athena":
        return "athena"
    if role == "worker":
        return "worker"
    return "zeus"
