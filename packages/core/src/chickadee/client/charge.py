"""What a call that failed after it was sent is known to have cost:
the one home of that charge, for every role. Zeus's loop and the shared request loop
(`tools/loop.ask`, which runs the worker, Athena and the scout) both charge through it.

A provider may bill a request that then fell silent, got back a body that was not a
completion, or was cancelled in flight; spend the ledger never hears of is spend the cap
never counts. A connect error, a 429, a 5xx and any refused status are attempts no provider
bills, and charging them would let an outage spend the cap. Which attempts may be billed is the
client's to say (`ClientError.billed_attempts`); what they cost is this module's.
"""

from __future__ import annotations

from collections.abc import Callable

from chickadee.client.errors import ClientError
from chickadee.schema.adapter import Usage

__all__ = ["charge_unanswered", "unanswered_usage"]


def unanswered_usage(sent_tokens: int, failure: ClientError) -> Usage | None:
    """What `failure` is known to have cost, None when no attempt of it may be billed.

    - **Input:** `sent_tokens`, the caller's local estimate of the whole request, once per
      billed attempt, since each sent all of it.
    - **Output:** what each billed attempt had streamed before it stopped, which streaming
      makes known.

    One usage for all the attempts, so a failed call is one charge and one `spend` line. It is
    `estimated`: neither count is the provider's."""
    billed = failure.billed_attempts
    if not billed:
        return None
    return Usage(
        prompt_tokens=sent_tokens * len(billed),
        completion_tokens=sum(billed),
        estimated=True,
    )


def charge_unanswered(
    charge: Callable[[Usage], float], sent_tokens: Callable[[], int], failure: ClientError
) -> float:
    """`failure`'s cost charged through `charge` (a role's ledger), and what it charged: 0.0,
    and nothing charged, when no attempt may be billed. `sent_tokens` estimates the request
    only then: during an outage every call fails unbilled, and a long conversation's estimate
    would be spent on each for nothing."""
    if not failure.billed_attempts:
        return 0.0
    usage = unanswered_usage(sent_tokens(), failure)
    return 0.0 if usage is None else charge(usage)
