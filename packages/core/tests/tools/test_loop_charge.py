"""What a failed call costs a role on the shared loop (`tools/loop.py`).

The worker, Athena and the scout ask through `ask`. Before D211 a call that failed after it
was sent charged them nothing, so the cap never counted a worker's timed-out attempts; now it
is charged through the one charge of a failed call (`client/charge.py`), as Zeus's is.
"""

from __future__ import annotations

from chickadee.client.errors import ClientError, RequestCancelled, TransportExhausted
from chickadee.client.request import ChatRequest, Message
from chickadee.context.request import request_tokens
from chickadee.schema import RetryParams
from chickadee.schema.adapter import Completion, Usage
from chickadee.tools.loop import AskResult, Messages, ask

from .tools_helpers import ScriptedRole


class _Failing:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        raise self.exc

    def close(self) -> None:
        pass


def _conversation() -> Messages:
    return Messages(
        [Message(role="user", content="q" * 350)], RetryParams(temperature=0.0, max_tokens=100)
    )


def _asked(exc: ClientError) -> tuple[AskResult[str], list[Usage], int]:
    """`ask` over a client that fails with `exc`: what it came to, what the failed-call charge
    was given, and the request's own estimate."""
    charged: list[Usage] = []
    client = _Failing(exc)
    asked = ask(
        client,
        _conversation(),
        ScriptedRole(),
        reasks=0,
        charge=lambda _u: 99.0,
        charge_failed=lambda u: charged.append(u) or 0.5,
    )
    return asked, charged, request_tokens(client.requests[0])


def test_moved_a_call_that_timed_out_charges_its_input_and_streamed_output() -> None:
    """Was: nothing charged. Now its input once per billed attempt and what each streamed,
    marked `estimated`, through `charge_failed`, never the answered calls' `charge`."""
    asked, charged, sent = _asked(
        TransportExhausted("no token within 60s", 2, billed_attempts=(0, 30))
    )
    assert asked.end == "failed"
    assert charged == [Usage(prompt_tokens=2 * sent, completion_tokens=30, estimated=True)]
    assert sent >= 100, "the request's whole estimate, its 350 characters included"
    assert asked.cost_usd == 0.5


def test_a_cancelled_attempt_is_charged_its_input_and_what_it_streamed() -> None:
    asked, charged, sent = _asked(RequestCancelled(1, billed_attempts=(12,)))
    assert asked.end == "cancelled"
    assert charged == [Usage(prompt_tokens=sent, completion_tokens=12, estimated=True)]


def test_a_failure_no_provider_bills_charges_nothing() -> None:
    for exc in (
        TransportExhausted("connect error: refused", 4),
        RequestCancelled(0),
    ):
        asked, charged, _ = _asked(exc)
        assert charged == []
        assert asked.cost_usd == 0.0


def test_without_a_failed_call_charge_nothing_is_charged() -> None:
    asked = ask(
        _Failing(TransportExhausted("deadline exceeded", 1, billed_attempts=(0,))),
        _conversation(),
        ScriptedRole(),
        reasks=0,
    )
    assert asked.end == "failed"
    assert asked.cost_usd == 0.0


def test_a_charge_the_session_log_refused_keeps_the_calls_own_ending() -> None:
    """Was: the `OSError` of a full disk left `ask` from inside its handler, the client's error
    lost and the role's thread ended. Now the role ends on the client's error, as Zeus's turn
    does; the ledger had counted the charge before it failed to write the line."""
    failure = TransportExhausted("no token within 60s", 1, billed_attempts=(5,))

    def full_disk(_usage: Usage) -> float:
        raise OSError(28, "No space left on device")

    asked = ask(
        _Failing(failure), _conversation(), ScriptedRole(), reasks=0, charge_failed=full_disk
    )
    assert asked.end == "failed"
    assert asked.failure is failure
