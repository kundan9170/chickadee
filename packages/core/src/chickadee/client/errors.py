"""Transport failure modes.

The split that matters: a `CredentialError` is an operator mistake and aborts the run, while
`TransportExhausted` is a runtime condition the orchestrator already has a path for. Retrying
a 401 forever is the failure this distinction prevents.

It was called `ConfigError` until 2026-09-12, which shadowed the public class of that name
without being catchable as one. `18-engine` §5 maps internal exceptions at the facade,
so a private class here is right — sharing the *name* of the public one is what would have
made that mapping easy to get wrong, and `CredentialError` says what both raise sites
actually are: an unset `api_key_env`, and a 401/403.
"""

from __future__ import annotations

from typing import Final

OPENAI_HEADERS_ENV: Final = "OPENAI_CUSTOM_HEADERS"
ANTHROPIC_HEADERS_ENV: Final = "ANTHROPIC_CUSTOM_HEADERS"
AMBIENT_HEADERS_ENV: Final = (OPENAI_HEADERS_ENV, ANTHROPIC_HEADERS_ENV)
"""The variables an SDK turns into request headers, which can carry a gateway's credential:
read by the clients (`ambient_header_names`), and withheld with the endpoints' keys from a
command run outside the sandbox (`EngineConfig.endpoint_keys`)."""


class ClientError(Exception):
    """Base for transport failures."""

    billed_attempts: tuple[int, ...] = ()
    """Each of the request's attempts the provider may have billed, as the output tokens it had
    streamed before it stopped: an attempt that fell silent past a limit
    while the provider was working, got back a body that was not a completion, or was
    cancelled in flight. A connect error, a 429, any other status and a credential refused are
    not billed, and a count of attempts cannot tell them apart. What a failed call
    is charged is `client/charge.unanswered_usage`'s."""


class TransportExhausted(ClientError):
    """A failure kind survived its retries, or a status no retry can fix.

    Surfaces to the orchestrator as its timeout path (row 7). The state machine sees a
    completion or a timeout and never HTTP anatomy — internal retries are invisible to it
    on purpose, because a retry ladder leaking into the machine would make attempt
    accounting mean two different things.
    """

    def __init__(
        self, detail: str, attempts: int, *, billed_attempts: tuple[int, ...] = ()
    ) -> None:
        self.attempts = attempts
        self.billed_attempts = billed_attempts
        super().__init__(f"transport exhausted after {attempts} attempt(s): {detail}")


class ContextOverflow(ClientError):
    """The provider refused the request as longer than the model takes.

    Its own failure, not a `TransportExhausted`: sending it again sends the same length, so the
    answer is never a retry of the same request, and reading it as a timeout gives the run the
    wrong taxonomy. Our local estimate said it fitted; the provider counted more. A refused
    request is not billed. Which refusals are this one is `openai_client.context_overflow`'s.
    """

    def __init__(self, detail: str, status: int, attempts: int) -> None:
        self.detail = detail
        self.status = status
        self.attempts = attempts
        self.billed_attempts = ()
        super().__init__(
            f"the provider refused the request as too long (HTTP {status}): {detail}"
        )


class ToolChoiceRefused(ClientError):
    """The provider refused a request's forced tool call: a 400 or 422 whose
    body names `tool_choice`. Never retried as it is: the client sends the request again
    unforced (`openai_client.Forcing`). A refused request is not billed."""

    def __init__(self, detail: str, status: int) -> None:
        self.status = status
        super().__init__(f"the provider refused tool_choice (HTTP {status}): {detail}")


class CredentialError(ClientError):
    """401/403, or an `api_key_env` variable that is unset.

    Credentials are an operator error, not retryable model behavior, so this raises to
    ABORT rather than burning the retry budget on a request that will never succeed.
    """


class RequestCancelled(ClientError):
    """The request's cancellation token was set: before an attempt, during a backoff, or
    while an attempt was in flight, which closes its stream at once. Never
    retried. An attempt cancelled in flight was sent, so it is billed with what it streamed.

    Raised rather than returned as an empty completion: a cancelled request produced no
    evidence, and a completion with no text would read downstream as a model that said
    nothing."""

    def __init__(self, attempts: int, *, billed_attempts: tuple[int, ...] = ()) -> None:
        self.attempts = attempts
        self.billed_attempts = billed_attempts
        super().__init__(f"cancelled after {attempts} attempt(s)")
