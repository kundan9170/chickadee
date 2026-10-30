"""What a wire client raises: a route's typed refusal, a server it cannot speak to, a stream it
could not keep, and no server at all."""

from __future__ import annotations

from chickadee_serve.frames import ErrorBody, ErrorCode
from chickadee_serve.refusals import Refusal

__all__ = [
    "Disconnected",
    "HostRefused",
    "NotServing",
    "Unanswered",
    "WireError",
    "WireMismatch",
]


class WireError(Exception):
    """A route answered with an `ErrorBody`: its status, its code and its message."""

    def __init__(self, status: int, error: ErrorBody) -> None:
        super().__init__(f"{status} {error.code}: {error.message}")
        self.status = status
        self.code: ErrorCode | None = error.code
        """`None` only on `Unanswered`: no route answered, so there is no `ErrorBody`."""
        self.message = error.message


class Unanswered(WireError):
    """A request the server did not answer: it timed out, or the connection broke before a
    response came. Not a route's refusal, so no `ErrorBody` and no `code`; a
    `WireError` all the same, so every caller that says a refusal and carries on says this too
    rather than crashing on the transport's own exception."""

    def __init__(
        self, method: str, path: str, why: str, *, maybe_delivered: bool = False
    ) -> None:
        Exception.__init__(self, f"{method} {path}: {why}")
        self.status = 0
        self.code = None
        self.maybe_delivered = maybe_delivered
        """Whether the server may have received and acted on the request: a timeout after it
        was sent, or a connection that broke mid-request. Such a request is not known to have
        failed, so a caller does not treat it as refused, nor resend it blindly (the project
        rules:
        never retry an ambiguous effect)."""
        outcome = "; it may have been received, so check before sending it again"
        self.message = (
            f"the server did not answer {method} {path}: {why}"
            f"{outcome if maybe_delivered else ''}"
        )


class WireMismatch(Exception):
    """The server speaks another major version of the wire (`WIRE_MISMATCH`)."""

    def __init__(self, client: str, server: str) -> None:
        super().__init__(f"this client speaks wire {client}, the server {server}")
        self.client = client
        self.server = server


class Disconnected(Exception):
    """A stream dropped more times in a row than its reconnect budget allows."""


class NotServing(Exception):
    """Nothing answers where a server was looked for. The message names every place tried."""


class HostRefused(Exception):
    """`host_in_process` would not start, and why, without a frontend importing the engine's
    own errors. `reason` is a `chickadee_serve.refusals.Refusal`."""

    def __init__(self, reason: Refusal, message: str):
        super().__init__(message)
        self.reason: Refusal = reason
