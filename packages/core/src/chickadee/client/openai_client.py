"""OpenAI-compatible transport.

Speaks one wire format to any endpoint — llama.cpp, vLLM, SGLang, or a hosted provider —
because supporting N adapters instead of one is what makes "local or hosted, your choice"
stop being a configuration question. The wire encoding is the official `openai` package's,
pointed at `endpoint.base_url`: the vendor maintains the serializer, and we keep the
request type, the usage probe and the retry ladder. The SDK's own retries are **off**
(`max_retries=0`) so AD-5's ladder is the only one and every attempt is counted.
Two rules carry the weight:

**Termination is read from the provider's stop signal, never from response text.**
Text and tool-call content coexist in one response, so a loop that ends when text appears
truncates work mid-flight. `MAX_TOKENS` is a truncated response and becomes
`ParseError(truncated)` upstream, never a terminal parse.

**Unknown is not zero.** A provider that does not report cached tokens yields `None`, not
`0`. Collapsing them would corrupt the D18 cache-hit metric, which is the number that
explains cost.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final, Literal, Protocol, cast, runtime_checkable

import httpx2
import openai
from pydantic import BaseModel

from chickadee.client.affinity import AffinityTable
from chickadee.client.errors import (
    OPENAI_HEADERS_ENV,
    ClientError,
    ContextOverflow,
    CredentialError,
    RequestCancelled,
    ToolChoiceRefused,
    TransportExhausted,
)
from chickadee.client.live import NewAttempt, Piece, Text, Thought, ToolArgs, hand
from chickadee.client.request import (
    ChatRequest,
    DocumentPart,
    ImagePart,
    Part,
    TextPart,
    ToolGrammar,
    ToolSchema,
    has_documents,
    has_images,
    prompt_text,
)
from chickadee.client.sleep import ModelConnections
from chickadee.client.stream import (
    Broke,
    Silences,
    Silent,
    StreamEnded,
    StreamStopped,
    read_stream,
)
from chickadee.errors import HarnessBug
from chickadee.limits import tokens_est
from chickadee.schema import Completion, StopReason, ToolCall, Usage
from chickadee.schema.adapter import CacheMode

_log = logging.getLogger(__name__)

MAX_BREAKPOINTS = 4
"""The provider limit on cache markers per request, on every `EXPLICIT` provider."""

# Providers spell the terminal signal differently; the raw string is preserved either way.
STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "end_turn": StopReason.END_TURN,
    "eos": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "tool_use": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_length": StopReason.MAX_TOKENS,
    # Anthropic: the prompt plus the output reached the window. Truncated all the same.
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
    "refusal": StopReason.REFUSAL,
}


@runtime_checkable
class EndpointSpec(Protocol):
    """What the client needs from a rung's configuration.

    Structural rather than an import of `EngineConfig.RungEndpoint`: the transport
    has no business depending on the config package, and a Protocol lets the real type
    satisfy this without either side knowing about the other.
    """

    @property
    def base_url(self) -> str: ...
    @property
    def model_id(self) -> str: ...
    @property
    def api_key_env(self) -> str | None: ...
    @property
    def provider(self) -> str: ...
    @property
    def first_token_s(self) -> float: ...
    @property
    def token_gap_s(self) -> float: ...
    @property
    def supports_images(self) -> bool: ...
    @property
    def supports_documents(self) -> bool: ...


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _obj(value: object) -> dict[str, Any]:
    """Narrow one JSON value to an object, or an empty one.

    A missing or wrong-typed field is treated as absent rather than raising: the response
    shape is the provider's business, and a surprising body should become a retry, not a
    stack trace inside the parse.
    """
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _arr(value: object) -> list[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else []


def map_stop_reason(raw: str | None) -> StopReason:
    """Map a provider's terminal signal onto `StopReason`.

    An unrecognized value maps to `OTHER` and the caller falls back to `tool_calls`
    emptiness — recorded rather than silently assumed, so the weaker signal is visible in
    telemetry instead of looking like a confident read.
    """
    if raw is None:
        return StopReason.OTHER
    return STOP_REASON_MAP.get(raw.strip().lower(), StopReason.OTHER)


def probe_usage(
    body: dict[str, Any], prompt_text: str, completion_text: str
) -> tuple[Usage, bool]:
    """Extract usage, or estimate it and say so. Returns (usage, estimated)."""
    # An empty or absent `usage` object are the same thing to us: neither carries counts.
    # Nor does one that names only a total, and reading its missing fields as 0 would be
    # the silently precise answer this fallback exists to avoid.
    usage = _obj(body.get("usage"))
    if not any(isinstance(usage.get(k), int) for k in ("prompt_tokens", "completion_tokens")):
        return (
            Usage(
                # P-2, as `04-adapter.md` §4 says: flagged estimated, and at least one
                # token, since a request that was sent is never free.
                prompt_tokens=max(1, tokens_est(prompt_text)),
                completion_tokens=max(1, tokens_est(completion_text)),
                cached_prompt_tokens=None,
                estimated=True,
            ),
            True,
        )
    cached: int | None = None
    detail_tokens = _obj(usage.get("prompt_tokens_details")).get("cached_tokens")
    if isinstance(detail_tokens, int):
        cached = detail_tokens
    # Provider-specific spellings, checked only when the OpenAI-style path found nothing.
    if cached is None:
        for key in ("cached_tokens", "prompt_cache_hit_tokens"):
            alt = usage.get(key)
            if isinstance(alt, int):
                cached = alt
                break
    return (
        Usage(
            prompt_tokens=_int(usage.get("prompt_tokens")),
            completion_tokens=_int(usage.get("completion_tokens")),
            cached_prompt_tokens=cached,
            estimated=False,
        ),
        False,
    )


def _arguments(value: object) -> dict[str, Any]:
    """Tool arguments, from either wire shape.

    The OpenAI schema types `function.arguments` as a **JSON-encoded string**, not an
    object, and most compatible servers follow it. Narrowing it with `_obj` yields `{}`
    for every standards-compliant response — a tool call that reaches the orchestrator
    stripped of its inputs, silently, which reads downstream as a model that asked to
    read nothing. Some providers do send the object inline, so both are accepted.
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    if isinstance(value, str):
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}
    return {}


def _tool_call(entry: dict[str, Any], grammars: dict[str, ToolGrammar]) -> ToolCall:
    """Accept both the `function`-nested shape and the flat one; providers differ. A custom
    call is its tool's one free-text argument, the grammar's `input`."""
    if entry.get("type") == "custom":
        custom = _obj(entry.get("custom"))
        name = str(custom.get("name") or "")
        grammar = grammars.get(name)
        text = custom.get("input")
        # A custom call to a tool the request did not constrain is not one the model was
        # offered: its arguments stay empty, and the dispatcher refuses it as invalid.
        arguments: dict[str, object] = (
            {grammar.argument: text} if grammar is not None and isinstance(text, str) else {}
        )
        return ToolCall(id=str(entry.get("id") or ""), name=name, arguments=arguments)
    fn = _obj(entry.get("function"))
    name = fn.get("name") or entry.get("name") or ""
    args = fn.get("arguments") if "arguments" in fn else entry.get("arguments")
    return ToolCall(
        id=str(entry.get("id") or ""),
        name=str(name),
        arguments=_arguments(args),
    )


FailureKind = Literal[
    "connect", "rate_limited", "server_error", "malformed", "first_token", "token_gap"
]
"""D211 decision 3's failure kinds, each retried `RetryPolicy.retries` times on its own count:
a connect error, a 429, a 5xx, a body that was not a completion, no first token within
`first_token_s`, and a gap past `token_gap_s` once the reply had started."""


class Retry(Exception):
    """One attempt failed in a way the ladder retries, of one `FailureKind`. Internal to the
    ladder; never escapes it. `billed` is the output tokens it had streamed when the provider
    may bill it (a silence, a body that was not a completion), None when it cannot have
    (a connect error, a 429, a 5xx). `retry_after` is what a 429 or 503 asked to wait."""

    def __init__(
        self,
        detail: str,
        kind: FailureKind,
        *,
        billed: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.kind: FailureKind = kind
        self.billed = billed
        self.retry_after = retry_after
        super().__init__(detail)


def retryable(status: int) -> bool:
    """AD-5's rule: 429 and every 5xx. Any other status is the request's fault, and sending
    it again sends the same fault."""
    return status == 429 or status >= 500


_OVERFLOW_MARKERS: Final = (
    "context_length_exceeded",  # OpenAI's error code
    "maximum context length",  # vLLM and other OpenAI-compatible servers
    "prompt is too long",  # Anthropic
    "exceeds the available context size",  # llama.cpp
)


def context_overflow(status: int, text: str) -> bool:
    """Whether a refusal says the request is longer than the model takes. A 413 always
    does; a 400 only when its body carries a provider's own marker. Any other 400 is the
    request's other fault, and is not guessed to be this one."""
    if status == 413:
        return True
    lowered = text.lower()
    return status == 400 and any(marker in lowered for marker in _OVERFLOW_MARKERS)


def _wait(delay_s: float, cancel: threading.Event | None) -> None:
    """A backoff. A cancel during it ends it: the wait is the longest stretch a cancelled
    session would otherwise sit through, and the ladder's next look at the token raises."""
    if cancel is None:
        time.sleep(delay_s)
    else:
        cancel.wait(delay_s)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times each failure kind is retried and how long each retry
    waits. `random` and `sleep` are the seams a test injects."""

    retries: int = 3
    waits_s: tuple[float, ...] = (1.0, 4.0, 16.0)
    """The kind's first, second and third retry's wait, before jitter."""
    jitter: float = 0.25
    """Up to this fraction added to each wait, so clients a failure hit together do not
    retry together."""
    retry_after_max_s: float = 60.0
    random: Callable[[], float] = random.random
    sleep: Callable[[float, threading.Event | None], None] = _wait

    def wait_s(self, retry: int, retry_after: float | None) -> float:
        """The wait before a kind's `retry`th retry (from 1): what a `Retry-After` asked, up
        to `retry_after_max_s`, else the ladder's step with its jitter."""
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.retry_after_max_s)
        step = self.waits_s[min(retry, len(self.waits_s)) - 1]
        return step * (1.0 + self.jitter * self.random())


RETRY_POLICY: Final = RetryPolicy()


def run_ladder[T](
    request: ChatRequest, attempt: Callable[[int], T], policy: RetryPolicy = RETRY_POLICY
) -> T:
    """Attempts until one answers, each failure kind retried `policy.retries`
    times on its own count, backing off between them. `attempt(n)` (n counts from 1) raises
    `Retry` for a failure the ladder retries and anything else to stop it. A kind's failure
    past its retries ends the ladder with `TransportExhausted`. Invisible to the orchestrator,
    and to the person (decision 4): it sees a completion or a typed error, never HTTP anatomy.

    What each attempt the provider may have billed had streamed is carried out on whatever
    error ends the ladder, every attempt's, not only the last: a body that came back and was
    not a completion was answered, so it may be billed even when the retry after it never
    connected, or was refused its credentials."""
    retried: dict[FailureKind, int] = {}
    billed: list[int] = []
    n = 0
    while True:
        if request.cancel is not None and request.cancel.is_set():
            raise RequestCancelled(n, billed_attempts=tuple(billed))
        n += 1
        if n > 1 and request.live is not None:
            # The live view is told a new attempt began, so what the last
            # one streamed is cleared.
            hand(request.live, (NewAttempt(n),))
        try:
            return attempt(n)
        except Retry as exc:
            if exc.billed is not None:
                billed.append(exc.billed)
            retry = retried.get(exc.kind, 0) + 1
            if retry > policy.retries:
                raise TransportExhausted(str(exc), n, billed_attempts=tuple(billed)) from exc
            retried[exc.kind] = retry
            delay = policy.wait_s(retry, exc.retry_after)
        except ClientError as exc:
            exc.billed_attempts = (*billed, *exc.billed_attempts)
            raise
        policy.sleep(delay, request.cancel)


@dataclass(frozen=True)
class SdkErrors:
    """One SDK's exception classes. The two vendors' hierarchies are parallel rather than
    shared, so the ladder's classification is written once against this."""

    timeout: type[Exception]
    connection: type[Exception]
    credential: tuple[type[Exception], ...]
    status: type[Exception]
    api: type[Exception]
    """The SDK's base error: what a stream raises for an error event inside it."""


OPENAI_ERRORS = SdkErrors(
    timeout=openai.APITimeoutError,
    connection=openai.APIConnectionError,
    credential=(openai.AuthenticationError, openai.PermissionDeniedError),
    status=openai.APIStatusError,
    api=openai.APIError,
)

_RETRY_AFTER_STATUSES: Final = frozenset({429, 503})


def retry_after_s(exc: Exception) -> float | None:
    """The wait a refusal's `Retry-After` asks for, in seconds (`retry-after-ms` first, as the
    SDKs read it; then seconds or an HTTP date), or None when it asks none."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    millis = headers.get("retry-after-ms")
    if millis is not None:
        try:
            waited = float(millis) / 1000.0
        except ValueError:
            pass
        else:
            if math.isfinite(waited):
                return waited
    said = headers.get("retry-after")
    if said is None:
        return None
    try:
        seconds = float(said)
    except ValueError:
        pass
    else:
        return seconds if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(said)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        # RFC 5322's `-0000` zone, which `email.utils` reads as no zone: HTTP dates are GMT.
        when = when.replace(tzinfo=UTC)
    return (when - datetime.now(UTC)).total_seconds()


_STREAM_ERRORS: Final[dict[str, FailureKind]] = {
    "overloaded_error": "server_error",
    "api_error": "server_error",
    "server_error": "server_error",
    "rate_limit_error": "rate_limited",
    "rate_limit_exceeded": "rate_limited",
}
"""An error event inside a stream, by its `type`, that is the server's state rather than the
reply's (Anthropic's `overloaded_error` arrives as a status error carrying the stream's 200):
retried as a 5xx or a 429 is, and not billed. Any other error event is a reply that was not a
completion."""


def _stream_error_kind(cause: Exception) -> FailureKind | None:
    """The kind an error event inside a stream names (`_STREAM_ERRORS`), or None. Anthropic's
    body is the event (`{"type": "error", "error": {"type":...}}`), OpenAI's the error."""
    body = _obj(getattr(cause, "body", None))
    inner = _obj(body.get("error"))
    said = inner.get("type") if inner else body.get("type")
    return _STREAM_ERRORS.get(said) if isinstance(said, str) else None


Unforced = Literal["no_capability", "thinking", "refused"]
"""Why a request that asked for a forced tool call went without it."""


class Forcing:
    """D220 decision 4, the one home of whether a client forces a request's tool call: only
    when the request asks (`ChatRequest.force_tool`) and offers tools, the endpoint declares
    `forced_tools`, the client has no reason of its own against it (`blocked`: Anthropic's
    thinking), and the provider has not refused it once. Otherwise the request goes unforced,
    and each reason is logged once. One per client, so a refusal holds for its session.

    Used on the client's caller's thread; two threads sharing a client race only on the log
    and on a refusal both would record alike."""

    def __init__(self, endpoint: EndpointSpec) -> None:
        self._model = endpoint.model_id
        self._capable = bool(getattr(endpoint, "forced_tools", False))
        self._refused = False
        self._said: set[Unforced] = set()

    def send(
        self,
        request: ChatRequest,
        send: Callable[[bool], Completion],
        *,
        blocked: Unforced | None = None,
    ) -> Completion:
        """`send(forced)`'s completion, forced when it may be; a refusal of the forced call
        sends it again at once unforced, and every later request through this client too."""
        if not (request.force_tool and request.tools):
            return send(False)
        why: Unforced | None = (
            "no_capability" if not self._capable else "refused" if self._refused else blocked
        )
        if why is not None:
            self._say(why)
            return send(False)
        try:
            return send(True)
        except ToolChoiceRefused:
            self._refused = True
            self._say("refused")
            return send(False)

    def _say(self, why: Unforced) -> None:
        if why not in self._said:
            self._said.add(why)
            _log.warning("%s: a forced tool call is sent unforced (D220): %s", self._model, why)


def _refused(status: int, text: str, n: int) -> ClientError:
    """A status no retry fixes, by what the provider refused: the request's length, its forced
    tool call, or anything else."""
    if context_overflow(status, text):
        return ContextOverflow(text[:200], status, n)
    if status in (400, 422) and "tool_choice" in text:
        return ToolChoiceRefused(text[:200], status)
    return TransportExhausted(f"HTTP {status}: {text[:200]}", n)


def failed_attempt(ended: StreamEnded, n: int, errors: SdkErrors, url: str) -> Exception:
    """An attempt whose stream ended before its reply, as the ladder reads it: a `Retry` of
    its kind, a terminal error, or `RequestCancelled`. The order of the clauses is
    load-bearing: in both SDKs a timeout is a connection error, a credential error is a
    status error, and every error is the SDK's base `api` error.

    A half-completion is not evidence; whatever streamed is discarded. An attempt is
    billed with what it streamed only once the provider had taken it on, answering with the
    stream's headers (`StreamEnded.opened`): a request that never reached it, or that it never
    answered (a connect that hung, a host that cannot be reached), is not billed, so an outage
    cannot spend the cap. A dropped connection is the SDK's connection error or, where the SDK
    does not wrap it (Anthropic's stream), the transport's own: before the reply started it is
    a connect error, after it a reply that died mid-way, a token gap. Any other error the SDK
    raised reading a stream the provider had opened is a malformed reply too; one raised
    before it opened is the request's own, and is raised as it is."""
    billed = ended.streamed_tokens if ended.opened else None
    if isinstance(ended, StreamStopped):
        return RequestCancelled(n, billed_attempts=() if billed is None else (billed,))
    if isinstance(ended, Silent):
        return Retry(str(ended), ended.kind, billed=billed)
    cause = ended.cause if isinstance(ended, Broke) else ended
    if isinstance(cause, (errors.timeout, httpx2.TimeoutException)):
        # The SDK's own read timeout, the backstop behind the two limits.
        kind: FailureKind = "token_gap" if ended.started else "first_token"
        return Retry(f"deadline exceeded: {cause}", kind, billed=billed)
    if isinstance(cause, (errors.connection, httpx2.TransportError)):
        if ended.started:
            return Retry(f"connection lost mid-reply: {cause}", "token_gap", billed=billed)
        return Retry(f"connect error: {cause}", "connect")
    if isinstance(cause, errors.credential):
        return CredentialError(
            f"{getattr(cause, 'status_code', '?')} from {url}: credentials are an operator "
            "error, not retryable model behavior"
        )
    if isinstance(cause, errors.status) and not ended.started:
        status = int(getattr(cause, "status_code", 0))
        if retryable(status):
            return Retry(
                f"HTTP {status}",
                "rate_limited" if status == 429 else "server_error",
                retry_after=retry_after_s(cause) if status in _RETRY_AFTER_STATUSES else None,
            )
        text = str(getattr(getattr(cause, "response", None), "text", ""))
        if status >= 400:
            return _refused(status, text, n)
    if isinstance(cause, errors.api):
        server = _stream_error_kind(cause)
        if server is not None:
            return Retry(f"stream error: {cause}", server)
    if isinstance(cause, (errors.api, ValueError)):
        # An error event inside the stream, or a body the SDK could not read as one (§4's
        # "malformed JSON" row): the provider answered, so it may bill it.
        return Retry(f"malformed stream: {cause}", "malformed", billed=billed)
    if ended.opened and isinstance(cause, Exception) and not isinstance(cause, HarnessBug):
        # The provider answered and the SDK could not assemble what it streamed: a server that
        # sends a tool call's later piece with `"type": null` or `"function": null` trips the
        # stream helper's own bare `assert`. Its reply is not a completion, whatever the SDK
        # raised; raised as it is, it read as the harness's failure, with no message.
        return Retry(
            f"malformed stream: {type(cause).__name__}: {cause}", "malformed", billed=billed
        )
    return cause if isinstance(cause, Exception) else ended


def completion(
    *,
    text: str,
    tool_calls: tuple[ToolCall, ...],
    raw_stop: str | None,
    usage: Usage,
    started: float,
) -> Completion:
    """The `Completion` both transports build, with D47's rules applied once."""
    stop = map_stop_reason(raw_stop)
    # D47's contract forbids the contradictory combinations; a provider that reports
    # `stop` while also returning tool calls is describing a turn that continues.
    if tool_calls and stop is StopReason.END_TURN:
        stop = StopReason.TOOL_USE
    if stop is StopReason.TOOL_USE and not tool_calls:
        stop = StopReason.OTHER
    cached_prefix = (
        None if usage.cached_prompt_tokens is None else usage.cached_prompt_tokens > 0
    )
    return Completion(
        text=text,
        usage=usage,
        latency_ms=int((time.monotonic() - started) * 1000),
        cached_prefix=cached_prefix,
        stop_reason=stop,
        raw_stop_reason=raw_stop,
        tool_calls=tool_calls,
    )


def dump(response: object) -> dict[str, Any] | None:
    """An SDK response as the provider sent it, or None when it is not one.

    `exclude_unset` is what keeps an absent field absent: the SDK models declare every
    usage counter, and a plain dump reports a counter the provider never sent as `None`
    beside the ones it did — enough to hide an empty `usage` from the estimator. Fields
    the model does not declare survive either way, which is where the provider-specific
    cache counters live (`prompt_cache_hit_tokens`)."""
    if not isinstance(response, BaseModel):
        return None
    return response.model_dump(warnings=False, exclude_unset=True)


_DELTA_TEXT: Final = ("content", "reasoning_content", "reasoning")
"""A chunk delta's text fields, all of them output: the reply's content, and the reasoning an
OpenAI-compatible server streams beside it (vLLM, DeepSeek, OpenRouter), billed as output as
Anthropic's thinking is."""


def _pieces(event: object) -> tuple[Piece, ...]:
    """What one of the chat stream helper's events streamed (`client/live.py`): a raw chunk's
    content as text, its reasoning as thought, and each tool call's arguments (a custom call's
    input) with its name where the chunk carries it. Its derived events (`content.delta`,...)
    repeat a chunk's, and say nothing."""
    if getattr(event, "type", None) != "chunk":
        return ()
    chunk: Any = getattr(event, "chunk", None)
    said: list[Piece] = []
    for choice in getattr(chunk, "choices", None) or ():
        delta: Any = getattr(choice, "delta", None)
        for field in _DELTA_TEXT:
            text = getattr(delta, field, None)
            if isinstance(text, str) and text:
                said.append(Text(text) if field == "content" else Thought(text))
        for call in getattr(delta, "tool_calls", None) or ():
            index = getattr(call, "index", 0)
            fn: Any = getattr(call, "function", None)
            custom = _obj(getattr(call, "custom", None))
            name = getattr(fn, "name", None) or custom.get("name")
            arguments = getattr(fn, "arguments", None)
            given = custom.get("input")
            text = "".join(t for t in (arguments, given) if isinstance(t, str))
            if text or isinstance(name, str):
                said.append(
                    ToolArgs(
                        index if isinstance(index, int) else 0,
                        text,
                        name if isinstance(name, str) else None,
                    )
                )
    return tuple(said)


def _snapshot(stream: object) -> dict[str, Any] | None:
    """What the chat stream helper accumulated, as the provider sent it (`dump`). Read from
    its snapshot rather than `get_final_completion`, which raises on a `length` stop and drops
    a custom tool's call: D47 reads the first as a truncated reply, and a constrained tool
    is the second."""
    return dump(getattr(stream, "current_completion_snapshot", None))


def ambient_header_names(variable: str) -> tuple[str, ...]:
    """The header names an SDK reads from `variable` when it is built — its own
    `Name: value` per line parse.

    The rung's config and the affinity table are the only sources of a header. A variable set in
    the shell for some other tool would otherwise put headers on
    every request to every endpoint, unseen by the config and by the byte test."""
    raw = os.environ.get(variable)
    if raw is None:
        return ()
    return tuple(line[: line.find(":")].strip() for line in raw.split("\n") if ":" in line)


def refuse_ambient_credential(variable: str, names: tuple[str, ...], *auth: str) -> None:
    """A credential header in `variable` would override the rung's key on the wire. Neither
    answer is safe to pick: dropping it may be dropping the only working key, and keeping it
    sends a credential the config never named."""
    clash = sorted(n for n in names if n.lower() in {a.lower() for a in auth})
    if clash:
        raise CredentialError(
            f"{variable} sets {', '.join(clash)}; a rung's api_key_env is its only credential"
        )


def api_key(endpoint: EndpointSpec) -> str | None:
    """The engine never stores secrets — it reads the named variable at call
    time, so a key lives in the process environment and nowhere in our state."""
    name = endpoint.api_key_env
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise CredentialError(
            f"endpoint declares api_key_env={name!r} but that variable is unset"
        )
    return value


def _ephemeral(part: dict[str, Any]) -> dict[str, Any]:
    return {**part, "cache_control": {"type": "ephemeral"}}


Attached = ImagePart | DocumentPart
"""What a tool result carries that this wire's tool messages cannot."""


def image_label(part: Attached) -> str:
    noun = "image" if isinstance(part, ImagePart) else "document"
    return f"[{noun} {part.sha[:8]}]"


def _content_part(part: Part) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    data = base64.b64encode(part.data).decode("ascii")
    url = f"data:{part.media_type};base64,{data}"
    if isinstance(part, DocumentPart):
        # Chat Completions takes a PDF as a `file` part, by a data URL and a filename.
        return {"type": "file", "file": {"filename": part.name, "file_data": url}}
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_content(parts: tuple[Part, ...]) -> list[dict[str, Any]]:
    """A tool result on this wire is text only (D110's compensation): each image or document
    becomes a line saying where it went, and it rides in the user message after the
    results."""
    return [
        {"type": "text", "text": p.text}
        if isinstance(p, TextPart)
        else {"type": "text", "text": f"{image_label(p)[:-1]} — attached below]"}
        for p in parts
    ]


def _attached_images(images: list[Attached]) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for img in images:
        content.append({"type": "text", "text": image_label(img)})
        content.append(_content_part(img))
    return {"role": "user", "content": content}


def openai_messages(request: ChatRequest, *, explicit: bool = False) -> list[dict[str, Any]]:
    """The request's messages in the chat-completions shape: what the SDK must emit, byte
    for byte (`test_wire_bytes.py`).

    `cache_breakpoint` has a spelling only when the provider caches `EXPLICIT`ly
    (OpenRouter): the marked message's last content part carries `cache_control`,
    which needs content parts, so only a marked message changes shape. Everywhere else
    this wire's providers cache automatically and a marker is ignored.

    D110's compensation: an image in a tool result is a text line in the result and the
    image itself in a `user` message after the run of results. After the run, not after
    each result, because the results answering one assistant turn must be contiguous on
    this wire; before the next assistant turn, which is all the model needs.
    """
    if explicit:
        marks = sum(1 for m in request.messages if m.cache_breakpoint)
        if marks > MAX_BREAKPOINTS:
            raise ValueError(
                f"{marks} cache breakpoints; the provider takes at most {MAX_BREAKPOINTS} "
                "(04-adapter.md §3.2 — stage 4 drops the innermost)"
            )
    out: list[dict[str, Any]] = []
    pending: list[Attached] = []
    grammars = _grammars(request)
    for m in request.messages:
        if pending and m.role != "tool":
            out.append(_attached_images(pending))
            pending = []
        marked = explicit and m.cache_breakpoint
        content: str | list[dict[str, Any]]
        if isinstance(m.content, str):
            content = m.content
            if marked and content:
                content = [_ephemeral({"type": "text", "text": content})]
        else:
            if m.role == "tool":
                content = _tool_content(m.content)
                pending.extend(p for p in m.content if isinstance(p, ImagePart | DocumentPart))
            else:
                content = [_content_part(p) for p in m.content]
            if marked:
                content[-1] = _ephemeral(content[-1])
        entry: dict[str, Any] = {"role": m.role, "content": content}
        if m.tool_call_id is not None:
            entry["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            entry["tool_calls"] = [_echoed(c, grammars) for c in m.tool_calls]
        out.append(entry)
    if pending:
        out.append(_attached_images(pending))
    return out


def openai_tools(request: ChatRequest) -> list[dict[str, Any]]:
    """The request's tools: a function each, but a tool decoded under a grammar,
    which is a Chat Completions custom tool, its one free text held to the Lark grammar."""
    return [_tool(t) for t in request.tools]


def _tool(t: ToolSchema) -> dict[str, Any]:
    if t.grammar is None:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.schema),
            },
        }
    return {
        "type": "custom",
        "custom": {
            "name": t.name,
            "description": t.description,
            "format": {
                "type": "grammar",
                "grammar": {"syntax": t.grammar.syntax, "definition": t.grammar.definition},
            },
        },
    }


def _grammars(request: ChatRequest) -> dict[str, ToolGrammar]:
    """The request's constrained tools by name: how a call to one is read and echoed."""
    return {t.name: t.grammar for t in request.tools if t.grammar is not None}


def _echoed(call: ToolCall, grammars: dict[str, ToolGrammar]) -> dict[str, Any]:
    """A call in history, in the shape it was made: a constrained tool's as its free text."""
    grammar = grammars.get(call.name)
    if grammar is None:
        return {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
    text = call.arguments.get(grammar.argument, "")
    return {
        "id": call.id,
        "type": "custom",
        "custom": {"name": call.name, "input": text if isinstance(text, str) else ""},
    }


def refuse_images(endpoint: EndpointSpec, request: ChatRequest) -> None:
    """`read` refuses an image upstream on a rung that cannot see. One reaching the
    client means that refusal did not run, and sending it anyway is a 400 at best. A PDF is
    the same: a model without `supports_documents` is sent the text layer upstream."""
    if has_images(request) and not endpoint.supports_images:
        raise ValueError(
            "an image reached a rung with supports_images = false; `read` refuses it "
            "upstream with NOT_SUPPORTED (D110)"
        )
    if has_documents(request) and not endpoint.supports_documents:
        raise ValueError(
            "a PDF reached a rung with supports_documents = false; it is sent its text layer "
            "upstream (D245)"
        )


class OpenAIClient:
    """One rung's transport over the official `openai` package. Satisfies `ModelClient`
    (`client/request.py`) through `send` and `close`."""

    def __init__(
        self,
        endpoint: EndpointSpec,
        affinity: AffinityTable | None = None,
        transport: httpx2.BaseTransport | None = None,
        *,
        retry: RetryPolicy = RETRY_POLICY,
    ) -> None:
        self.endpoint = endpoint
        self.affinity = affinity or AffinityTable()
        self._retry = retry
        self._silences = Silences(endpoint.first_token_s, endpoint.token_gap_s)
        self._forcing = Forcing(endpoint)
        # `transport` is the test seam: a mock server with real HTTP semantics. Made anew after
        # the machine sleeps, so no request waits on a connection the sleep left dead.
        self._http = ModelConnections(
            lambda: httpx2.Client(
                timeout=httpx2.Timeout(self._silences.backstop_s), transport=transport
            )
        )

    def close(self) -> None:
        self._http.close()

    def _sdk(self, key: str | None) -> openai.OpenAI:
        # Built per call so the key is read at call time and never held. The
        # placeholder only satisfies the SDK's constructor; `_build` omits the header it
        # would become.
        return openai.OpenAI(
            api_key=key or "unused",
            base_url=self.endpoint.base_url,
            timeout=self._silences.backstop_s,
            max_retries=0,
            http_client=self._http.get(),
        )

    # -- request assembly ---------------------------------------------------

    def _build(
        self, request: ChatRequest, key: str | None, *, forced: bool = False
    ) -> dict[str, Any]:
        refuse_images(self.endpoint, request)
        params = request.params
        row = self.affinity.row(self.endpoint.provider)
        kwargs: dict[str, Any] = {
            "model": self.endpoint.model_id,
            "messages": openai_messages(request, explicit=row.cache_mode is CacheMode.EXPLICIT),
            "max_tokens": params.max_tokens,
        }
        # A rung that refuses the parameter never receives it.
        if bool(getattr(self.endpoint, "supports_temperature", True)):
            kwargs["temperature"] = params.temperature
        if params.stop:
            kwargs["stop"] = list(params.stop)
        # `tools` rides iff the request declares some and the rung speaks native
        # tools; otherwise the TOOL text protocol carries them in the prompt.
        if request.tools and bool(getattr(self.endpoint, "native_tools", True)):
            if _grammars(request) and not bool(getattr(self.endpoint, "grammar_tools", False)):
                # The load rule refuses `constrain` on such a model: one reaching
                # here means that rule did not run, and the provider would reject the tool.
                raise HarnessBug(
                    f"{self.endpoint.model_id} has no grammar_tools, and a request "
                    "constrains a tool"
                )
            kwargs["tools"] = openai_tools(request)
            if forced:
                kwargs["tool_choice"] = "required"
        extra_body: dict[str, Any] = {}
        if params.reasoning is not None:
            # Provider-neutral in the contract, mapped here. A provider without a
            # reasoning parameter drops it, and the drop is recorded by the caller rather
            # than silently ignored.
            extra_body["reasoning_effort"] = params.reasoning.level
        headers: dict[str, str] = {}
        row.inject(extra_body, headers, request.session_key)
        extra_headers: dict[str, Any] = dict(headers)
        ambient = ambient_header_names(OPENAI_HEADERS_ENV)
        refuse_ambient_credential(OPENAI_HEADERS_ENV, ambient, "Authorization")
        # The SDK also turns OPENAI_ORG_ID and OPENAI_PROJECT_ID into headers; an OpenAI
        # organization id is nothing a third-party endpoint should be sent.
        for name in (*ambient, "OpenAI-Organization", "OpenAI-Project"):
            extra_headers.setdefault(name, openai.Omit())
        if key is None:
            # A local server has nothing to check a key against, and a placeholder bearer
            # is a string someone will one day mistake for a key.
            extra_headers["Authorization"] = openai.Omit()
        kwargs["extra_body"] = extra_body or None
        kwargs["extra_headers"] = extra_headers
        # Every reply streams, and its last chunk carries the usage, so the
        # usage capture reads the provider's counts as it did a whole reply's.
        kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    # -- the call -----------------------------------------------------------

    def send(self, request: ChatRequest) -> Completion:
        return self._forcing.send(request, lambda forced: self._send(request, forced))

    def _send(self, request: ChatRequest, forced: bool) -> Completion:
        key = api_key(self.endpoint)
        sdk = self._sdk(key)
        kwargs = self._build(request, key, forced=forced)
        started = time.monotonic()
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        # The SDK's overloads type each keyword; the request is built as one dict so the
        # byte test can see exactly what was handed over.
        stream = cast(Callable[..., Any], sdk.chat.completions.stream)

        def attempt(n: int) -> Completion:
            try:
                read = read_stream(
                    lambda: stream(**kwargs),
                    pieces=_pieces,
                    final=_snapshot,
                    silences=self._silences,
                    cancel=request.cancel,
                    live=request.live,
                )
            except StreamEnded as ended:
                raise failed_attempt(ended, n, OPENAI_ERRORS, url) from ended
            parsed = self._parse(read.final, request, started)
            if parsed is None:
                raise Retry(
                    "malformed JSON or missing choices",
                    "malformed",
                    billed=read.streamed_tokens,
                )
            return parsed

        return run_ladder(request, attempt, self._retry)

    def _parse(
        self, payload: object, request: ChatRequest, started: float
    ) -> Completion | None:
        """Build a Completion from the stream's accumulated response, as the provider sent it
        (`_snapshot`), or None to signal one more internal retry: a stream that carried no
        event, or no choice."""
        if not isinstance(payload, dict):
            return None
        payload = cast("dict[str, Any]", payload)
        choices = _arr(payload.get("choices"))
        if not choices:
            return None
        choice = _obj(choices[0])
        message = _obj(choice.get("message"))
        raw_text = message.get("content")
        text = raw_text if isinstance(raw_text, str) else ""
        grammars = _grammars(request)
        tool_calls = tuple(
            _tool_call(_obj(entry), grammars) for entry in _arr(message.get("tool_calls"))
        )
        finish = choice.get("finish_reason") or choice.get("stop_reason")
        usage, _estimated = probe_usage(payload, prompt_text(request), text)
        return completion(
            text=text,
            tool_calls=tool_calls,
            raw_stop=finish if isinstance(finish, str) else None,
            usage=usage,
            started=started,
        )
