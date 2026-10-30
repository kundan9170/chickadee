"""Contract 5 — the model call's value types. `ModelClient` retired from here to
`client/request.py`; the `ModelAdapter` protocol, its `ToolRequest` and `ParseResult`
retired with the packet loop that parsed a coder's patch.

The model-agnostic seam. Two rules here carry most of the weight:

Nothing that reads model text raises on it. Every malformed output is a `ParseError`
*value*, so a refusal is data the role is told, never a harness crash.

Turn termination is read from `stop_reason` and `tool_calls`, never from response text.
Text and tool-call content coexist in one response — a model routinely says "I'll
check the imports first" *and* requests a read in the same turn — so a loop that ends when
text appears kills the turn mid-work.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from chickadee.schema.ids import sha256_hex

WIRE = ConfigDict(extra="forbid", frozen=True)


class Prompt(BaseModel):
    model_config = WIRE

    prefix: str
    """Byte-stable across calls within a run (invariant 8), so a provider's cache holds it."""
    volatile: str
    """What changes every call."""

    def full(self) -> str:
        return self.prefix + self.volatile

    def prefix_sha(self) -> str:
        return sha256_hex(self.prefix)


class ParseError(BaseModel):
    model_config = WIRE

    kind: Literal["no_block", "multiple_blocks", "truncated", "bad_format"]
    hint: str
    """Goes into the format-nudge template, so it must read as an instruction to a model,
    not as an exception message to a developer."""


class ReasoningSpec(BaseModel):
    """Provider-neutral; the client maps it per endpoint."""

    model_config = WIRE

    level: Literal["minimal", "low", "medium", "high", "max"]
    budget_tokens: int | None = None


class RetryParams(BaseModel):
    model_config = WIRE

    temperature: float
    """Same-or-higher on retry, never lower. A retry exists because the last sample
    failed; lowering temperature makes the next one more like it."""
    max_tokens: int
    stop: tuple[str, ...] = ()
    reasoning: ReasoningSpec | None = None
    """Per-rung from config, never chosen by a model mid-run, and never a retry lever
    — escalating reasoning on retry would confound the failing-set-stability signal
    the classifier reads. A provider without a reasoning parameter drops it, and the drop
    is recorded rather than silently ignored."""


class CacheMode(StrEnum):
    """How prompt caching is *caused* at an endpoint.

    Distinct from whether it can be *observed* — a provider may cache automatically and
    report nothing, which is why `CacheObs.warm` is tri-state rather than a bool.
    """

    EXPLICIT = "explicit"
    """We mark the prefix. The provider caches exactly what we mark and nothing else
    (Anthropic native, Bedrock, OpenRouter). Placing no marker means caching no bytes."""

    AUTOMATIC = "automatic"
    """The provider matches prefixes itself. Our only duty is keeping them byte-stable
    (OpenAI-compatible hosted, llama.cpp, vLLM). A marker would have nowhere to go."""

    NONE = "none"
    """No prompt cache here, *declared*. The default for an unrecognized provider, for the
    same reason `AffinityTable.row` injects nothing rather than guessing a field name: an
    assumed cache that does not exist is a cost model quietly measuring the wrong thing."""


class Usage(BaseModel):
    model_config = WIRE

    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int | None = None
    """None = the provider does not report it. Distinct from 0, which means a genuine cache
    miss — collapsing them would corrupt the cache-hit metric."""
    estimated: bool = False
    """True when the server reported no usage and these are client-side estimates.
    Cost derived from estimated tokens is not the same evidence as
    cost derived from reported ones, and a number that cannot tell you which it is has no
    business in an economics claim."""
    cache_write_tokens: int | None = None
    """None = the provider does not report it. Distinct from 0, which means no cache was
    created this call — collapsing them would corrupt the cache metric, exactly as
    `cached_prompt_tokens`'s docstring says of reads. Cache CREATION, priced separately by
    Anthropic-style providers."""


class CacheObs(BaseModel):
    """What one response told us about the cache.

    `should_compact` consumes this; compaction triggers
    on a cache-cold *transition* rather than on size, so the caller compares consecutive
    observations. This type is one observation, never the comparison.
    """

    model_config = WIRE

    mode: CacheMode
    read_tokens: int | None = None
    """Served from cache. None = the provider does not report it — NOT zero. Carried
    through from `Usage.cached_prompt_tokens`, which draws the same distinction."""
    prompt_tokens: int = 0

    @property
    def warm(self) -> bool | None:
        """True / False / **None = unknowable**.

        The third state is the point. Under `NONE`, or wherever the provider reports
        nothing, we cannot distinguish a cold cache from an absent one — and invariant 20
        forbids returning a negative as a fact when coverage is not total. A `False` here
        would let the compaction trigger read "no cache" as "cache went cold" and compact on a
        transition
        that never happened.
        """
        if self.mode is CacheMode.NONE or self.read_tokens is None:
            return None
        return self.read_tokens > 0

    @classmethod
    def from_usage(cls, usage: Usage, mode: CacheMode) -> CacheObs:
        return cls(
            mode=mode,
            read_tokens=usage.cached_prompt_tokens,
            prompt_tokens=usage.prompt_tokens,
        )


class ToolCall(BaseModel):
    """One requested tool invocation within a turn."""

    model_config = WIRE

    id: str
    name: str
    arguments: dict[str, object] = {}


CallForm = Literal["native", "tool_line", "need", "json", "json_fenced"]
"""How a call was written: a native `tool_calls` entry, a `TOOL` line, the `NEED:` alias,
or a JSON object, bare or in one fence. Its journal line records it, so a model
calling in a form its protocol did not show it is counted."""


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"
    """A provider value we do not map, recorded verbatim in `raw_stop_reason`. The loop
    then falls back to tool_calls emptiness alone — recorded, so the weaker signal shows up
    in telemetry rather than being silently assumed."""


class Completion(BaseModel):
    model_config = WIRE

    text: str
    usage: Usage
    latency_ms: int
    cached_prefix: bool | None = None
    stop_reason: StopReason
    raw_stop_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode="after")
    def _termination_is_coherent(self) -> Completion:
        # The termination table, as an assertion. TOOL_USE with nothing requested is a provider
        # or
        # adapter bug that would otherwise surface much later as an empty turn.
        if self.stop_reason is StopReason.TOOL_USE and not self.tool_calls:
            raise ValueError("stop_reason=tool_use requires at least one tool_call")
        if self.stop_reason is StopReason.END_TURN and self.tool_calls:
            raise ValueError(
                "stop_reason=end_turn with tool_calls is contradictory; a turn requesting "
                "tools is not terminal (D47)"
            )
        return self

    def is_terminal(self) -> bool:
        """Whether this response should be parsed as the rung's output.

        MAX_TOKENS is deliberately not terminal: it is truncated mid-response and becomes
        `ParseError(truncated)`, never a terminal parse.
        """
        if self.tool_calls:
            return False
        return self.stop_reason in (StopReason.END_TURN, StopReason.REFUSAL)
