"""Termination is read from the provider's signal, never from response text."""

from __future__ import annotations

import httpx2
import pytest
from client_helpers import FakeEndpoint, ok_body, scripted

from chickadee.client import OpenAIClient, map_stop_reason, to_request
from chickadee.schema import Prompt, RetryParams, StopReason

PROMPT = Prompt(prefix="P", volatile="V")
PARAMS = RetryParams(temperature=0.2, max_tokens=512)
# The OpenAI schema types `function.arguments` as a JSON-encoded **string**. Using the
# inline-object form here — which some providers also send — is what let the string case
# go untested while every standards-compliant tool call arrived with its inputs stripped.
TOOL = [{"id": "c1", "function": {"name": "read", "arguments": '{"path": "a.py"}'}}]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("stop", StopReason.END_TURN),
        ("end_turn", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("model_context_window_exceeded", StopReason.MAX_TOKENS),
        ("tool_calls", StopReason.TOOL_USE),
        ("content_filter", StopReason.REFUSAL),
        ("something_new", StopReason.OTHER),
        (None, StopReason.OTHER),
    ],
)
def test_stop_reason_mapping(raw: str | None, expected: StopReason) -> None:
    assert map_stop_reason(raw) is expected


def _complete(**kw: object):
    transport, _ = scripted(httpx2.Response(200, json=ok_body(**kw)))  # type: ignore[arg-type]
    return OpenAIClient(FakeEndpoint(), transport=transport).send(to_request(PROMPT, PARAMS))


def test_text_alongside_tool_calls_is_not_terminal() -> None:
    """The documented anti-pattern: a model says "I'll check the imports first" *and*
    requests a read in the same turn. Ending on text presence truncates work mid-flight."""
    c = _complete(
        text="I'll check the imports first", finish_reason="tool_calls", tool_calls=TOOL
    )
    assert c.text
    assert c.tool_calls
    assert c.is_terminal() is False


def test_provider_saying_stop_while_returning_tools_is_corrected() -> None:
    """Some providers report `stop` with tool calls attached. The turn continues; taking
    the word over the evidence would drop the request."""
    c = _complete(text="", finish_reason="stop", tool_calls=TOOL)
    assert c.stop_reason is StopReason.TOOL_USE
    assert c.is_terminal() is False


def test_tool_use_without_calls_degrades_to_other() -> None:
    c = _complete(text="hmm", finish_reason="tool_calls", tool_calls=None)
    assert c.stop_reason is StopReason.OTHER
    assert c.raw_stop_reason == "tool_calls"


def test_max_tokens_is_never_terminal() -> None:
    """Truncated mid-response; it becomes ParseError(truncated) upstream, never a
    terminal parse."""
    c = _complete(text="def f(", finish_reason="length")
    assert c.stop_reason is StopReason.MAX_TOKENS
    assert c.is_terminal() is False


def test_refusal_is_terminal_evidence_not_a_parse_failure() -> None:
    c = _complete(text="I can't help with that", finish_reason="content_filter")
    assert c.stop_reason is StopReason.REFUSAL
    assert c.is_terminal() is True


def test_empty_text_with_end_turn_is_terminal() -> None:
    """Termination does not depend on there being text — an empty terminal response is a
    parse problem for the adapter, not a reason to keep the turn open."""
    c = _complete(text="", finish_reason="stop")
    assert c.is_terminal() is True


def test_raw_stop_reason_is_preserved() -> None:
    c = _complete(finish_reason="eos_token_reached")
    assert c.stop_reason is StopReason.OTHER
    assert c.raw_stop_reason == "eos_token_reached"
