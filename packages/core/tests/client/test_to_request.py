"""`to_request`: the frozen `Prompt` rendered into one system and one user message,
content byte-identical; `tools=()` by default; the coder's four when the caller passes them."""

from __future__ import annotations

import pytest

from chickadee.client import ChatRequest, ImagePart, Message, TextPart, ToolSchema, to_request
from chickadee.client.request import prompt_text, text_of
from chickadee.schema import Prompt, RetryParams, ToolCall

PARAMS = RetryParams(temperature=0.2, max_tokens=2048)
CODER_TOOLS = tuple(
    ToolSchema(name, f"{name} (coder)", {"type": "object", "properties": {}})
    for name in ("read", "search", "symbol", "verifier_query")
)


def test_one_system_and_one_user_message_byte_identical() -> None:
    prompt = Prompt(prefix="SYSTEM ✓\r\n  trailing  \n", volatile="TASK t1\n\x00é\n")
    req = to_request(prompt, PARAMS)
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.messages[0].content == prompt.prefix
    assert req.messages[1].content == prompt.volatile
    assert text_of(req.messages[0]).encode() + text_of(req.messages[1]).encode() == (
        prompt.full().encode()
    )


def test_no_tools_by_default() -> None:
    req = to_request(Prompt(prefix="p", volatile="v"), PARAMS)
    assert req.tools == ()
    assert req.session_key is None and req.cancel is None
    assert req.params is PARAMS


def test_the_coders_four_when_the_caller_passes_them() -> None:
    req = to_request(
        Prompt(prefix="p", volatile="v"), PARAMS, tools=CODER_TOOLS, session_key="s1"
    )
    assert [t.name for t in req.tools] == ["read", "search", "symbol", "verifier_query"]
    assert req.session_key == "s1"


def test_no_cache_marker_is_placed_by_the_rendering() -> None:
    """Stage 4 places markers; `to_request` is stage 1 for the coder."""
    req = to_request(Prompt(prefix="p", volatile="v"), PARAMS)
    assert not any(m.cache_breakpoint for m in req.messages)


def test_message_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="x")
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="user", content="x", tool_call_id="c1")
    with pytest.raises(ValueError, match="assistant"):
        Message(role="user", content="x", tool_calls=(ToolCall(id="c", name="read"),))


def test_a_request_is_not_empty_and_names_each_tool_once() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ChatRequest(messages=(), params=PARAMS)
    with pytest.raises(ValueError, match="repeat"):
        ChatRequest(
            messages=(Message(role="user", content="x"),),
            params=PARAMS,
            tools=(CODER_TOOLS[0], CODER_TOOLS[0]),
        )


# --- D110: content parts --------------------------------------------------------------------

PNG = ImagePart(media_type="image/png", sha="4c1a9e0b", data=b"png")


def test_a_user_or_tool_message_carries_an_image() -> None:
    Message(role="user", content=(TextPart("look"), PNG))
    Message(role="tool", content=(PNG,), tool_call_id="c1")


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_a_system_or_assistant_message_carries_no_image(role: str) -> None:
    with pytest.raises(ValueError, match="no image"):
        Message(role=role, content=(PNG,))  # type: ignore[arg-type]


def test_parts_content_is_never_empty() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        Message(role="user", content=())


def test_an_image_type_neither_transport_takes_is_refused() -> None:
    with pytest.raises(ValueError, match="image/tiff"):
        ImagePart(media_type="image/tiff", sha="x", data=b"")


def test_the_estimator_reads_text_parts_only() -> None:
    req = ChatRequest(
        messages=(Message(role="user", content=(TextPart("ab"), PNG, TextPart("cd"))),),
        params=PARAMS,
    )
    assert prompt_text(req) == "abcd"
