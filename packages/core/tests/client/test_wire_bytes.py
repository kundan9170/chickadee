"""The stage-4 byte invariant, per transport: the messages and
tools a `ChatRequest` carries are the bytes the vendor's SDK puts on the wire. An SDK upgrade
that re-serializes fails here instead of showing up as a larger bill."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx2
import pytest
from client_helpers import FakeEndpoint, anthropic_body, ok_body, scripted

from chickadee.client import (
    AnthropicClient,
    ChatRequest,
    ImagePart,
    Message,
    OpenAIClient,
    TextPart,
    ToolSchema,
)
from chickadee.client.anthropic_client import (
    anthropic_messages,
    anthropic_system,
    anthropic_tools,
)
from chickadee.client.openai_client import EndpointSpec, openai_messages, openai_tools
from chickadee.config import RungEndpoint
from chickadee.schema import RetryParams, ToolCall

PARAMS = RetryParams(temperature=0.3, max_tokens=512, stop=("</patch>",))
TOOLS = (
    ToolSchema(
        "read",
        "Read a file — whole, or a line span.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    ToolSchema(
        "symbol", "Look a symbol up in the graph.", {"type": "object", "properties": {}}
    ),
)


def _conversation(*, tools: tuple[ToolSchema, ...] = ()) -> ChatRequest:
    """Every message shape: a system prefix, user text with non-ASCII bytes, an assistant
    turn with a call, the tool's answer, and the next user turn."""
    return ChatRequest(
        messages=(
            Message(role="system", content="You are Zeus.\n  indent ✓ «quotes» \t tab\n"),
            Message(role="user", content="Fix `parse()` — it drops é and 𝄞.\n"),
            Message(
                role="assistant",
                content="Reading it first.",
                tool_calls=(ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),),
            ),
            Message(
                role="tool", content="def parse():\n    return None\n", tool_call_id="call_1"
            ),
            Message(role="user", content="and the tests?"),
        ),
        params=PARAMS,
        tools=tools,
    )


def _compact(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _sent_openai(req: ChatRequest, endpoint: EndpointSpec | None = None) -> bytes:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    OpenAIClient(endpoint or FakeEndpoint(), transport=transport).send(req)
    return seen[0].content


def _sent_anthropic(req: ChatRequest, endpoint: EndpointSpec | None = None) -> bytes:
    transport, seen = scripted(httpx2.Response(200, json=anthropic_body()))
    ep = endpoint or FakeEndpoint(api_key_env="CHICKADEE_TEST_ANTHROPIC_KEY")
    AnthropicClient(ep, transport=transport).send(req)
    return seen[0].content


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHICKADEE_TEST_ANTHROPIC_KEY", "sk-ant-test")


# --- OpenAI-compatible ----------------------------------------------------------------------


@pytest.mark.parametrize("tools", [(), TOOLS])
def test_openai_messages_and_tools_are_the_bytes_the_sdk_sends(
    tools: tuple[ToolSchema, ...],
) -> None:
    req = _conversation(tools=tools)
    raw = _sent_openai(req)
    body = json.loads(raw)
    assert body["messages"] == openai_messages(req)
    assert _compact(openai_messages(req)) in raw, "the SDK re-serialized the messages"
    if tools:
        assert body["tools"] == openai_tools(req)
        assert _compact(openai_tools(req)) in raw, "the SDK re-serialized the tools"
    else:
        assert "tools" not in body, "`tools` rides iff the request declared one"


def test_openai_tools_stay_off_the_wire_when_the_rung_has_no_native_tools() -> None:
    body = json.loads(
        _sent_openai(_conversation(tools=TOOLS), FakeEndpoint(native_tools=False))
    )
    assert "tools" not in body


def test_openai_sends_no_authorization_without_a_key() -> None:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    OpenAIClient(FakeEndpoint(), transport=transport).send(_conversation())
    assert "authorization" not in seen[0].headers


def test_openai_rung_without_temperature_never_receives_it() -> None:
    body = json.loads(_sent_openai(_conversation(), FakeEndpoint(supports_temperature=False)))
    assert "temperature" not in body


# --- Anthropic native -----------------------------------------------------------------------


@pytest.mark.parametrize("tools", [(), TOOLS])
def test_anthropic_system_messages_and_tools_are_the_bytes_the_sdk_sends(
    tools: tuple[ToolSchema, ...],
) -> None:
    req = _conversation(tools=tools)
    raw = _sent_anthropic(req)
    body = json.loads(raw)
    assert body["system"] == anthropic_system(req)
    assert body["messages"] == anthropic_messages(req)
    assert _compact(anthropic_messages(req)) in raw, "the SDK re-serialized the messages"
    assert _compact(anthropic_system(req)) in raw
    if tools:
        assert body["tools"] == anthropic_tools(req)
        assert _compact(anthropic_tools(req)) in raw
    else:
        assert "tools" not in body


def test_anthropic_wire_shape_alternates_and_carries_the_call() -> None:
    wire = anthropic_messages(_conversation())
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read",
        "input": {"path": "a.py"},
    }
    # The tool's answer and the next user text are one user turn on this API.
    assert [b["type"] for b in wire[2]["content"]] == ["tool_result", "text"]
    assert wire[2]["content"][0]["tool_use_id"] == "call_1"


def test_anthropic_places_exactly_the_markers_stage_4_declared() -> None:
    req = ChatRequest(
        messages=(
            Message(role="system", content="frozen", cache_breakpoint=True),
            Message(role="user", content="per session", cache_breakpoint=True),
            Message(role="assistant", content="ok"),
            Message(role="user", content="per turn"),
        ),
        params=PARAMS,
    )
    body = json.loads(_sent_anthropic(req))
    assert body["system"] == [
        {"type": "text", "text": "frozen", "cache_control": {"type": "ephemeral"}}
    ]
    marked = [b for m in body["messages"] for b in m["content"] if "cache_control" in b]
    assert marked == [
        {"type": "text", "text": "per session", "cache_control": {"type": "ephemeral"}}
    ]


def test_anthropic_places_no_marker_the_request_did_not_declare() -> None:
    raw = _sent_anthropic(_conversation(tools=TOOLS))
    assert b"cache_control" not in raw


def test_a_fifth_marker_is_refused_not_dropped() -> None:
    req = ChatRequest(
        messages=tuple(
            Message(
                role="user" if i % 2 == 0 else "assistant",
                content=str(i),
                cache_breakpoint=True,
            )
            for i in range(5)
        ),
        params=PARAMS,
    )
    with pytest.raises(ValueError, match="at most 4"):
        _sent_anthropic(req)


def test_openai_ignores_markers_its_providers_cache_automatically() -> None:
    req = ChatRequest(
        messages=(Message(role="system", content="s", cache_breakpoint=True),),
        params=PARAMS,
    )
    assert b"cache_control" not in _sent_openai(req)


def test_openrouter_spells_the_declared_markers_and_only_those() -> None:
    """OpenRouter caches `EXPLICIT`ly on this wire: a marked message's last part
    carries `cache_control`, and an unmarked message keeps its plain-string bytes."""
    req = ChatRequest(
        messages=(
            Message(role="system", content="frozen", cache_breakpoint=True),
            Message(role="user", content="per turn"),
        ),
        params=PARAMS,
    )
    raw = _sent_openai(req, FakeEndpoint(provider="openrouter"))
    body = json.loads(raw)
    assert body["messages"] == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "frozen", "cache_control": {"type": "ephemeral"}}
            ],
        },
        {"role": "user", "content": "per turn"},
    ]
    assert _compact(openai_messages(req, explicit=True)) in raw


def test_openrouter_refuses_a_fifth_marker() -> None:
    req = ChatRequest(
        messages=tuple(
            Message(role="user", content=str(i), cache_breakpoint=True) for i in range(5)
        ),
        params=PARAMS,
    )
    with pytest.raises(ValueError, match="at most 4"):
        _sent_openai(req, FakeEndpoint(provider="openrouter"))


def test_anthropic_sends_no_empty_text_block() -> None:
    """The API rejects an empty text block with a 400; an empty message carries no bytes."""
    req = ChatRequest(
        messages=(
            Message(role="system", content=""),
            Message(role="user", content="task"),
            Message(role="assistant", content=""),
            Message(role="user", content="more"),
        ),
        params=PARAMS,
    )
    body = json.loads(_sent_anthropic(req))
    assert "system" not in body
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "task"}, {"type": "text", "text": "more"}],
        }
    ]


# --- D110: images ---------------------------------------------------------------------------

PNG = ImagePart(media_type="image/png", sha="4c1a9e0b77f2", data=b"\x89PNG fake")
PNG_B64 = base64.b64encode(PNG.data).decode()


def _looked() -> ChatRequest:
    """Zeus reads a screenshot through a tool, then a second tool answers beside it."""
    return ChatRequest(
        messages=(
            Message(role="system", content="You are Zeus."),
            Message(role="user", content="How does the page look?"),
            Message(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="read", arguments={"path": "shot.png"}),
                    ToolCall(id="c2", name="read", arguments={"path": "a.css"}),
                ),
            ),
            Message(role="tool", content=(TextPart("shot.png:"), PNG), tool_call_id="c1"),
            Message(role="tool", content="body { margin: 0 }", tool_call_id="c2"),
            Message(role="assistant", content="The margin is gone."),
        ),
        params=PARAMS,
    )


def test_openai_image_compensation_has_one_shape() -> None:
    """D110's compensation, pinned: the tool result names the image, and the pixels ride in
    a user message after the run of results — never between two results, which this wire
    requires to be contiguous — and before the next assistant turn."""
    req = _looked()
    raw = _sent_openai(req)
    wire = json.loads(raw)["messages"]
    assert [m["role"] for m in wire] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ]
    assert wire[3]["content"] == [
        {"type": "text", "text": "shot.png:"},
        {"type": "text", "text": "[image 4c1a9e0b — attached below]"},
    ]
    assert wire[4]["content"] == "body { margin: 0 }", "a text-only result keeps its bytes"
    assert wire[5]["content"] == [
        {"type": "text", "text": "[image 4c1a9e0b]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
    ]
    assert _compact(openai_messages(req)) in raw


def test_openai_user_image_is_an_image_part() -> None:
    req = ChatRequest(
        messages=(Message(role="user", content=(TextPart("this mockup"), PNG)),), params=PARAMS
    )
    wire = json.loads(_sent_openai(req))["messages"]
    assert wire == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "this mockup"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
            ],
        }
    ]


def test_anthropic_image_rides_inside_the_tool_result() -> None:
    req = _looked()
    raw = _sent_anthropic(req)
    wire = json.loads(raw)["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user", "assistant"]
    result = wire[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "c1"
    assert result["content"] == [
        {"type": "text", "text": "shot.png:"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
        },
    ]
    assert _compact(anthropic_messages(req)) in raw


def test_the_client_reads_supports_images_from_the_models_own_entry() -> None:
    """The field the config declares, not a fallback for a missing attribute."""
    blind = RungEndpoint(
        base_url="https://api.example.test/v1",
        model_id="m",
        context_window=65_536,
        max_output_tokens=16_384,
        supports_images=False,
    )
    with pytest.raises(ValueError, match="supports_images"):
        _sent_openai(_looked(), blind)
    _sent_openai(_looked(), blind.model_copy(update={"supports_images": True}))


def test_an_image_on_a_rung_that_cannot_see_is_refused() -> None:
    with pytest.raises(ValueError, match="supports_images"):
        _sent_openai(_looked(), FakeEndpoint(supports_images=False))
    with pytest.raises(ValueError, match="supports_images"):
        _sent_anthropic(
            _looked(),
            FakeEndpoint(api_key_env="CHICKADEE_TEST_ANTHROPIC_KEY", supports_images=False),
        )
