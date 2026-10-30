"""A PDF sent whole is a document part on both transports, the bytes each SDK
puts on the wire, and a model without `supports_documents` is never sent one.
"""

from __future__ import annotations

import base64
import json

import httpx2
import pytest
from client_helpers import FakeEndpoint, anthropic_body, ok_body, scripted

from chickadee.client import AnthropicClient, ChatRequest, Message, OpenAIClient, TextPart
from chickadee.client.anthropic_client import anthropic_messages
from chickadee.client.openai_client import openai_messages
from chickadee.client.request import DocumentPart, ImagePart, has_documents
from chickadee.schema import RetryParams

PARAMS = RetryParams(temperature=0.3, max_tokens=64, stop=())
PDF = b"%PDF-1.4 the bytes do not matter to a transport"
DOC = DocumentPart("application/pdf", "ab" * 32, "spec.pdf", PDF)
B64 = base64.b64encode(PDF).decode()


def request(*messages: Message) -> ChatRequest:
    return ChatRequest(messages=messages, params=PARAMS)


def attached() -> ChatRequest:
    return request(Message("user", (TextPart("[turn 1]\nread this"), DOC)))


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHICKADEE_TEST_ANTHROPIC_KEY", "sk-ant-test")


def test_openai_sends_a_file_part_by_data_url_and_filename() -> None:
    (user,) = openai_messages(attached())
    assert user["content"][1] == {
        "type": "file",
        "file": {"filename": "spec.pdf", "file_data": f"data:application/pdf;base64,{B64}"},
    }


def test_anthropic_sends_a_document_block() -> None:
    (user,) = anthropic_messages(attached())
    assert user["content"][1] == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": B64},
        "title": "spec.pdf",
    }


def test_openai_moves_a_tool_results_document_after_the_results() -> None:
    from chickadee.schema import ToolCall

    req = request(
        Message("user", "go"),
        Message(
            "assistant",
            "",
            tool_calls=(ToolCall(id="c1", name="read", arguments={"path": "spec.pdf"}),),
        ),
        Message("tool", (TextPart("spec.pdf — application/pdf"), DOC), tool_call_id="c1"),
    )
    *_, tool, after = openai_messages(req)
    assert tool["content"][1] == {
        "type": "text",
        "text": f"[document {DOC.sha[:8]} — attached below]",
    }
    assert after["role"] == "user"
    assert after["content"][1]["type"] == "file"


def test_a_document_rides_no_system_or_assistant_message() -> None:
    with pytest.raises(ValueError, match="no image or document"):
        Message("assistant", (DOC,))
    with pytest.raises(ValueError, match="not a document"):
        DocumentPart("image/png", "s", "x.png", b"")


def test_a_model_without_supports_documents_is_never_sent_one() -> None:
    assert has_documents(attached())
    transport, _seen = scripted(httpx2.Response(200, json=ok_body()))
    with pytest.raises(ValueError, match="supports_documents"):
        OpenAIClient(FakeEndpoint(), transport=transport).send(attached())
    transport, _seen = scripted(httpx2.Response(200, json=anthropic_body()))
    blind = FakeEndpoint(api_key_env="CHICKADEE_TEST_ANTHROPIC_KEY")
    with pytest.raises(ValueError, match="supports_documents"):
        AnthropicClient(blind, transport=transport).send(attached())


def test_a_model_that_reads_pdfs_is_sent_the_bytes() -> None:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    OpenAIClient(FakeEndpoint(supports_documents=True), transport=transport).send(attached())
    body = json.loads(seen[0].content)
    assert body["messages"][0]["content"][1]["file"]["file_data"].endswith(B64)


def test_a_part_round_trips_through_the_box_conversation() -> None:
    from chickadee.boxctx.conversation import message_bytes, message_of

    m = Message("user", (TextPart("t"), DOC, ImagePart("image/png", "cd" * 32, b"\x89PNG")))
    assert message_of(message_bytes(m)) == m
