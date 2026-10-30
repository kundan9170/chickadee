"""`est_tools` is the one measure of a request's `tools` array: each tool's name and
schema weighed as a call is, its description, and a constrained tool's grammar. The per-call
estimate, P-9's prefix term, the load's fit and Zeus's prefix all take it."""

from __future__ import annotations

from context_helpers import png

from chickadee.client.request import (
    ChatRequest,
    ImagePart,
    Message,
    TextPart,
    ToolGrammar,
    ToolSchema,
)
from chickadee.context.outline import image_est
from chickadee.context.request import est_call, est_text, est_tools, request_tokens
from chickadee.schema import RetryParams

SCHEMA = {"type": "object", "properties": {"patch": {"type": "string"}}}
PLAIN = ToolSchema(name="apply_patch", description="Apply a V4A patch.", schema=SCHEMA)
GRAMMAR = ToolGrammar(argument="patch", definition="start: LINE+\nLINE: /[^\\n]+\\n/\n")


def test_a_tool_weighs_its_name_schema_and_description() -> None:
    assert est_tools([PLAIN]) == est_call(PLAIN.name, SCHEMA) + est_text(PLAIN.description)
    assert est_tools([]) == 0


def test_a_constrained_tool_also_weighs_its_grammar() -> None:
    # The grammar's definition is sent with the tool (OpenAI's custom tool).
    constrained = ToolSchema(
        name=PLAIN.name, description=PLAIN.description, schema=SCHEMA, grammar=GRAMMAR
    )
    assert est_tools([constrained]) == est_tools([PLAIN]) + est_text(GRAMMAR.definition)


def test_a_requests_estimate_counts_its_tools_by_the_same_measure() -> None:
    params = RetryParams(temperature=0.0, max_tokens=64)
    bare = ChatRequest(messages=(Message(role="user", content="go"),), params=params)
    with_tools = ChatRequest(messages=bare.messages, params=params, tools=(PLAIN,))
    assert request_tokens(with_tools) - request_tokens(bare) == est_tools([PLAIN])


def test_a_requests_estimate_counts_its_images_by_the_one_image_rule() -> None:
    """Was: an image counted nothing, so a failed box request was charged less for its
    screenshots than the same request from Zeus, whose estimate counts them (`assemble`)."""
    params = RetryParams(temperature=0.0, max_tokens=64)
    data = png(640, 480)
    words = Message(role="user", content=(TextPart("look"),))
    shown = Message(role="user", content=(TextPart("look"), ImagePart("image/png", "s", data)))
    plain = request_tokens(ChatRequest(messages=(words,), params=params))
    seen = request_tokens(ChatRequest(messages=(shown,), params=params))
    assert seen == plain + image_est(data)
    assert image_est(data) > 100
