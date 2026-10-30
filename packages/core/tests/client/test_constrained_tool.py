"""A grammar-constrained tool through the chosen transport: OpenAI's Chat
Completions custom tool. The request, the call and its echo are held to the shapes the `openai`
SDK types (`ChatCompletionCustomToolParam`, `ChatCompletionMessageCustomToolCall`), and one
constrained `apply_patch` call round-trips from the policy to the task worker's edit port over
a fake server. No live call is made here (the task's Log says why)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx2
import pytest
from client_helpers import FakeEndpoint, ok_body, scripted
from lark import Lark
from openai.types.chat import ChatCompletionCustomToolParam
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from pydantic import TypeAdapter
from refusals import stated
from repos import init_repo
from tools.tools_helpers import EditWorker, refused, said, worktree_edits

from chickadee.client import ChatRequest, Message, OpenAIClient
from chickadee.client.anthropic_client import anthropic_tools
from chickadee.client.openai_client import openai_messages
from chickadee.edits.formats import grammar
from chickadee.errors import HarnessBug
from chickadee.roles import TASK_WORKER
from chickadee.schema import RetryParams, ToolCall
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.policy import RolePolicy
from chickadee.tools.validate import resident

PARAMS = RetryParams(temperature=0.0, max_tokens=512)
PATCH = (
    "*** Begin Patch\n*** Update File: app.py\n-    return 1\n+    return 2\n*** End Patch\n"
)
APP = "def go() -> int:\n    return 1\n"


@dataclass(frozen=True)
class GrammarEndpoint(FakeEndpoint):
    grammar_tools: bool = True


def _policy(*, constrain: tuple[str, ...] = ("apply_patch",)) -> RolePolicy:
    base = TASK_WORKER
    return replace(base, enabled=(*base.enabled, "apply_patch"), constrain=constrain)


def _request(policy: RolePolicy, *messages: Message) -> ChatRequest:
    tools = resident(policy.surface(BUILTINS), {})
    return ChatRequest(
        messages=messages or (Message(role="user", content="fix go()"),),
        params=PARAMS,
        tools=tools,
    )


def _custom_call(patch: str = PATCH) -> dict[str, Any]:
    return {
        "id": "call_1",
        "type": "custom",
        "custom": {"name": "apply_patch", "input": patch},
    }


def _sent(seen: list[httpx2.Request]) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(seen[0].content)
    return body


# -- the request ------------------------------------------------------------------------------


def test_a_constrained_tool_is_sent_as_a_custom_tool_the_sdk_types() -> None:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    OpenAIClient(GrammarEndpoint(), transport=transport).send(_request(_policy()))
    tools = _sent(seen)["tools"]
    custom = [t for t in tools if t["type"] == "custom"]
    assert len(custom) == 1 and all(t["type"] == "function" for t in tools if t not in custom)
    TypeAdapter(ChatCompletionCustomToolParam).validate_python(custom[0])
    shape = custom[0]["custom"]
    assert shape["name"] == "apply_patch"
    assert shape["format"] == {
        "type": "grammar",
        "grammar": {"syntax": "lark", "definition": grammar("apply_patch")},
    }


def test_unconstrained_the_same_tool_is_a_function() -> None:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    OpenAIClient(GrammarEndpoint(), transport=transport).send(_request(_policy(constrain=())))
    assert all(t["type"] == "function" for t in _sent(seen)["tools"])


def test_anthropic_refuses_a_constrained_tool_rather_than_send_it_unconstrained() -> None:
    # T-3.135 review: sent as a plain tool, the constrained arm would be measured unconstrained.
    with pytest.raises(HarnessBug, match="no tool under a grammar"):
        anthropic_tools(_request(_policy()))
    assert anthropic_tools(_request(_policy(constrain=())))  # unconstrained, sent as before


def test_a_model_without_grammar_tools_is_never_sent_one() -> None:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    with pytest.raises(HarnessBug, match="no grammar_tools"):
        OpenAIClient(FakeEndpoint(), transport=transport).send(_request(_policy()))
    assert not seen


# -- the call, and its echo --------------------------------------------------------------------


def test_a_custom_call_is_read_as_the_tools_one_argument() -> None:
    ChatCompletionMessageCustomToolCall.model_validate(_custom_call())
    body = ok_body(text="", finish_reason="tool_calls", tool_calls=[_custom_call()])
    transport, _ = scripted(httpx2.Response(200, json=body))
    done = OpenAIClient(GrammarEndpoint(), transport=transport).send(_request(_policy()))
    assert done.tool_calls == (
        ToolCall(id="call_1", name="apply_patch", arguments={"patch": PATCH}),
    )


def test_a_custom_call_to_a_tool_not_constrained_has_no_arguments() -> None:
    body = ok_body(text="", finish_reason="tool_calls", tool_calls=[_custom_call()])
    transport, _ = scripted(httpx2.Response(200, json=body))
    done = OpenAIClient(GrammarEndpoint(), transport=transport).send(
        _request(_policy(constrain=()))
    )
    assert done.tool_calls[0].arguments == {}  # the dispatcher refuses it as invalid


def test_a_constrained_call_is_echoed_in_history_as_it_was_made() -> None:
    call = ToolCall(id="call_1", name="apply_patch", arguments={"patch": PATCH})
    request = _request(
        _policy(),
        Message(role="user", content="fix go()"),
        Message(role="assistant", content="", tool_calls=(call,)),
        Message(role="tool", content="ok: app.py", tool_call_id="call_1"),
    )
    echoed = openai_messages(request)[1]["tool_calls"][0]
    assert echoed == _custom_call()
    ChatCompletionMessageCustomToolCall.model_validate(echoed)


# -- the round trip: policy, request, call, the worker's draft ---------------------------------


def _tools(root: Path, policy: RolePolicy) -> EditWorker:
    """The task worker's tools under `policy` over a worktree holding `app.py`."""
    init_repo(root, {"app.py": APP})
    return EditWorker(worktree_edits(root), policy.enabled, constrain=policy.constrain)


def test_a_constrained_apply_patch_call_round_trips_to_the_worktree(tmp_path: Path) -> None:
    policy = _policy()
    body = ok_body(text="", finish_reason="tool_calls", tool_calls=[_custom_call()])
    transport, _ = scripted(httpx2.Response(200, json=body))
    (call,) = (
        OpenAIClient(GrammarEndpoint(), transport=transport).send(_request(policy)).tool_calls
    )
    # The model's text is in the grammar's language, as the provider decoded it.
    Lark(grammar("apply_patch"), parser="earley", lexer="dynamic").parse(PATCH)
    tools = _tools(tmp_path / "repo", policy)
    answer = tools.dispatcher.call(call)
    assert said(answer) == "ok: app.py\n    return 2"
    assert (tmp_path / "repo" / "app.py").read_text(encoding="utf-8") == APP.replace("1", "2")


def test_constrained_a_unified_diff_is_not_its_language(tmp_path: Path) -> None:
    diff = "--- a/app.py\n+++ b/app.py\n@@ -2 +2 @@\n-    return 1\n+    return 2\n"
    # Unconstrained, the model is shown `why` and sends it; constrained, the text is the call.
    call = ToolCall(id="c1", name="apply_patch", arguments={"patch": diff})
    denied = _tools(tmp_path / "a", _policy()).dispatcher.call(call)
    assert stated(refused(denied, "BAD_FORMAT")) == "BAD_FORMAT: the patch holds no V4A section"
    # Unconstrained, the same diff applies.
    said(_tools(tmp_path / "b", _policy(constrain=())).dispatcher.call(call))
