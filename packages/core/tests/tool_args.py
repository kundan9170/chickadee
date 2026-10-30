"""A tool's arguments as `validate` passes them, for tests that call a tool directly.

Since D151 a tool takes the `Arguments` its schema check returned, never a dict, so a test
that hands a tool its arguments goes through the same check the loop runs first. Arguments
the schema refuses fail here, in the test's setup, rather than reaching the tool.
`zeus_answer` then answers such a call as Zeus's loop does once it is admitted.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import cast

from chickadee.context.recall import Recall
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session, ToolLine
from chickadee.engine import NonInteractive
from chickadee.roles import role
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import FileView, HandleStore, Locator
from chickadee.tools.rules import Pending
from chickadee.tools.spec import spec
from chickadee.tools.tool import Invocation
from chickadee.tools.validate import Arguments, validate
from chickadee.zeus.calls import ResponseFacts, ZeusSink, ZeusTools, zeus_dispatcher
from chickadee.zeus.loop import LoopConfig


def checked(tool: str, args: Mapping[str, object]) -> Arguments:
    """`args` for `tool` on Zeus's surface, valid against its full schema."""
    found = spec(tool)
    assert found is not None, f"no tool {tool!r} on the surface"
    result = validate(found, args)
    assert isinstance(result, Arguments), result
    return result


def zeus_answer(
    session: Session,
    reader: FileView,
    call_id: str,
    tool: str,
    args: Arguments,
    *,
    handles: HandleStore | None = None,
    recall: Recall | None = None,
    locator: Locator | None = None,
    cancel: threading.Event | None = None,
) -> ToolLine:
    """Answer one call the session already records through Zeus's own dispatcher, sink and
    built-in tool — what the loop does once the call is admitted — and return its line.

    `args` are handed on as given: a test that builds them directly, past `validate`, reaches
    the reader's own refusal behind the schema. Each answer gets a dispatcher of its own, so
    answers may be given from several threads at once, as the read path's own test does."""
    policy = role("zeus").tool_defaults
    assert policy is not None
    tools = ZeusTools(
        session,
        reader,
        handles if handles is not None else cast("HandleStore", reader),
        recall if recall is not None else Recall(session.root, ReplayCache(session.root)),
        NonInteractive(),
        locator=locator,
    )
    dispatcher = zeus_dispatcher(
        tools,
        ZeusSink(session),
        ResponseFacts(),
        policy=policy,
        offered=[s.name for s in policy.surface(BUILTINS)],
        limits=LoopConfig(max_result_tokens=0).limits(),
        generation=lambda _state: 0,
        cancel=cancel,
        executors={},
    )
    found = BUILTINS.get(tool)
    assert found is not None, f"no tool {tool!r} is registered"
    at = session.state.last_seq
    pending = Pending(
        call=Invocation(call_id, call_id, at),
        sent=tool,
        raw=args.sent,
        tool=tool,
        spec=found.spec,
        arguments=args,
    )
    return dispatcher.execute(pending)
