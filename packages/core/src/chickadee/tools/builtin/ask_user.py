"""`ask_user`, a built-in.

The typed question channel (contract 6, `InteractionPort.ask`). Only a person's own answer
is the person's words: a default, or a reused answer, is `Defaulted`, recorded as not
run.

A thin run over the `user` port. What the call may touch is the port's to decide; whether it may
run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from typing import Final

from chickadee.context.results import CallCancelled
from chickadee.schema.interaction import AnswerSource, Question, QuestionKind
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Defaulted, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["DEFAULT_ANSWER", "SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "ask_user")

DEFAULT_ANSWER: Final = "no answer: proceed on the default the question states"
"""CL-6 requires every question to carry a default. `ask_user` takes one string, and the
template tells the model to state its default in it, so this is the choice that says so."""


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    user = required(ctx.user, "user")
    if ctx.cancel is not None and ctx.cancel.is_set():
        return CallCancelled(SPEC.name)
    question = Question(
        id=call.id,
        # The closed list has no row for "the model asks"; a question whose answer
        # changes what gets built is a clarification of the goal.
        kind=QuestionKind.CLARIFY_GOAL,
        text=args.text("question"),
        default=DEFAULT_ANSWER,
    )
    answer = user.ask(question, cancel=ctx.cancel)
    if ctx.cancel is not None and ctx.cancel.is_set():
        # A hard stop reached the question while it waited: whatever the port
        # returned to let go, nobody answered it.
        return CallCancelled(SPEC.name)
    if answer.source is AnswerSource.HUMAN:
        return Said(answer.note if answer.note is not None else answer.choice)
    return Defaulted(f"[{answer.source.value}] {answer.choice}")


TOOL = Tool(SPEC, frozenset({"user"}), run)
