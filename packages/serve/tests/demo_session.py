"""The recorded session a frontend is built against (`--replay`), and how it is made.

`build(dest)` writes it through core's own writers, with the clock held to a fixed sequence, so
the bytes are the same on every machine and `test_serve_fixture.py` can hold the committed copy
to them: the fixture is regenerated from the code, never edited by hand, and cannot rot.

It holds one of each record a frontend must render: a user turn, an assistant turn calling a
tool, the tool's result, a question asked and answered, an episode, a second exchange, and an
undo taking that exchange back.

    uv run python packages/serve/tests/demo_session.py # rewrites tests/fixtures/demo
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from chickadee.context import store
from chickadee.context.lines import EpisodeLine, UndoLine
from chickadee.context.questions import (
    QuestionAnsweredLine,
    QuestionAskedLine,
    record_answered,
    record_asked,
)
from chickadee.context.rewind import record_rewind
from chickadee.context.session import Session
from chickadee.context.store import LOCK_NAME, sessions_root
from chickadee.schema import ToolCall
from chickadee.schema.interaction import Option, Question, QuestionKind
from chickadee.state.roots import Roots

SESSION_ID: Final = "demo"
QUESTION_ID: Final = "q-demo-1"
RUN_ID: Final = "2026-10-10-0123abcd"
FIXTURE: Final = Path(__file__).resolve().parent / "fixtures" / "demo"
START: Final = datetime(2026, 10, 10, 10, 0, 0, tzinfo=UTC)
STEP: Final = timedelta(seconds=2)


@contextlib.contextmanager
def _held_clock() -> Iterator[None]:
    ticks = (START + n * STEP for n in range(10_000))
    held = store._now  # pyright: ignore[reportPrivateUsage]
    store._now = lambda: next(ticks).strftime("%Y-%m-%dT%H:%M:%SZ")  # pyright: ignore[reportPrivateUsage]
    try:
        yield
    finally:
        store._now = held  # pyright: ignore[reportPrivateUsage]


def roots(state: Path) -> Roots:
    return Roots(project=state, repo_local=state, checkout_local=state, user=state)


def build(dest: Path) -> Path:
    """Write the demo session under `dest` (a repository-local state directory) and return
    `dest`. Anything already there is replaced."""
    shutil.rmtree(dest, ignore_errors=True)
    root = sessions_root(roots(dest))
    root.mkdir(parents=True)
    with _held_clock():
        session = Session.create(root, SESSION_ID)
        try:
            session.record_user("Where is the retry limit set?")
            call = ToolCall(id="call-1", name="grep_in", arguments={"pattern": "max_retries"})
            # No purpose line on the call.
            session.record_assistant("Searching for it.", (call,))
            session.record_tool("call-1", "src/client.py:41: max_retries = 3")
            question = Question(
                id=QUESTION_ID,
                kind=QuestionKind.CLARIFY_GOAL,
                text="Raise the retry limit to 5?",
                options=(
                    Option(key="yes", label="Yes, raise it"),
                    Option(key="no", label="No"),
                ),
                default="no",
            )
            record_asked(session, QuestionAskedLine(question=question))
            record_answered(
                session,
                QuestionAnsweredLine(
                    id=QUESTION_ID, choice="yes", source="human", elapsed_ms=4200
                ),
            )
            session.record_assistant("It is `max_retries` in src/client.py, line 41.")
            session.record_episode(
                EpisodeLine(
                    run_id=RUN_ID,
                    path="exact",
                    tier=None,
                    design_sha=None,
                    constraints=(),
                    opened_at_seq=None,
                )
            )
            second = session.record_user("Also lower the timeout.")
            session.record_assistant("Lowered it.")
            record_rewind(
                session,
                UndoLine(to_turn=second.seq, code=False, conversation=True, workspaces=()),
                kept_paths=(),
                runs=(),
            )
            session.sync()
        finally:
            session.close()
    (root / SESSION_ID / LOCK_NAME).unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    print(build(FIXTURE))
