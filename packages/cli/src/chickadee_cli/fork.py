"""`chickadee fork <session> [--at <seq>] [--reason <why>]`: a new session that branches
another, over the wire.

**The engine asks whether the code goes back** to the fork point (restore) or stays (keep).
On a terminal that question is the person's, shown and answered as the REPL answers one
(`interaction.answering`). Off a terminal nobody can answer it, so the command says so and
asks for its own default, `keep`, rather than leave the question to time out.

What the person must see rather than infer: the child's id, the notices its first turn
carries (the effects the fork cannot undo), and what happened to each workspace's code. An
unsafe `--at` is refused by the engine, with its reason; that is exit `1`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Final

from chickadee_cli.interaction import Interaction, answering
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.repl import InputSource
from chickadee_cli.runs import pump
from chickadee_cli.surface import DONE, USAGE
from chickadee_serve.frames import RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.events import Frame
from chickadee_wire.models import Forked

__all__ = ["DEFAULT_REASON", "NOBODY_TO_ASK", "run"]

DEFAULT_REASON: Final = "forked from the terminal"
NOBODY_TO_ASK: Final = (
    "not a terminal, so nobody can say whether the code goes back: it stays as it is "
    "(the default)"
)


def run(
    client: Client,
    session_id: str,
    *,
    at: int | None,
    reason: str | None,
    source: InputSource | None,
    say: Callable[[str, Tone], None],
    style: Style,
) -> int:
    """Fork `session_id` at `at` (its last seq by default) and say what came of it. `source`
    answers the engine's question; None, there is nobody to ask. The exit code."""
    why = reason or DEFAULT_REASON
    try:
        at_seq = at if at is not None else client.session_info(session_id).last_seq
        if source is None:
            say(NOBODY_TO_ASK, Tone.DIM)
            forked = client.fork(session_id, at_seq, why, "keep")
        else:
            forked = _asked(client, session_id, at_seq, why, source, say, style)
    except WireError as exc:
        say(f"chickadee fork: {exc.message}", Tone.BAD)
        return USAGE
    _show(forked, say)
    return DONE


def _asked(
    client: Client,
    session_id: str,
    at_seq: int,
    why: str,
    source: InputSource,
    say: Callable[[str, Tone], None],
    style: Style,
) -> Forked:
    """The fork, while its restore-or-keep question is followed on the session's stream and
    answered from `source`."""
    questions = Interaction(client, session_id, style, lambda line: say(line, Tone.PLAIN))
    events = client.session_events(session_id, client.session_info(session_id).last_seq)
    done = threading.Event()

    def take(frame: Frame) -> None:
        if isinstance(frame, RecordFrame):
            questions.feed(frame)

    reader = threading.Thread(
        target=pump, args=(events, take, done), name="fork-events", daemon=True
    )
    reader.start()
    try:
        return answering(
            lambda: client.fork(session_id, at_seq, why), questions, _default_on_end(source)
        )
    finally:
        # The reader stops here; its read ends when the caller closes the client, as the
        # REPL's does (an `Events` is closed only from the thread iterating it).
        done.set()


def _default_on_end(source: InputSource) -> Callable[[str], str]:
    """`source`'s lines, with Ctrl-D or Ctrl-C taking the question's default, as the REPL's
    `ask` does: the fork is already under way on its thread, so the answer must still come."""

    def read(prompt_text: str) -> str:
        try:
            return source.read(prompt_text)
        except (EOFError, KeyboardInterrupt):
            return ""

    return read


def _show(forked: Forked, say: Callable[[str, Tone], None]) -> None:
    say(f"forked: {forked.child_id} (code {forked.code})", Tone.GOOD)
    if forked.worktree:
        say(f"  worktree {forked.worktree}", Tone.DIM)
    for outcome in forked.workspaces:
        restored = ", ".join(outcome.restored) or "nothing"
        say(f"  {outcome.where} {outcome.root}: restored {restored}", Tone.DIM)
        if outcome.note:
            say(f"    {outcome.note}", Tone.DIM)
    for notice in forked.notices:
        say(f"  {notice}", Tone.PLAIN)
