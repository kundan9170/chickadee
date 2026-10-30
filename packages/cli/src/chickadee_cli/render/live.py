"""What the REPL shows while a turn runs:
replies as they are written, a thinking line, a tool call being written, and the working
indicator.

**Fed the live progress frames and every record** (`progress`, `record`), from whichever thread
renders them; read by the prompt's bottom toolbar (`bottom`), on its own. One lock guards it.

**Zeus's words are printed as they are written:** each line, once it is finished, goes above
the prompt (`write`), as his record would print it. When his record arrives, `record` hands
back the frame with what was printed taken off the front, so nothing is printed twice. A reply
whose shown text the record does not begin with (a frame lost, a retry after lines printed) is
printed whole: shown twice in part rather than never in part.

**The box's words stay in the toolbar:** Athena's and a worker's reply is drawn there, labelled,
line by line, and goes when that role's `model_call` record arrives, which prints its turn as
it always has.

**Thinking** is one line, `zeus · thinking… 12s`; at `-v` the reasoning is printed too, dimmed,
line by line. **A tool call being written** is `→ create core/ai.py · writing… 4.1 KB`, once it
has been written for `TOOL_AFTER_S`, so a short call never flashes. **The working indicator**
names who is working and what at, how long the turn has run and the tokens streamed.

Nothing here is drawn to a pipe: the REPL builds a `LiveView` for a terminal only.
"""

from __future__ import annotations

import ast
import codeop
import re
import textwrap
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import ValidationError

from chickadee_cli.render.stream import LIVE_PROGRESS, Verbosity, who
from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import ProgressFrame, RecordFrame
from chickadee_wire.models import AssistantTurnP, ModelCallP, ToolCallP
from chickadee_wire.records import payload

__all__ = [
    "GAP",
    "STOPPED",
    "TOOL_AFTER_S",
    "LiveTool",
    "LiveView",
    "Replies",
    "StreamingReply",
    "size_text",
]

TOOL_AFTER_S: Final = 0.5
"""How long a tool call is written before its live line shows."""
GAP: Final = "…"
"""Where a lost frame's text was: the record fills it in (decision 5)."""
STOPPED: Final = " (stopped)"
"""After what was shown of a reply the person stopped (decision 6)."""
_FENCE: Final = re.compile(r"^[ \t]*(```|~~~)[ \t]*([A-Za-z0-9_+.-]*)[ \t]*$")
"""A fence's opening line, its marker and its tag: the engine's reading (`pycall.split`). The
REPL never imports the engine (`test_boundaries`), so a test holds the
blocks read here equal to the engine's."""
_PYTHON_TAGS: Final = frozenset({"python", "python3", "py", "py3"})
ENDED_KEPT: Final = 256
"""Finished replies remembered, so a late frame of one is not drawn again."""


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


@dataclass
class LiveTool:
    tool: str
    path: str | None
    size: int
    since: float


@dataclass
class StreamingReply:
    """One model call's live state."""

    who: str
    role: str
    task: str | None
    text: str = ""
    next: int = 0
    """The offset the next piece is expected at; past it, a frame was lost."""
    printed: int = 0
    """How much of `text` is on the screen above the prompt (Zeus's), or printed dim (the
    reasoning, at `-v`)."""
    thought: str = ""
    thought_next: int = 0
    thought_printed: int = 0
    thinking_s: float | None = None
    tools: dict[int, LiveTool] = field(default_factory=dict[int, LiveTool])
    tokens: int = 0
    diverged: bool = False
    """A new attempt began after lines were printed: the record is printed whole."""
    headed: bool = False
    """The speaker's label is printed above its words (a worker's, under Python calls)."""
    read_at: int = -1
    """The length of `text` that `read` was split at: a redraw at the same length splits
    nothing again."""
    read: tuple[str, list[str]] = ("", [])
    """`text`'s words and blocks while it streams (`_split`, never the fenceless reading)."""


type Moved = Literal["text", "restart", "thought", "tool"]
"""What a live frame moved in its reply."""


class Replies:
    """The live replies of one stream, folded from its progress frames:
    each model call's text, reasoning and tool calls as its frames build them, a lost frame's
    gap marked, a new attempt voiding what it replaces, and the tokens of the replies already
    over. **The one fold of live frames**: the REPL's `LiveView` prints from it, the TUI's feed
    draws from it. Not thread-safe: its holder guards it."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.open: dict[str, StreamingReply] = {}
        """The replies being written, by the model call's id, oldest first."""
        self._ended: dict[str, None] = {}
        """The replies whose record has arrived, newest last, at most `ENDED_KEPT`."""
        self.tokens_done = 0
        """Tokens of the replies that have finished, or a new attempt voided, since its holder
        last set it to zero (a turn's start)."""

    @property
    def tokens(self) -> int:
        """Every token streamed: the finished replies' and the open ones'."""
        return self.tokens_done + sum(r.tokens for r in self.open.values())

    def take(self, frame: ProgressFrame) -> tuple[StreamingReply, Moved] | None:
        """A live frame (`text`, `thought`, `tool_started`) folded into its reply: the reply,
        and what moved in it. None for a frame of a reply already over."""
        p = frame.payload
        role = _str(p.get("role")) or "zeus"
        task = p.get("task")
        task = task if isinstance(task, str) else None
        key = _str(p.get("reply"))
        if key in self._ended:
            return None
        reply = self.open.get(key)
        if reply is None:
            reply = self.open[key] = StreamingReply(who(role, task), role, task)
        reply.tokens = max(reply.tokens, _int(p.get("tokens")))
        if frame.kind == "text":
            if p.get("restart"):
                self._restart(reply)
                return reply, "restart"
            reply.text, reply.next = _joined(
                reply.text, reply.next, _int(p.get("offset")), _str(p.get("piece"))
            )
            return reply, "text"
        if frame.kind == "thought":
            elapsed = p.get("elapsed_s")
            reply.thinking_s = float(elapsed) if isinstance(elapsed, int | float) else 0.0
            reply.thought, reply.thought_next = _joined(
                reply.thought, reply.thought_next, _int(p.get("offset")), _str(p.get("piece"))
            )
            return reply, "thought"
        index = _int(p.get("index"))
        path = p.get("path")
        held = reply.tools.get(index)
        since = held.since if held is not None else self._clock()
        reply.tools[index] = LiveTool(
            _str(p.get("tool")),
            path if isinstance(path, str) else None,
            _int(p.get("size")),
            since,
        )
        return reply, "tool"

    def _restart(self, reply: StreamingReply) -> None:
        """A new attempt (decision 6): its text so far is void and cleared. What was printed of
        Zeus's stays on the screen, so his record is then printed whole."""
        reply.diverged = reply.diverged or reply.printed > 0
        reply.text, reply.next = reply.text[: reply.printed], 0
        reply.thought, reply.thought_next, reply.thinking_s = "", 0, None
        reply.tools.clear()
        self.tokens_done += reply.tokens
        reply.tokens = 0

    def end(self, key: str) -> StreamingReply:
        """Reply `key` is over: its tokens counted, and a frame of it arriving after its record
        (the two travel apart) ignored rather than shown again."""
        reply = self.open.pop(key)
        self.tokens_done += reply.tokens
        self._ended[key] = None
        while len(self._ended) > ENDED_KEPT:
            del self._ended[next(iter(self._ended))]
        return reply

    def finish(self, role: str, task: str | None) -> None:
        """`role`'s replies at `task` are over: its record arrived. Athena's are hers whatever
        task the record names."""
        for key, reply in list(self.open.items()):
            if reply.role == role and (reply.task == task or role == "athena"):
                self.end(key)


class LiveView:
    """The live area of one REPL. `write` prints a line above the prompt; `verbosity` is read
    as each frame arrives, so `/verbose` takes effect at once."""

    def __init__(
        self,
        write: Callable[[str], None],
        style: Style,
        verbosity: Callable[[], Verbosity],
        clock: Callable[[], float] = time.monotonic,
        *,
        python_calls: bool = False,
    ) -> None:
        """`python_calls`: the worker calls tools by Python blocks (`chickadee/pycall.py`), so
        its prose is printed whole as its words, as Zeus's are, and its blocks as its calls.
        Learnt, when not given, from a worker's live piece marked `calls: "python"`, or else
        its first `model_call` whose tool is `python`."""
        self._python_calls = python_calls
        self._write = write
        self._style = style
        self._verbosity = verbosity
        self._clock = clock
        self._lock = threading.Lock()
        """Guards everything below: frames arrive on the REPL's reader threads, and the
        toolbar is drawn on the prompt's."""
        self._replies = Replies(clock)
        self._turn_since: float | None = None
        self._doing = ""
        self._who = "zeus"

    # -- what arrives --------------------------------------------------------------------------

    def progress(self, frame: ProgressFrame) -> None:
        """A progress frame: a live one drawn, a turn event starting or ending the turn."""
        with self._lock:
            if frame.kind in ("answer", "error"):
                if frame.kind == "error":
                    self._stop_all()
                self._turn_since = None
                return
            if frame.kind in ("alive", "context"):
                return  # the keep-alive; Zeus's context is the status line's
            if self._turn_since is None:
                self._turn_since, self._replies.tokens_done = self._clock(), 0
            if frame.kind in LIVE_PROGRESS:
                self._live(frame)
            elif frame.kind == "tool_call":
                self._who, self._doing = "zeus", f"running {_str(frame.payload.get('tool'))}"
            elif frame.kind == "delegation":
                self._who, self._doing = "zeus", "delegating"

    def record(self, frame: RecordFrame) -> RecordFrame:
        """A record: the live state it finishes is cleared, and the frame to print returned, for
        Zeus's reply less what was already printed of it."""
        try:
            typed = payload(frame)
        except ValidationError:
            return frame
        with self._lock:
            match typed:
                case AssistantTurnP():
                    return self._finish_zeus(frame, typed)
                case ModelCallP():
                    if typed.role == "worker" and "python" in typed.tools:
                        # The engine's own word that the worker calls by Python blocks: the
                        # config is the engine's, never read here.
                        self._python_calls = True
                    if self._speaks(typed.role):
                        called = "python" in typed.tools
                        return self._finish_worker(
                            frame, typed.role, frame.task_id, called, typed.text_preview
                        )
                    self._finish_role(typed.role, frame.task_id)
                case ToolCallP():
                    self._who = who(typed.role, frame.task_id)
                    self._doing = f"running {typed.tool}"
                case _:
                    pass
        return frame

    def stopped(self) -> None:
        """The person stopped the turn: what was shown stays, marked (decision 6)."""
        with self._lock:
            self._stop_all()

    # -- what the toolbar draws ----------------------------------------------------------------

    def bottom(self) -> list[str]:
        """The toolbar's live lines, top to bottom: each reply's line being written and its tool
        calls, then the working indicator. Nothing while no turn runs."""
        with self._lock:
            if self._turn_since is None and not self._replies.open:
                return []
            now = self._clock()
            lines: list[str] = []
            for reply in self._replies.open.values():
                lines += self._reply_lines(reply, now)
            if self._turn_since is not None:
                lines.append(self._indicator(now))
            return lines

    # -- under the lock ------------------------------------------------------------------------

    def _live(self, frame: ProgressFrame) -> None:
        got = self._replies.take(frame)
        if got is None:
            return
        reply, moved = got
        self._who = reply.who
        if reply.role == "worker" and frame.payload.get("calls") == "python":
            # The engine's word on the first piece, so even the session's
            # first reply prints its words live and its blocks as calls.
            self._python_calls = True
        if moved == "text":
            self._doing = "writing"
            self._text(reply)
        elif moved == "thought":
            self._doing = "thinking"
            if self._verbosity() is Verbosity.VERBOSE:
                reply.thought_printed = self._print_lines(
                    reply.thought, reply.thought_printed, Tone.DIM, "  "
                )
        elif moved == "tool":
            self._doing = f"writing {_str(frame.payload.get('tool'))}"

    def _text(self, reply: StreamingReply) -> None:
        """What the reply's new text prints: Zeus's finished lines, and a speaking worker's."""
        if reply.role == "zeus":
            reply.printed = self._print_lines(reply.text, reply.printed, Tone.PLAIN, "")
        elif self._speaks(reply.role):
            if _may_be_code(reply.text):
                return
            prose = _streamed(reply)[0]
            if not reply.headed and prose.rfind("\n") >= reply.printed and prose.strip():
                self._head(reply)
            reply.printed = self._print_lines(prose, reply.printed, Tone.PLAIN, "")

    def _speaks(self, role: str) -> bool:
        """Whether `role`'s words are printed as they are written: a worker's, under Python
        calls; Zeus's always, by `_text`."""
        return self._python_calls and role == "worker"

    def _head(self, reply: StreamingReply) -> None:
        reply.headed = True
        for out in self._style.lines(f"{reply.who}:", Tone.DIM):
            self._write(out)

    def _finish_worker(
        self,
        frame: RecordFrame,
        role: str,
        task: str | None,
        called: bool,
        preview: str | None,
    ) -> RecordFrame:
        """A worker's reply under Python calls is over: the rest of its prose printed, then its
        blocks as its calls; the record's preview of its words dropped when they were printed.
        The record says what the engine did: `called`, it made the reply's blocks calls (at
        report time it makes none, and a fence is the report's); with no `preview` of words
        left, a fenceless reply was one block (`_split`'s `whole`)."""
        printed = False
        for key, reply in list(self._replies.open.items()):
            if reply.role != role or reply.task != task:
                continue
            prose, codes = _split(reply.text, whole=called and not preview)
            rest = prose[reply.printed :].strip("\n")
            if rest.strip() and not reply.diverged:
                if not reply.headed:
                    self._head(reply)
                for out in self._style.lines(rest):
                    self._write(out)
            printed = printed or (reply.headed and not reply.diverged)
            for code in codes:
                if not called:
                    # Part of the report, never run: shown as its words, not as a call.
                    for out in self._style.lines(code, indent="    "):
                        self._write(out)
                    continue
                for out in self._style.lines("→ python", Tone.DIM, indent="    "):
                    self._write(out)
                for out in self._style.lines(code, Tone.DIM, indent="        "):
                    self._write(out)
            self._replies.end(key)
        if not printed:
            return frame
        return frame.model_copy(update={"payload": {**frame.payload, "text_preview": None}})

    def _print_lines(self, text: str, printed: int, tone: Tone, indent: str) -> int:
        """Each finished line of `text` past `printed`, printed; returns the new mark."""
        end = text.rfind("\n")
        if end < printed:
            return printed
        for line in text[printed:end].split("\n"):
            for out in self._style.lines(line, tone, indent=indent):
                self._write(out)
        return end + 1

    def _finish_zeus(self, frame: RecordFrame, turn: AssistantTurnP) -> RecordFrame:
        shown, whole = "", False
        for key, reply in list(self._replies.open.items()):
            if reply.role == "zeus":
                shown += reply.text[: reply.printed]
                whole = whole or reply.diverged
                self._replies.end(key)
        if whole or not shown or not turn.text.startswith(shown):
            return frame
        rest = turn.text[len(shown) :]
        return frame.model_copy(update={"payload": {**frame.payload, "text": rest}})

    def _finish_role(self, role: str, task: str | None) -> None:
        self._replies.finish(role, task)

    def _stop_all(self) -> None:
        for key, reply in list(self._replies.open.items()):
            if reply.role == "zeus" and reply.printed < len(reply.text):
                partial = reply.text[reply.printed :].rstrip("\n")
                for out in self._style.lines(partial + STOPPED):
                    self._write(out)
            self._replies.end(key)

    def _reply_lines(self, reply: StreamingReply, now: float) -> list[str]:
        lines: list[str] = []
        prose_now = _streamed(reply)[0] if self._speaks(reply.role) else reply.text
        if reply.role == "zeus" or self._speaks(reply.role):
            current = prose_now[prose_now.rfind("\n") + 1 :]
        else:
            current = reply.text.rstrip("\n").rsplit("\n", 1)[-1]
        if reply.thinking_s is not None and not reply.text:
            lines.append(
                self._style.verbatim(
                    f"{reply.who} · thinking… {reply.thinking_s:.0f}s", Tone.DIM
                )
            )
        if current:
            lines.append(self._style.verbatim(f"{reply.who} · {current}"))
        if self._speaks(reply.role) and len(prose_now) < len(reply.text):
            code = len(reply.text) - len(prose_now)
            lines.append(self._style.verbatim(f"→ python · writing… {_size(code)}", Tone.DIM))
        for call in reply.tools.values():
            if now - call.since >= TOOL_AFTER_S:
                where = f" {call.path}" if call.path else ""
                lines.append(
                    self._style.verbatim(
                        f"→ {call.tool}{where} · writing… {_size(call.size)}", Tone.DIM
                    )
                )
        return lines

    def _indicator(self, now: float) -> str:
        assert self._turn_since is not None
        tokens = self._replies.tokens
        doing = f" · {self._doing}" if self._doing else ""
        text = (
            f"{self._who}{doing} · working {_elapsed(now - self._turn_since)} · "
            f"{tokens:,} tokens · Ctrl-C to stop"
        )
        return self._style.verbatim(text, Tone.DIM)


def _streamed(reply: StreamingReply) -> tuple[str, list[str]]:
    """`reply`'s words and blocks as it streams, split once per length of its text."""
    if reply.read_at != len(reply.text):
        reply.read, reply.read_at = _split(reply.text), len(reply.text)
    return reply.read


def _split(text: str, *, whole: bool = False) -> tuple[str, list[str]]:
    """A reply's words and its Python blocks, as the engine reads them (`pycall.split`): a
    fence tagged `python`, `python3`, `py` or `py3` in any case, or an untagged one when it is
    the reply's only fence, each to its closing fence or the reply's end. The words are every
    line outside those blocks, other fences included, so nothing between or after a block is
    lost. A `whole` reply with no fence that parses as Python and calls something is one
    block: the engine also asks that it call an offered tool, which only it knows. Only a whole
    reply is parsed so, once, at its end: a streaming one is held by `_may_be_code`."""
    lines = text.split("\n")
    fences: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        opened = _FENCE.match(lines[i])
        if opened is None:
            i += 1
            continue
        marker, tag = opened.group(1), opened.group(2).lower()
        close = next((j for j in range(i + 1, len(lines)) if lines[j].strip() == marker), None)
        end = len(lines) if close is None else close
        fences.append((i, end, tag))
        i = end + 1
    taken = [
        (s, e)
        for s, e, tag in fences
        if tag in _PYTHON_TAGS
        or (tag == "" and len(fences) == 1 and _calls_something("\n".join(lines[s + 1 : e])))
    ]
    if whole and not fences and _calls_something(text):
        return "", [textwrap.dedent(text).strip("\n")]
    codes = [textwrap.dedent("\n".join(lines[s + 1 : e])).rstrip() for s, e in taken]
    inside = {n for s, e in taken for n in range(s, min(e + 1, len(lines)))}
    return "\n".join(line for n, line in enumerate(lines) if n not in inside), codes


def _calls_something(text: str) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(text))
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(n, ast.Call) for n in ast.walk(tree))


def _may_be_code(text: str) -> bool:
    """Whether a fenceless reply so far may still be a block, so its words wait for the
    reply's end: its first line is Python, whole or begun. Prose (`I'll read the file.`) is
    not, and prints as it comes."""
    if any(_FENCE.match(line) for line in text.split("\n")):
        return False
    first = text.lstrip("\n").split("\n", 1)[0]
    if not first.strip():
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            codeop.compile_command(first, "<reply>", "exec")
    except (SyntaxError, ValueError, OverflowError):
        return False
    return True


def _joined(text: str, expected: int, offset: int, piece: str) -> tuple[str, int]:
    """`piece` at `offset` joined to `text`, whose next piece was `expected`: a piece past it
    marks the gap (decision 5); one before it repeats what is held, and only its new end is
    kept."""
    if offset > expected:
        text += GAP
    elif offset < expected:
        piece = piece[expected - offset :]
        offset = expected
    return text + piece, offset + len(piece)


def size_text(size: int) -> str:
    """A tool call's arguments so far, as the live lines say it: `812 B`, `4.1 KB`."""
    return _size(size)


def _size(size: int) -> str:
    return f"{size} B" if size < 1024 else f"{size / 1024:.1f} KB"


def _elapsed(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole}s" if whole < 60 else f"{whole // 60}m {whole % 60:02d}s"
