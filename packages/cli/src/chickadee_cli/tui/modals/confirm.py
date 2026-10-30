"""A command's confirmation as a modal: a
command above the safe tier, asked of the person when it is dispatched, by Zeus, Athena or
the worker, drawn over the screen as the mockup draws it.

**Every row is a field of the question** (`ActionApproval`: an observation of the action,
never a claim about it). The command band is its `command` as it runs, else `display`,
whole, wrapped, never cut: what the person approves must be seen entire. `in`, `network`,
`writes`, `sandbox` and `credentials` are drawn only from what the record says; a row the
record cannot fill is left out rather than drawn empty (absent is unknown). `sandboxed` is
false on every command a shell asks about: approved, it runs outside the sandbox, and the row
says so.

**The answers are the wire's** (`render.question.choices`, over `approval_choices`, the one
home every client reads): *once*, *this session* and *this repo* where offered, *deny* and
*abort*. The mockup's `e` Edit command has no answer on
the wire: drawn
dimmed, bound to nothing. Deny is the default, highlighted when it opens:
`Enter` submits the highlighted answer, and Esc denies, as the mockup says.
"""

from __future__ import annotations

from typing import ClassVar, Final

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.events import Resize
from textual.widgets import Static

from chickadee_cli.render.question import choices, covers, default_of, grant_breadth
from chickadee_cli.tui.modals import Answer, QuestionModal, Send, answer_bindings
from chickadee_cli.tui.names import asker_words
from chickadee_cli.tui.panes import border_title
from chickadee_wire.models import ActionApproval
from chickadee_wire.questions import OpenQuestion

__all__ = ["CONFIRM_ANSWERS", "SANDBOX_OUTSIDE", "ConfirmModal", "confirm_rows"]

SANDBOX_OUTSIDE: Final = "none once approved · it runs as you"
"""The `sandbox` row of a command that runs outside the sandbox once approved."""

CONFIRM_ANSWERS: Final = (
    Answer("y", "approve"),
    Answer("a", "repo"),
    Answer("s", "session"),
    Answer("e", None, wire_request="a later release"),
    Answer("n", "deny"),
    Answer("x", "abort"),
)
"""The mockup's `y` Run once, `a` Always allow … here, `e` Edit command, `n` Deny; and the
wire's `session` and `abort`, which the mockup does not draw (the owner's choice)."""

_VERB: Final = {"command": "wants to run", "mcp_tool": "wants to call", "git": "wants to run"}
_TITLE: Final = {
    "command": "CONFIRM A COMMAND",
    "mcp_tool": "CONFIRM A TOOL CALL",
    "git": "CONFIRM A GIT ACTION",
}
_ROWS: Final = (("y", "a"), ("s",), ("e", "n", "x"))
"""The mockup's two rows of answers, with *this session* on its own: its label is long."""
_LABEL_W: Final = 10
"""The rows' names, padded to the mockup's column (`network yes`)."""


def risk(approval: ActionApproval) -> str:
    """The mockup's `T2 · above T0, so you decide`; a T0 command asked anyway (with no sandbox
    every command asks) is not above T0, so its tier stands alone."""
    tier = approval.tier.upper()
    return tier if approval.tier == "t0" else f"{tier} · above T0, so you decide"


def confirm_rows(approval: ActionApproval) -> list[tuple[str, str]]:
    """The rows under the command, each from a field, in the mockup's order."""
    binding = approval.grant or approval.exact
    # A binding names where the call goes (`destination`), not a host list.
    where = binding.destination if binding is not None else None
    network = "no"
    if approval.network:
        network = f"yes · {where}" if where else "yes"
    rows = [("in", approval.cwd)] if approval.cwd else []
    rows.append(("network", network))
    writes = approval.effect_surface or (binding.effects if binding is not None else [])
    if writes:
        rows.append(("writes", ", ".join(writes)))
    if approval.sandboxed is not None:
        rows.append(("sandbox", "inside" if approval.sandboxed else SANDBOX_OUTSIDE))
    if approval.credentials:
        rows.append(("credentials", ", ".join(approval.credentials)))
    rows.append(("risk", risk(approval)))
    return rows


def answer_labels(approval: ActionApproval) -> dict[str, str]:
    prefix = " ".join(approval.grant.prefix) if approval.grant is not None else ""
    session = (
        "Allow this exact command this session"
        if approval.exact is not None
        else f"Always allow '{prefix}' this session"
    )
    return {
        "y": "Run once",
        "a": f"Always allow '{prefix}' here",
        "s": session,
        "e": "Edit command",
        "n": "Deny",
        "x": "Stop the turn",
    }


class ConfirmModal(QuestionModal):
    TIMES_OUT_AS: ClassVar[str] = "denied"
    ANSWERS: ClassVar[tuple[Answer, ...]] = CONFIRM_ANSWERS
    BINDINGS: ClassVar[list[BindingType]] = [
        *answer_bindings(CONFIRM_ANSWERS),
        Binding("up", "move(-1)", show=False, priority=True),
        Binding("down", "move(1)", show=False, priority=True),
        Binding("escape", "default", show=False),
    ]

    DEFAULT_CSS = """
    ConfirmModal > #box { width: 65%; max-width: 100; height: auto; max-height: 100%; }
    ConfirmModal Static { height: auto; }
    ConfirmModal #band { background: $btn; margin-bottom: 1; }
    ConfirmModal #answers { border-top: solid $faint; padding-top: 1; }
    ConfirmModal #foot { margin-top: 1; }
    """

    def __init__(
        self,
        asked: OpenQuestion,
        colours: dict[str, str],
        send: Send,
        *,
        deadline: str | None = None,
    ) -> None:
        if asked.approval is None:
            raise ValueError("a confirm modal shows an action approval")
        super().__init__(asked, colours, send, deadline=deadline)
        self.approval = asked.approval
        self._offered = {c.key for c in choices(asked)}
        self.answers = Static("", id="answers")

    def offered(self, key: str) -> bool:
        answer = next((a for a in CONFIRM_ANSWERS if a.key == key), None)
        return answer is not None and answer.choice in self._offered

    def compose(self) -> ComposeResult:
        a, c = self.approval, self._colours
        with Vertical(id="box"):
            yield Static(
                Content(f"{asker_words(a.asker)} {_VERB.get(a.kind, 'wants to run')}"),
                classes="dim",
            )
            yield Static(
                Content.assemble(
                    ("$ ", c["shell"]), (a.command or a.display, f"bold {c['fg']}")
                ),
                id="band",
            )
            rows = [
                Content.assemble((name.ljust(_LABEL_W), c["dim"]), self._value(name, value))
                for name, value in confirm_rows(a)
            ]
            yield Static(Content("\n").join(rows))
            if a.grant is not None and "repo" in self._offered:
                covered = f"{covers(a.grant)} · {grant_breadth(a)}"
                yield Static(
                    Content.assemble(("always".ljust(_LABEL_W), c["dim"]), (covered, c["dim"]))
                )
            yield self.answers
            yield Static(Content(self._foot()), id="foot", classes="dim")
            yield self.why
            yield self.deadline

    def _value(self, name: str, value: str) -> Content:
        c = self._colours
        if name == "network" and value != "no":
            head, _, rest = value.partition(" · ")
            return Content.assemble((head, c["bad"]), (f" · {rest}" if rest else "", c["fg"]))
        if name == "risk":
            head, _, rest = value.partition(" · ")
            return Content.assemble((head, c["zeus"]), (f" · {rest}" if rest else "", c["fg"]))
        return Content.styled(value, c["fg"])

    def _foot(self) -> str:
        who = asker_words(self.approval.asker)
        saves = " Always-allow saves to this repo." if "repo" in self._offered else ""
        return f"{who} waits. Esc denies.{saves}"

    def on_mount(self) -> None:
        super().on_mount()
        self.redraw()

    def first_selected(self) -> str | None:
        """Deny, the default (`APPROVAL_DEFAULT`), highlighted as the mockup draws it: a stray
        `Enter` denies, and the command runs only on a person's choice."""
        default = default_of(self.asked)
        return next((a.key for a in CONFIRM_ANSWERS if a.choice == default), None)

    def order(self) -> tuple[str, ...]:
        return tuple(k for row in _ROWS for k in row)

    def on_resize(self, _event: Resize) -> None:
        self._retitle()

    def _retitle(self) -> None:
        a, c = self.approval, self._colours
        box = self.query_one("#box")
        left = Content.styled(
            f" {_TITLE.get(a.kind, _TITLE['command'])} ", f"bold {c['ink']} on {c['athena']}"
        )
        right = asker_words(a.asker).removeprefix("The ").lower()
        right += f" · {a.task_id}" if a.task_id else ""
        box.border_title = border_title(
            left, Content.styled(right, c["dim"]), box.outer_size.width, double=True
        )

    def redraw(self) -> None:
        labels = answer_labels(self.approval)
        rows: list[Content] = []
        for keys in _ROWS:
            chips = [
                self.chip(k, labels[k])
                for k in keys
                if self.offered(k) or self._choice(k) is None
            ]
            if chips:
                rows.append(Content("  ").join(chips))
        rows.append(self.confirm_chip())
        self.answers.update(Content("\n\n").join(rows))
        if self.is_mounted:
            self._retitle()

    @staticmethod
    def _choice(key: str) -> str | None:
        return next(a.choice for a in CONFIRM_ANSWERS if a.key == key)

    # -- answering -------------------------------------------------------------------------

    def submit(self, key: str) -> None:
        choice = self._choice(key)
        if choice is not None:
            self.send(choice)

    def action_default(self) -> None:
        """Esc: the default, deny (`APPROVAL_DEFAULT`), at once, as the mockup says."""
        self.send(default_of(self.asked))
