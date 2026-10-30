"""The BACKGROUND pane: Zeus's
spawned agents and his background jobs, under the CODE pane, only while one of them runs.

**A list, two lines an item** (the owner's choice, 2026-10-26): what it is and what it was
asked, then how it stands. An agent's words are its `spawn_agent` call's `task`; a job's, its
shell call's command. Both are read from his turn by the call the record names, never kept
twice. There are no details: `Enter` opens nothing.
"""

from __future__ import annotations

from typing import ClassVar, Final

from textual.binding import Binding, BindingType
from textual.content import Content

from chickadee_cli.render.stream import preview
from chickadee_cli.tui.keys import BACKGROUND_KEYS
from chickadee_cli.tui.model import AgentItem, TuiState, call_of
from chickadee_cli.tui.names import PANE_WORDS
from chickadee_cli.tui.panes import RoleFrame
from chickadee_cli.tui.theme import Glyphs
from chickadee_wire.models import JobView

__all__ = ["HEIGHT", "BackgroundFrame", "background_body", "background_label"]

HEIGHT: Final = 10
"""Four items of two lines, and the border: the rest scroll (the owner's choice)."""

_ASKED: Final = ("task", "command")
"""The argument that says what an agent (`spawn_agent`) or a job (`shell`) was asked."""


def _asked(state: TuiState, call_id: str) -> str:
    call = state.zeus.calls.get(call_id)
    if call is None:
        return ""
    said = next((v for k in _ASKED if isinstance(v := call.arguments.get(k), str)), "")
    return preview(said, whole=False)


def _agent_lines(
    agent: AgentItem, state: TuiState, glyphs: Glyphs, colours: dict[str, str], chosen: bool
) -> list[Content]:
    dim, fg = colours["dim"], colours["fg"]
    if agent.running:
        dot, tone = glyphs.in_flight, colours["worker"]
        how = f"running · {agent.calls} call{'s' if agent.calls != 1 else ''}"
    elif agent.status in ("complete", "partial"):
        dot, tone = glyphs.accepted, colours["worker"]
        how = f"{agent.status} · {agent.findings or 0} findings"
    else:
        dot, tone = glyphs.dropped, colours["bad"]
        how = agent.status or "ended"
    if agent.cost_usd is not None:
        how += f" · ${agent.cost_usd:.2f}"
    head = Content.assemble(
        ("▸ " if chosen else f"{dot} ", colours["zeus"] if chosen else tone),
        (f"{agent.agent} {agent.profile}", f"bold {fg}"),
        (f"  {_asked(state, agent.call_id)}", dim),
    )
    return [head, Content.styled(f"  {how}", dim)]


def _job_lines(
    job: JobView, state: TuiState, glyphs: Glyphs, colours: dict[str, str], chosen: bool
) -> list[Content]:
    dim, fg = colours["dim"], colours["fg"]
    call = call_of(state.zeus, job.step)
    asked = _asked(state, call.id) if call is not None else ""
    if job.running:
        dot, tone = glyphs.in_flight, colours["shell"]
    elif job.report is not None and job.report.exit_code == 0:
        dot, tone = glyphs.accepted, colours["worker"]
    else:
        dot, tone = glyphs.dropped, colours["bad"]
    head = Content.assemble(
        ("▸ " if chosen else f"{dot} ", colours["zeus"] if chosen else tone),
        (f"job {job.job_id}", f"bold {fg}"),
        (f"  {asked}", dim),
    )
    return [head, Content.styled(f"  {job.kind} · {job.state}", dim)]


def background_body(
    state: TuiState, glyphs: Glyphs, colours: dict[str, str], chosen: str | None = None
) -> Content:
    """Each item on two lines, agents first; `chosen` is the row marked while focused."""
    background = state.background
    jobs = {f"job:{j.job_id}": j for j in background.jobs}
    lines: list[Content] = []
    for item in background.items:
        mark = item == chosen
        if item in background.agents:
            lines += _agent_lines(background.agents[item], state, glyphs, colours, mark)
        elif item in jobs:
            lines += _job_lines(jobs[item], state, glyphs, colours, mark)
    return Content("\n").join(lines)


def background_label(state: TuiState) -> str:
    """The title's right: `2 running · 1 done`."""
    background = state.background
    jobs = {f"job:{j.job_id}": j for j in background.jobs}
    running = done = 0
    for item in background.items:
        agent = background.agents.get(item)
        on = agent.running if agent is not None else item in jobs and jobs[item].running
        running, done = (running + 1, done) if on else (running, done + 1)
    return f"{running} running" + (f" · {done} done" if done else "")


class BackgroundFrame(RoleFrame):
    role = "background"
    colour = "athena"
    KEYS = BACKGROUND_KEYS
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(k.key, k.action, show=False) for k in BACKGROUND_KEYS if k.action is not None
    ]

    DEFAULT_CSS = f"""
    BackgroundFrame {{ border: round $athena; height: {HEIGHT}; display: none; }}
    BackgroundFrame:focus {{ border: double $athena; }}
    BackgroundFrame > .body {{ text-wrap: nowrap; text-overflow: ellipsis; }}
    """

    def idle(self, state: TuiState) -> bool:
        return False  # it is shown only while something runs

    def title(self, state: TuiState) -> Content:
        return Content.styled(
            PANE_WORDS["background"].upper(), f"bold {self._colours['athena']}"
        )

    def label(self, state: TuiState) -> Content:
        return Content.styled(background_label(state), self._colours["dim"])

    def content(self, state: TuiState) -> Content:
        return background_body(state, self._glyphs, self._colours, self.chosen())

    def rows(self, state: TuiState) -> list[str]:
        return state.background.items

    def default_row(self, rows: list[str]) -> int:
        return 0
