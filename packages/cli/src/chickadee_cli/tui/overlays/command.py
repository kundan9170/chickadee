"""A command's details ("a command's output is a first line and its exit
code; `Enter` expands it"), from the worker's pane.

**What its record holds**: its label, its exit code and whether it
timed out, where it ran, under which sandbox and policy, with or without the network, and the
escalation it ran under, if any. Its output is its run blobs (`stdout_sha`, `stderr_sha`),
read as the overlay shows them (`/run/{run_id}/blob/{sha}`).
"""

from __future__ import annotations

from typing import ClassVar

from textual.binding import BindingType
from textual.content import Content

from chickadee_cli.tui.overlays import CLOSE, Blobs, Host, Overlay, OverlayKey, bindings
from chickadee_cli.tui.overlays.task import blob_lines
from chickadee_wire.models import ExecP

__all__ = ["CommandDetail"]


class CommandDetail(Overlay):
    WORD = "COMMAND"
    COMMAND = "Enter on a command"
    WIDTH = 72
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (CLOSE,)
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def __init__(self, host: Host, ran: ExecP, task_id: str) -> None:
        super().__init__(host)
        self.ran = ran
        self.task_id = task_id
        self.blobs = Blobs(host, host.state.athena.run_id, self.redraw)

    def heading(self) -> str:
        return self.task_id

    def right(self) -> str:
        return "what the record holds"

    def body(self, width: int) -> Content:
        c, ran = self.colours, self.ran
        late = " (timed out)" if ran.timed_out else ""
        ok = ran.exit_code == 0 and not ran.timed_out
        rows = [
            ("ran in", ran.cwd),
            ("sandbox", ran.sandbox),
            ("network", "on" if ran.network else "off"),
            ("policy", ran.policy),
        ]
        if ran.escalation_id is not None:
            rows.append(("escalation", ran.escalation_id))
        lines = [
            Content.assemble((ran.label, f"bold {c['shell']}")),
            Content.styled(f"exit {ran.exit_code}{late}", c["worker"] if ok else c["bad"]),
            Content(""),
            *(Content.assemble((f"{k:<12}", c["dim"]), v) for k, v in rows),
            Content(""),
            Content.styled("STDOUT", f"bold {c['fg']}"),
            *blob_lines(self.blobs.said(ran.stdout_sha), c, indent="  "),
            Content(""),
            Content.styled("STDERR", f"bold {c['fg']}"),
            *blob_lines(self.blobs.said(ran.stderr_sha), c, indent="  "),
        ]
        return Content("\n").join(lines)
