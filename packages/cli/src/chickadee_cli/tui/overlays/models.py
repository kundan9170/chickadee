"""The models (`/rung`): each rung's model and the catalog to switch
it from.

**By rung, as the wire serves it.** `/rung` (`ModelWiring`) names each rung's model, the
engine catalog, which role runs on which rung (`roles`, the config's assignment) and each
model's window and provider (`models`). A server that sends no assignment
lists a role under a rung only where its own calls name it, never guessed. A rung's model is
the one configured or switched to, never claimed as the one that served: that is each pane's
label, from its calls.

**A switch is `/rung`'s**: `Enter` sends `/rung <rung> <model>`, so the server's
refusal (not in the catalog, too small a window, something in flight) is said in its words.
Nothing here says *locked*: while a turn or a coder run is in flight the overlay says that the
server refuses a switch then, the spec's rule, and the server decides.
"""

from __future__ import annotations

from typing import ClassVar, Final

from textual.binding import BindingType
from textual.content import Content

from chickadee_cli.tui.model import StaleRoute
from chickadee_cli.tui.overlays import CLOSE, Host, Overlay, OverlayKey, bindings, fit, row
from chickadee_cli.tui.panes import window_text
from chickadee_cli.tui.theme import UNKNOWN

__all__ = ["IN_FLIGHT", "Models", "used_by"]

IN_FLIGHT: Final = (
    "A turn or a coder run is in flight: the server refuses a switch until nothing runs (D185)."
)
_RULES: Final = (
    "Takes effect from the next call. A model must be in the engine catalog and fit the "
    "role's window. Defaults are changed in the engine config, not here."
)
_COLUMNS: Final = (("RUNG", 12), ("MODEL", 24), ("WINDOW", 8), ("SERVED BY", 12))


def used_by(state_rungs: dict[str, str | None], rung: str) -> tuple[str, ...]:
    """The roles whose own calls name `rung` (`{role: rung}`, from the panes)."""
    return tuple(role for role, named in state_rungs.items() if named == rung)


class Models(Overlay):
    WORD = "MODELS"
    COMMAND = "/rung"
    WIDTH = 79
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("up", "down"), "Up/Down", "choose", ("move(-1)", "move(1)")),
        OverlayKey(("right", "left"), "Right", "catalog", ("catalog(True)", "catalog(False)")),
        OverlayKey(("enter",), "Enter", "switch", ("switch",)),
        OverlayKey(("r",), "r", "reload catalog", ("reload",)),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def __init__(self, host: Host) -> None:
        super().__init__(host)
        self.in_catalog = False
        """`Right` was pressed: Up/Down choose the rung's new model."""
        self.model_cursor = 0

    def right(self) -> str:
        return "this session only"

    def rungs(self) -> list[tuple[str, str]]:
        wiring = self.state.wiring
        return sorted(wiring.rungs.items()) if wiring is not None else []

    def catalog(self) -> list[str]:
        wiring = self.state.wiring
        return list(wiring.catalog) if wiring is not None else []

    def rows(self) -> int:
        return len(self.catalog()) if self.in_catalog else len(self.rungs())

    def action_move(self, step: int) -> None:
        if not self.in_catalog:
            super().action_move(step)
            return
        count = len(self.catalog())
        if count:
            self.model_cursor = (self.model_cursor + step) % count
            self.redraw()

    def _role_rungs(self) -> dict[str, str | None]:
        """Role to rung: the config's assignment when the server sends it, else what each
        box role's own calls name."""
        wiring = self.state.wiring
        if wiring is not None and wiring.roles:
            return dict(wiring.roles)
        task = self.state.worker.task
        return {"athena": self.state.athena.rung, "worker": task.rung if task else None}

    def _about(self, model: str) -> tuple[str, str]:
        """A catalog model's window and provider, or unknown."""
        wiring = self.state.wiring
        known = (wiring.models or {}).get(model) if wiring is not None else None
        if known is None:
            return UNKNOWN, UNKNOWN
        return window_text(known.context_window), known.provider

    # -- what it shows ---------------------------------------------------------------------

    def body(self, width: int) -> Content:
        c = self.colours
        rungs = self.rungs()
        if not rungs:
            why = "not served here" if StaleRoute.RUNG in self.state.not_served else UNKNOWN
            return Content.styled(f"the wiring: {why}", c["dim"])
        head = "".join(fit(name, w) for name, w in _COLUMNS) + "USED BY"
        lines = [Content.styled(head, c["dim"])]
        roles = self._role_rungs()
        for at, (rung, model) in enumerate(rungs):
            users = ", ".join(used_by(roles, rung)) or UNKNOWN
            window, provider = self._about(model)
            text = f"{fit(rung, 12)}{fit(model, 24)}{fit(window, 8)}{fit(provider, 12)}{users}"
            chosen = at == self.cursor
            lines.append(row(Content(text), width, c, chosen=chosen and not self.in_catalog))
        lines += [Content(""), *self._choices(width)]
        if self.state.zeus.busy is not None or self.state.episode_open:
            lines += [Content(""), Content.styled(IN_FLIGHT, c["zeus"])]
        lines += [Content(""), Content.styled(_RULES, c["dim"])]
        return Content("\n").join(lines)

    def _choices(self, width: int) -> list[Content]:
        c = self.colours
        rung, current = self.rungs()[self.cursor % len(self.rungs())]
        lines = [
            Content.assemble((rung, f"bold {c['athena']}"), (" · choose a model", c["dim"]))
        ]
        for at, model in enumerate(self.catalog()):
            mark = "●" if model == current else "○"
            note = "  current" if model == current else ""
            text = Content(f"  {mark} {model}{note}")
            chosen = self.in_catalog and at == self.model_cursor
            lines.append(row(text, width, c, chosen=chosen))
        return lines

    # -- switching -------------------------------------------------------------------------

    def applies(self, action: str) -> bool:
        if action == "switch":
            return self.in_catalog and self._switch_to() is not None
        if action == "catalog":
            return bool(self.rungs())
        return True

    def action_catalog(self, into: bool) -> None:
        self.in_catalog = into
        if into:
            current = self.rungs()[self.cursor][1]
            catalog = self.catalog()
            self.model_cursor = catalog.index(current) if current in catalog else 0
        self.redraw()

    def _switch_to(self) -> tuple[str, str] | None:
        rungs, catalog = self.rungs(), self.catalog()
        if not rungs or not catalog:
            return None
        rung, current = rungs[self.cursor % len(rungs)]
        model = catalog[self.model_cursor % len(catalog)]
        return None if model == current else (rung, model)

    def action_switch(self) -> None:
        chosen = self._switch_to()
        if chosen is not None:
            self.dismiss()
            self.host.command(f"/rung {chosen[0]} {chosen[1]}")

    def action_reload(self) -> None:
        self.host.refresh_routes({StaleRoute.RUNG})
