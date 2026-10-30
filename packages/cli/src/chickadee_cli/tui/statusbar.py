"""The status bar (the mockup on the diff
screen): every key of the view's table, lit or dimmed, never hidden, so the bar does not jump.
A lit key is a button too: clicked, it runs the key's action.

**Narrow, it gives way without hiding a key**: where the keys and their labels do not
fit, the labels go and every key's chip stays, lit or dimmed as before.
"""

from __future__ import annotations

from rich.cells import cell_len
from textual.content import Content
from textual.events import Resize
from textual.widgets import Static

from chickadee_cli.tui.keys import DIFF_OFF_BAR, Key, ScreenView, bar_for, diff_bound, lit
from chickadee_cli.tui.model import TuiState

__all__ = ["StatusBar", "bar_line", "key_hint", "pane_hint"]


def bar_line(
    state: TuiState,
    colours: dict[str, str],
    *,
    view: ScreenView = ScreenView.BASE,
    hunks: bool = False,
    width: int | None = None,
) -> Content:
    """The view's keys in their groups: a lit key's chip in Zeus amber, a dimmed one's faint;
    with no labels when they would not fit in `width`."""
    full = _line(state, colours, view, hunks, labels=True)
    if width is None or cell_len(full.plain) <= width:
        return full
    return _line(state, colours, view, hunks, labels=False)


def _line(
    state: TuiState, colours: dict[str, str], view: ScreenView, hunks: bool, *, labels: bool
) -> Content:
    ink, zeus, faint, dim = colours["ink"], colours["zeus"], colours["faint"], colours["dim"]
    parts: list[tuple[str, str] | str] = [" "]
    for g, group in enumerate(bar_for(view)):
        if g:
            parts.append(("│ ", ink))
        for key in (k for k in group if k.drawn(state)):
            on = lit(key, state, view=view, hunks=hunks)
            chip = f"{ink} on {zeus}" if on else f"{dim} on {faint}"
            # A lit key's chip and label are its button; a dimmed one does nothing, as its key.
            # A chip drawn for two keys (`[ ]`) is two buttons, each its own key's.
            halves = _halves(key, view) if on else ()
            if len(halves) == 2:
                (first, own), (second, other) = halves
                parts += [
                    (f" {first}", f"{chip} @click={_click(own, view)}"),
                    (" ", chip),
                    (f"{second} ", f"{chip} @click={_click(other, view)}"),
                ]
            else:
                click = f" @click={_click(key, view)}" if on and key.action else ""
                parts.append((f" {key.shown} ", chip + click))
            click = f" @click={_click(key, view)}" if on and key.action else ""
            parts.append((f" {key.label} " if labels else " ", (ink if on else dim) + click))
    return Content.assemble(*parts)


def _halves(key: Key, view: ScreenView) -> tuple[tuple[str, Key], ...]:
    """A chip that shows two keys (`[ ]`: `[` the previous hunk, `]` the next, which the bar
    draws once): each half and the key it is, when the view binds the second off the bar."""
    first, _, second = key.shown.partition(" ")
    if not second or not key.action:
        return ()
    off = DIFF_OFF_BAR if view is ScreenView.DIFF else ()
    other = next((k for k in off if k.shown == second and k.action), None)
    return ((first, key), (second, other)) if other is not None else ()


def _click(key: Key, view: ScreenView) -> str:
    """What clicking a key runs: its action where its key is bound, the app's or, for a key
    the diff screen binds itself (`keys.diff_bound`: `n`, `t`, `[`, `]`, `Esc`), the diff
    screen's, through the app (`ChickadeeApp.action_on_diff`)."""
    if view is ScreenView.DIFF and key in diff_bound():
        return f"app.on_diff({key.action!r})"
    return f"app.{key.action}"


def key_hint(
    state: TuiState | None,
    colours: dict[str, str],
    keys: tuple[tuple[str, str], ...],
    *,
    view: ScreenView = ScreenView.BASE,
    hunks: bool = False,
) -> Content:
    """A hint naming keys in a pane (`^D full diff · ^U undo task`), each lit or dimmed as the
    view's bar draws it (the same table), so a hint never shows an unbuilt key as live. With
    no state, every key dimmed."""
    table = {k.key: k for group in bar_for(view) for k in group}
    table.update({k.key: k for k in DIFF_OFF_BAR if view is ScreenView.DIFF})
    parts: list[Content] = []
    for name, label in keys:
        key = table[name]
        on = state is not None and lit(key, state, view=view, hunks=hunks)
        shown = (key.shown, colours["zeus"] if on else colours["dim"])
        parts.append(Content.assemble(shown, (f" {label}", colours["dim"])))
    return Content.styled(" · ", colours["dim"]).join(parts)


def pane_hint(state: TuiState, colours: dict[str, str], keys: tuple[Key, ...]) -> Content:
    """A focused pane's own keys (`keys.ATHENA_KEYS`), keys sharing a label drawn as one
    (`↑↓ choose · Enter details`), each lit or dimmed by the same `lit`."""
    groups: dict[str, list[Key]] = {}
    for key in keys:
        groups.setdefault(key.label, []).append(key)
    parts: list[Content] = []
    for label, same in groups.items():
        on = all(lit(k, state) for k in same)
        shown = "".join(k.shown for k in same)
        parts.append(
            Content.assemble(
                (shown, colours["zeus"] if on else colours["faint"]),
                (f" {label}", colours["dim"]),
            )
        )
    return Content.styled(" · ", colours["dim"]).join(parts)


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar { height: 1; background: $fg; }
    """

    def __init__(self, colours: dict[str, str]) -> None:
        super().__init__()
        self._colours = colours
        self._drawn: tuple[TuiState, ScreenView, bool] | None = None

    def show(
        self, state: TuiState, *, view: ScreenView = ScreenView.BASE, hunks: bool = False
    ) -> None:
        self._drawn = (state, view, hunks)
        width = self.size.width or None
        self.update(bar_line(state, self._colours, view=view, hunks=hunks, width=width))

    def on_resize(self, _event: Resize) -> None:
        if self._drawn is not None:
            state, view, hunks = self._drawn
            self.show(state, view=view, hunks=hunks)
