"""The brand's pictures: the bird, a grid of pixels drawn two to a cell
with half blocks, and the name in block letters.

A chickadee is a black cap and bib, white cheek, buff flank. On a dark terminal a black cap is
lost against the background, so the bird's cap has a slate rim and its eye a white glint. The
header and the front page draw the same bird (the owner's call).

**The front page gives way by width** (`front_mark`): the bird centred over the wordmark, then
over the spaced name in text, then the text alone. A terminal without the half blocks
(`Glyphs.mark` is ASCII) gets the text alone. Art is never wrapped: a wrapped row is noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from textual.content import Content

__all__ = [
    "BIRD",
    "NAME",
    "SPACED",
    "TAGLINE",
    "WORDMARK",
    "bird",
    "front_mark",
    "half_blocks",
    "wordmark",
]

NAME: Final = "CHICKADEE"
SPACED: Final = " ".join(NAME)
"""The name where the art cannot be drawn: the header's, and the front page's fallback."""
TAGLINE: Final = "design·plan·code"

BIRD: Final = (
    ".......rrrr...",
    "......rkkkkr..",
    ".....rkkkkEwr.",
    ".....kkkkkEEhh",
    "....gKWWWWWkk.",
    "...gGGWWWWkkk.",
    "..gGDDDBbWWk..",
    ".hDDwwDBBbW...",
    "hDDDDDDcBb....",
    "h..nnnndnndnn.",
)
"""The chickadee, 14 by 10 pixels (five rows), in the header and on the front page, perched in
profile, drawn by hand (the owner's choice): the cap
and its rim, the eye and its glint, the bill, the cheek between the cap and the bib, the grey
back, a dark wing with one pale bar, the buff flank, the long tail, the twig."""

_INK: Final = {
    "K": "#16181d",
    "k": "#262a33",
    "r": "#5d6577",
    "E": "#030405",
    "W": "#fbfbfc",
    "w": "#e6e9ee",
    "G": "#b2b8c5",
    "g": "#959dae",
    "h": "#788092",
    "D": "#474d5a",
    "d": "#363b46",
    "b": "#f3e5cd",
    "B": "#e6c490",
    "c": "#d2a46b",
    "n": "#9a7350",
}
"""The bird's colours, a few shades of each: black (`K k`), its slate rim (`r`) so the cap's
outline reads on a dark terminal, the eye (`E`) blacker still with a white glint; white
(`W w`), grey (`G g h`), the wing's dark (`D d`), buff (`b B c`) and the twig (`n`).
The bill is slate (`h`), as a chickadee's is dark: an amber bill reads as a duck's."""

WORDMARK: Final = (
    "░█▀▀░█░█░▀█▀░█▀▀░█░█░█▀█░█▀▄░█▀▀░█▀▀",
    "░█░░░█▀█░░█░░█░░░█▀▄░█▀█░█░█░█▀▀░█▀▀",
    "░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀░▀░▀▀░░▀▀▀░▀▀▀",
)
"""The name in shaded block letters, after figlet's Pagga (the owner's choice), drawn in
gold: 36 columns, which a 120-column screen's front page holds under the bird."""


def half_blocks(grid: Sequence[str], ink: Mapping[str, str]) -> list[Content]:
    """`grid` as rows of `▀`: each cell its top pixel as the colour, its bottom one as the
    background, a blank pixel (`.`) the terminal's own. `grid` has an even number of rows."""
    rows: list[Content] = []
    for top, bottom in zip(grid[0::2], grid[1::2], strict=True):
        cells: list[tuple[str, str] | str] = []
        for up, down in zip(top, bottom, strict=True):
            if up == "." and down == ".":
                cells.append(" ")
            elif up == ".":
                cells.append(("▄", ink[down]))
            elif down == ".":
                cells.append(("▀", ink[up]))
            else:
                cells.append(("▀", f"{ink[up]} on {ink[down]}"))
        rows.append(Content.assemble(*cells))
    return rows


def bird() -> list[Content]:
    """The header's bird: five rows, fourteen columns."""
    return half_blocks(BIRD, _INK)


def wordmark(colour: str) -> list[Content]:
    """`WORDMARK`, three rows of 36 columns, in `colour`."""
    return [Content.styled(row, f"bold {colour}") for row in WORDMARK]


def centred(rows: Sequence[Content]) -> list[Content]:
    """`rows` each centred on the widest of them."""
    widest = max((row.cell_length for row in rows), default=0)
    return [Content.assemble(" " * ((widest - row.cell_length) // 2), row) for row in rows]


def front_mark(colours: dict[str, str], *, width: int, pictures: bool) -> list[Content]:
    """The front page's head, at most `width` columns, each row centred on the widest: the bird
    over the wordmark, or over the spaced name in text, or the text alone; the tagline last."""
    gold, dim = colours["gold"], colours["dim"]
    name = Content.styled(SPACED, f"bold {gold}")
    tagline = Content.styled(TAGLINE, dim)
    if not pictures or width < len(BIRD[0]):
        return centred([name, tagline])
    word = wordmark(gold) if width >= len(WORDMARK[0]) else [name]
    return centred([*bird(), Content(""), *word, tagline])
