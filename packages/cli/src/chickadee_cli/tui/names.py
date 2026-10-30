"""What the screen calls the cast (the owner's words, 2026-10-25): one home, so a pane's title,
its key, its tab and an approval's asker never name a role two ways.

The code keeps its own names (`zeus`, `athena`, `worker`); only what the person reads moves.
`VOICE` is the one the person talks to: Zeus's replies, the composer, his asks.
"""

from __future__ import annotations

from typing import Final

__all__ = ["PANE_WORDS", "ROLE_WORDS", "VOICE", "asker_words"]

ROLE_WORDS: Final = {"zeus": "Designer", "athena": "Planner", "worker": "Worker"}
"""Each role as the screen names it: its pane's title (upper case), its key, its tab."""
PANE_WORDS: Final = {"diff": "Code", "background": "Background"}
"""The panes that are no role's, as the screen names them: their title (upper case), their key
and their tab."""
VOICE: Final = "Chickadee"
"""The voice the person talks to: the label over Zeus's replies, the composer's placeholder,
the asker of his approvals."""


def asker_words(asker: str) -> str:
    """Who asks, as a sentence starts it: `Chickadee` for Zeus, else `The planner` and so on."""
    if asker == "zeus":
        return VOICE
    return f"The {ROLE_WORDS.get(asker, asker).lower()}"
