"""How an episode ended: one vocabulary for the box's report, the run's
`run_end` line, and the design record that remembers it.

Its home is contract 3 (`schema.journal.Outcome`), since `run_end` carries it; named here so
`boxctx`, which `box` builds on, can name it without importing the box.
"""

from __future__ import annotations

from chickadee.schema.journal import Outcome

__all__ = ["Outcome"]
