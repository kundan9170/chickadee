"""`coder`, a built-in.

A delegation to the coder box, which is the only writer (`zeus/delegate.py`). It streams: the
episode opens and the delegation is yielded before the
report comes back, so only its role's executor runs it.

A thin run over the `box` port. What the call may touch is the port's to decide; whether it may
run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Tool, streams_only

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "coder")


TOOL = Tool(SPEC, frozenset(), streams_only, streams=True)
