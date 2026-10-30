"""Context inside the box: what each box role
can see, how long each part lives, and where it leaves when eviction is not enough.

The rule behind every part: a context can live as long as the world it describes stays still,
and only records cross a boundary. One conversation for both roles (`conversation`), receipts
for tool results (`evict`), the named exits (`exits`), the task records and the symbol table
(`records`), the worker's head (`worker`), and Athena's head and notes (`athena`).

The packet loop's compaction ladder, its records, the earlier designs' history and the fenced
summaries are parked as they stood.
"""

from __future__ import annotations

from chickadee.boxctx.athena import AthenaNote, NumberedLog, athena_head, write_note
from chickadee.boxctx.conversation import (
    BoxConversation,
    ContextFull,
    JournaledTranscript,
    rebuilt,
)
from chickadee.boxctx.exits import BoxExit, ExitKind
from chickadee.boxctx.records import (
    SymbolGraph,
    SymbolTable,
    TaskRecord,
    render_task_full,
    render_task_line,
    symbol_table,
    task_record,
)
from chickadee.boxctx.worker import render_design_layer, task_head

__all__ = [
    "AthenaNote",
    "BoxConversation",
    "BoxExit",
    "ContextFull",
    "ExitKind",
    "JournaledTranscript",
    "NumberedLog",
    "SymbolGraph",
    "SymbolTable",
    "TaskRecord",
    "athena_head",
    "rebuilt",
    "render_design_layer",
    "render_task_full",
    "render_task_line",
    "symbol_table",
    "task_head",
    "task_record",
    "write_note",
]
