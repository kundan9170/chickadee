"""A tool is a plug-in: one registry, one dispatcher, and a role is a policy over them.

Five parts, each with one home:

- **tools** — what can be called: `spec.py` (`ToolSpec`, `SURFACE`), `validate.py`
  (`validate`, `Arguments`), `tool.py` (`Tool`, `Invocation`, `Answer`), `registry.py`, and one
  module per built-in under `builtin/`;
- **ports** — what a tool may touch, data policy: `context.py`;
- **rules** — whether, when and how a call runs, and what it costs, call policy: `rules.py`;
- **sinks** — how a role shows, records and replays a call: `sink.py`;
- **policies** — which of each a role gets: `policy.py`.

`dispatch.py` runs a role's pipeline over them: `admit → dispatch → around(run) → result →
record → observe`.

This package imports no role package (`zeus`, `scout`, `box`, `packer`, `orchestrator`,
`engine`): each role builds its context, sink and dispatcher at its own
composition root. Every name is imported from its module; this package re-exports nothing.
"""
