Which tools exist for a task, or one tool's full schema. Exactly one of `query` or `name` is required.

`query` returns the name and one-line description of every tool — built-in or registered at
run time — whose name or description matches, case-insensitively, under `result_cap` lines.
An empty match is an empty list: the registry is in-process and total, so the empty set is a
fact, not a failure.

`name` returns that one tool's full schema as content — the cheap way back to a schema that
compaction evicted, and the only way to a dynamic tool's schema past `dynamic_stub_max`.
Giving both, or neither, is `INVALID_ARGUMENTS`.
