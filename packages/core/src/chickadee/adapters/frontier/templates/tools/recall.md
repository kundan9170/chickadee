Whether this was done before: prior designs, runs and outcomes from the journal, dated. `query` is required.

Reads the journal, not the graph — the one navigate tool that answers questions about the
past. Returns pointers into prior sessions: the design that was approved, the run that
executed it, what it cost, how it ended, and when. Dated, so an old answer reads as old.

`UNRESOLVED` means no prior session matched. A pointer whose blob was pruned by `gc` says
so (`HANDLE_EXPIRED{reason: pruned}`) rather than re-fetching content that is gone.
