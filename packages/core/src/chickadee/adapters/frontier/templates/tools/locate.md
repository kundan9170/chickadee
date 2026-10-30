Where the thing you can only describe is: pointers found by a bounded search agent, never prose. `question` is required.

The one navigate tool that spends a model call — on the cheap rung, in a discarded context,
bounded at `locate_max_calls` tool calls. Describe what you are looking for in a sentence;
it returns pointers (path, line, why it matched) and nothing else. The scout reads the
repository through its own reader: name paths, symbols and what to look for, never one of
your handles, which mean nothing to it.

Dispatch is `capped`: several `locate` calls in one turn run one after another, up to the
fan-out cap; the excess are refused with the cap named. `UNRESOLVED` reports what it searched;
`TIMED_OUT` returns the pointers found so far, marked partial. `UNAVAILABLE` means the scout
could not search at all and names why; search yourself instead, with `symbol`, `grep`,
`glob` or `grep_in`, or ask the user. Calling `locate` again helps only once the
scout's rung is back.
