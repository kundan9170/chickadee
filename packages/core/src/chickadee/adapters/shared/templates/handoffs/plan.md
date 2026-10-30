# Plan: <what the design builds>

<These sections are the minimum: fill in each one. Add any other `##` sections the work needs, anywhere. This file is `plan/README.md`; each task's brief is its own file in `plan/tasks/`, written to the brief template.>

## Tasks
<The tasks in the order they run, one line each, exactly in this form. Each is one unit a coding agent can finish and test on its own; its file says what it changes, where, and how it is tested. Each id is the task's for good: T1, T2, …, a piece of a re-cut task listed right after it as T2.1, T2.2; a task depends only on earlier lines.>
1. [T1: Build and update](tasks/T1-build-and-update.md): `SegmentTree(values)` and `update(i, value)`, with tests. Depends on: none
2. [T2: Range query](tasks/T2-range-query.md): `query(lo, hi)`, with tests for the edges. Depends on: T1
3. [T3: Use it in stats](tasks/T3-use-it-in-stats.md): `stats.total()` uses it; `naive_sum()` goes. Depends on: T2

## Final check
<What runs at the end, for example `tests_for` on every changed file, then the full suite.>
