# T<n>: <title>

<These sections are the minimum: fill in each one. Add any other `##` sections the task needs, anywhere. In a plan, this file is `plan/tasks/T<n>-<slug>.md`, its title the same as its line in the plan's Tasks.>

## Goal
<One or two sentences: what is true when this task is done.>
`SegmentTree.query(lo, hi)` returns the sum of values[lo:hi] in O(log n).

## Where to look
<The files, functions and lines the coding agent needs, with what you found in them. Be specific: the coding agent starts with no context beyond this brief and what earlier tasks changed.>
- `segtree.py`: `__init__` builds `self.tree` (1-indexed, leaves start at `self.size`); `update()` is lines 30-42.
- `tests/test_segtree.py`: the existing tests for build and update.

## Approach
<How to do it, as precisely as the work needs: the algorithm, the shape of the change, what to reuse.>

## Don't
<What the agent must leave alone, and what it must not do to get a green run.>
- Edit an existing test to make it pass. If one looks wrong, stop and say so.

## Done when
<The commands to run and what they must show. Where no command can run, the tests that would cover the change and what to check by reading it.>
`pytest tests/test_segtree.py -q` passes, with new tests for an empty range, one element and the whole array.

## Report
What you changed, the commands you ran and what they printed, anything you're unsure of.
