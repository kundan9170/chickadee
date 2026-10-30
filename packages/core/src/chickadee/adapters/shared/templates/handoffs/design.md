# <A short title for the change>

<These sections are the minimum: fill in each one. Add any other `##` sections the work needs, anywhere. This file is `design/README.md`; put long detail in other files of `design/` and list each under Files.>

## Request
<Filled in with the user's request, word for word. Leave this section as it is.>

## Goal
<One or two sentences: what is true for the user once the change is done.>

## What changes
<Each change, one bullet each: the file or module, and what it gains, loses or does differently. Name new files, functions and classes with their signatures.>
- New `segtree.py`: `SegmentTree(values)`, `update(i, value)`, `query(lo, hi)` (sum over [lo, hi)).
- `stats.total()` uses `SegmentTree` instead of `naive_sum()`; `naive_sum()` is deleted.

## What must not change
<What the work must leave as it is: public signatures, behaviour others rely on, dependencies.>
- The signature of `stats.total()`. No new dependencies.

## Behaviour
<What is true once it is done, as cases someone could check: inputs and what they give, edge cases, costs.>
- `query` on an empty range returns 0; `SegmentTree([1, 2, 3]).query(0, 2) == 3`.

## Outside the repository
<Anything the work does beyond the repository's files: the network, a service, a package installed. Write "None." when there is nothing.>

## Constraints from our conversation
<Filled in with the constraints from the conversation that bind the work. Leave this section as it is.>

## Files
<Every other file of `design/`, one line each, exactly in this form; write "None." when there is none.>
- [api.md](api.md): the endpoints, their arguments and answers.
