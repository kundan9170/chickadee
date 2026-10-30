"""Context economy: handles, slices, compaction, invalidation.

The session is the unit: an append-only log (`store`), folded (`session`). Reads go
through handles and bounded views (`handle`, `spans`, `files`, `results`, `resolver`,
`outline`, and the Reader in `reader` with `terms`, `opener`, `views`, `parts`, `fit`,
`homes`, `diagnostics` and `pattern`), stay fresh per file (`freshness`), and say what a later
re-read can recover (`recovery`). Each turn's request is assembled from the log (`request`).
A long conversation is compacted by one cut (the `compaction` package) whose layers are typed
projection (`project`), never a model's paraphrase, with the constraint ledger, the rejected
alternatives and the concerns exempt from every layer (`ledger`, `concerns`) and the standing
set re-stated by notice (`standing`). A served
answer's citations are checked for location (`citations`); what an earlier session knew is
found by asking (`recall`), never injected.

Every name is imported from its module; this package re-exports nothing. Nothing
imported the 70 names it once bound here, and a split then had to edit this file as well as
the importers it moved.
"""
