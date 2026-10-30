"""Compaction: one cut, the layers a priority order inside it.

The layers are not four passes. A cut is an edit to the middle of the conversation and a prompt
cache is a prefix cache, so N passes would pay N rebuilds. A cut takes ONE boundary,
puts every element of the region before it into exactly one bucket, renders ONE block and
commits ONE line:

- **lifted** — a pinned skill body: re-emitted intact after the standing snapshot.
- **layer 2** — anything inside a completed unit: rendered from typed fields.
- **dropped** — an older duplicate of a call made again: its result goes.
- **layer 1** — any other tool exchange: a receipt of its finding and recovery.
- **layer 3** — prose: only what the record does not already hold, marked a claim.

**The log is the context.** `assemble` builds `messages[]` from the session log alone: each
cut's block and standing snapshot in order, then the pinned bodies the newest cut re-emits,
then every line from the newest boundary on, verbatim. Nothing is ever deleted, because there
is nothing to delete from: removing a thing is not including it in the next build, and the
only build that omits anything is one a cut committed.

**Commit, then emit**. The cut's line carries the block, the standing snapshot and the
boundary together, so a crash leaves the whole cut or none of it, and the model sees a cut only
in a request assembled after that line is durable.

The package re-exports nothing: `trigger` decides when, `boundary` where,
`receipts` and `residue` the layers, `cutter` builds the block and `cut` commits it, over
`context/request.py`'s weights and pieces.
"""
