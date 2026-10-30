What breaks if these symbols change: the blast radius through the graph. `symbols` is required.

`symbols` is a non-empty list. Returns every symbol reachable through callers and importers
within the graph's default depth, grouped by file, with the edge confidence on each step —
so an `unknown` edge reads as *possibly affected* rather than *affected*.

Consult it before a design that changes a public contract, and before deleting anything. It
does not decide; it reports reach.
