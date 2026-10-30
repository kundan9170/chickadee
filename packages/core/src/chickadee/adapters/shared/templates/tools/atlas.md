What this codebase is: its components, public contracts and module edges. No arguments.

Returns the map: components (one per directory, by language), each with its public contracts,
and the edges between them with their confidence (`resolved`, `fallback` or `unknown`). It also
names what the index does not cover (unprofiled files and parse gaps), so a missing edge reads
as not indexed rather than no dependency.

Use it first, to see where the work lands. It never returns file content; `read` does that.
