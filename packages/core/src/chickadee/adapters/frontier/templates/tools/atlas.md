What this codebase is: its components, public contracts and module edges. No arguments.

Returns the map: components (directory-derived, per language profile), each with its public
contracts, and the edges between them with their confidence (`resolved` / `fallback` / `unknown`).
Also names what the index does not cover — unprofiled files and parse gaps — so a missing
edge reads as *not indexed* rather than *no dependency*.

Use it first in an unfamiliar repository, and again after `coder` lands a design that adds a
component. It never returns file content; `read` does that.
