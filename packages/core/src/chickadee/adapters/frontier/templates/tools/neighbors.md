What surrounds a symbol or path, one hop out: names and signatures, no bodies. `target` is required.

`target` is a symbol or a path; `depth` (default 1, at most 3) is how many hops. Returns the
neighbourhood as names and signatures — what calls it, what it calls, what shares its file —
so you can decide what to `read` without reading everything.

Bounded inline; a large neighbourhood is cut with a note naming the total.
