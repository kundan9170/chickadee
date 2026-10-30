Read a file. Give `path` (from the repository root), or `handle=` for a reference a result
gave you (long shell output, an earlier read).

With no lines, you get a preview (the outline, or the head of the file) and a handle, never
the whole file. `start_line` and `end_line` (both, inclusive; one line: the same number twice)
return those lines. `symbol="name"` returns one outline entry's lines. `depth` picks
"outline", "signatures" or "full". Every line comes back with its number.

Examples: read("src/app.py"); read("src/app.py", start_line=40, end_line=80);
read("src/app.py", symbol="parse").
