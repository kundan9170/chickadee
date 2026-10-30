Search inside a file already opened by `read`; every hit is addressable as a slice. `handle` and `pattern` are required.

`handle` is what `read` returned; `pattern` is a regular expression; `context` (default 2)
is lines around each hit. Hits are `line: text` with the line numbers you can pass straight
to `read(handle=…, lines=)`. Bounded by `result_cap` lines, with the total named past it.

For a file you have not opened, `grep` searches every file.
