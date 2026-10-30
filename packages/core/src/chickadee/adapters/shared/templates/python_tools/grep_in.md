Search inside a file you already opened with `read`: `handle` is what `read` returned,
`pattern` a regular expression, `context_lines` the lines shown around each hit. Each hit is
`line: text`; pass its line numbers to read(handle=..., start_line=..., end_line=...).
For a file you have not opened, use `grep`.
