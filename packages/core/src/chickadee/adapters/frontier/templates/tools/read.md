Read a file or a handle: a preview and a handle with no span, or the span itself with `lines=` or `at=`. Give exactly one of `path` and `handle`.

`path` is a file from the repository root (opens it, returns a handle); `handle` is one a result
returned (reuses it). With no span the
result is a preview — the outline, or the head for a file without one — plus the handle, never
the whole file. `lines="10-40"` returns that span in the same call; `at=` names an outline
entry; `depth=` picks `outline` / `signatures` / `full`; `max_bytes=` may only *lower* the
inline cap. Every line carries its number. The file's diagnostics come back too,
stamped with the bytes they were computed against.

An image returns as one block at `image_max_edge` on a rung that can see, and
`NOT_SUPPORTED` on one that cannot; any other non-text file is `NOT_TEXT{kind, bytes}`.
`HANDLE_EXPIRED` and `STALE_GENERATION` say re-read; `SECRET_PATH` says do not.
