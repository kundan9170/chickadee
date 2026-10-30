# Python conventions

These are rendered into the coder's stable prefix (R-2). They describe how code in this
repository is written, not what the packet must do.

- Target Python 3.12+. Use `X | None`, not `Optional[X]`.
- Type public functions. Internal helpers may be inferred where obvious.
- Prefer explicit, boring code over clever code.
- Comments explain **why**, not what. A comment restating the line below it is noise.
- Raise specific exceptions; never `except Exception: pass`.
- Use `pathlib.Path` for filesystem work, not string concatenation.
- No wildcard imports.
