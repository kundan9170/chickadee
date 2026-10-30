# Rust conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Code is `rustfmt`-formatted and `clippy`-clean.
- Return `Result` for a fallible operation; propagate with `?`. No `unwrap()` or `expect()`
  outside tests, unless an invariant makes failure impossible and a comment says which.
- Borrow over clone; clone only where ownership is genuinely needed.
- Public items carry a `///` doc comment.
- Unit tests in a `#[cfg(test)] mod tests` beside the code; integration tests in `tests/`.
- `unsafe` only with a `// SAFETY:` comment stating what holds.
- Comments explain **why**, not what.
