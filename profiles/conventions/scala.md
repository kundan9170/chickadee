# Scala conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Follow the repository's scalafmt settings when it has them.
- Immutable by default: `val`, immutable collections, case classes for data.
- `Option`, `Either` or `Try` for absence and failure; never `null`, never `.get` on an
  `Option` outside tests.
- Pattern matching over type tests and casts; sealed traits for closed sets of cases.
- Keep side effects at the edges; pure functions in the core.
- Tests in `src/test/scala`, mirroring the package under test.
- Comments explain **why**, not what.
