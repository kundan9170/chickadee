# Java conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Target the Java version the build declares; use `var` only where the type is obvious.
- Prefer immutable values: `final` fields, records for plain data.
- Never return `null` for a collection; return an empty one. Use `Optional` for an absent
  single result, never as a field or parameter.
- Catch the specific exception; never swallow one with an empty `catch`.
- One public top-level class per file, named for the file.
- Tests in `src/test/java`, mirroring the package of the code under test.
- Comments explain **why**, not what.
