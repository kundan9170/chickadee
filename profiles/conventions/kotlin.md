# Kotlin conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- `val` over `var`; immutable collections unless mutation is the point.
- Null safety: no `!!` unless an invariant makes null impossible and a comment says which.
- Data classes for plain data; sealed classes for a closed set of cases, matched with `when`.
- Coroutines for asynchronous work; never block a coroutine with `runBlocking` in library code.
- Expression bodies for one-line functions.
- Tests in `src/test/kotlin`, mirroring the package under test.
- Comments explain **why**, not what.
