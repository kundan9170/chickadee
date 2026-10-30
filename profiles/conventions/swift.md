# Swift conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- `let` over `var`; value types (`struct`, `enum`) unless identity is needed.
- No force unwrap (`!`) or `try!` outside tests, unless an invariant makes failure impossible
  and a comment says which; use `guard let` to exit early.
- Throw typed errors for expected failures; `async`/`await` for asynchronous work.
- Access control as narrow as works: `private` by default, `public` only for the API.
- Protocols for capabilities; extensions to group conformances.
- Tests in `Tests/<Target>Tests`, one test case class per type under test.
- Comments explain **why**, not what.
