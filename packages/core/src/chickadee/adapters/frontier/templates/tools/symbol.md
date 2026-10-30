Where a function, class or other symbol is defined, and everything about it: signature, doc, callers, callees, tests. `name` is required.

Resolves `name` (a bare name or a qualified one) through the graph and returns its definition
pointer, signature, first doc line, callers, callees and the tests that touch it — with the
**edge confidence** on every relation. An ambiguous bare name returns every candidate rather
than guessing; qualify it to pick one. With no symbol of that name, it lists the closest by
name (`get_user` finds `getUserName` and `user_get`): call it again with the one you meant.
It searches names only; for other text, use `grep`.

Nothing here is file content. To see the body, `read(path=…, at=name)`.
