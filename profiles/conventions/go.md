# Go conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Code is `gofmt`-formatted; imports grouped standard library first.
- Return errors, never panic for an expected failure; wrap with `fmt.Errorf("...: %w", err)`.
- Check every returned error; `_ =` only with a comment saying why it is safe.
- Accept interfaces, return concrete types. Keep interfaces small, defined where used.
- Exported names carry a doc comment starting with the name.
- Table-driven tests in `*_test.go`, beside the code they test.
- Comments explain **why**, not what.
