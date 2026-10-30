# C# conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Nullable reference types are on: annotate what may be null, and handle it.
- `async`/`await` all the way down; never `.Result` or `.Wait()` on a task.
- `PascalCase` for types and members, `camelCase` for locals and parameters, `_camelCase` for
  private fields.
- Dispose what is disposable: `using` declarations.
- Catch the specific exception; never swallow one with an empty `catch`.
- Prefer records for plain data and `readonly` where state does not change.
- Comments explain **why**, not what.
