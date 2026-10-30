# Ruby conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Follow the repository's RuboCop settings when it has them.
- Two-space indentation; `snake_case` methods and variables, `CamelCase` classes.
- `# frozen_string_literal: true` at the top of a file, where the codebase uses it.
- Rescue a specific error class, never a bare `rescue`; never rescue `Exception`.
- Prefer small methods and guard clauses over nested conditionals.
- Specs in `spec/` mirror the file under test (`lib/a/b.rb` → `spec/a/b_spec.rb`).
- Comments explain **why**, not what.
