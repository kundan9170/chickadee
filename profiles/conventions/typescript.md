# TypeScript conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- `strict` typing: no implicit `any`; reach for `unknown` and narrow it, never `any`.
- Prefer `type` aliases and interfaces at module boundaries; let local types be inferred.
- Named exports over a default export.
- `async`/`await` over raw promise chains; a floating promise is a silent failure.
- Strict equality (`===`) unless a null-ish check genuinely wants `== null`.
- Comments explain **why**, not what.
- Follow the repository's own tsconfig, lint and format settings over these.
