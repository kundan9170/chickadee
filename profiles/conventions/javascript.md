# JavaScript conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Modern syntax: `const`/`let`, arrow functions, async/await over raw promise chains.
- Strict equality (`===`) unless a null-ish check genuinely wants `==`.
- Comments explain **why**, not what.
- Handle rejected promises; a floating promise is a silent failure.
- Prefer named exports over a default export.
