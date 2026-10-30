# C conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Follow the C standard the build declares; no compiler extensions unless the code has them.
- Check every return value that can signal failure (`malloc`, I/O, system calls).
- Every allocation has one clear owner and one `free`; document who frees in the header.
- Bound every buffer write: `snprintf`, never `sprintf`; no `gets`.
- `static` for anything not used outside its file; headers have include guards.
- Prefer `const` for pointers to data a function does not change.
- Comments explain **why**, not what.
