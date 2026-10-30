# C++ conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- Follow the standard the build declares (`CMAKE_CXX_STANDARD`).
- RAII for every resource: `std::unique_ptr` and containers, never a bare `new`/`delete`.
- Pass by `const&` for read-only parameters larger than a pointer; return by value.
- Mark single-argument constructors `explicit`, overrides `override`, and what does not
  change state `const`.
- Prefer `enum class` and `std::optional`/`std::variant` over sentinels.
- No `using namespace` in a header.
- Comments explain **why**, not what.
