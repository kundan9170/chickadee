# PHP conventions

Rendered into the coder's stable prefix (R-2). How code here is written, not what to do.

- `declare(strict_types=1);` at the top of every file.
- PSR-12 style and PSR-4 autoloading: one class per file, the namespace matching its path.
- Type every parameter, return and property the language allows.
- Throw a specific exception; never suppress errors with `@`.
- Use prepared statements for every query; never interpolate input into SQL.
- Tests in `tests/`, one `*Test.php` per class under test.
- Comments explain **why**, not what.
