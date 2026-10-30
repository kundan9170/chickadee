List the files whose paths match a glob. `pattern` is required.

`**/*.py` matches Python files at any depth, the root included; `src/*.ts` the TypeScript files
directly in `src`; a pattern with no `/` matches file names anywhere (`test_*.py`). Every file
the repository does not ignore is listed, one path per line. Open one with `read(path=…)`.

Hits past the result cap are counted, not shown: narrow the pattern. `UNRESOLVED` means no file
matched.
