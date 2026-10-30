Find text in the repository's files: a regular expression over file contents, one `path:line: text` per hit. `pattern` is required.

It reads every file the repository does not ignore, configuration and documentation included,
not only the indexed code. Matching is case-sensitive, and hidden files are included. `paths`
limits it to files matching its globs (`["src/**/*.py"]`). The result's first line names the
engine: ripgrep reads `pattern` as a regular expression; without it, as literal text.

Hits past the result cap are counted, not shown: narrow the pattern or the paths. A hit's line
can be passed to `read(path=…, lines=)`. `UNRESOLVED` names what was searched.
