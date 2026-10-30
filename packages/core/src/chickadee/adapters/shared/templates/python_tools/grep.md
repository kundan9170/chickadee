Find text in the repository's files: `pattern` is a regular expression (case-sensitive),
`paths` optional globs to limit it (["src/**/*.py"]). One `path:line: text` per hit; pass a
hit's path and line to `read`. Hits past the cap are counted, not shown: narrow the search.

Returns text that also has `.matches`: a list of `Match(path, line, text)`, one per hit shown.
    for m in grep(pattern="def decide").matches:
        print(read(path=m.path, start_line=m.line, end_line=m.line + 20))
