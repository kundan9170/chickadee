List files whose paths match a glob, one per line: "**/*.py" (any depth), "src/*.ts"
(directly in src), "test_*.py" (that name anywhere). Open one with read(path).

Returns text that also has `.paths`: the list of paths shown, as `glob.glob` returns one.
