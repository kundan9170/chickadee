"""Text operations on a file's lines, each in one home.

A leaf: it reads no file, calls no model and holds no packet state, and it imports only
`schema/`, `graph/` and `context/spans`. The Applier, the adapters, the packer and Zeus's
context import it; it imports none of them.

- `lines`: one splitter, one newline policy, and the `N│` line renderer every view uses.
- `match`: the exact text matcher a `search_replace` hunk is resolved by.
- `elision`: whether new text stands a placeholder in for code it omits.
"""
