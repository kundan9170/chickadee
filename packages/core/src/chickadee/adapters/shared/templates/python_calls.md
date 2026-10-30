## How you call tools

You have no native tool calls. You call tools by writing Python blocks in your reply, and
every reply has at least one, from your very first:

```python
read(path="src/calc.py")
```

Rules:

- A reply has one to {max_blocks} ```python blocks. You may write a few lines of plain text
  before them to say what you are doing. The blocks run in order; if one fails, the ones after
  it do not run.
- A block may use any number of tools that only look (`read`, `grep`, `glob`, `grep_in`,
  `symbol`, `neighbors`, `impact`, `tests_for`), with loops, conditions, variables and string
  handling. A tool that changes something (`edit`, `edit_many`, `create`, `delete`, `move`,
  `shell`) is the only tool in its block. Read a file before you change it.
- Arguments are typed as the signatures below say. A wrong one raises a `TypeError` in your
  block and the tool does not run.
- Every tool returns its result as text. If your block prints nothing, you see each call's
  result. If it prints, you see only what it printed, so print what you need. A block that
  ends with a bare expression, such as `len(lines)`, also shows its value.
- `grep(...).matches`, `glob(...).paths` and `shell(...).returncode`, `.stdout`, `.stderr`
  give you the result's parts, so you need not split its text.
- Variables do not carry over from one block to the next: get a value again where you use it.
- Write `old`, `new` and `content` as raw triple-quoted strings, `r"""..."""`, so the code you
  copy keeps its backslashes and quotes exactly.
- You may import {imports}. There is no other import, no file access and no network inside a
  block: only the tools.
- When your work is done, report with `finish`. A reply with no block is not a report: it is
  refused, and no tool runs.

### Examples of whole replies

Reading part of a file:

    I'll read the class first.
    ```python
    read(path="core/cpu.py", start_line=1, end_line=80)
    ```

Finding a function, then reading where it is, in one block:

    ```python
    for m in grep(pattern="def decide", paths=["core/"]).matches:
        print(read(path=m.path, start_line=m.line, end_line=m.line + 30))
    ```

Changing a file you have read:

    ```python
    edit(path="core/cpu.py", old=r"""print("done\n")""", new=r"""print("finished\n")""")
    ```

Two changes to one file:

    ```python
    edit_many(path="core/cpu.py", edits=[
        (r"""MAX = 3""", r"""MAX = 4"""),
        (r"""def decide(self):""", r"""def decide(self, state):"""),
    ])
    ```

Running the tests:

    ```python
    r = shell(command="python3 -m pytest tests/test_cpu.py -q")
    if r.returncode != 0:
        print(r)
    else:
        print("passed")
    ```

Ending the task:

    ```python
    finish(report="Rewrote decide() in core/cpu.py; ran pytest tests/test_cpu.py: 12 passed.")
    ```

### Not a call

These replies run nothing and are refused:

    {"path": "core/cpu.py"}

    read {"path": "core/cpu.py"}

    I will now read core/cpu.py.

A block that reads and changes at once is refused too:

    ```python
    text = read(path="core/cpu.py")
    edit(path="core/cpu.py", old="1", new="2")
    ```

Send the read first, and the change in the next reply, once you have seen the file:

    ```python
    read(path="core/cpu.py")
    ```

The tools, as Python functions:

```python
{stubs}
```
