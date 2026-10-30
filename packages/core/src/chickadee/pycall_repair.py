"""Reading a worker's Python block that does not parse: the one
home of the arm's text handling before a block runs.

Three outcomes, by how sure the reading is:

- `Clean`: the block parses as written, or after a cleanup with one reading (`\\r\\n` line
  endings, `>>> ` prompts on every line, curly quotes outside a string). Runs for any tool.
- `Repaired`: the only smallest repair that makes it parse. A guess, so it runs only when every
  tool the block uses changes nothing; otherwise the refusal names it.
- `Broken`: no repair, or several. The model is shown the error (`error_text`).

The search is bounded (`BUDGET` parses), and a block past the bound is `Broken`: the prototype
D226 times took 16.5 s unbounded on a 12-line block.
"""

from __future__ import annotations

import ast
import io
import itertools
import tokenize
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

__all__ = [
    "BUDGET",
    "Broken",
    "Clean",
    "Repaired",
    "error_text",
    "read_block",
]

BUDGET: Final = 2000
"""Parses one block's repair may try: about 200 ms at worst."""
_ONE: Final = (")", "]", "}", ",", ":")
"""The tokens a single insertion may add: what a model drops at the end of a call or a line."""
_TWO: Final = (")", "]", "}")
"""Two insertions add closers only: two missing closers is the two-token case seen, and every
other pair is past the bound on any block of length."""
_ROUNDS: Final = 8
_CURLY: Final = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_PRINT: Final = frozenset({"sep", "end", "file", "flush"})


@dataclass(frozen=True)
class Clean:
    """Runs as it is: `cleaned` names each cleanup made, empty when it parsed as written."""

    code: str
    cleaned: tuple[str, ...] = ()


@dataclass(frozen=True)
class Repaired:
    """The only smallest repair: `what` says each insertion, as the result's note does."""

    code: str
    what: tuple[str, ...]
    cleaned: tuple[str, ...] = ()
    error: str = ""
    """The error the block as written raised, for a refusal that names the repair."""


@dataclass(frozen=True)
class Broken:
    error: str


def read_block(code: str, params: Mapping[str, frozenset[str]]) -> Clean | Repaired | Broken:
    """`code` read as D226 decisions 1 and 5 say. `params` maps each function the block may
    call to its keyword names: a repair that passes one a keyword it does not take is no
    repair (`print(grep(pattern="y"), paths=["a"])`)."""
    first = _error(code)
    if first is None:
        return Clean(code)
    cleaned, notes = _cleaned(code)
    error = _error(cleaned)
    if error is None:
        return Clean(cleaned, notes)
    shown = error_text(code, first)
    budget = [BUDGET]
    what: list[str] = []
    current = cleaned
    for _ in range(_ROUNDS):
        step = _string(current, error) or _insertion(current, error, params, budget)
        if step is None:
            return Broken(shown)
        current, said = step
        what.append(said)
        error = _error(current)
        if error is None:
            return Repaired(current, tuple(what), notes, shown)
    return Broken(shown)


def error_text(code: str, exc: SyntaxError) -> str:
    """The block's line, a caret under the column, and Python's message."""
    lines = code.splitlines()
    n = exc.lineno or 0
    if not 1 <= n <= len(lines):
        return f"SyntaxError: {exc.msg}"
    text = lines[n - 1]
    start = max((exc.offset or 1) - 1, 0)
    same_line = exc.end_lineno in (None, n)
    end = (exc.end_offset or 0) - 1 if same_line else len(text)
    width = max(end - start, 1)
    return f"line {n}:\n    {text}\n    {' ' * start}{'^' * width}\nSyntaxError: {exc.msg}"


# -- the cleanups: one reading each ------------------------------------------------------------


def _cleaned(code: str) -> tuple[str, tuple[str, ...]]:
    notes: list[str] = []
    if "\r" in code:
        code = code.replace("\r\n", "\n")
        notes.append("line endings")
    unprompted = _unprompted(code)
    if unprompted is not None:
        code = unprompted
        notes.append(">>> prompts")
    unquoted = _straight_quotes(code)
    if unquoted != code:
        code = unquoted
        notes.append("curly quotes")
    return code, tuple(notes)


def _unprompted(code: str) -> str | None:
    """Each line without its `>>> ` or `... `, when every line that is not blank has one and
    one of them is `>>> `: a block copied from an interactive session."""
    lines = code.split("\n")
    marked = [line for line in lines if line.strip()]
    prompts = (">>> ", "... ")
    if not marked or not all(
        line.startswith(prompts) or line in (">>>", "...") for line in marked
    ):
        return None
    if not any(line.startswith(">>>") for line in marked):
        return None
    return "\n".join(line[4:] if line.startswith(prompts) else "" for line in lines)


def _straight_quotes(code: str) -> str:
    """Each curly quote outside a string made straight, and the string it opens closed by its
    curly or straight partner: a curly quote is never valid Python there, and one inside a
    string is the string's text, left as it is."""
    out: list[str] = []
    i = 0
    while i < len(code):
        ch = code[i]
        if ch == "#":
            end = code.find("\n", i)
            end = len(code) if end < 0 else end
            out.append(code[i:end])
            i = end
            continue
        if ch in "\"'" or ch in _CURLY:
            quote = _CURLY.get(ch, ch)
            delimiter = quote * 3 if ch == quote and code.startswith(ch * 3, i) else quote
            end = _string_end(code, i + len(delimiter), delimiter, curly=ch != quote)
            body = code[i + len(delimiter) : end]
            closer = code[end : end + len(delimiter)]
            if closer and closer[0] in _CURLY:
                closer = _CURLY[closer[0]]
            out.append(delimiter + body + closer)
            i = end + len(delimiter)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _string_end(code: str, start: int, delimiter: str, *, curly: bool) -> int:
    """Where a string's closing delimiter starts, or the block's end. A string a curly quote
    opened closes on either quote of its kind, curly or straight."""
    closers: set[str] = {delimiter[0]}
    if curly:
        closers |= {k for k, v in _CURLY.items() if v == delimiter[0]}
    i = start
    while i < len(code):
        if code[i] == "\\":
            i += 2
            continue
        if len(delimiter) == 3:
            if code.startswith(delimiter, i):
                return i
        elif code[i] in closers or code[i] == "\n":
            return i
        i += 1
    return len(code)


# -- the repairs: a guess each, taken only when it is the only smallest one --------------------


def _string(code: str, error: SyntaxError) -> tuple[str, str] | None:
    """A `"..."` or `'...'` string that runs across lines, made triple-quoted, when the very
    next same quote is followed by what ends an argument: the only place it can close."""
    if not error.msg.startswith("unterminated string literal"):
        return None
    at = _offset(code, error.lineno or 0, (error.offset or 1) - 1)
    if at is None or code[at] not in "\"'":
        return None
    quote = code[at]
    if code.startswith(quote * 3, at):
        return None
    close = _next_quote(code, at + 1, quote)
    if close is None or "\n" not in code[at:close]:
        return None
    rest = code[close + 1 :].lstrip(" \t")
    if not rest or rest[0] not in ",)]}":
        return None
    body = code[at + 1 : close]
    if quote * 3 in body or body.endswith("\\"):
        return None
    fixed = code[:at] + quote * 3 + body + quote * 3 + code[close + 1 :]
    return fixed, f"made the string at line {error.lineno} triple-quoted"


def _next_quote(code: str, start: int, quote: str) -> int | None:
    i = start
    while i < len(code):
        if code[i] == "\\":
            i += 2
            continue
        if code[i] == quote:
            return i
        i += 1
    return None


def _insertion(
    code: str,
    error: SyntaxError,
    params: Mapping[str, frozenset[str]],
    budget: list[int],
) -> tuple[str, str] | None:
    """The only smallest insertion that parses: one token, else two closers (`grep("x",
    paths=["a"` needs `])`). When none parses, the only smallest one that moves the error
    later, so the next round can go on. None when there are several, or none, or the budget
    runs out."""
    places = _places(code, error)
    here = _where(error)
    progress: list[dict[str, str]] = []
    for size, tokens in ((1, _ONE), (2, _TWO)):
        parsed: dict[str, str] = {}
        moved: dict[str, str] = {}
        for at, put in _candidates(places, tokens, size):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            fixed = _inserted(code, at, put)
            failed = _error(fixed)
            if failed is None:
                if _calls_fit(fixed, params):
                    parsed[fixed] = _said(code, at, put)
            elif _where(failed) > here:
                moved[fixed] = _said(code, at, put)
        if parsed:
            return _only(parsed)
        progress.append(moved)
    for moved in progress:
        if moved:
            return _only(moved)
    return None


def _only(found: dict[str, str]) -> tuple[str, str] | None:
    if len(found) != 1:
        return None
    [(fixed, said)] = found.items()
    return fixed, said


def _candidates(
    places: list[int], tokens: tuple[str, ...], size: int
) -> Iterator[tuple[tuple[int, ...], tuple[str, ...]]]:
    for at in itertools.combinations_with_replacement(places, size):
        for put in itertools.product(tokens, repeat=size):
            yield at, put


def _inserted(code: str, at: tuple[int, ...], put: tuple[str, ...]) -> str:
    """`put` inserted at `at`, in the order given where two share an offset: `]` then `)` is
    `])`, a different repair from `)]`, and both are tried."""
    order = sorted(range(len(at)), key=lambda i: (at[i], i), reverse=True)
    for i in order:
        code = code[: at[i]] + put[i] + code[at[i] :]
    return code


def _said(code: str, at: tuple[int, ...], put: tuple[str, ...]) -> str:
    parts: list[str] = []
    for offset, token in zip(at, put, strict=True):
        line = code.count("\n", 0, offset) + 1
        column = offset - (code.rfind("\n", 0, offset) + 1) + 1
        parts.append(f"added `{token}` at line {line}, column {column}")
    return "; ".join(parts)


def _places(code: str, error: SyntaxError) -> list[int]:
    """Where a missing token may go: the end of each token on the error's line and the line
    before it, every line after an opener Python says was never closed, and the block's end."""
    n = error.lineno or 1
    last = code.count("\n") + 1
    lines = (
        range(n, last + 1) if "was never closed" in error.msg else range(max(n - 1, 1), n + 1)
    )
    wanted = set(lines)
    out: set[int] = {len(code.rstrip())}
    for row, column in _token_ends(code):
        if row in wanted:
            at = _offset(code, row, column)
            if at is not None:
                out.add(at)
    return sorted(out)


def _token_ends(code: str) -> list[tuple[int, int]]:
    """Each token's end, as far as the tokenizer reads a block that does not parse."""
    skip = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    ends: list[tuple[int, int]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type not in skip:
                ends.append(token.end)
    except (tokenize.TokenError, SyntaxError):
        pass
    return ends


def _calls_fit(code: str, params: Mapping[str, frozenset[str]]) -> bool:
    """Whether every call to an offered function, and to `print`, passes only keywords it
    takes, and nothing calls what one of them returned (`print(grep)(pattern="y")`): a tool
    returns text and `print` returns None, so neither result is callable."""
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        inner = node.func
        called = inner.func if isinstance(inner, ast.Call) else None
        if isinstance(called, ast.Name) and (called.id == "print" or called.id in params):
            return False
        if not isinstance(inner, ast.Name):
            continue
        takes = _PRINT if inner.id == "print" else params.get(inner.id)
        if takes is None:
            continue
        if any(k.arg is not None and k.arg not in takes for k in node.keywords):
            return False
    return True


# -- positions ---------------------------------------------------------------------------------


def _error(code: str) -> SyntaxError | None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return exc
    except ValueError:  # a null byte: no repair reads past it
        return SyntaxError("source code cannot contain null bytes", ("<block>", 1, 1, "", 1, 1))
    return None


def _where(error: SyntaxError) -> tuple[int, int]:
    return (error.lineno or 0, error.offset or 0)


def _offset(code: str, line: int, column: int) -> int | None:
    """The index into `code` of a 1-based line and 0-based column."""
    if line < 1:
        return None
    start = 0
    for _ in range(line - 1):
        nl = code.find("\n", start)
        if nl < 0:
            return None
        start = nl + 1
    at = start + column
    return at if at <= len(code) else None
