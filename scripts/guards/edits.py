"""The edits guard (under D148 decision 4): each text operation an edit or a view
needs has one home in `chickadee/edits/`.

`T-3.127` moved the text matcher out of the Applier and the line splitter and renderer out of
Zeus's context package into `edits/`, so what the worker is shown and what its anchors are
matched against cannot disagree; `T-3.128` gave each reply format one parser in
`edits/formats/`; `T-3.130` gave each tool protocol one module in `adapters/protocols/`. This
fails the next copy, by eight rows, over `packages/core/src/chickadee/**/*.py`, the adapter
families' templates and every grammar file under `src/`. E7 came with the draft,
E5's line tag with the one tag function, E6 with the first grammar.

- **E6, a grammar outside `edits/formats/`, or beside no parser:** a `.lark` or `.gbnf` file
  outside `edits/formats/`, or one there with no `<name>.py` beside it; or a string constant,
  outside a docstring and outside `edits/formats/`, holding a grammar's own syntax (a line
  opening `%import`, or a `start:` rule). A grammar lives beside the parser that holds it to
  its language, so a second copy, or one with no parser, is a language nobody
  checks.

- **E1, a second format parser, outside `edits/formats/`:** a string constant, outside a
  docstring, holding a format's marker: `<<<<<<<` or `>>>>>>>` (SEARCH/REPLACE), `*** Begin
  Patch`, `*** End Patch`, `*** Update File`, `*** Add File`, `*** Delete File`, `*** Move to`
  (V4A), or one that starts with `@@` (a hunk header, a regex's `^` allowed before it). A
  module that names a marker is reading a format.
- **E4, a tool's arguments written outside its spec, in a template:** a template holding a
  hand-written JSON schema (`"properties":`) or a typed argument list (`TOOL edit
  path=<string>`, `<a path>` too). A tool's arguments are its spec's, rendered by its protocol;
  an example line, in a family's `examples/`, names values, never types.
- **E8, a second protocol, outside `adapters/protocols/`:** a string constant, outside a
  docstring, that reads a text call (`TOOL ` at its start, after a regex's flags, `^` and
  `\\s*`; `tool\\s` in any case; or `NEED:`); a test of a word against `"TOOL"` or `"NEED:"`
  (`==`, `in`, `startswith`); or a comparison or `match` case on a protocol's
  name: `"prompt_simple"` or `"prompt_schema"` anywhere, `"native"` beside a name or attribute
  saying `protocol` (`native` is a sandbox backend and a call's form too, which a count of
  forms compares). Reading a call or branching on the protocol is the protocol's.
- **E7, draft text changed outside `CandidateDraft.apply`:** in `edits/draft.py`, a store to,
  a delete from, or a mutating call on a draft's text (an attribute `_texts`, or an item of
  it) anywhere but `CandidateDraft.apply`, its constructor, `_hold` (a file taken on as read)
  and the call scope's restore;
  anywhere else, any reference to it, a read included, and `setattr`/`getattr`/`delattr` by
  its name: the text is the draft's alone. A tool that wrote the draft itself would skip the
  scope and elision checks.
- **E2, a second text matcher, outside `edits/match.py`:** a call `.count`, `.find`, `.rfind`,
  `.index`, `.rindex`, `.replace`, `.split`, `.rsplit`, `.partition` or `.rpartition` whose
  first argument is an anchor: a name or attribute spelled `old` or `anchor`, alone or with
  `_text`, `_str`, `_string`, `_line` or `_lines` (`hunk.old`, `old_string`). The name is the
  signal because what carries an anchor names it so; `text.find(needle)` is not one, and
  `old in by_id` is a comparison, not a call.
- **E3, a diff reaching the Applier as a diff, outside `schema/`:** `ApplyMode.DIFF`. A
  unified diff is translated to hunks first; code branching on the mode keeps a
  second path.
- **E5, a second line renderer or splitter, outside `edits/lines.py`:**
  - a string holding the bar `│`, in a plain string or an f-string's literal part, outside a
    docstring: building a numbered row and parsing one both need it;
  - a `path:N:` hit, an f-string running `{P}:{L}:…`, `P` a name or attribute whose last word
    is `path`, `file`, `filename`, `rel` or `target`, `L` one whose last word is `line`,
    `lineno`, `number`, `row` or `n`. By name, since shape alone would take an id
    (`{kind}:{packet_id}:{attempt}:{n}`) or a mount (`{src}:{dst}:ro`) for a hit; a call
    (`row.get("line")`) is not a name, so a lint tool's own diagnostics repeated are not hits;
  - a numbering splitter: `enumerate(` over `X.splitlines(…)` or `X.split("\\n")`, or either
    indexed or sliced by anything but constants. `splitlines` also splits on a form feed and
    `\\u2028`, and `split("\\n")` keeps a CR, so the lines they number are not the file's.
    Iterated, counted, or taken at `[0]`, they number nothing (process output, a config file,
    model text) and pass;
  - any such splitter in `boxctx/` or `chickadee/tools/`, where the worker's views and tools
    hold a target's text and its anchors are made: there a split that numbers
    nothing yet still decides what a line is;
  - a line tag: a hex digest cut from its start to four digits or fewer (`.hexdigest()[:3]`),
    or to `TAG_DIGITS`. A tag made twice is a tag that can drift, and an anchor copied from
    one view then fails in another. A longer cut is an id or a digest, not a tag.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. What it cannot see: an anchor held in a name none of those, matched
through `operator` or `re.search(re.escape(old))`; a hit built with `%` or `":".join`; lines
split once and numbered later through a variable; a splitter written as a loop over
`find("\\n")`; a marker built from pieces (`"<" * 7`), held in a bytes literal, or imported
from a format's module (`from …search_replace import SEARCH`); a tag cut by a variable bound
or through `[0:n]` spelled another way. That is the review checklist's
first question. E8 cannot see a JSON call read through a `["tool"]` key (a TOML `[tool]` table
reads the same), a protocol held in a constant and compared through it, or `"native"` compared
beside a name that does not say `protocol`. E7 cannot see a draft's text aliased inside
`edits/draft.py` and written through the alias, or reached through `vars()` or `__dict__`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import docstrings, functions

REPO_ROOT = Path(__file__).resolve().parents[2]

SRC = "packages/core/src/chickadee/"
FORMATS = f"{SRC}edits/formats/"
"""The reply formats' parsers: E1's home."""
PROTOCOLS = f"{SRC}adapters/protocols/"
"""The tool protocols: E8's home."""
DRAFT = f"{SRC}edits/draft.py"
"""The worker's draft: E7's home."""
DRAFT_WRITERS = frozenset(
    {
        "CandidateDraft.apply",
        "CandidateDraft.__init__",
        "CandidateDraft._hold",
        "CandidateDraft._restore",
    }
)
"""The draft's one write, and the places that set its text whole: made (the constructor, and
`_hold`, which takes a file on as `read` gives it, for the constructor and for D190's open
scope), and undone. None of them writes text a model gave."""
MATCH = f"{SRC}edits/match.py"
"""The one text matcher: E2's home."""
LINES = f"{SRC}edits/lines.py"
"""The one splitter and line renderer: E5's home."""
WORKER_VIEWS = (f"{SRC}boxctx/", f"{SRC}tools/")
"""Where E5 fails every line split, not only one that numbers."""
SCHEMA = f"{SRC}schema/"
"""Where `ApplyMode` is declared, `DIFF` with it: E3's one place."""

BAR = "│"
"""What separates a row's number from its text; restated, as a guard imports only the
standard library. The test holds it to `edits.lines.BAR`."""

_MARKERS = (
    "<<<<<<<",
    ">>>>>>>",
    "*** Begin Patch",
    "*** End Patch",
    "*** Update File",
    "*** Add File",
    "*** Delete File",
    "*** Move to",
)
"""What opens or closes a format's edit, in SEARCH/REPLACE and V4A; a hunk header is `_HUNK`."""
_HUNK = "@@"

_ONLY_PROTOCOLS = frozenset({"prompt_simple", "prompt_schema"})
_PROTOCOL_NAMES = _ONLY_PROTOCOLS | {"native"}
_TOOL_OPENING = re.compile(r"(?:\(\?[a-zA-Z]+\))?\^?(?:\\s[*+])?TOOL ")
_TOOL_PATTERN = re.compile(r"tool\\s", re.I)
_KEYWORDS = frozenset({"TOOL", "NEED"})
_SCHEMA_IN_TEMPLATE = re.compile(r'"properties"\s*:')
_TYPED_ARGS = re.compile(r"^\s*TOOL\s+\S+.*=<[^<>\n]+>")

GRAMMAR_SUFFIXES = frozenset({".lark", ".gbnf"})
"""Grammar files: Lark (what OpenAI's decoder reads) and GBNF (llama.cpp's). E6's corpus."""
_GRAMMAR_SYNTAX = re.compile(r"(?m)^[ \t]*%import\b|^start[ \t]*:")

_DRAFT_TEXT = "_texts"
"""The draft's text store, by its attribute's name: E7's signal."""
_MUTATORS = frozenset({"update", "pop", "popitem", "clear", "setdefault", "__setitem__"})
"""`dict` methods that change it."""

_SEARCHES = frozenset(
    {"count", "find", "rfind", "index", "rindex", "replace", "split", "rsplit"}
    | {"partition", "rpartition"}
)
"""`str` methods that look for their first argument in the text."""
_ANCHOR = re.compile(r"(?:old|anchor)(?:_(?:text|str|string|line|lines))?")
_TAG_DIGITS_MAX = 4
"""The longest digest prefix E5 reads as a line tag; a tag is three today."""
_TAG_BOUND = "TAG_DIGITS"
_PATHISH = frozenset({"path", "file", "filename", "rel", "target"})
_LINEISH = frozenset({"line", "lineno", "number", "row", "n"})

EXEMPT: dict[tuple[str, str], str] = {
    (f"{SRC}tools/validate.py", "first_line"): (
        "E5: a tool description's first line, the stub's summary: a spec's prose, not a "
        "target's text"
    ),
    (f"{SRC}graph/indexer.py", "git_recency"): (
        "E5: ranks the paths `git log --name-only` prints, newest first; process output, "
        "whose position is a rank, not a file's line"
    ),
}


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST
    what: str


def _name(node: ast.AST) -> str | None:
    """A name's or an attribute's own name: `old`, or `old` in `hunk.old`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _last_word(node: ast.AST) -> str | None:
    name = _name(node)
    return name.rsplit("_", 1)[-1].lower() if name is not None else None


def _marker(node: ast.AST) -> str | None:
    """The format marker a string constant holds, if any."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return None
    hit = next((m for m in _MARKERS if m in node.value), None)
    if hit is None and node.value.lstrip("^").startswith(_HUNK):
        hit = _HUNK
    return hit


def _text_call(node: ast.AST) -> str | None:
    """A string that reads a text call: `TOOL ` opening it (after a regex's flags, `^` and
    `\\s*`), `tool\\s` in any case, or `NEED:`."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return None
    value = node.value
    if _TOOL_OPENING.match(value) or _TOOL_PATTERN.search(value):
        return "a TOOL line"
    return "NEED:" if "NEED:" in value else None


def _keyword_test(node: ast.AST) -> str | None:
    """A reply's word tested against a call's keyword: `word == "TOOL"`, or
    `text.startswith("TOOL")`. The bare word is everywhere else (an `__all__`, an enum's value),
    so only a test of it reads a call; `role == "tool"` is a message's role, so the spelling is
    the keyword's."""
    if isinstance(node, ast.Compare):
        constants = [node.left, *node.comparators]
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "startswith"
    ):
        constants = list(node.args)
    else:
        return None
    for c in constants:
        if (
            isinstance(c, ast.Constant)
            and isinstance(c.value, str)
            and c.value.rstrip(": ") in _KEYWORDS
        ):
            return f"a test for {c.value!r}"
    return None


def _protocol_branch(node: ast.AST) -> str | None:
    """A comparison or a `match` case on a protocol's name. `prompt_simple` and
    `prompt_schema` name nothing else; `native` names a sandbox backend and a call's form too,
    so it is a protocol only beside a name or attribute that says `protocol`."""
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
    elif isinstance(node, ast.Match):
        operands = [node.subject, *(c.pattern for c in node.cases)]
    else:
        return None
    about = any("protocol" in ast.unparse(o).lower() for o in operands)
    for operand in operands:
        for name in _protocol_names(operand):
            if name in _ONLY_PROTOCOLS or about:
                return f"== {name!r}"
    return None


def _protocol_names(node: ast.AST) -> Iterator[str]:
    """The protocol names a comparison's operand or a case's pattern holds."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value in _PROTOCOL_NAMES:
            yield sub.value


def _draft_text(node: ast.AST) -> bool:
    """`x._texts`, or an item of it."""
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Attribute) and node.attr == _DRAFT_TEXT


def _draft_write(node: ast.AST) -> str | None:
    """A store, a delete or a mutating call on a draft's text."""
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = node.targets
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _MUTATORS
        and _draft_text(node.func.value)
    ):
        return f"{_DRAFT_TEXT}.{node.func.attr}"
    return f"a store to {_DRAFT_TEXT}" if any(_draft_text(t) for t in targets) else None


def _draft_reached(node: ast.AST) -> str | None:
    """A draft's text reached from outside its module, a read included: it is the draft's
    alone, and a reference held elsewhere is a write waiting to happen (`t = d._texts`)."""
    if isinstance(node, ast.Attribute) and node.attr == _DRAFT_TEXT:
        return f"{_DRAFT_TEXT} outside edits/draft.py"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("setattr", "getattr", "delattr")
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == _DRAFT_TEXT
    ):
        return f"{node.func.id}(…, {_DRAFT_TEXT!r})"
    return None


def _matcher(node: ast.AST) -> str | None:
    """`X.find(old)` and its kin, as written."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SEARCHES
        and node.args
    ):
        anchor = _name(node.args[0])
        if anchor is not None and _ANCHOR.fullmatch(anchor):
            return f".{node.func.attr}({anchor})"
    return None


def _diff_mode(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "DIFF"
        and _name(node.value) == "ApplyMode"
    )


def _hit(node: ast.AST) -> bool:
    """An f-string with `{path}:{line}:` in it."""
    if not isinstance(node, ast.JoinedStr):
        return False
    parts = node.values
    for i in range(len(parts) - 3):
        path, colon, line, after = parts[i : i + 4]
        if (
            isinstance(path, ast.FormattedValue)
            and _last_word(path.value) in _PATHISH
            and isinstance(colon, ast.Constant)
            and colon.value == ":"
            and isinstance(line, ast.FormattedValue)
            and _last_word(line.value) in _LINEISH
            and isinstance(after, ast.Constant)
            and isinstance(after.value, str)
            and after.value.startswith(":")
        ):
            return True
    return False


def _splitter(node: ast.AST) -> str | None:
    """`X.splitlines(…)` or `X.split("\\n")`, as written."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr == "splitlines":
        return ".splitlines()"
    if (
        node.func.attr == "split"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "\n"
    ):
        return '.split("\\n")'
    return None


def _line_tag(node: ast.AST) -> str | None:
    """A hex digest cut from its start to a tag's length: `.hexdigest()[:3]`."""
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Slice)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "hexdigest"
    ):
        return None
    cut = node.slice
    lower = cut.lower
    if cut.step is not None or not (
        lower is None or (isinstance(lower, ast.Constant) and lower.value == 0)
    ):
        return None
    upper = cut.upper
    if isinstance(upper, ast.Constant) and isinstance(upper.value, int):
        if 0 < upper.value <= _TAG_DIGITS_MAX:
            return f".hexdigest()[:{upper.value}]"
        return None
    if upper is not None and _name(upper) == _TAG_BOUND:
        return f".hexdigest()[:{_TAG_BOUND}]"
    return None


def _constant(node: ast.AST | None) -> bool:
    """A literal index or bound: `0`, `-1`, or none."""
    if node is None or isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)


def _numbering(node: ast.AST) -> str | None:
    """Lines numbered by a splitter other than `split_lines`."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enumerate"
        and node.args
    ):
        split = _splitter(node.args[0])
        if split is not None:
            return f"enumerate({split})"
    if isinstance(node, ast.Subscript):
        split = _splitter(node.value)
        index = node.slice
        constant = (
            _constant(index.lower) and _constant(index.upper) and _constant(index.step)
            if isinstance(index, ast.Slice)
            else _constant(index)
        )
        if split is not None and not constant:
            return f"{split}[…]"
    return None


def _sites(tree: ast.Module, rel: str) -> Iterator[_Site]:
    prose = docstrings(tree)
    for scope, node in functions(tree):
        if not rel.startswith(PROTOCOLS):
            read = _text_call(node) if id(node) not in prose else None
            if read is not None:
                yield _Site("E8", scope, node, read)
            test = _keyword_test(node)
            if test is not None:
                yield _Site("E8", scope, node, test)
            branch = _protocol_branch(node)
            if branch is not None:
                yield _Site("E8", scope, node, f"a protocol branched on {branch}")
        if rel == DRAFT:
            write = _draft_write(node) if scope not in DRAFT_WRITERS else None
            if write is not None:
                yield _Site("E7", scope, node, write)
        else:
            reached = _draft_reached(node)
            if reached is not None:
                yield _Site("E7", scope, node, reached)
        if not rel.startswith(FORMATS) and id(node) not in prose and _grammar_text(node):
            yield _Site("E6", scope, node, "a grammar's text")
        if not rel.startswith(FORMATS) and id(node) not in prose:
            marker = _marker(node)
            if marker is not None:
                yield _Site("E1", scope, node, f"the marker {marker!r}")
        if rel != MATCH:
            matcher = _matcher(node)
            if matcher is not None:
                yield _Site("E2", scope, node, matcher)
        if not rel.startswith(SCHEMA) and _diff_mode(node):
            yield _Site("E3", scope, node, "ApplyMode.DIFF")
        if rel == LINES:
            continue
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and BAR in node.value
            and id(node) not in prose
        ):
            yield _Site("E5", scope, node, "the bar")
        if _hit(node):
            yield _Site("E5", scope, node, "a path:N: hit")
        cut = _line_tag(node)
        if cut is not None:
            yield _Site("E5", scope, node, f"a line tag, {cut}")
        if rel.startswith(WORKER_VIEWS):
            split = _splitter(node)
            if split is not None:
                yield _Site("E5", scope, node, f"{split} in a worker view")
            continue
        numbering = _numbering(node)
        if numbering is not None:
            yield _Site("E5", scope, node, numbering)


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every E1, E2, E3, E5, E7 and E8 site in `tree`, the module at repo-relative path
    `rel`."""
    return list(_sites(tree, rel))


def template_sites(lines: Sequence[str], rel: str) -> list[Violation]:
    """E4: a template's hand-written schema or typed argument list. An example line names
    values, so a type in one (`path=<string>`) is a typed argument list too."""
    out: list[Violation] = []
    for n, line in enumerate(lines, start=1):
        if _SCHEMA_IN_TEMPLATE.search(line) or _TYPED_ARGS.match(line):
            out.append(
                Violation(path=rel, line=n, text=f"E4 (a tool's arguments): {line.strip()}")
            )
    return out


def grammar_sites(path: Path, rel: str) -> list[Violation]:
    """E6: a grammar file outside `edits/formats/`, or one there with no parser beside it."""
    if not rel.startswith(FORMATS):
        return [Violation(path=rel, line=0, text="E6 (a grammar outside edits/formats/)")]
    if not path.with_suffix(".py").is_file():
        parser = path.with_suffix(".py").name
        return [Violation(path=rel, line=0, text=f"E6 (a grammar with no parser: {parser})")]
    return []


def _grammar_text(node: ast.AST) -> bool:
    """A string constant holding a grammar's own syntax: a `%import` line or a `start:` rule."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _GRAMMAR_SYNTAX.search(node.value) is not None
    )


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one."""
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        if path.suffix == ".md":
            violations.extend(template_sites(lines, rel))
            continue
        if path.suffix in GRAMMAR_SUFFIXES:
            violations.extend(grammar_sites(path, rel))
            continue
        for site in sites(ast.parse(source), rel):
            key = (rel, site.function)
            if key in EXEMPT:
                used.add(key)
                continue
            line = getattr(site.node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(
                Violation(path=rel, line=line, text=f"{site.rule} ({site.what}): {text}")
            )
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


# E5: the scout's search hit, spelled `path:N: text`; the worker's (`path:N:line`) moved onto
# `edits.lines.hit`.
BASELINE: dict[str, int] = {
    "packages/core/src/chickadee/scout/search.py": 1,
}

GUARD = Guard(
    name="edits",
    rule=(
        "one home per text operation (D172 §15): read a call or branch on a tool protocol only "
        "in chickadee/adapters/protocols (E8); write a tool's arguments only in its spec, "
        "never a schema or typed argument list in a template (E4); read a reply format only in "
        "chickadee/edits/formats, never its markers elsewhere (E1); keep a grammar there, "
        "beside its parser, never a .lark or .gbnf file or a grammar's text elsewhere (E6); "
        "change the worker's draft "
        "only in CandidateDraft.apply (E7); match an anchor through "
        "chickadee.edits.match.resolve, never str.count/find/replace of it (E2); no "
        "ApplyMode.DIFF outside schema/, a diff becomes hunks first (E3); number, render "
        "and parse lines and tag them through chickadee.edits.lines (split_lines, row, "
        "parse_row, tag), never the bar, a path:N: hit, a line tag, an enumerated splitlines, "
        "or any line split in boxctx/ or tools/ outside it (E5)"
    ),
    corpus=(
        "packages/core/src/**/*.py",
        "packages/core/src/chickadee/adapters/*/templates/**/*.md",
        "packages/core/src/**/*.lark",
        "packages/core/src/**/*.gbnf",
    ),
    find=find,
    baseline=BASELINE,
)
