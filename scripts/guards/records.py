"""The records guard: a record is read through its type, never as a dict.

Journal payloads were read back by dumping them to dicts or casting them to a payload class,
and session-log lines and tool arguments by narrowing a decoded JSON object key by key
(#276 T2.1 to T2.3). `T-3.56` to `T-3.58` moved every such read onto the types. This fails
the next one, by three rules:

- **R1:** `model_dump` is for writing. A call anywhere in `packages/core/src` fails unless its
  function is exempt as a serializer.
- **R2:** no `cast` to a journal payload class: a class `schema/journal.py`'s `KIND_TABLE`
  lists. The narrowing (`isinstance`) is the check a cast skips.
- **R3:** in the records corpus, no key is read off a parameter annotated as a decoded JSON
  object (`Payload`, `Mapping[str, object]`, `Mapping[str, Any]`, `dict[str, object]`,
  `dict[str, Any]`) — a subscript by a string, or `.get` with one — and nothing is cast to
  such a type.

Exemptions are keyed by file and qualified function name, each with its reason; one that no
longer exempts anything fails, as the storage guard's does.

What it cannot see: a dict reached some other way and then read. That is the review
checklist's question 6.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = "packages/core/src/chickadee/"
SCHEMA = REPO_ROOT / CORE / "schema" / "journal.py"

R1_EXEMPT: dict[tuple[str, str], str] = {
    (CORE + "journal/writer.py", "Journal.append"): "the journal writer: a payload to its line",
    (CORE + "journal/gc.py", "blob_shas"): (
        "every field of a payload walked for the blobs it names: gc's and the inspector's"
    ),
    (CORE + "boxctx/sent.py", "record_sent"): (
        "a request's sampling params written to a run blob as they were sent (D224); no field "
        "is read back"
    ),
    (CORE + "inspect.py", "_row"): (
        "a journal event written out for the inspector's page as stored; its fields are read "
        "through the payload's type, never off this dump"
    ),
    (CORE + "client/openai_client.py", "dump"): "the OpenAI client's wire dump of a response",
    (CORE + "box/episode.py", "_snapshot"): (
        "a box run's config snapshot: the request it was opened with, read back by its type"
    ),
    (CORE + "scopes.py", "_run_record"): (
        "a run event's payload as a host reads it, written out as stored for the wire "
        "(30-serve §8.3); no field is read back"
    ),
}
"""D151 decision 7's serializers: each writes a model out, and none reads a field back."""

R3_CORPUS: tuple[str, ...] = (
    CORE + "context/",
    CORE + "boxctx/records.py",
    CORE + "scout/tools.py",
    CORE + "scout/escapes.py",
    CORE + "zeus/",
    CORE + "tools/",
    CORE + "journal/projection.py",
    CORE + "journal/metrics.py",
)
"""The modules that read records and tool arguments. A prefix ending in
`/` is a directory."""

R3_EXEMPT_FILES: dict[str, str] = {
    CORE + "tools/validate.py": (
        "validating against a schema means reading untyped keys off the call and the schema; "
        "its walkers do that by construction (D151 decision 7)"
    ),
}

_JSON_FILE = "outlines a JSON file, whose content has no type"
_FOREIGN = (
    "another harness's call, refused before any schema could type it: its arguments as sent "
    "are all there is to read the suggestion from (D209, T-3.181)"
)

R3_EXEMPT: dict[tuple[str, str], str] = {
    (CORE + "context/recall.py", "_find_key"): (
        "`_pick`'s reader of a call's arguments as sent (invariant 19, D151 decision 6)"
    ),
    (CORE + "context/recall.py", "_strings"): (
        "every string a call or a box record holds, for recall's search text"
    ),
    (CORE + "tools/meta.py", "with_meta"): (
        "reads a tool's schema, not a record (D151); moved with `with_meta` from "
        "`zeus/prefix.py` (T-3.166)"
    ),
    (CORE + "context/record_lines.py", "RecordLine._typed"): (
        "a journal-record line's own validator (the `scout` and `shell_effect` lines): it "
        "types the payload by its event's `KIND_TABLE` model, which is the parse itself; "
        "moved here with the validator from `ScoutLine._typed` (T-3.144)"
    ),
    (CORE + "tools/foreign.py", "translate"): _FOREIGN,
    (CORE + "tools/foreign.py", "_command"): _FOREIGN,
    (CORE + "tools/foreign.py", "_takes"): (
        "reads a tool's schema for the names it takes, not a record (D209, T-3.181)"
    ),
    (CORE + "context/outline.py", "_type_of"): _JSON_FILE,
    (CORE + "context/outline.py", "_walk"): _JSON_FILE,
    (CORE + "context/store.py", "_parse_line"): (
        "the session log's envelope (seq, at, kind, payload sha, run id), each field checked; "
        "the payload it names is typed by `lines.parse` (T-3.56)"
    ),
    (CORE + "context/store.py", "_first_line"): (
        "a child log's first envelope, read for its kind and seq before `_parse_line` reads it"
    ),
}
"""Readers of something that is not a record, each with its reason: a call's arguments as
sent, a tool's schema, a JSON file's content, and the session log's
envelope. D151 also names `compact._subject`, `recall._pick`, `prefix.split_meta` and
`boxctx/evict._subject`; none reads a key R3 can see, so none has an entry: an entry that
exempts nothing fails."""

_JSON_OBJECT = frozenset(
    {
        "Payload",
        "Mapping[str, object]",
        "Mapping[str, Any]",
        "dict[str, object]",
        "dict[str, Any]",
    }
)


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST


def payload_classes(schema: Path = SCHEMA) -> frozenset[str]:
    """The payload class names `KIND_TABLE` lists, read from its source: a guard imports
    only the standard library, so it reads the table rather than importing it."""
    tree = ast.parse(schema.read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "KIND_TABLE" and value is not None:
            if not isinstance(value, ast.Dict):
                break
            names = {
                v.elts[0].id
                for v in value.values
                if isinstance(v, ast.Tuple) and v.elts and isinstance(v.elts[0], ast.Name)
            }
            if names:
                return frozenset(names)
    raise ValueError(f"no KIND_TABLE of payload classes in {schema}")


def _cast_target(node: ast.Call) -> str | None:
    """`cast(T, x)`'s `T` as written, whether a string or an expression."""
    func = node.func
    named = isinstance(func, ast.Name) and func.id == "cast"
    attr = isinstance(func, ast.Attribute) and func.attr == "cast"
    if not (named or attr) or not node.args:
        return None
    target = node.args[0]
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value.strip()
    return ast.unparse(target)


def _json_params(fn: ast.AST) -> frozenset[str]:
    if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
        return frozenset()
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    return frozenset(
        p.arg
        for p in params
        if p.annotation is not None and _annotation(p.annotation) in _JSON_OBJECT
    )


def _annotation(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ast.unparse(node)


def _key_read(node: ast.AST, params: frozenset[str]) -> bool:
    """`p["key"]` or `p.get("key",...)` on one of `params`."""
    if isinstance(node, ast.Subscript):
        key = node.slice
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in params
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        owner = node.func.value
        return (
            node.func.attr == "get"
            and isinstance(owner, ast.Name)
            and owner.id in params
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
    return False


def sites(tree: ast.Module, payloads: frozenset[str], *, records: bool) -> list[_Site]:
    """Every R1 and R2 site in `tree`, and every R3 site when it is in the records corpus."""
    out: list[_Site] = []
    params: dict[str, frozenset[str]] = {}
    for scope, node in functions(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            params[scope] = _json_params(node)
            continue
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "model_dump":
                out.append(_Site("R1", scope, node))
            target = _cast_target(node)
            if target is not None and target in payloads:
                out.append(_Site("R2", scope, node))
            if records and target is not None and target in _JSON_OBJECT:
                out.append(_Site("R3", scope, node))
        if records and _key_read(node, params.get(scope, frozenset())):
            out.append(_Site("R3", scope, node))
    return out


def _in_corpus(rel: str) -> bool:
    return any(rel == c or (c.endswith("/") and rel.startswith(c)) for c in R3_CORPUS)


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one.

    Parses core's source, which is Python 3.12: run it with 3.12 or later, as CI's
    `python3` is."""
    payloads = payload_classes(SCHEMA)
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    used_files: set[str] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        records = _in_corpus(rel)
        found = sites(ast.parse(source), payloads, records=records)
        lines = source.splitlines()
        for site in found:
            key = (rel, site.function)
            if site.rule == "R1" and key in R1_EXEMPT:
                used.add(key)
                continue
            if site.rule == "R3" and rel in R3_EXEMPT_FILES:
                used_files.add(rel)
                continue
            if site.rule == "R3" and key in R3_EXEMPT:
                used.add(key)
                continue
            line = getattr(site.node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=f"{site.rule}: {text}"))
    for rel, fn in sorted((set(R1_EXEMPT) | set(R3_EXEMPT)) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    for rel in sorted(set(R3_EXEMPT_FILES) - used_files):
        violations.append(Violation(path=rel, line=0, text="stale exemption: the whole file"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="records",
    rule=(
        "a record is read through its type: narrow a journal payload with isinstance, never "
        "dump (R1) or cast (R2) it, and read no key off a decoded JSON object (R3) (D151)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
