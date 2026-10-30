"""`validate(spec, arguments)` — full-schema validation, and the tier-2 stub.

Validation is against the **full** schema regardless of what the model saw: the stub is what
the model reads, the spec is what the harness checks, and a call the stub would admit and the
schema refuses is `INVALID_ARGUMENTS` — the mechanism working, not a gap in it. The failure
carries the tool, each violated field with what was wrong, and the full schema, so the retry
is against a schema now in context. Never prose.

The checker is a small, closed JSON-schema subset — the keywords `SURFACE`'s schemas use and
a dynamic tool's stub may use — rather than a dependency: `type`, `properties`, `required`,
`additionalProperties`, `enum`, `minimum`, `maximum`, `minLength`, `minItems`, `pattern`,
`items`, `oneOf`. A keyword outside the subset is ignored, which is the permissive direction
and is stated here so it is a known edge rather than a surprise.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast, overload

from chickadee.client.request import ToolGrammar, ToolSchema
from chickadee.edits.formats import grammar
from chickadee.errors import HarnessBug
from chickadee.tools.foreign import call_text, foreign_argument, ours_for
from chickadee.tools.spec import (
    BLOCK,
    INVALID_ARGUMENTS,
    RESULT_LINES_DEFAULT,
    SURFACE,
    ToolSpec,
)

__all__ = [
    "INVALID_ARGUMENTS",
    "ArgumentViolation",
    "Arguments",
    "InvalidArguments",
    "ToolBrief",
    "ToolBriefs",
    "resident",
    "stub",
    "tools",
    "validate",
]


@dataclass(frozen=True)
class ArgumentViolation:
    field: str
    """A dotted path from the root; `""` for the root object itself."""
    problem: str


@dataclass(frozen=True)
class InvalidArguments:
    """`INVALID_ARGUMENTS{tool, violations, schema}` — a typed value, never a string."""

    tool: str
    violations: tuple[ArgumentViolation, ...]
    schema: Mapping[str, object]
    failure: str = INVALID_ARGUMENTS
    send: str | None = None
    """The call to send instead, when renaming or dropping the refused arguments makes it pass:
    `create {"path": "core/ai.py", "content": <as you sent it>}`."""

    def as_content(self) -> str:
        """What the model reads: every violation, the call to send when one is known, then the
        schema it is now correct against."""
        lines = [f"{self.failure}: {self.tool}"]
        lines.extend(f"- {v.field or '<arguments>'}: {v.problem}" for v in self.violations)
        if self.send is not None:
            lines.append(f"Send instead: {self.send}")
        else:
            # The instruction, when no call to send is known.
            lines.append(
                f"Do: send {self.tool} again with each argument above fixed, "
                "as the schema says."
            )
        lines.append(f"schema: {self.schema}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Arguments:
    """A call's arguments once `validate` has passed them: read through the types its schema
    names, so no tool narrows them again by hand.

    The schema stays the one declaration of a tool's shape; this only reads what it
    promised. A shape the schema let through but names otherwise is a harness bug, never a
    guess: the schema and the read disagree, and that is fixed in one of them."""

    tool: str
    sent: Mapping[str, object]
    """Exactly as the model sent them (invariant 19): what a digest hashes, a log shows and a
    delegation hands on. Read a value through the methods below, never from here."""

    def text(self, key: str) -> str:
        value = self.sent.get(key)
        if not isinstance(value, str):
            raise self._bug(key, value)
        return value

    def text_or_none(self, key: str) -> str | None:
        value = self.sent.get(key)
        if value is not None and not isinstance(value, str):
            raise self._bug(key, value)
        return value

    @overload
    def integer(self, key: str, default: int) -> int: ...
    @overload
    def integer(self, key: str, default: None) -> int | None: ...
    def integer(self, key: str, default: int | None) -> int | None:
        if key not in self.sent:
            return default
        value = self.sent[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise self._bug(key, value)
        return value

    def anchor(self, key: str) -> int | str:
        """A line anchor, as the policy's style's schema let it through: a number,
        or a `12:a3f` text."""
        value = self.sent.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise self._bug(key, value)
        return value

    @overload
    def boolean(self, key: str, default: bool) -> bool: ...
    @overload
    def boolean(self, key: str, default: None) -> bool | None: ...
    def boolean(self, key: str, default: bool | None) -> bool | None:
        if key not in self.sent:
            return default
        value = self.sent[key]
        if not isinstance(value, bool):
            raise self._bug(key, value)
        return value

    def texts(self, key: str) -> tuple[str, ...]:
        """An absent list is empty: every list a schema here declares is optional or has a
        `minItems`, and the schema has already said which."""
        value = self.sent.get(key, [])
        if not isinstance(value, list):
            raise self._bug(key, value)
        items = cast("list[object]", value)
        if not all(isinstance(v, str) for v in items):
            raise self._bug(key, items)
        return tuple(cast("list[str]", items))

    def integers(self, key: str) -> tuple[int, ...]:
        """A list of integers (`tell`'s `from`); an absent list is empty, as `texts` reads
        one."""
        value = self.sent.get(key, [])
        if not isinstance(value, list):
            raise self._bug(key, value)
        items = cast("list[object]", value)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in items):
            raise self._bug(key, items)
        return tuple(cast("list[int]", items))

    def object(self, key: str) -> Arguments:
        value = self.sent.get(key)
        if not isinstance(value, Mapping):
            raise self._bug(key, value)
        return Arguments(f"{self.tool}.{key}", cast("Mapping[str, object]", value))

    def objects(self, key: str) -> tuple[Arguments, ...]:
        """A list of objects, each read as the arguments it is (`edit`'s `edits`). An
        absent list is empty, as `texts` reads one."""
        value = self.sent.get(key, [])
        if not isinstance(value, list):
            raise self._bug(key, value)
        items = cast("list[object]", value)
        if not all(isinstance(v, Mapping) for v in items):
            raise self._bug(key, items)
        return tuple(
            Arguments(f"{self.tool}.{key}[{i}]", cast("Mapping[str, object]", v))
            for i, v in enumerate(items)
        )

    def _bug(self, key: str, value: object) -> HarnessBug:
        return HarnessBug(f"{self.tool}: {key!r} passed validation as {_name_of(value)}")


_NOT_AN_ARGUMENT = "not an argument of this tool"

_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def validate(spec: ToolSpec, arguments: Mapping[str, object]) -> Arguments | InvalidArguments:
    """`arguments` as the tool reads them when they satisfy the full schema; otherwise the
    typed failure."""
    violations = _violations(spec, arguments)
    if not violations:
        return Arguments(spec.name, arguments)
    send = _send(spec, arguments, violations)
    return InvalidArguments(spec.name, tuple(violations), spec.schema, send=send)


def _send(
    spec: ToolSpec, arguments: Mapping[str, object], violations: Sequence[ArgumentViolation]
) -> str | None:
    """The whole call to send, not only what is wrong with it: a small model told `create`
    takes `path, content` re-sent `command` four times (the 2026-10-19 log), and one shown the
    whole call sent it. Given only when every refused argument has a known fix (an old name's
    replacement, another harness's argument renamed or dropped) and the fixed call passes."""
    fixed: dict[str, object] = {}
    # A violation of the whole call (`exactly one of old and new`) is judged by the fixed call.
    wrong = {v.field for v in violations if v.field}
    for name, value in arguments.items():
        if name not in wrong:
            fixed[name] = value
            continue
        new = spec.renamed.get(name)
        if new is None and foreign_argument(name) is not None:
            new = ours_for(spec.name, name) or ""
        if new is None:
            return None
        if new:
            fixed[new] = value
    if fixed == arguments or _violations(spec, fixed):
        return None
    return call_text(spec.name, fixed)


def _violations(spec: ToolSpec, arguments: Mapping[str, object]) -> list[ArgumentViolation]:
    violations = [_renamed(spec, v) for v in _check(spec.schema, arguments, "")]
    if spec.exactly_one:
        given = sum(all(n in arguments for n in group) for group in spec.exactly_one)
        if given != 1:
            names = ", or ".join(" and ".join(group) for group in spec.exactly_one)
            violations.append(
                ArgumentViolation("", f"exactly one of {names} must be given; got {given}")
            )
    return violations


def _renamed(spec: ToolSpec, violation: ArgumentViolation) -> ArgumentViolation:
    """An old argument name's refusal names its replacement, not the whole list; an
    argument of a changed type, the change; another harness's argument, whose it is
    and what to send here."""
    new = spec.renamed.get(violation.field)
    if new is not None:
        return ArgumentViolation(violation.field, f"renamed: use {new}")
    changed = spec.retyped.get(violation.field)
    if changed is not None:
        return ArgumentViolation(violation.field, changed)
    # Only a name at the top is another harness's argument; one inside an object is not.
    top = "." not in violation.field and "[" not in violation.field
    if top and violation.problem.startswith(_NOT_AN_ARGUMENT):
        return _foreign(spec, violation)
    return violation


def _foreign(spec: ToolSpec, violation: ArgumentViolation) -> ArgumentViolation:
    """`command` on `create` (OpenHands' `str_replace_editor`'s): an instruction not to send it,
    with what this tool takes, or our name for it when the table has one (`old_str` on `edit`
    is `old`). Whose habit it was is not said: the model can act on nothing in it."""
    arg = violation.field
    if foreign_argument(arg) is None:
        return violation
    refused = f"not an argument of {spec.name}. Do not send it"
    mine = ours_for(spec.name, arg)
    if mine is not None:
        return ArgumentViolation(arg, f"{refused}; use {mine}.")
    props = spec.schema.get("properties")
    empty: Mapping[str, object] = {}
    named = cast("Mapping[str, object]", props) if isinstance(props, Mapping) else empty
    takes = ", ".join(named)
    return ArgumentViolation(arg, f"{refused}; {spec.name} takes {takes}.")


def _check(schema: Mapping[str, object], value: object, path: str) -> list[ArgumentViolation]:
    out: list[ArgumentViolation] = []
    given: object = value
    expected = schema.get("type")
    if isinstance(expected, str) and not _is_type(value, expected):
        out.append(ArgumentViolation(path, f"expected {expected}, got {_name_of(value)}"))
        return out
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in cast("list[object]", enum):
        out.append(ArgumentViolation(path, f"must be one of {enum!r}, got {value!r}"))
    if isinstance(value, str):
        out.extend(_check_string(schema, value, path))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.extend(_check_number(schema, value, path))
    if isinstance(value, list):
        out.extend(_check_array(schema, cast("list[object]", value), path))
    if isinstance(value, dict):
        out.extend(_check_object(schema, cast("dict[str, object]", value), path))
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        branches = cast("list[Mapping[str, object]]", one_of)
        matching = sum(1 for b in branches if not _check(b, given, path))
        if matching != 1:
            out.append(ArgumentViolation(path, _one_of_problem(branches, given, matching)))
    return out


def _one_of_problem(branches: list[Mapping[str, object]], value: object, matching: int) -> str:
    """An object's branches differ by what they require; a value's (a line anchor's) by their
    shape, which is what the model must be told."""
    if any("required" in b for b in branches):
        return f"exactly one of {_required_of(branches)} must be given; {matching} matched"
    shapes = [_shape(b) for b in branches]
    return f"must be one of: {'; or '.join(shapes)}; got {value!r}"


def _shape(branch: Mapping[str, object]) -> str:
    """One value branch as the model reads it: its type, and its enum or pattern."""
    enum = branch.get("enum")
    if isinstance(enum, list):
        return " or ".join(repr(v) for v in cast("list[object]", enum))
    pattern = branch.get("pattern")
    kind = branch.get("type", "a value")
    return f"{kind} matching {pattern!r}" if isinstance(pattern, str) else str(kind)


def _required_of(branches: list[Mapping[str, object]]) -> list[list[str]]:
    return [sorted(cast("list[str]", b.get("required", []))) for b in branches]


def _is_type(value: object, expected: str) -> bool:
    kind = _TYPES.get(expected)
    if kind is None:
        return True
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, kind)


def _name_of(value: object) -> str:
    for name, kind in _TYPES.items():
        if isinstance(value, kind) and not (
            name in ("integer", "number") and isinstance(value, bool)
        ):
            return name
    return type(value).__name__


def _check_string(
    schema: Mapping[str, object], value: str, path: str
) -> list[ArgumentViolation]:
    out: list[ArgumentViolation] = []
    least = schema.get("minLength")
    if isinstance(least, int) and len(value) < least:
        out.append(
            ArgumentViolation(path, f"must be at least {least} character(s), got {len(value)}")
        )
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        out.append(ArgumentViolation(path, f"must match {pattern!r}, got {value!r}"))
    return out


def _check_number(
    schema: Mapping[str, object], value: float, path: str
) -> list[ArgumentViolation]:
    """`integer` and `number` alike: a bound is a number too, and `{"type": "number",
    "minimum": 0}` must refuse `-1.5` — tier 2 validates a dynamic tool's full schema, and
    numeric ranges are where floats arrive."""
    out: list[ArgumentViolation] = []
    low, high = schema.get("minimum"), schema.get("maximum")
    if _is_number(low) and value < low:
        out.append(ArgumentViolation(path, f"must be at least {low}, got {value}"))
    if _is_number(high) and value > high:
        out.append(ArgumentViolation(path, f"must be at most {high}, got {value}"))
    return out


def _is_number(bound: object) -> TypeGuard[float]:
    return isinstance(bound, (int, float)) and not isinstance(bound, bool)


def _check_array(
    schema: Mapping[str, object], value: Sequence[object], path: str
) -> list[ArgumentViolation]:
    out: list[ArgumentViolation] = []
    least = schema.get("minItems")
    if isinstance(least, int) and len(value) < least:
        out.append(
            ArgumentViolation(path, f"must have at least {least} item(s), got {len(value)}")
        )
    items = schema.get("items")
    if isinstance(items, Mapping):
        for i, item in enumerate(value):
            out.extend(_check(cast("Mapping[str, object]", items), item, f"{path}[{i}]"))
    return out


def _check_object(
    schema: Mapping[str, object], value: Mapping[str, object], path: str
) -> list[ArgumentViolation]:
    out: list[ArgumentViolation] = []
    props_raw = schema.get("properties")
    empty: Mapping[str, Mapping[str, object]] = {}
    props = (
        cast("Mapping[str, Mapping[str, object]]", props_raw)
        if isinstance(props_raw, Mapping)
        else empty
    )
    required = schema.get("required")
    if isinstance(required, list):
        for name in cast("list[str]", required):
            if name not in value:
                out.append(ArgumentViolation(_join(path, name), "required"))
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in props:
                out.append(
                    ArgumentViolation(
                        _join(path, name),
                        f"{_NOT_AN_ARGUMENT}; known: {sorted(props)}",
                    )
                )
    for name, sub in props.items():
        if name in value:
            out.extend(_check(sub, value[name], _join(path, name)))
    return out


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


# -- the stub and the resident set (tier 1) ------------------------------------


def first_line(description: str) -> str:
    return description.strip().splitlines()[0].strip() if description.strip() else ""


def stub(spec: ToolSpec, description: str) -> ToolSchema:
    """The ~20-token form a dynamic tool is shown as: name, the first line of its
    description, `{"type": "object"}`. `description` is the template's text."""
    return ToolSchema(spec.name, first_line(description), {"type": "object"})


def resident(
    surface: tuple[ToolSpec, ...], descriptions: Mapping[str, str]
) -> tuple[ToolSchema, ...]:
    """The built-ins' full schemas for the prefix (tier 1). `descriptions` maps a template
    name to its text; the whole text is the description, the first line being the stub."""
    return tuple(
        ToolSchema(
            s.name,
            descriptions.get(s.description, "").strip(),
            dict(s.schema),
            _grammar(s),
        )
        for s in surface
    )


def _grammar(spec: ToolSpec) -> ToolGrammar | None:
    """A constrained spec's grammar (`RolePolicy.resolved` keeps one only then), and
    the argument its free text is: the spec's one block argument (`BLOCK`)."""
    if spec.grammar is None:
        return None
    props = spec.schema.get("properties")
    typed = cast(
        "Mapping[str, Mapping[str, object]]", props if isinstance(props, Mapping) else {}
    )
    blocks = [k for k, v in typed.items() if v.get(BLOCK)]
    if len(blocks) != 1 or spec.schema.get("required") != blocks:
        raise HarnessBug(
            f"{spec.name} carries a grammar but not one required block argument: {blocks}"
        )
    return ToolGrammar(argument=blocks[0], definition=grammar(spec.grammar))


# -- `tools` itself --------------------------------------------------


@dataclass(frozen=True)
class ToolBrief:
    name: str
    line: str


@dataclass(frozen=True)
class ToolBriefs:
    """`tools(query=)`'s answer: the briefs, and how many matched — so a cut list says it
    was cut (truncation is a typed value, never an absence the reader has to infer).
    `matched > len(briefs)` is the cut; the briefs are in `surface` order, not relevance
    order, so a model that sees `truncated` narrows the query rather than assuming it saw
    everything."""

    briefs: tuple[ToolBrief, ...]
    matched: int

    @property
    def truncated(self) -> bool:
        return self.matched > len(self.briefs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.briefs)


def tools(
    *,
    query: str | None = None,
    name: str | None = None,
    surface: tuple[ToolSpec, ...] = SURFACE,
    descriptions: Mapping[str, str] | None = None,
    entry_cap: int = RESULT_LINES_DEFAULT,
) -> ToolBriefs | dict[str, object] | InvalidArguments:
    """`tools(query=)` → the name and first line of every tool whose name or first line
    matches, case-insensitively, as `ToolBriefs` — at most `entry_cap` of them, and the
    count that matched, so a cut is typed; `tools(name=)` → that one tool's full schema as
    content, or `{}` for a name the registry does not hold. Both, or neither, is
    `INVALID_ARGUMENTS`. An empty match is `matched == 0`: the registry is in-process and
    total, so the empty set is a fact (invariant 20).

    `entry_cap` is `result_cap`'s number applied as a count — one brief is one line, so
    the lines bound §3 row 8 names and the entry cap are one fact (`RESULT_LINES_DEFAULT`),
    named for what it counts here.

    Pure over `surface` plus whatever dynamic registry the loop hands it later; built here
    over `SURFACE` alone. `descriptions` maps template names to text; absent, the first
    line is the template name. A `surface` without a `tools` entry is a harness bug — the
    tool validates its own arguments against its own schema.
    """
    me = next((s for s in surface if s.name == "tools"), None)
    if me is None:
        raise HarnessBug(
            "tools() was given a surface with no `tools` entry to validate against"
        )
    args: dict[str, object] = {}
    if query is not None:
        args["query"] = query
    if name is not None:
        args["name"] = name
    checked = validate(me, args)
    if isinstance(checked, InvalidArguments):
        return checked
    texts = descriptions or {}

    def line_of(s: ToolSpec) -> str:
        return first_line(texts.get(s.description, "")) or s.description

    if name is not None:
        found = next((s for s in surface if s.name == name), None)
        if found is None:
            return {}
        return {
            "name": found.name,
            "description": texts.get(found.description, "").strip(),
            "parameters": dict(found.schema),
        }
    needle = (query or "").lower()
    hits = [
        ToolBrief(s.name, line_of(s))
        for s in surface
        if needle in s.name.lower() or needle in line_of(s).lower()
    ]
    return ToolBriefs(tuple(hits[:entry_cap]), len(hits))
