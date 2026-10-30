"""Validation is against the full schema regardless of what the model saw: a call the stub
admits and the schema refuses is `INVALID_ARGUMENTS` carrying the
full schema; a valid call passes untouched; `note_concern` with an empty `anchor` is
`INVALID_ARGUMENTS`, not a prose refusal."""

from __future__ import annotations

from chickadee.tools.builtin import move
from chickadee.tools.spec import SURFACE, Bound, ToolSpec, spec
from chickadee.tools.validate import Arguments, InvalidArguments, validate


def _spec(name: str) -> ToolSpec:
    found = spec(name)
    assert found is not None
    return found


def test_a_call_the_stub_admits_and_the_schema_refuses_is_invalid_arguments() -> None:
    """The stub is `{"type": "object"}`: it admits anything. The schema does not."""
    result = validate(_spec("read"), {"file": "src/app.py"})
    assert isinstance(result, InvalidArguments)
    assert result.tool == "read" and result.failure == "INVALID_ARGUMENTS"
    assert result.schema == _spec("read").schema, "the full schema travels with the failure"
    fields = {v.field for v in result.violations}
    assert fields == {"", "file"}, result.violations
    problems = {v.field: v.problem for v in result.violations}
    assert problems[""] == "exactly one of path, or handle must be given; got 0"
    assert "not an argument" in problems["file"]


def _passes(tool: ToolSpec, args: dict[str, object]) -> bool:
    """Valid, and handed on untouched: since a valid call is its `Arguments`, holding
    exactly what was sent, where it was `None` before."""
    result = validate(tool, args)
    return isinstance(result, Arguments) and result.sent == args


def _refused(tool: ToolSpec, args: dict[str, object]) -> bool:
    return isinstance(validate(tool, args), InvalidArguments)


def test_valid_arguments_pass_untouched() -> None:
    assert _passes(_spec("read"), {"path": "src/app.py"})
    assert _passes(_spec("read"), {"handle": "h1", "lines": "10-40", "depth": "full"})
    assert _passes(_spec("shell"), {"command": "pytest -q"})
    assert _passes(_spec("atlas"), {})
    assert _passes(_spec("neighbors"), {"target": "go", "depth": 2})


def test_note_concern_with_an_empty_anchor_is_invalid_arguments_not_a_string() -> None:
    result = validate(_spec("note_concern"), {"text": "fragile test", "anchor": []})
    assert isinstance(result, InvalidArguments)
    (violation,) = result.violations
    assert violation.field == "anchor" and "at least 1" in violation.problem
    assert not isinstance(result, str)


def test_note_rejected_with_an_empty_reason_is_invalid_arguments() -> None:
    result = validate(_spec("note_rejected"), {"text": "use redis", "reason": ""})
    assert isinstance(result, InvalidArguments)
    assert [v.field for v in result.violations] == ["reason"]


def test_read_takes_exactly_one_of_path_and_handle() -> None:
    # `target` was both; each now has its own name, and a call gives one.
    for args in ({}, {"path": "a.py", "handle": "h1"}, {"lines": "1-2"}):
        result = validate(_spec("read"), args)
        assert isinstance(result, InvalidArguments)
        assert (
            "",
            "exactly one of path, or handle must be given; got "
            + str(sum(k in args for k in ("path", "handle"))),
        ) in [(v.field, v.problem) for v in result.violations]


def test_no_tool_schema_carries_a_top_level_combinator() -> None:
    """Anthropic's `input_schema` refuses a top-level `oneOf`, `anyOf` or `allOf`, and
    every schema is sent to every provider as it is. Exactly-one rules are `exactly_one`."""
    from chickadee.tools.builtin import BUILTINS

    for name in BUILTINS.names():
        tool = BUILTINS.get(name)
        assert tool is not None
        found = {"oneOf", "anyOf", "allOf"} & set(tool.spec.schema)
        assert not found, f"{name}'s schema has a top-level {sorted(found)}"


def test_edit_takes_old_and_new_or_edits_never_both() -> None:
    from chickadee.tools.builtin import edit

    one = {"path": "a.py", "old": "x", "new": "y"}
    many = {"path": "a.py", "edits": [{"old": "x", "new": "y"}]}
    assert isinstance(validate(edit.SPEC, one), Arguments)
    assert isinstance(validate(edit.SPEC, many), Arguments)
    for bad, got in (
        ({**one, **many}, 2),
        ({"path": "a.py", "old": "x"}, 0),
        ({"path": "a"}, 0),
    ):
        result = validate(edit.SPEC, bad)
        assert isinstance(result, InvalidArguments)
        assert (
            "",
            f"exactly one of old and new, or edits must be given; got {got}",
        ) in [(v.field, v.problem) for v in result.violations]


def test_an_old_argument_name_is_refused_naming_its_replacement() -> None:
    # D198, a moved answer: the refusal names the new name, not the whole list.
    read = validate(_spec("read"), {"target": "a.py"})
    assert isinstance(read, InvalidArguments)
    assert ("target", "renamed: use path, or handle") in [
        (v.field, v.problem) for v in read.violations
    ]
    moved = validate(move.SPEC, {"source": "a.py", "target": "b.py"})
    assert isinstance(moved, InvalidArguments)
    problems = {v.field: v.problem for v in moved.violations}
    assert problems["source"] == "renamed: use path"
    assert problems["target"] == "renamed: use destination"
    assert problems["path"] == problems["destination"] == "required"


def test_each_keyword_of_the_subset_refuses_what_it_should() -> None:
    assert _refused(_spec("read"), {"path": "x", "depth": "everything"})
    assert _refused(_spec("read"), {"path": "x", "lines": "ten to twenty"})
    assert _refused(_spec("read"), {"path": "x", "max_bytes": 0})
    assert _refused(_spec("neighbors"), {"target": "x", "depth": 9})
    assert _refused(_spec("neighbors"), {"target": "x", "depth": True})
    assert _refused(_spec("impact"), {"symbols": []})
    assert _refused(_spec("impact"), {"symbols": ["go", ""]})
    assert _refused(_spec("shell"), {"command": ""})
    assert _refused(_spec("shell"), {"command": "ls", "idempotent": "yes"})
    assert _refused(_spec("coder"), {"request": "not an object"})


def test_number_bounds_hold_for_floats_too() -> None:
    """`minimum` / `maximum` were checked only when the value was an `int`, so a
    `{"type": "number", "minimum": 0}` schema admitted `-1.5` — silently under-enforced
    on the schemas tier 2 exists to validate in full. A bound may be a float as well."""
    number = ToolSpec(
        name="probe",
        family="navigate",
        parallel="any",
        mutates="nothing",
        tier=_spec("atlas").tier,
        model_calls="none",
        schema={
            "type": "object",
            "properties": {
                "ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "scale": {"type": "number", "minimum": 0.5},
                "count": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        description="probe.md",
        bound=Bound("none"),
        failures=frozenset({"INVALID_ARGUMENTS"}),
        effects="none",
        idempotent=True,
    )
    assert _passes(number, {"ratio": 0.5, "scale": 0.5, "count": 1})
    below = validate(number, {"ratio": -1.5})
    assert isinstance(below, InvalidArguments) and "at least 0" in below.violations[0].problem
    above = validate(number, {"ratio": 1.5})
    assert isinstance(above, InvalidArguments) and "at most 1" in above.violations[0].problem
    half = validate(number, {"scale": 0})
    assert isinstance(half, InvalidArguments) and "at least 0.5" in half.violations[0].problem
    assert _refused(number, {"count": 0})
    assert _refused(number, {"count": True}), "a bool is not a number"


def test_violations_name_nested_fields() -> None:
    result = validate(_spec("impact"), {"symbols": ["go", 3]})
    assert isinstance(result, InvalidArguments)
    assert result.violations[0].field == "symbols[1]"


def test_as_content_is_the_violations_then_the_schema() -> None:
    result = validate(_spec("grep"), {})
    assert isinstance(result, InvalidArguments)
    text = result.as_content()
    assert text.startswith("INVALID_ARGUMENTS: grep")
    assert "- pattern: required" in text and "schema:" in text


def test_every_surface_schema_is_an_object_with_declared_properties() -> None:
    for s in SURFACE:
        assert s.schema["type"] == "object"
        assert isinstance(s.schema.get("properties"), dict)
        assert s.schema.get("additionalProperties") is False
        assert isinstance(s.bound, Bound)
