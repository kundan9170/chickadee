"""A key the hand readers let a line omit still may be omitted.

The readers `T-3.56` deleted read an optional string or int as `None` when its key was
absent (`opt_str`, `opt_int`) and a list as empty (`str_tuple`, `_ints`, `p.get(key, [])`).
The types keep that: each case below takes a recorded line holding the key, drops it, and
reads what `main` read. A line that never omitted it is unaffected; one written by hand, or
by an older writer, still reads.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from chickadee.context.lines import SessionKind, parse
from chickadee.context.recall import record_head

RECORDING = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "records"
LINES = RECORDING / "session_lines.jsonl"

OMITTED: list[tuple[SessionKind, tuple[str, ...], object]] = [
    ("episode", ("design_sha",), None),
    ("episode", ("constraints",), ()),
    ("workspace", ("branch",), None),
    ("workspace", ("worktree_path",), None),
    ("compaction", ("lifted",), ()),
    ("constraint", ("scope_id",), None),
    ("constraint", ("valid_until",), None),
    ("constraint", ("superseded_by",), None),
    ("rejected", ("scope_id",), None),
    ("concern", ("anchor",), ()),
    ("concern", ("run_id",), None),
    ("fork", ("at",), None),
    ("tool", ("handle", "ops"), ()),
    ("tool", ("handle", "shape", "lines"), None),
    ("tool", ("handle", "shape", "symbols"), None),
]


def _recorded(kind: str, path: tuple[str, ...]) -> dict[str, Any]:
    """The first recorded `kind` line that holds `path`."""
    for raw in LINES.read_text(encoding="utf-8").splitlines():
        line = json.loads(raw)
        if line["kind"] != kind:
            continue
        payload: dict[str, Any] = json.loads(line["bytes"])
        node: Any = payload
        for key in path[:-1]:
            node = node.get(key)
        if isinstance(node, dict) and path[-1] in node:
            return payload
    raise AssertionError(f"the recording has no {kind} line holding {'.'.join(path)}")


@pytest.mark.parametrize(("kind", "path", "reads_as"), OMITTED)
def test_a_key_main_let_a_line_omit_still_may_be(
    kind: SessionKind, path: tuple[str, ...], reads_as: object
) -> None:
    payload = _recorded(kind, path)
    holder: dict[str, Any] = payload
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    record: Any = parse(kind, json.dumps(payload).encode())
    for key in path:
        record = getattr(record, key)
    assert record == reads_as


def test_a_box_record_s_head_field_that_is_not_text_says_nothing() -> None:
    """Recall read `id`, `outcome`, `goal`, `task` and `failure` only when they were text,
    and a box record is its own writer's to validate, so a number there shows nothing."""
    head = record_head(b'{"id": 7, "outcome": ["x"], "goal": null, "task": "t"}')
    assert (head.id, head.outcome, head.goal, head.task, head.failure) == (
        None,
        None,
        None,
        "t",
        None,
    )
