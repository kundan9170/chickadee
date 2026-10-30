"""No call carries a purpose line (`tools/meta.py`): no role's schema names `why` or
`intent`, a call without one runs, and one a model still sends is refused as any argument a
tool does not name is.

Moved from `tools/test_why.py`, each on purpose: every schema listed
`why` first and required it; a call without one was refused with an example; a long one was
recorded cut and said; the records read it as `intent`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from box.box_helpers import delegate, episode_rig, worker_rig
from box.box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo
from zeus.zeus_helpers import env, rig

from chickadee.boxctx.athena import athena_head
from chickadee.client.request import ToolGrammar, ToolSchema
from chickadee.roles import role
from chickadee.schema import ToolCall
from chickadee.schema.journal import BoxMessageP
from chickadee.tools.meta import shown, with_meta
from chickadee.tools.spec import SURFACE
from chickadee.work import DEFAULT_DIR
from chickadee.zeus.prefix import assemble

APP = "def go():\n    return 1\n"
PURPOSE = {"why", "intent"}


def _repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _tool_texts(journal: object) -> list[str]:
    lines = journal.projection().lines(BoxMessageP)  # type: ignore[attr-defined]
    return [
        json.loads(journal.get_blob(p.payload.sha))["content"]  # type: ignore[attr-defined]
        for p in lines
        if p.payload.side == "tool"
    ]


def _no_purpose(schema: Mapping[str, Any], name: str) -> None:
    named = set(schema.get("properties", {})) | set(schema.get("required", []))
    assert not named & PURPOSE, name


# -- the schemas ---------------------------------------------------------------------------


def test_no_zeus_tool_names_a_purpose_line_and_findings_stay_last(tmp_path: Path) -> None:
    p = assemble(role("zeus"), SURFACE, "", env(tmp_path / "ws"), work_dir=DEFAULT_DIR)
    for t in p.tools:
        _no_purpose(t.schema, t.name)
        assert list(t.schema["properties"])[-1] == "findings", t.name


def test_no_worker_tool_names_a_purpose_line(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    worker, journal, rung = worker_rig(root, [answer("nothing to do")])
    worker.start("T1", "Do it.", delegate(journal, root))
    [sent] = rung.requests
    assert sent.tools
    for t in sent.tools:
        _no_purpose(t.schema, t.name)
        assert "findings" not in t.schema["properties"], t.name
    shell = {t.name: t for t in sent.tools}["shell"].schema
    # T-3.178 adds `sandbox`, after the others.
    assert list(shell["properties"]) == ["command", "cwd", "network", "idempotent", "sandbox"]
    assert list(shell["required"]) == ["command"]


def test_no_athena_tool_names_a_purpose_line(tmp_path: Path) -> None:
    r = episode_rig(_repo(tmp_path), [answer("nothing")], [])
    r.athena.start(athena_head("You are Athena.", "P", "B", design="# Design"))
    sent = r.athena_rung.requests[0]
    assert {"plan", "delegate", "review"} <= {t.name for t in sent.tools}
    for t in sent.tools:
        _no_purpose(t.schema, t.name)


def test_a_tool_under_a_grammar_is_shown_as_it_is() -> None:
    free = ToolSchema("patch", "d", {"type": "object"}, ToolGrammar("text", "g"))
    assert shown((free,)) == (free,)


def test_only_zeus_is_shown_findings() -> None:
    own = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert with_meta(own) == own
    zeus = with_meta(own, findings=True)
    assert list(zeus["properties"]) == ["a", "findings"]  # type: ignore[call-overload]
    assert zeus["required"] == ["a"]


# -- a call without one runs ---------------------------------------------------------------


def test_zeus_runs_a_call_without_a_purpose_line(tmp_path: Path) -> None:
    r = rig(
        tmp_path, [calls(ToolCall(id="c1", name="glob", arguments={"pattern": "*"})), answer()]
    )
    r.run("x")
    got = r.session.state.result("c1")
    assert got is not None and got.ran


def test_the_worker_runs_a_call_without_a_purpose_line(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    bare = ToolCall(id="c1", name="glob", arguments={"pattern": "**/*.py"})
    worker, journal, _ = worker_rig(root, [calls(bare), answer("listed")])
    worker.start("T1", "List the code.", delegate(journal, root))
    [text] = _tool_texts(journal)
    assert "app.py" in text


def test_athena_runs_a_call_without_a_purpose_line(tmp_path: Path) -> None:
    bare = ToolCall(id="a1", name="read", arguments={"path": "app.py"})
    r = episode_rig(_repo(tmp_path), [calls(bare), answer("read it")], [])
    assert r.athena.start(athena_head("You are Athena.", "P", "B", design="# D")).report == (
        "read it"
    )
    [text] = _tool_texts(r.journal)
    assert not text.startswith("INVALID_ARGUMENTS")


def test_a_refusal_of_bad_arguments_shows_no_purpose_line(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    worker, journal, _ = worker_rig(root, [calls(call("glob", "c1", pattern=5)), answer("ok")])
    worker.start("T1", "List the code.", delegate(journal, root))
    [text] = _tool_texts(journal)
    assert text.startswith("INVALID_ARGUMENTS: glob") and "'why'" not in text


# -- one still sent is refused ------------------------------------------------------------


def test_zeus_refuses_a_why_or_an_intent_as_an_argument_glob_does_not_take(
    tmp_path: Path,
) -> None:
    stray = [
        ToolCall(id="c1", name="glob", arguments={"why": "look", "pattern": "*"}),
        ToolCall(id="c2", name="glob", arguments={"intent": "look", "pattern": "*.md"}),
    ]
    r = rig(tmp_path, [calls(*stray), answer()])
    r.run("x")
    for cid, key in (("c1", "why"), ("c2", "intent")):
        got = r.session.state.result(cid)
        assert got is not None and not got.ran, cid
        assert got.text.startswith("INVALID_ARGUMENTS: glob") and f"- {key}:" in got.text, cid


def test_the_worker_refuses_a_why_as_an_argument_glob_does_not_take(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    stray = ToolCall(id="c1", name="glob", arguments={"why": "look", "pattern": "**/*.py"})
    worker, journal, _ = worker_rig(root, [calls(stray), answer("listed")])
    worker.start("T1", "List the code.", delegate(journal, root))
    [text] = _tool_texts(journal)
    assert text.startswith("INVALID_ARGUMENTS: glob") and "- why:" in text


# -- no model-facing text labels a reason `why` -----------------------------------


def test_no_failure_a_model_reads_labels_its_reason_why() -> None:
    """A refusal that showed `why:` as a key taught the model to send `why` again."""
    from chickadee.context.results import HandleExpired, Interrupted, InvalidArgument
    from chickadee.schema.execution import SandboxLevel
    from chickadee.shell import NetworkDenied, SandboxRequired, WriteDenied
    from chickadee.tools.rules import BoundReached

    rendered = [
        InvalidArgument("path", "a file path", example="'a.py'").render(),
        HandleExpired("h_1", "absent").render(),
        Interrupted("shell", "the person stopped it").render(),
        BoundReached("tool_calls", 4, 4, "turn").render(),
        NetworkDenied("looks like a network failure").render(),
        WriteDenied("x", "read-only").render(),
        SandboxRequired(SandboxLevel.NONE, "require_sandbox is set").render(),
    ]
    for text in rendered:
        assert "why" not in text and "reason: " in text, text
