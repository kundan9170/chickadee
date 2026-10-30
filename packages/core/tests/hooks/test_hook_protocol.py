"""A script's input and output (`hooks/protocol.py`).

What these pin: exit 0 with nothing is `Proceed`, exit 2 the point's blocking verdict in
stderr's words, anything else a hook error; each JSON field Claude Code names maps to the
verdict it means there, the strongest one winning; `updatedInput` and `updatedToolOutput` are
refused by name, and an unknown key is an error, not a silence; the input carries Claude Code's
names beside ours, validates against its point's model, and holds no secret; and every golden
matches what the models render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chickadee.hooks.export import GOLDENS
from chickadee.hooks.fire import HookError
from chickadee.hooks.payloads import AfterTool, BeforeStop, BeforeTool, HookEvent
from chickadee.hooks.points import HookPoint
from chickadee.hooks.protocol import from_output, input_model, to_input
from chickadee.hooks.verdicts import (
    Ask,
    Continue,
    Deny,
    Halt,
    HookVerdict,
    Note,
    Proceed,
    RetryTask,
    Warn,
)

RUN = "2026-10-24-0c1d2e3f"
TOOL = HookPoint.BEFORE_TOOL


def _decided(decision: str) -> dict[str, str]:
    return {"permissionDecision": decision, "permissionDecisionReason": "r"}


def _json(point: HookPoint, body: object) -> HookVerdict:
    return from_output(point, 0, json.dumps(body), "")


def test_exit_0_with_nothing_or_plain_text_is_proceed() -> None:
    assert from_output(TOOL, 0, "", "") == Proceed()
    assert from_output(TOOL, 0, "all fine\n", "") == Proceed()


@pytest.mark.parametrize(
    ("point", "verdict"),
    [
        (HookPoint.BEFORE_TOOL, Deny("no, use edit", None)),
        (HookPoint.BEFORE_STOP, Continue("no, use edit")),
        (HookPoint.AFTER_TOOL, Note("no, use edit")),
    ],
)
def test_exit_2_is_the_points_blocking_verdict_in_stderrs_words(
    point: HookPoint, verdict: HookVerdict
) -> None:
    assert from_output(point, 2, "", "no, use edit\n") == verdict


def test_exit_2_with_nothing_to_say_is_an_error() -> None:
    with pytest.raises(HookError, match="nothing on stderr"):
        from_output(TOOL, 2, "", "")


def test_any_other_exit_is_an_error_carrying_the_tail() -> None:
    with pytest.raises(HookError, match="exited 1: boom"):
        from_output(TOOL, 1, "", "boom")


@pytest.mark.parametrize(
    ("point", "body", "verdict"),
    [
        (TOOL, {"continue": False, "stopReason": "enough"}, Halt("enough")),
        (TOOL, {"decision": "block", "reason": "r", "do": "send x"}, Deny("r", "send x")),
        (
            TOOL,
            {
                "hookSpecificOutput": {
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "r",
                }
            },
            Deny("r", None),
        ),
        (
            TOOL,
            {
                "hookSpecificOutput": {
                    "permissionDecision": "ask",
                    "permissionDecisionReason": "r",
                }
            },
            Ask("r"),
        ),
        (TOOL, {"hookSpecificOutput": {"permissionDecision": "allow"}}, Proceed()),
        (HookPoint.AFTER_TOOL, {"hookSpecificOutput": {"additionalContext": "n"}}, Note("n")),
        (HookPoint.AFTER_TOOL, {"systemMessage": "w"}, Warn("w")),
        (HookPoint.BEFORE_STOP, {"decision": "block", "reason": "go on"}, Continue("go on")),
        (
            HookPoint.ROUND_ENDED,
            {"retry_task": {"text": "again, smaller", "fresh": True}},
            RetryTask("again, smaller", True),
        ),
    ],
)
def test_each_json_field_means_its_verdict(
    point: HookPoint, body: dict[str, object], verdict: HookVerdict
) -> None:
    assert _json(point, body) == verdict


def test_the_strongest_field_wins() -> None:
    body = {"continue": False, "stopReason": "s", "systemMessage": "w", "decision": "block"}
    assert _json(TOOL, body) == Halt("s")


@pytest.mark.parametrize(
    "body",
    [
        {"hookSpecificOutput": {"updatedInput": {"path": "b.py"}}},
        {"updatedToolOutput": "replaced"},
    ],
)
def test_a_rewrite_is_refused_by_name(body: dict[str, object]) -> None:
    with pytest.raises(HookError, match="is not taken"):
        _json(TOOL, body)


def test_an_unknown_key_is_an_error_not_a_silence() -> None:
    with pytest.raises(HookError, match="systemMesage"):
        _json(TOOL, {"systemMesage": "typo"})


def test_json_that_does_not_parse_is_an_error() -> None:
    with pytest.raises(HookError, match="does not parse"):
        from_output(TOOL, 0, "{not json", "")


def test_the_input_carries_claude_codes_names_beside_ours(tmp_path: Path) -> None:
    payload = AfterTool(
        tool="read", tool_sent="Read", input={"path": "a.py"}, output="text", failed=False
    )
    event = HookEvent(HookPoint.AFTER_TOOL, "worker", payload, run_id=RUN, call_id="c1")
    body = json.loads(to_input(event, tmp_path))
    assert body["hook_event_name"] == "PostToolUse"
    assert body["chickadee_event"] == "after_tool"
    assert (body["tool_name"], body["tool_response"]) == ("read", "text")
    assert body["session_id"] == RUN and body["cwd"] == str(tmp_path)
    input_model(HookPoint.AFTER_TOOL).model_validate(body)


def test_the_stop_input_says_whether_a_continue_was_already_given(tmp_path: Path) -> None:
    payload = BeforeStop(reply="done", continues_used=1, continues_left=0)
    event = HookEvent(HookPoint.BEFORE_STOP, "athena", payload, run_id=RUN)
    body = json.loads(to_input(event, tmp_path))
    assert (body["stop_hook_active"], body["last_assistant_message"]) == (True, "done")
    input_model(HookPoint.BEFORE_STOP).model_validate(body)


def test_no_secret_reaches_a_script(tmp_path: Path) -> None:
    key = "AKIAIOSFODNN7EXAMPLE"
    payload = BeforeTool(tool="shell", tool_sent="shell", input={key: f"export K={key}"})
    event = HookEvent(TOOL, "worker", payload, run_id=RUN, call_id="c1")
    text = to_input(event, tmp_path).decode()
    assert key not in text and "REDACTED" in text


@pytest.mark.parametrize("name", sorted(GOLDENS.contracts))
def test_golden_matches_regeneration(name: str) -> None:
    golden = GOLDENS.path(name)
    assert golden.exists(), f"missing {golden}; run python -m chickadee.hooks.export"
    assert golden.read_text(encoding="utf-8") == GOLDENS.render(name), (
        f"{name} drifted from its golden: regenerate with `python -m chickadee.hooks.export` "
        "and review the diff as a contract change (D54)"
    )


def test_every_point_has_an_input_golden() -> None:
    assert {f"{p.value}_input" for p in HookPoint} | {"output"} == set(GOLDENS.contracts)
