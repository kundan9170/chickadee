"""`chickadee inspect`'s reader (`chickadee.inspect`): a run as the page reads it, each call's
exact request with its messages and tools given once, and only the lines past `after`. Read
through the facade. A session's reading is `tests/zeus/test_inspect_session.py`."""

from __future__ import annotations

from box_helpers import BoxRig, design, worker_call
from box_helpers import athena_call as acall
from engines import answer, calls

from chickadee.engine import inspect_runs

BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"


def test_a_run_reads_with_each_calls_request_and_only_new_lines(rig: BoxRig) -> None:
    rig.athena_script = [
        calls(acall("plan", "a1", text="1. **Return 2.**\n")),
        calls(acall("delegate", "a2", title="Return 2", brief=BRIEF)),
        calls(acall("review", "a3", task="T1", answer="accept")),
        answer("done"),
    ]
    rig.worker_script = [
        calls(worker_call("edit", "w1", path="app.py", old="return 1", new="return 2")),
        answer("T1: go() returns 2."),
    ]
    report = rig.box().open(design()).report()
    inspection = inspect_runs(rig.workspaces[0].root)

    [listed] = inspection.index()["runs"]
    assert (listed["id"], listed["outcome"]) == (report.run_id, "COMPLETED")

    whole = inspection.run(listed["id"])
    assert whole["damage"] is None
    events = whole["events"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    system = next(e for e in events if e["kind"] == "box_message")
    assert system["message"]["role"] == "system" and system["message"]["content"]
    first_call = next(e for e in events if e["kind"] == "model_call")
    sent = first_call["sent"]
    assert sent["messages"] and all(m in whole["messages"] for m in sent["messages"])
    assert [t["name"] for t in whole["tools"][sent["tools"]]]

    later = inspection.run(listed["id"], after=events[-3]["seq"])
    assert [e["seq"] for e in later["events"]] == [e["seq"] for e in events[-2:]]
