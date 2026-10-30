"""The head of a task's conversation: `boxctx/worker.task_head`.

What these pin: the fixed instructions are the shipped `worker.md`, the same bytes for every
task, so the provider's cache holds them; the head is they, then "done so far" (each committed
task's record, oldest first, collapsing to one line each over the budget), then the brief.
"""

from __future__ import annotations

from pathlib import Path

from box_helpers import delegate, worker_rig
from engines import answer
from repos import init_repo

from chickadee.box.worker import instructions
from chickadee.boxctx.records import TaskRecord
from chickadee.boxctx.worker import DONE_SO_FAR, TASKS_OMITTED, task_head
from chickadee.context.request import est_text
from chickadee.schema.journal import FileFactP
from chickadee.work import DEFAULT_DIR

T1 = TaskRecord(
    "T1",
    "add the parser",
    files=(FileFactP(path="app/parse.py", added=40, removed=0, kind="created"),),
    tests_added=("tests/test_parse.py",),
    commit="0123456789abcdef",
)
T2 = TaskRecord(
    "T2",
    "use it in stats",
    files=(FileFactP(path="app/stats.py", added=3, removed=1, kind="edited"),),
)


def test_the_fixed_instructions_are_the_same_bytes_for_every_task(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    worker, journal, rung = worker_rig(root, [answer("one"), answer("two")], fixed=None)
    worker.start("T1", "First brief.", delegate(journal, root, "T1"))
    worker.start("T2", "Second brief.", delegate(journal, root, "T2"), done=(T1,))
    first, second = (r.messages[0] for r in rung.requests)
    assert first == second
    assert (first.role, first.content) == ("system", instructions(work_dir=DEFAULT_DIR))
    assert "call `finish` with your report" in instructions(work_dir=DEFAULT_DIR)


def test_the_head_is_the_instructions_then_done_so_far_then_the_brief() -> None:
    system, user = task_head("FIXED", (T1, T2), "Make stats use the parser.", 10_000)
    assert (system.role, system.content) == ("system", "FIXED")
    text = str(user.content)
    assert text.startswith(DONE_SO_FAR) and text.endswith("Make stats use the parser.")
    assert text.index("[T1] add the parser") < text.index("[T2] use it in stats")
    assert "tests/test_parse.py" in text and "commit 0123456" in text


def test_done_so_far_collapses_to_one_line_each_over_its_budget() -> None:
    lines = (
        f"{DONE_SO_FAR}\n[T1] ✓ add the parser · app/parse.py\n"
        "[T2] ✓ use it in stats · app/stats.py"
    )
    _, user = task_head("FIXED", (T1, T2), "Brief.", est_text(lines))
    assert user.content == f"{lines}\n\nBrief."


def test_with_no_room_even_for_lines_the_newest_are_kept_and_the_rest_counted() -> None:
    _, user = task_head("FIXED", (T1, T2), "Brief.", 1)
    assert user.content == f"{DONE_SO_FAR}\n{TASKS_OMITTED.format(k=2)}\n\nBrief."


def test_the_first_task_has_no_done_so_far() -> None:
    _, user = task_head("FIXED", (), "Brief.", 10_000)
    assert user.content == "Brief."
