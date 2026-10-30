"""What the TUI reads from the wire (T-5.23 Part 8), end to end through a hosted engine:
the question timeout on `GET /`, spend by role and the warning levels, each model's window and
the role-to-rung wiring, the header's sandbox level and uncommitted paths, and a rung's switch
as a readable record."""

from __future__ import annotations

import pytest
from engines import answer
from hosting import SID, answered, hosted, until
from pydantic import ValidationError

from chickadee.config import InteractionPolicy
from chickadee_serve.frames import ErrorCode
from chickadee_wire.errors import WireError
from chickadee_wire.models import ModelSwitchedRecord, TitleRecord
from chickadee_wire.records import payload


def test_hello_says_how_long_a_question_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        assert client.server is not None
        assert client.server.question_timeout_s == InteractionPolicy().question_timeout_s


def test_the_budget_has_spend_by_role_and_the_warning_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch, [answer("hello")]) as host, host.client() as client:
        client.turn(SID, "hi")
        answered(client, 1)
        budget = client.budget(SID)
        assert budget.warning_fractions == [0.8]
        # Zeus's conversation is the session log's own charges; no run has spent yet.
        assert budget.roles is not None and set(budget.roles) == {"zeus"}
        assert budget.roles["zeus"] == pytest.approx(budget.spent_usd)


def test_models_name_each_roles_rung_and_each_models_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        wiring = client.models(SID)
        assert wiring.roles is not None
        assert wiring.roles["zeus"] in wiring.rungs and wiring.roles["worker"] in wiring.rungs
        assert wiring.models is not None and set(wiring.models) == set(wiring.catalog)
        for model in wiring.models.values():
            assert model.context_window > model.max_output_tokens > 0
            assert model.provider


def test_the_header_says_the_sandbox_level_and_the_uncommitted_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        (host.repo / "scratch.txt").write_text("not committed\n")
        header = client.header(SID)
        assert header.sandbox, "the level commands run at here, a value"
        assert any(line.kind == "sandbox" for line in header.lines)
        assert "scratch.txt" in header.uncommitted


def test_a_rung_switch_is_a_record_a_client_can_read(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        wiring = client.models(SID)
        assert wiring.roles is not None
        # To the model Zeus's rung runs on: the one the test config prices.
        rung, other = wiring.roles["worker"], wiring.rungs[wiring.roles["zeus"]]
        assert wiring.rungs[rung] != other
        client.switch_model(SID, rung, other)
        [frame] = [f for f in client.session_journal(SID) if f.kind == "model_switched"]
        moved = payload(frame)
        assert isinstance(moved, ModelSwitchedRecord)
        assert (moved.rung, moved.to_model) == (rung, other)


def test_a_runs_blob_and_diff_are_refused_by_reason_over_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run with no journal is not found, as its records are. The
    reads themselves, and their other refusals, are `run_reads`' tests."""
    with hosted(monkeypatch) as host, host.client() as client:
        with pytest.raises(WireError) as missing:
            client.run_blob("2026-10-24-0a1b2c3d", "0" * 64)
        assert missing.value.code == ErrorCode.NOT_FOUND
        with pytest.raises(WireError) as also_missing:
            client.task_diff("2026-10-24-0a1b2c3d", "T1")
        assert also_missing.value.code == ErrorCode.NOT_FOUND


def test_a_listed_file_is_read_and_any_other_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CODE pane reads a file `/files` lists, and no other: an ignored or absent
    path is not found, as `@` never offers it."""
    with hosted(monkeypatch) as host, host.client() as client:
        (host.repo / "notes.md").write_text("one\ntwo\n")
        (host.repo / ".gitignore").write_text("ignored.txt\n")
        (host.repo / "ignored.txt").write_text("hidden\n")
        assert "notes.md" in client.files(SID)
        read = client.file_text(SID, "notes.md")
        said = (read.path, read.size, read.text, read.withheld)
        assert said == ("notes.md", 8, "one\ntwo\n", None)
        for path in ("ignored.txt", "absent.py", "../outside.txt"):
            with pytest.raises(WireError) as refused:
                client.file_text(SID, path)
            assert refused.value.code == ErrorCode.NOT_FOUND, path


def test_a_binary_file_is_named_and_not_inlined(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        (host.repo / "logo.bin").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
        read = client.file_text(SID, "logo.bin")
        assert read.text is None and read.size == 10
        assert read.withheld is not None and "UTF-8" in read.withheld


def test_a_session_is_named_after_its_first_turn_and_renamed_by_its_person(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The title role's answer is the session's name, a readable `title` record; the
    person's `/rename` wins, and a name with no words is refused before it is sent."""
    script = [answer("PR #670 adds the question card."), answer("PR 670 UI review")]
    with hosted(monkeypatch, script, auto_title=True) as host, host.client() as client:
        client.turn(SID, "Can you look at the latest pr 670. Which is about ui.")
        answered(client, 1)
        named = until(client, "title")
        record = payload(named)
        assert isinstance(record, TitleRecord) and (record.text, record.by) == (
            "PR 670 UI review",
            "model",
        )
        assert client.session_info(SID).title == "PR 670 UI review"
        assert client.rename(SID, "TUI panes").title == "TUI panes"
        with pytest.raises(ValidationError):
            client.rename(SID, "   ")  # checked as the server checks it, before it is sent
        assert client.session_info(SID).title == "TUI panes"


def test_the_work_is_none_until_the_session_opens_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/session/{sid}/work`, the engine's fold of the open work across its runs, what
    the screen draws a continued work's earlier tasks from; None while the session has none."""
    with hosted(monkeypatch, [answer("hello")]) as host, host.client() as client:
        client.turn(SID, "hi")
        answered(client, 1)
        assert client.work(SID) is None
        with pytest.raises(WireError):
            client.work("S-nobody")


def test_the_work_s_fold_reads_back_as_the_wire_s_model() -> None:
    """What `_work` answers, read by the client's model: each task, its state and its run."""
    from chickadee.work_progress import PartState, TaskStatus, WorkKey
    from chickadee.work_progress import WorkProgress as Fold
    from chickadee_serve.app import _wire  # pyright: ignore[reportPrivateUsage]
    from chickadee_wire.models import WorkProgress

    fold = Fold(
        key=WorkKey(dir="docs", name="w"),
        runs=("2026-10-09-00000001", "2026-10-10-00000002"),
        design=PartState(1, "a", "a"),
        plan=PartState(1, "b", "b"),
        plan_run="2026-10-09-00000001",
        plan_version=1,
        plan_sha="s",
        tasks=(TaskStatus("T1", "one", "committed", commit="c1", run="2026-10-09-00000001"),),
        run=None,
    )
    read = WorkProgress.model_validate(_wire(fold))
    assert [(t.id, t.state, t.run, t.commit) for t in read.tasks] == [
        ("T1", "committed", "2026-10-09-00000001", "c1")
    ]
