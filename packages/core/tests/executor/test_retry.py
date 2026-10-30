"""EX-5 — a non-idempotent effect is never retried, as a property."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import init_repo

from chickadee.executor.reconcile import Landed, Reconciliation, may_retry


@pytest.mark.parametrize("landed", list(Landed))
def test_a_non_idempotent_effect_is_never_retried(landed: Landed) -> None:
    """The property, over every reconciliation outcome.

    `alembic revision` twice yields two migrations. The rule is not that a retry would
    probably be wrong — it is that deciding it is *safe* needs knowledge the harness does
    not have.
    """
    ok, why = may_retry(False, Reconciliation(landed, "any evidence"))
    assert not ok
    assert "not idempotent" in why


def test_an_idempotent_effect_retries_only_when_proven_not_to_have_landed() -> None:
    ok, _ = may_retry(True, Reconciliation(Landed.NO, "no artifacts"))
    assert ok


@pytest.mark.parametrize("landed", [Landed.YES, Landed.UNKNOWN])
def test_unknown_is_treated_as_landed(landed: Landed) -> None:
    """`UNKNOWN` must not permit a retry. Retrying an effect that already landed is the
    failure this exists to prevent, so ambiguity resolves against retrying."""
    ok, why = may_retry(True, Reconciliation(landed, "could not tell"))
    assert not ok
    assert "could not prove" in why


def test_a_missing_completion_event_is_not_permission(tmp_path: Path) -> None:
    """§6.1 states it directly: the process may have died after the effect landed and
    before the event was durable."""
    repo = init_repo(tmp_path / "r", commit=None)
    (repo / "artifact.txt").write_text("it landed\n", encoding="utf-8")

    from chickadee.executor.reconcile import by_file_evidence

    reconciliation = by_file_evidence(repo, ("artifact.txt",))
    assert reconciliation.landed is Landed.YES
    assert not may_retry(True, reconciliation)[0]


def test_git_failing_is_unknown_not_no(tmp_path: Path) -> None:
    """`UNKNOWN`, never `NO`. Git being unavailable says nothing about whether the effect
    landed, and `NO` is the one answer that permits a retry."""
    from chickadee.executor.reconcile import by_operation_trailer

    reconciliation = by_operation_trailer(tmp_path / "not-a-repo", "op-1")
    assert reconciliation.landed is Landed.UNKNOWN
    assert not may_retry(True, reconciliation)[0]


def test_trailer_timeout_is_unknown_not_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee.process.supervisor import Completed

    def hung(
        cwd: Path,
        *args: str,
        extra_env: object = None,
        supervisor: object = None,
        timeout_s: int = 30,
    ) -> Completed:
        del cwd, args, extra_env, supervisor, timeout_s
        return Completed(b"", b"", 124, True)

    monkeypatch.setattr("chickadee.executor.reconcile.run_git", hung)
    from chickadee.executor.reconcile import by_operation_trailer

    reconciliation = by_operation_trailer(tmp_path, "op-1")
    assert reconciliation.landed is Landed.UNKNOWN
    assert not may_retry(True, reconciliation)[0]
