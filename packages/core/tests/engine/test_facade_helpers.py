"""D150's pins: three helpers the facade held untested, pinned before they moved out.

`workspace.head_and_branch` gives Zeus's turn frame its head and branch,
`zeus.reconcile.run_outcome` words a run a crash left open for the session's repair, and
`client.client_for` picks a rung's transport. They were the engine's `_head`,
`_reconcile` and `_client_for`, which nothing tested. These cases landed first and passed on
those, and pass unchanged here: only the import and the two calling helpers below followed
the move.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.client import client_for
from chickadee.client.anthropic_client import AnthropicClient
from chickadee.client.openai_client import OpenAIClient
from chickadee.config import RungEndpoint
from chickadee.context.project import JournalRecords
from chickadee.journal import Journal, paths
from chickadee.journal.runs import RunProjections
from chickadee.schema import EventKind
from chickadee.schema.journal import EffectCompletedP, EffectIntendedP, RunEndP
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import head_and_branch
from chickadee.zeus.reconcile import run_outcome

RUN = "2026-10-03-0b0c0d0e"


def head_of(root: Path) -> tuple[str | None, str | None]:
    """What the turn frame reads: the head and the branch."""
    return head_and_branch(root)


def outcome(records: JournalRecords, run_id: str) -> str:
    """What the repair records for a run left open."""
    return run_outcome(records, run_id)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", commit=None)


@pytest.fixture
def committed(repo: Path) -> str:
    """`repo` with one commit on `main`: its sha. Empty, so nothing is written in the tree."""
    git(repo, "commit", "-q", "--allow-empty", "-m", "first")
    return git(repo, "rev-parse", "HEAD")


# -- the turn frame's head and branch --------------------------------------------------


def test_a_commit_on_a_branch_gives_both(repo: Path, committed: str) -> None:
    assert head_of(repo) == (committed, "main")


def test_a_detached_head_gives_the_commit_and_no_branch(repo: Path, committed: str) -> None:
    git(repo, "checkout", "-q", "--detach", committed)
    assert head_of(repo) == (committed, None)


def test_an_unborn_head_gives_neither(repo: Path) -> None:
    assert head_of(repo) == (None, None)


# -- the repair's words for a run a crash left open ------------------------------------------


def _intended(op: str) -> EffectIntendedP:
    return EffectIntendedP(
        operation_id=op,
        effect_kind="bootstrap",
        approved_sha="a" * 64,
        preconditions=(),
        idempotent=True,
        declared_surface=(),
    )


def test_a_run_with_no_journal_did_no_work(repo: Path) -> None:
    records = JournalRecords(resolve_roots(repo), RunProjections(resolve_roots(repo)))
    assert outcome(records, RUN) == f"run {RUN} left no journal: it did no work"


def test_an_unreadable_journal_is_unknown(repo: Path) -> None:
    roots = resolve_roots(repo)
    Journal.create(roots, RUN, {}).close()
    paths.journal_path(roots, RUN).write_text("{not json\n", encoding="utf-8")
    said = outcome(JournalRecords(roots, RunProjections(roots)), RUN)
    assert said.startswith(f"run {RUN}'s journal could not be read (")
    assert said.endswith("); what it did is unknown")


def test_a_run_that_ended_names_its_outcome(repo: Path) -> None:
    roots = resolve_roots(repo)
    with Journal.create(roots, RUN, {}) as j:
        j.append(
            EventKind.RUN_END,
            RunEndP(
                outcome="PARTIAL",
                why="T2 'query' is open, uncommitted (delegated)",
                tasks=2,
                committed=1,
                tokens=10,
                cost_usd=0.5,
                wall_ms=1,
                handoff_path="blobs/x",
            ),
        )
    assert (
        outcome(JournalRecords(roots, RunProjections(roots)), RUN)
        == f"run {RUN} ended PARTIAL: 1 of 2 task(s) committed"
    )


def test_a_run_with_no_terminal_counts_its_effects(repo: Path) -> None:
    roots = resolve_roots(repo)
    with Journal.create(roots, RUN, {}) as j:
        j.append(EventKind.EFFECT_INTENDED, _intended("op-1"))
        j.append(EventKind.EFFECT_INTENDED, _intended("op-2"))
        j.append(
            EventKind.EFFECT_COMPLETED,
            EffectCompletedP(operation_id="op-1", outcome="succeeded", observed_surface=()),
        )
    assert outcome(JournalRecords(roots, RunProjections(roots)), RUN) == (
        f"run {RUN} reached no terminal: 2 effect(s) intended, 1 completed; resume the "
        "session to reconcile the rest"
    )


# -- a rung's transport ----------------------------------------------------------


def _endpoint(provider: str) -> RungEndpoint:
    return RungEndpoint(
        base_url="http://localhost:8000/v1",
        model_id="m",
        context_window=131_072,
        max_output_tokens=16_384,
        provider=provider,
    )


@pytest.mark.parametrize(
    ("provider", "kind"),
    [("anthropic", AnthropicClient), ("openai", OpenAIClient), ("openrouter", OpenAIClient)],
)
def test_the_provider_picks_the_transport(provider: str, kind: type) -> None:
    client = client_for(_endpoint(provider))
    assert isinstance(client, AnthropicClient | OpenAIClient)
    try:
        assert type(client) is kind
    finally:
        client.close()
