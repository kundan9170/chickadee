"""A host's run in Zeus's session: its `episode` line has no turn, so it answers no
`coder` call, opens no unit of Zeus's conversation and holds no cut back. The readers that
serve the person — recall, replay — still find it."""

from __future__ import annotations

import pytest
from context_helpers import exchange, recall_of, reopen, run_id, work_unit

from chickadee.context.compaction.boundary import Boundary, choose_boundary
from chickadee.context.lines import EpisodeLine, PayloadError
from chickadee.context.project import completed_units
from chickadee.context.recall import Recalled
from chickadee.context.session import Session
from chickadee.schema import Tier


def host_run(name: str) -> EpisodeLine:
    return EpisodeLine(run_id(name), "exact", Tier.T0, None, (), None)


def test_a_hosts_episode_answers_no_coder_call_and_opens_once(session: Session) -> None:
    session.record_user("anything")
    line = session.record_episode(host_run("host"))
    assert (line.run_id, line.opened_at_seq) == (run_id("host"), None)
    with pytest.raises(PayloadError, match="already has an episode"):
        session.record_episode(host_run("host"))
    assert reopen(session).state.episodes() == (line,), "replay folds it the same"


def test_zeus_units_are_the_same_with_a_hosts_run_among_them(session: Session) -> None:
    """Without the skip, a host's episode never gets a `coder` result, so it would hold every
    later unit open for ever."""
    alpha = work_unit(session, "alpha")
    session.record_episode(host_run("host"))
    beta = work_unit(session, "beta")
    units = completed_units(session.state)
    assert [u.run_id for u in units] == [alpha, beta]
    assert units[1].seq_open == units[0].seq_end, "still contiguous"


def test_a_hosts_unfinished_run_holds_no_cut_back(session: Session) -> None:
    """§7.6.6 rule 1 is about Zeus's own units: a host's run in flight is not one."""
    session.record_episode(host_run("host"))
    for name in ("alpha", "beta", "gamma"):
        work_unit(session, name)
    exchange(session, "late", "shell", {"command": "ls"}, "a\n")
    session.record_user("what is left?")
    session.record_assistant("nothing")
    assert isinstance(choose_boundary(session.state, 0), Boundary)


def test_recall_finds_a_hosts_run_and_says_who_started_it(session: Session) -> None:
    session.record_user("anything")
    session.record_episode(host_run("host"))
    session.sync()
    found = recall_of(session.root)(run_id("host"))
    assert isinstance(found, Recalled)
    (hit,) = [h for h in found.hits if h.kind == "run"]
    assert "started by the host, not by a coder call" in hit.text
