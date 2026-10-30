"""A unit runs from the previous RUN_END to this RUN_END, units are contiguous, and the design
half contains PLAN_DECISION.

An earlier revision spanned RUN_START..PLAN_DECISION, which runs backwards: the plan is
decided before the run starts, so that range was empty and the whole design conversation
fell outside the unit written to preserve it. From the previous RUN_END, every line belongs to
exactly one unit, and nothing can fall between two blocks and be lost.
"""

from __future__ import annotations

import random

import pytest
from context_helpers import FakeRecords, exchange, run_end, run_id, run_start, work_unit

from chickadee.context.ledger import note_constraint
from chickadee.context.lines import EpisodeLine, PayloadError
from chickadee.context.project import Unit, completed_units, project_unit
from chickadee.context.session import CODER, Session
from chickadee.schema import Tier, ToolCall


def result_seq(session: Session, name: str) -> int:
    """The seq of the `coder` result `work_unit(name)` recorded."""
    line = session.state.result(f"c-{name}")
    assert line is not None
    return line.seq


def assert_contiguous(units: tuple[Unit, ...]) -> None:
    """Every line from 1 to the last unit's end is in exactly one unit."""
    expected_open = 0
    for unit in units:
        assert unit.seq_open == expected_open
        assert unit.turn_start == unit.seq_open + 1
        assert unit.turn_end == unit.seq_end
        assert unit.seq_open < unit.seq_decision < unit.seq_run_start < unit.seq_end
        expected_open = unit.seq_end
    covered = [s for u in units for s in range(u.turn_start, u.turn_end + 1)]
    assert covered == list(range(1, expected_open + 1))


def test_a_unit_runs_from_the_previous_run_end_to_this_one(session: Session) -> None:
    session.record_user("before any code: should the cache be per-user?")
    session.record_assistant("per-user, since the keys differ")
    alpha = work_unit(session, "alpha")
    session.record_user("now the eviction policy")
    beta = work_unit(session, "beta")
    session.record_user("and what is left?")
    first, second = completed_units(session.state)
    assert (first.run_id, second.run_id) == (alpha, beta)
    assert (first.seq_open, first.turn_start) == (0, 1)
    assert first.seq_end == result_seq(session, "alpha")
    assert (second.seq_open, second.seq_end) == (first.seq_end, result_seq(session, "beta"))
    assert first.turn_start <= 2 <= first.turn_end, "the design talk is in the unit it designed"
    assert session.state.last_seq > second.seq_end, "a line past the last RUN_END is in none"


@pytest.mark.parametrize("seed", range(20))
def test_units_are_contiguous_whatever_the_conversation(session: Session, seed: int) -> None:
    rng = random.Random(seed)
    made = 0
    for step in range(rng.randint(5, 25)):
        roll = rng.random()
        if roll < 0.25:
            said = session.record_user(f"user {step}")
            if rng.random() < 0.3:
                assert isinstance(
                    note_constraint(session, f"rule {step}", [said.seq], "session"), str
                )
        elif roll < 0.4:
            session.record_assistant(f"assistant {step}")
        elif roll < 0.7:
            exchange(session, f"s{step}", "shell", {"command": f"ls {step}"}, "a\n")
        else:
            work_unit(session, f"unit {step}")
            made += 1
    units = completed_units(session.state)
    assert len(units) == made
    assert_contiguous(units)
    for unit in units:
        after = completed_units(session.state, after=unit.seq_end)
        assert after[0] == unit, "`after` names the first unit not yet projected"


def test_the_design_half_holds_the_plan_decision(session: Session) -> None:
    """The plan is decided at dispatch — the turn that called `coder` — and the episode line
    after it opens the execution half. The projection renders both halves in ONE block."""
    session.record_user("add a retry to the fetcher")
    alpha = work_unit(session, "alpha")
    (unit,) = completed_units(session.state)
    decided = session.state.turn(unit.seq_decision)
    assert decided is not None
    assert [c.tool for c in decided.calls] == [CODER]
    episode = session.state.episode(alpha)
    assert episode is not None
    assert (episode.seq, episode.opened_at_seq) == (unit.seq_run_start, unit.seq_decision)
    assert unit.seq_open < unit.seq_decision < unit.seq_run_start < unit.seq_end

    # The journal's `plan_decision` row sat between the two (`goal`, `decision`, `outcome`);
    # it is parked with the packet loop's rows.
    records = FakeRecords({alpha: (run_start(alpha, "retry"), run_end(alpha, 2))})
    rows = [row.split()[0] for row in project_unit(session.state, unit, records).splitlines()]
    assert rows.count("▸") == 1
    assert rows.index("goal") < rows.index("outcome")


# -- the fold keeps a unit closable ----------------------------------------------------


def dispatch(session: Session, name: str, *calls: ToolCall) -> int:
    coder = ToolCall(id=f"c-{name}", name="coder", arguments={"request": {"goal": name}})
    return session.record_assistant("", [coder, *calls]).seq


def episode(name: str, at: int) -> EpisodeLine:
    return EpisodeLine(run_id(name), "planned", Tier.T1, None, (), at)


def test_an_episode_opens_once_at_its_dispatch(session: Session) -> None:
    session.record_user("go")
    talk = session.record_assistant("thinking").seq
    with pytest.raises(PayloadError, match="opens at the turn that called coder"):
        session.record_episode(episode("alpha", talk))
    at = dispatch(session, "alpha")
    session.record_episode(episode("alpha", at))
    with pytest.raises(PayloadError, match="already opened an episode"):
        session.record_episode(episode("beta", at))
    session.record_tool("c-alpha", "SOLVED", run_id=run_id("alpha"))
    before = session.state.last_seq
    with pytest.raises(PayloadError, match="is answered; an episode opens at dispatch"):
        session.record_episode(episode("late", at))
    assert session.state.last_seq == before


def test_a_coder_result_names_the_run_its_episode_opened(session: Session) -> None:
    """Otherwise that episode would wait for ever, and every unit after it with it."""
    session.record_user("go")
    at = dispatch(session, "alpha")
    session.record_episode(episode("alpha", at))
    with pytest.raises(PayloadError, match=f"opened run {run_id('alpha')}; name that run"):
        session.record_tool("c-alpha", "SOLVED")
    with pytest.raises(PayloadError, match="name that run"):
        session.record_tool("c-alpha", "SOLVED", run_id=run_id("other"))
    session.record_tool("c-alpha", "SOLVED", run_id=run_id("alpha"))
    assert [u.run_id for u in completed_units(session.state)] == [run_id("alpha")]


def test_a_refused_coder_call_opens_no_episode_and_names_no_run(session: Session) -> None:
    session.record_user("go")
    dispatch(session, "alpha")
    with pytest.raises(PayloadError, match="opened no episode, so no run"):
        session.record_tool("c-alpha", "SOLVED", run_id=run_id("alpha"))
    session.record_tool("c-alpha", "INVALID_ARGUMENTS{field: request}", ran=False)
    assert completed_units(session.state) == ()


def test_a_turn_makes_at_most_one_coder_delegation(session: Session) -> None:
    session.record_user("go")
    second = ToolCall(id="c-beta", name="coder", arguments={"request": {"goal": "beta"}})
    with pytest.raises(PayloadError, match="at most one coder delegation"):
        dispatch(session, "alpha", second)
    assert session.state.last_seq == 1
