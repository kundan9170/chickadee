"""Rungs are declared in config, and a role's rung is resolved once: a person's own rung
is one `[rungs]` line, any role may be moved onto it and capped there; a rung a role needs must
be declared, and one nothing runs on is reported; a former name is refused naming its
successor, while old logs that name it keep counting on that successor; and the critic never
shares the worker's model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo

from chickadee.config import EngineConfig
from chickadee.context.lines import CapLine
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError
from chickadee.schema.adapter import Usage
from chickadee.state.roots import roots as resolve_roots

SID = "S-open"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", commit=None)


def engine_for(repo: Path, raw: dict[str, Any], engines: list[PacketEngine]) -> PacketEngine:
    engine = PacketEngine(repo, EngineConfig.with_defaults(raw))
    engines.append(engine)
    return engine


def created(repo: Path) -> Path:
    root = sessions_root(resolve_roots(repo))
    Session.create(root, SID).close()
    return root


# -- a rung of the person's own ----------------------------------------------------------


def test_a_persons_own_rung_takes_a_role_and_a_cap(
    repo: Path, raw: dict[str, Any], engines: list[PacketEngine]
) -> None:
    """`nemotron = "…"` declares a rung; `[roles]` moves `locate` onto it; it is capped from
    the config and by the person, with no code change and no decision."""
    raw["models"]["nemotron3-ultra"] = dict(raw["models"][raw["rungs"]["mid-param"]])
    raw["rungs"]["nemotron"] = "nemotron3-ultra"
    raw["roles"] = {"locate": "nemotron"}
    raw["spend"]["rungs"] = {"nemotron": 2.0}
    cfg = EngineConfig.with_defaults(raw)

    assert cfg.assignment["locate"] == "nemotron"
    assert cfg.rung_for("locate") is cfg.models["nemotron3-ultra"]
    assert "nemotron" in cfg.rung_names and cfg.notices == ()

    created(repo)
    engine = engine_for(repo, raw, engines)
    assert engine.budget(SID).rungs["nemotron"].cap_usd == 2.0
    capped = engine.set_cap(SID, 0.5, rung="nemotron")
    assert (capped.rungs["nemotron"].cap_usd, capped.rungs["nemotron"].cap_source) == (
        0.5,
        "session",
    )


def test_a_role_on_an_undeclared_rung_is_refused(raw: dict[str, Any]) -> None:
    raw["roles"] = {"locate": "nemotron"}
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(raw)
    message = str(err.value)
    assert "role 'locate' runs on rung 'nemotron', which [rungs] does not declare" in message
    assert "the declared rungs are ['high-param', 'mid-param', 'low-param']" in message


def test_a_config_need_not_declare_a_default_no_role_runs_on(raw: dict[str, Any]) -> None:
    """The defaults are not otherwise special."""
    raw["roles"] = {
        "locate": "high-param",
        "critic": "high-param",
        "reader": "high-param",
        "reviewer": "high-param",
        "researcher": "high-param",
        "checkpoint": "high-param",
    }
    del raw["rungs"]["mid-param"]
    assert "mid-param" not in EngineConfig.with_defaults(raw).rung_names


# -- an unused rung is reported, not refused ------------------------------------------------


def test_an_unused_rung_loads_and_is_reported_once_with_its_near_name(
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw["rungs"]["hgh-param"] = raw["rungs"]["high-param"]
    raw["rungs"]["spare"] = raw["rungs"]["low-param"]

    with caplog.at_level(logging.WARNING, logger="chickadee.engine"):
        engine = engine_for(repo, raw, engines)

    assert engine.config.notices == (
        "rung 'hgh-param' is declared and no role runs on it; did you mean 'high-param'?",
        "rung 'spare' is declared and no role runs on it",
    )
    said = [r.getMessage() for r in caplog.records if r.name == "chickadee.engine"]
    assert said == [f"config: {n}" for n in engine.config.notices], "each once"


# -- former names --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "edit", "says"),
    [
        ("[rungs]", lambda raw: raw["rungs"].update(coder="m"), "'coder'"),
        ("[roles]", lambda raw: raw.update(roles={"locate": "scout"}), "'scout'"),
        ("[spend.rungs]", lambda raw: raw["spend"].update(rungs={"zeus": 1.0}), "'zeus'"),
    ],
    ids=["rungs", "roles", "spend-rungs"],
)
def test_a_former_name_refuses_the_load_naming_its_successor(
    raw: dict[str, Any], section: str, edit: Any, says: str
) -> None:
    edit(raw)
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(raw)
    message = str(err.value)
    assert f"{section}: {says} is a former rung" in message
    successor = {"'coder'": "low-param", "'scout'": "mid-param", "'zeus'": "high-param"}[says]
    assert f"its successor is '{successor}'" in message


def test_set_cap_refuses_a_former_name_naming_its_successor(
    repo: Path, raw: dict[str, Any], engines: list[PacketEngine]
) -> None:
    created(repo)
    engine = engine_for(repo, raw, engines)
    with pytest.raises(ConfigError, match=r"'planner' is a former rung.*'high-param'"):
        engine.set_cap(SID, 1.0, rung="planner")


def test_an_old_logs_coder_cap_and_spend_apply_to_low_param(
    repo: Path, raw: dict[str, Any], engines: list[PacketEngine]
) -> None:
    """A family's lines written before the rungs were renamed name former
    rungs. They are read through `FORMER_RUNGS` and never rewritten, so a cap set on `coder`
    caps `low-param` and the spend on it counts there."""
    root = created(repo)
    with Session.open(root, SID) as old:
        old.record_cap(1.5, rung="coder")
        old.record_spend("m", Usage(prompt_tokens=1, completion_tokens=1), 0.25, "coder")
        old.record_spend("m", Usage(prompt_tokens=1, completion_tokens=1), 0.5, "zeus")

    budget = engine_for(repo, raw, engines).budget(SID)

    low = budget.rungs["low-param"]
    assert (low.spent_usd, low.cap_usd, low.cap_source) == (0.25, 1.5, "session")
    assert budget.rungs["high-param"].spent_usd == 0.5
    assert "coder" not in budget.rungs and "zeus" not in budget.rungs
    with Session.open(root, SID) as reread:
        caps = [line.rung for _, line in reread.state.lines() if isinstance(line, CapLine)]
    assert caps == ["coder"], "the line itself is never rewritten"


# -- distinct models ------------------------------------------------------------------------


def test_the_critic_and_the_worker_on_one_model_are_refused(raw: dict[str, Any]) -> None:
    """The critic does not check work on the model that did it."""
    raw["rungs"]["mid-param"] = raw["rungs"]["low-param"]
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(raw)
    message = str(err.value)
    assert "roles 'critic' and 'worker' must run on different models" in message
    assert "critic on rung 'mid-param', worker on 'low-param'" in message


def test_moving_the_critic_onto_the_workers_rung_is_refused_too(raw: dict[str, Any]) -> None:
    raw["roles"] = {"critic": "low-param"}
    with pytest.raises(ConfigError, match="'critic' and 'worker' must run on different models"):
        EngineConfig.with_defaults(raw)
