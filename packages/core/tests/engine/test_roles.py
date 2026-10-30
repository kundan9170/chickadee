"""The cast as a value: seven roles on five rungs, every
rung with a role on it; critic and worker on different rungs; exactly one role has the d5
ladder and it is worker; exactly two have judgment and they are zeus and athena; every cap is
the limits register's; the cast table in the architecture notes, parsed, names exactly
`ROLES`; a config naming a retired role or rung refuses, naming the decision; a config with an
unknown role key refuses; a config missing the zeus rung refuses; an unknown role is a config
error, not a `KeyError` escaping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from chickadee.config import EngineConfig
from chickadee.errors import ConfigError
from chickadee.limits import limit
from chickadee.roles import (
    CAST_HEADER,
    DEFAULT_RUNGS,
    ROLES,
    Role,
    Sampling,
    render_cast,
    role,
)


def test_eleven_roles_in_the_documented_order() -> None:
    """D121 retired the `elaborator`: elaboration is a request on Athena's session. D213 added
    the two built-in agent profiles, `reader` and `reviewer`, and D232 the third,
    `researcher`. D239 added `title`, a session's name."""
    assert [r.name for r in ROLES] == [
        "zeus",
        "athena",
        "worker",
        "locate",
        "critic",
        "reader",
        "reviewer",
        "researcher",
        "residue",
        "checkpoint",
        "title",
    ]


def test_three_default_tiers_and_every_one_has_a_role() -> None:
    """The cast defaults to three tiers, and each is some role's default, so
    the defaults declare no rung a config must wire for nothing."""
    assert DEFAULT_RUNGS == ("high-param", "mid-param", "low-param")
    defaults = {r.name: r.default_rung for r in ROLES}
    assert defaults == {
        "zeus": "high-param",
        "athena": "high-param",
        "worker": "low-param",
        "locate": "mid-param",
        "critic": "mid-param",
        "reader": "mid-param",
        "reviewer": "mid-param",
        "researcher": "mid-param",
        "residue": "high-param",
        "checkpoint": "mid-param",
        "title": "low-param",
    }
    assert set(defaults.values()) == set(DEFAULT_RUNGS)


def test_the_critic_is_distinct_from_the_worker() -> None:
    """The critic does not run on the worker's model, which
    the cast declares and the load checks; their default rungs differ too."""
    assert role("critic").distinct_from == ("worker",)
    assert role("critic").default_rung != role("worker").default_rung
    assert all(r.distinct_from == () for r in ROLES if r.name != "critic")


def test_no_role_has_the_d5_ladder_since_the_packet_attempts_retired() -> None:
    """D5's ladder raised a packet attempt's temperature per retry; the task worker has no
    attempts to climb, so every role's sampling is fixed."""
    ladders = [r.name for r in ROLES if r.sampling.ladder == "d5"]
    assert ladders == []


def test_exactly_two_roles_have_judgment_and_they_are_zeus_and_athena() -> None:
    assert [r.name for r in ROLES if r.judgment] == ["zeus", "athena"]


def test_tools_is_a_name_never_a_list() -> None:
    for r in ROLES:
        assert r.tools in ("zeus", "athena", "coder", "scout", "agent", "none"), r.name
    assert role("zeus").tools == "zeus" and role("worker").tools == "coder"
    assert role("residue").tools == "none" and role("residue").max_tool_calls == 0
    assert role("checkpoint").tools == "none" and role("checkpoint").max_tool_calls == 0


def test_athena_has_her_own_surface_and_template() -> None:
    """Her port held more than the scout's table, so the surface has her name; her
    instructions are a prompt tied to no model family."""
    assert role("athena").tools == "athena"
    assert role("athena").prompt == "shared/athena.md"
    assert role("athena").default_rung == "high-param"


def test_what_each_role_may_propose() -> None:
    assert role("zeus").may_propose == {"command", "design"}
    assert role("athena").may_propose == {"design", "patch"}
    assert role("worker").may_propose == {"patch"}
    assert role("critic").may_propose == {"check"}
    for name in ("locate", "residue", "checkpoint"):
        assert role(name).may_propose == frozenset(), name


def test_the_residue_and_the_checkpoint_default_apart() -> None:
    """L3_MODEL and CHECKPOINT_MODEL are these two roles' rungs, apart
    since so an experiment moves one without the other; the residue on the model that
    held the conversation (the owner's call)."""
    assert role("residue").default_rung == "high-param"
    assert role("checkpoint").default_rung == "mid-param"


def test_caps_are_the_limits_registers_defaults() -> None:
    """A role's caps are rows of the register, read when `ROLES` is built — not typed
    in `roles.py`, where the old D48 values (8 worker calls, 15 for locate) had drifted from
    the register the config resolves. A role with no tool surface has neither."""
    for r in ROLES:
        if r.tools == "none":
            assert (r.max_tool_calls, r.max_cumulative_result_tokens) == (0, 0.0), r.name
            continue
        assert r.max_tool_calls == limit(f"{r.name}.max_tool_calls").value, r.name
        assert (
            r.max_cumulative_result_tokens
            == limit(f"{r.name}.max_cumulative_result_tokens").value
        ), r.name
    assert role("zeus").max_tool_calls == 64 and role("worker").max_tool_calls == 200
    assert role("locate").max_tool_calls == 40, "the second copy of the locate cap is gone"


def test_an_unknown_role_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="no role named 'nope'"):
        role("nope")


def test_a_role_defaults_only_to_a_default_tier() -> None:
    """A role's default is one of `DEFAULT_RUNGS`; a former rung is none."""
    with pytest.raises(ConfigError, match="not one of"):
        Role("x", "gpu5", "none", 0, 0.0, "p", Sampling(), False, frozenset())
    with pytest.raises(ConfigError, match="not one of"):
        Role("x", "coder", "none", 0, 0.0, "p", Sampling(), False, frozenset())


def test_a_role_with_no_surface_has_no_caps_and_a_fraction_is_a_fraction() -> None:
    with pytest.raises(ConfigError, match="no tool surface but a cap"):
        Role("x", "high-param", "none", 3, 0.0, "p", Sampling(), False, frozenset())
    with pytest.raises(ConfigError, match="no tool surface but a cap"):
        Role("x", "high-param", "none", 0, 0.2, "p", Sampling(), False, frozenset())
    with pytest.raises(ConfigError, match="window fraction"):
        Role("x", "high-param", "scout", 3, 1.5, "p", Sampling(), False, frozenset())


@pytest.mark.parametrize(("calls", "results"), [(0, 0.3), (3, 0.0)])
def test_a_tool_using_role_with_a_zero_cap_is_refused_when_built(
    calls: int, results: float
) -> None:
    """A zero register row is found where it is, not later as a config load that blames the
    person for a cap they never set."""
    with pytest.raises(ConfigError, match="tool surface but a zero cap"):
        Role("x", "high-param", "scout", calls, results, "p", Sampling(), False, frozenset())


# --- the config that refuses the rest (the two refusals, on `EngineConfig`) ---


def test_a_config_with_an_unknown_role_key_refuses_naming_the_cast(raw: dict[str, Any]) -> None:
    """`31-state.md` rule 4: a role is added by a decision, never by configuration."""
    raw["roles"] = {"summarizer": "high-param"}
    with pytest.raises(
        ConfigError, match=r"not roles.*\['summarizer'\]|\['summarizer'\].*not roles"
    ):
        EngineConfig.with_defaults(raw)


@pytest.mark.parametrize("table", ["roles", "tools"])
def test_a_config_naming_the_elaborator_refuses(raw: dict[str, Any], table: str) -> None:
    """A config written in older versions is told why, not only that the name is unknown."""
    entry: object = "high-param" if table == "roles" else {"max_tool_calls": 12}
    raw[table] = {"elaborator": entry}
    with pytest.raises(ConfigError, match=r"'elaborator', a role that was retired") as err:
        EngineConfig.with_defaults(raw)
    assert "Athena's session" in str(err.value)


def test_a_config_naming_the_mid_rung_refuses(raw: dict[str, Any]) -> None:
    raw["rungs"]["mid"] = raw["rungs"]["high-param"]
    with pytest.raises(
        ConfigError, match=r"\[rungs\]: 'mid' is a former rung \(retired by D131"
    ):
        EngineConfig.with_defaults(raw)
    del raw["rungs"]["mid"]
    raw["roles"] = {"locate": "mid"}
    with pytest.raises(ConfigError, match=r"\[roles\]: 'mid' is a former rung .*no successor"):
        EngineConfig.with_defaults(raw)


@pytest.mark.parametrize(
    ("extra_rung", "extra_table", "decision"),
    [
        ("", '[roles]\nelaborator = "high-param"\n', "retired"),
        ("", "[tools.elaborator]\nmax_tool_calls = 12\n", "retired"),
        ('mid = "m"\n', "", "D131"),
    ],
    ids=["roles", "tools", "rungs"],
)
def test_a_retired_name_in_a_file_refuses_naming_its_decision_and_the_file(
    tmp_path: Path, extra_rung: str, extra_table: str, decision: str
) -> None:
    """Through the layered loader: a retired name passes the per-layer name check, which
    would call it a typo, and the model refuses it naming the decision, with the file."""
    with pytest.raises(ConfigError, match=decision) as err:
        EngineConfig.from_toml(_file(tmp_path, extra_rung, extra_table))
    assert "config.toml" in str(err.value)


def _file(tmp_path: Path, extra_rung: str = "", extra_table: str = "") -> Path:
    """A complete engine config — model `m` behind every rung but the worker's, which runs on
    `w` (the critic may not share the worker's model) — plus what a test adds."""
    model = 'base_url="u"\nmodel_id="m"\ncontext_window=65536\nmax_output_tokens=16384\n'
    rungs = (
        "".join(f'{name} = "{"w" if name == "low-param" else "m"}"\n' for name in DEFAULT_RUNGS)
        + extra_rung
    )
    f = tmp_path / "config.toml"
    f.write_text(
        f"[models.m]\n{model}[models.w]\n{model}[rungs]\n{rungs}[spend]\nhard_cost_usd = 5.0\n"
        f"{extra_table}"
    )
    return f


def test_a_role_typo_is_never_offered_a_retired_name(tmp_path: Path) -> None:
    """A misspelt role lists the roles that exist. Offering `elaborator` would send a person
    to a name that is refused next, under D121. (A rung is declared by naming it, D176, so a
    misspelt rung loads and is reported: `test_open_rungs.py`.)"""
    with pytest.raises(ConfigError, match="Known here") as err:
        EngineConfig.from_toml(_file(tmp_path, extra_table='[roles]\nlocat = "high-param"\n'))
    assert "'elaborator'" not in str(err.value)


def test_a_retired_entry_with_a_bad_value_still_hears_why(tmp_path: Path) -> None:
    """Refused before its fields are validated: a cap of 0 on `[tools.elaborator]` must not
    send a person to fix the value of an entry that should be deleted."""
    bad = "[tools.elaborator]\nmax_tool_calls = 0\n"
    with pytest.raises(ConfigError, match=r"'elaborator', a role that was retired") as err:
        EngineConfig.from_toml(_file(tmp_path, extra_table=bad))
    assert "greater than 0" not in str(err.value)


@pytest.mark.parametrize(("rung", "first"), [("high-param", "zeus"), ("mid-param", "locate")])
def test_a_config_missing_a_rung_a_role_runs_on_refuses(
    raw: dict[str, Any], rung: str, first: str
) -> None:
    """A rung is required when a role resolves to it, and the refusal names
    the first role that does, the rung, and the rungs declared. This is the rule a user meets
    first."""
    raw["rungs"] = {k: v for k, v in raw["rungs"].items() if k != rung}
    with pytest.raises(ConfigError, match=rf"role {first!r} runs on rung {rung!r}"):
        EngineConfig.with_defaults(raw)


def test_render_cast_has_one_row_per_role_with_every_column() -> None:
    lines = render_cast(ROLES).splitlines()
    assert lines[0] == CAST_HEADER
    assert len(lines) == 2 + len(ROLES)
    worker = next(ln for ln in lines if ln.startswith("| `worker`"))
    assert "| fixed |" in worker and "| 200 |" in worker and "patch" in worker
    zeus = next(ln for ln in lines if ln.startswith("| `zeus`"))
    assert "| yes |" in zeus and "command, design" in zeus
    assert re.search(r"\| `residue` \| `[\w-]+` \| `none` \| 0 \|", render_cast(ROLES))


def test_every_role_s_prompt_exists_unless_named_unwritten() -> None:
    """A role naming a prompt file that does not exist is on `UNWRITTEN_PROMPTS`, with
    why, or it is a bug; and an entry whose file now exists comes off: the list only shrinks."""
    import chickadee.adapters
    from chickadee.roles import ROLES, UNWRITTEN_PROMPTS

    adapters = Path(chickadee.adapters.__file__).parent
    missing = {
        r.name for r in ROLES if not (adapters / r.family / "templates" / r.template).is_file()
    }
    assert missing == set(UNWRITTEN_PROMPTS)
    assert all(UNWRITTEN_PROMPTS.values()), "each says why"
