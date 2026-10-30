"""A model with no price row raises; invalid caps are rejected at load."""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from itertools import pairwise
from typing import Any, get_args

import pytest
from budget_helpers import ledger, usage
from pydantic import BaseModel

from chickadee.budget.manager import BudgetCaps
from chickadee.config import EngineConfig, SpendPolicy, is_cap
from chickadee.errors import ConfigError
from chickadee.roles import DEFAULT_RUNGS


def test_an_unpriced_model_raises_on_charge_not_silently_zero() -> None:
    led = ledger(cap=1.0)
    with pytest.raises(ConfigError, match="unpriced model"):
        led.for_rung("low-param").charge_usage("p1", usage(), "mystery")
    assert led.spend("session").cost_usd == 0.0, "nothing was charged"


def test_a_rung_is_not_a_model_and_has_no_row() -> None:
    """The table is keyed by the catalog's models. The rung a call ran on may have run
    on two models in one session, so a charge made against the rung's name would price the
    call at whichever row happened to share it — and with none, refuses, naming what it was
    given."""
    led = ledger(cap=1.0)
    with pytest.raises(ConfigError, match="no price row for model 'coder'"):
        led.for_rung("low-param").charge_usage("p1", usage(), "coder")
    assert led.spend("session").cost_usd == 0.0


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_a_zero_or_negative_cap_is_rejected(cap: float) -> None:
    with pytest.raises(ConfigError, match="positive"):
        BudgetCaps(hard_cost_cap_usd=cap)


@pytest.mark.parametrize("cap", [float("inf"), float("nan")])
def test_a_cap_that_is_not_finite_is_rejected(cap: float) -> None:
    """`> 0` alone passed `inf`, though the refusal already said finite."""
    with pytest.raises(ConfigError, match="finite"):
        BudgetCaps(hard_cost_cap_usd=cap)


@pytest.mark.parametrize("fractions", [(0.8, 0.5), (0.5, 0.5), (1.0,), (0.0,), (1.5,)])
def test_bad_warning_fractions_are_rejected(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ConfigError, match="strictly increase"):
        BudgetCaps(hard_cost_cap_usd=1.0, warning_fractions=fractions)


def test_a_non_positive_packet_warning_is_rejected() -> None:
    with pytest.raises(ConfigError, match="packet_warning_cost_usd"):
        BudgetCaps(hard_cost_cap_usd=1.0, packet_warning_cost_usd=0.0)


def test_caps_load_from_the_engine_spend_policy() -> None:
    policy = SpendPolicy(
        hard_cost_usd=5.0, warning_fractions=(0.5, 0.9), packet_warning_cost_usd=0.2
    )
    caps = BudgetCaps.from_config(policy)
    assert caps == BudgetCaps(5.0, (0.5, 0.9), 0.2)
    assert caps.hard_cost_cap_usd is not None, "D42: the policy makes the cap required"


# -- a cap per rung -------------------------------------------------------------------


def test_rung_caps_load_beneath_the_total() -> None:
    policy = SpendPolicy(hard_cost_usd=5.0, rungs={"high-param": 2.0, "low-param": 1.5})
    assert policy.rungs == {"high-param": 2.0, "low-param": 1.5}
    assert SpendPolicy(hard_cost_usd=5.0).rungs == {}, "a rung not named has no rung cap"


@pytest.mark.parametrize(
    ("rungs", "says"),
    [
        ({"zeus": 1.0}, "successor is 'high-param'"),
        ({"mid": 1.0}, "D131"),
        ({"low-param": 0.0}, "finite and above zero"),
        ({"low-param": -1.0}, "finite and above zero"),
        ({"low-param": float("inf")}, "finite and above zero"),
        ({"low-param": float("nan")}, "finite and above zero"),
    ],
)
def test_a_rung_cap_no_config_could_hold_is_refused(rungs: dict[str, float], says: str) -> None:
    with pytest.raises(ValueError, match=says):
        SpendPolicy(hard_cost_usd=5.0, rungs=rungs)


def test_a_cap_on_an_undeclared_rung_is_refused_by_the_config() -> None:
    """`[spend.rungs]` cannot see `[rungs]`, so the config checks a cap names
    a declared rung — a misspelt one would cap nothing."""
    raw = _minimal()
    raw["spend"]["rungs"] = {"hgh-param": 1.0}
    with pytest.raises(ConfigError, match=r"\['hgh-param'\], which \[rungs\] does not declare"):
        EngineConfig.with_defaults(raw)


@pytest.mark.parametrize(
    "value", [1.0, 0.01, 5, 1e9, 0.0, -1.0, -0.0, float("inf"), float("-inf"), float("nan")]
)
def test_is_cap_agrees_with_the_total_caps_declared_rule(value: float) -> None:
    """`is_cap` is the one predicate the hand-written checks ask, and
    `hard_cost_usd`'s `Field(gt=0, allow_inf_nan=False)` states the same rule to pydantic.
    The two must never disagree on a number."""
    try:
        SpendPolicy(hard_cost_usd=value)
    except (ConfigError, ValueError):
        declared = False
    else:
        declared = True
    assert is_cap(value) is declared


# -- T-3.107: a number is never a boolean ------------------------------------------------------
#
# Pydantic's default mode reads `true` as 1.0 and `false` as 0.0 for a `float` field, so a
# slip in the person's own config became a $1.00 cap, a price of 1.0, or a one-second timeout,
# with no warning. Every float the engine config reads refuses a boolean at load, and names
# the key it was given for.


def _minimal() -> dict[str, Any]:
    """A config every rung can run, from `tests/engine/conftest.py`'s shape: one model per
    rung, each priced, and a cap; and one profile of the person's own, so its numbers are
    swept too."""
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 65_536,
        "max_output_tokens": 16_384,
    }
    return {
        "models": {f"{name}-model": dict(model) for name in DEFAULT_RUNGS},
        "rungs": {name: f"{name}-model" for name in DEFAULT_RUNGS},
        "prices": {
            f"{name}-model": {"price_in": 1.0, "price_out": 2.0} for name in DEFAULT_RUNGS
        },
        "spend": {"hard_cost_usd": 5.0},
        "agents": {"profiles": {"mine": {"base": "reader"}}},
    }


HOOK_ROW: dict[str, Any] = {"event": "before_stop", "name": "swept", "use": "owed_review"}
"""The `[[hooks]]` row a hook's field is set on: an array of tables, so its path names row 0."""


def _set(raw: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    node: Any = raw
    for key, then in pairwise(path):
        if isinstance(key, int):
            while len(node) <= key:
                node.append(dict(HOOK_ROW))
            node = node[key]
        else:
            node = node.setdefault(key, [] if isinstance(then, int) else {})
    node[path[-1]] = value
    return raw


MONEY: dict[str, tuple[str, ...]] = {
    "[spend] hard_cost_usd": ("spend", "hard_cost_usd"),
    "[spend.rungs] low-param": ("spend", "rungs", "low-param"),
    "[spend] packet_warning_cost_usd": ("spend", "packet_warning_cost_usd"),
    "price_in": ("prices", "low-param-model", "price_in"),
    "price_out": ("prices", "low-param-model", "price_out"),
    "price_cached_in": ("prices", "low-param-model", "price_cached_in"),
    "price_cache_write": ("prices", "low-param-model", "price_cache_write"),
}


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("key", list(MONEY))
def test_a_boolean_for_money_is_refused_naming_the_key(key: str, flag: bool) -> None:
    path = MONEY[key]
    raw = _set(_minimal(), path, flag)
    with pytest.raises(ConfigError) as refused:
        EngineConfig.with_defaults(raw)
    message = str(refused.value)
    assert path[-1] in message, message
    assert "boolean" in message, message


@pytest.mark.parametrize("path", list(MONEY.values()), ids=list(MONEY))
def test_a_whole_number_still_loads_where_an_amount_is_read(path: tuple[str, ...]) -> None:
    """Only booleans are refused: `hard_cost_usd = 5` is five dollars, as it always was."""
    EngineConfig.with_defaults(_set(_minimal(), path, 3))


def _fields_reading(
    kind: str, model: type[BaseModel], seen: set[type[BaseModel]]
) -> Iterator[tuple[str, str]]:
    """Every `(Model, field)` in the config's tree whose annotation reads `kind` (`float`
    or `int`)."""
    if model in seen:
        return
    seen.add(model)
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if re.search(rf"\b{kind}\b", repr(annotation)):
            yield model.__name__, name
        stack: list[Any] = [annotation]
        while stack:
            arg = stack.pop()
            stack.extend(get_args(arg))
            if inspect.isclass(arg) and issubclass(arg, BaseModel):
                yield from _fields_reading(kind, arg, seen)


_PROFILE: tuple[str, ...] = ("agents", "profiles", "mine")

SWEPT: dict[tuple[str, str], tuple[str, ...]] = {
    ("RungEndpoint", "first_token_s"): ("models", "low-param-model", "first_token_s"),
    ("RungEndpoint", "token_gap_s"): ("models", "low-param-model", "token_gap_s"),
    ("RungEndpoint", "temperature"): ("models", "low-param-model", "temperature"),
    ("PriceRow", "price_in"): MONEY["price_in"],
    ("PriceRow", "price_out"): MONEY["price_out"],
    ("PriceRow", "price_cached_in"): MONEY["price_cached_in"],
    ("PriceRow", "price_cache_write"): MONEY["price_cache_write"],
    ("SpendPolicy", "hard_cost_usd"): MONEY["[spend] hard_cost_usd"],
    ("SpendPolicy", "warning_fractions"): ("spend", "warning_fractions"),
    ("SpendPolicy", "packet_warning_cost_usd"): MONEY["[spend] packet_warning_cost_usd"],
    ("SpendPolicy", "rungs"): MONEY["[spend.rungs] low-param"],
    ("PlanningPolicy", "score_weights"): ("planning", "score_weights", "a"),
    ("PlanningPolicy", "threshold"): ("planning", "threshold"),
    ("ExecutionDifficultyPolicy", "score_weights"): (
        "difficulty",
        "score_weights",
        "a",
    ),
    ("ExecutionDifficultyPolicy", "frontier_threshold"): (
        "difficulty",
        "frontier_threshold",
    ),
    ("InteractionPolicy", "question_timeout_s"): ("interaction", "question_timeout_s"),
    ("AgentProfileConfig", "max_cumulative_result_tokens"): (
        *_PROFILE,
        "max_cumulative_result_tokens",
    ),
}
"""Where each float the config reads is set from TOML. No float in the engine config is one a
boolean could mean anything for, so the sweep names no exemption: a float added later fails
`test_every_float_the_config_reads_is_swept` until it is listed here, and is then refused a
boolean like every other."""


def test_every_float_the_config_reads_is_swept() -> None:
    assert sorted(_fields_reading("float", EngineConfig, set())) == sorted(SWEPT)


@pytest.mark.parametrize("field", list(SWEPT), ids=[".".join(k) for k in SWEPT])
def test_no_float_the_config_reads_takes_a_boolean(field: tuple[str, str]) -> None:
    path = SWEPT[field]
    value: object = [True] if field == ("SpendPolicy", "warning_fractions") else True
    raw = _set(_minimal(), path, value)
    with pytest.raises(ConfigError, match="boolean"):
        EngineConfig.with_defaults(raw)


COUNTS: dict[tuple[str, str], tuple[str | int, ...]] = {
    ("RungEndpoint", "context_window"): ("models", "low-param-model", "context_window"),
    ("RungEndpoint", "max_output_tokens"): ("models", "low-param-model", "max_output_tokens"),
    ("RungEndpoint", "input_limit"): ("models", "low-param-model", "input_limit"),
    ("PriceRow", "cache_ttl_s"): ("prices", "low-param-model", "cache_ttl_s"),
    ("ExecutionDifficultyPolicy", "starting_candidates"): ("difficulty", "starting_candidates"),
    ("RescuePolicy", "enrich_retry_attempts"): ("rescue", "enrich_retry_attempts"),
    ("RescuePolicy", "max_replans"): ("rescue", "max_replans"),
    ("InteractionPolicy", "max_questions_per_run"): ("interaction", "max_questions_per_run"),
    ("UndoPolicy", "keep_days"): ("undo", "keep_days"),
    ("UndoPolicy", "keep_turns"): ("undo", "keep_turns"),
    ("UndoPolicy", "untracked_max_bytes"): ("undo", "untracked_max_bytes"),
    ("AgentProfileConfig", "max_tool_calls"): (*_PROFILE, "max_tool_calls"),
    ("AgentProfileConfig", "seconds"): (*_PROFILE, "seconds"),
    ("AgentProfileConfig", "max_cost_cents"): (*_PROFILE, "max_cost_cents"),
    ("AgentProfileConfig", "report_reasks"): (*_PROFILE, "report_reasks"),
    ("HookSpec", "timeout_s"): ("hooks", 0, "timeout_s"),
}
"""Where each int the config reads is set from TOML, as `SWEPT` is for floats. `true` was 1:
a 1-token window, one tool call, one retry."""

COUNTS_EXEMPT: dict[tuple[str, str], str] = {
    ("ReasoningSpec", "budget_tokens"): (
        "schema/adapter.py, a frozen contract: schema/** changes only through T-0.2 under an "
        "owner grant (D54, D67), so it keeps reading `true` as 1 until one is given"
    ),
}


def test_every_int_the_config_reads_is_swept() -> None:
    found = sorted(_fields_reading("int", EngineConfig, set()))
    assert found == sorted([*COUNTS, *COUNTS_EXEMPT])


@pytest.mark.parametrize("field", list(COUNTS), ids=[".".join(k) for k in COUNTS])
def test_no_int_the_config_reads_takes_a_boolean(field: tuple[str, str]) -> None:
    raw = _set(_minimal(), COUNTS[field], True)
    with pytest.raises(ConfigError, match="boolean"):
        EngineConfig.with_defaults(raw)


@pytest.mark.parametrize("field", list(COUNTS), ids=[".".join(k) for k in COUNTS])
def test_a_count_still_loads_as_its_number(field: tuple[str, str]) -> None:
    """Only booleans are refused: every count still takes a whole number in its range."""
    fits = {"context_window": 131_072, "max_output_tokens": 16_384, "input_limit": 65_536}
    value = fits.get(field[1], 1)
    EngineConfig.with_defaults(_set(_minimal(), COUNTS[field], value))


FLAGS: dict[tuple[str, str], tuple[str | int, ...]] = {
    ("InteractionPolicy", "auto_title"): ("interaction", "auto_title"),
    ("PriceRow", "cache_affinity"): ("prices", "low-param-model", "cache_affinity"),
    ("RungEndpoint", "forced_tools"): ("models", "low-param-model", "forced_tools"),
    ("RungEndpoint", "grammar_tools"): ("models", "low-param-model", "grammar_tools"),
    ("RungEndpoint", "json_mode"): ("models", "low-param-model", "json_mode"),
    ("RungEndpoint", "native_tools"): ("models", "low-param-model", "native_tools"),
    ("RungEndpoint", "supports_images"): ("models", "low-param-model", "supports_images"),
    ("RungEndpoint", "supports_documents"): ("models", "low-param-model", "supports_documents"),
    ("RungEndpoint", "supports_temperature"): (
        "models",
        "low-param-model",
        "supports_temperature",
    ),
    ("SandboxPolicy", "allow_unsandboxed_escapes"): ("sandbox", "allow_unsandboxed_escapes"),
    ("SandboxPolicy", "allow_unsandboxed_python"): ("sandbox", "allow_unsandboxed_python"),
    ("SandboxPolicy", "never_ask"): ("sandbox", "never_ask"),
    ("SandboxPolicy", "require_sandbox"): ("sandbox", "require_sandbox"),
    ("SecretsPolicy", "allow_secret_read"): ("secrets", "allow_secret_read"),
    ("ShellLift", "athena"): ("sandbox", "allow_unsandboxed_shell", "athena"),
    ("ShellLift", "worker"): ("sandbox", "allow_unsandboxed_shell", "worker"),
    ("ShellLift", "zeus"): ("sandbox", "allow_unsandboxed_shell", "zeus"),
    ("SkillsPolicy", "external"): ("skills", "external"),
    ("ToolPolicy", "force_tool_call"): ("tools", "worker", "force_tool_call"),
    ("ToolSelection", "force_tool_call"): (
        "tools",
        "worker",
        "models",
        "low-param-model",
        "force_tool_call",
    ),
    ("WorkPolicy", "commit"): ("work", "commit"),
    ("WorkspacePolicy", "require_lease"): ("workspace", "require_lease"),
    ("HookSpec", "enabled"): ("hooks", 0, "enabled"),
}
"""Where each boolean the config reads is set from TOML (`config.Flag`). A quoted
`"false"` or a `0` was read as a boolean: `require_sandbox = "no"` loaded unsandboxed."""


def test_every_boolean_the_config_reads_is_swept() -> None:
    assert sorted(_fields_reading("bool", EngineConfig, set())) == sorted(FLAGS)


@pytest.mark.parametrize("value", ["yes", "false", 1, 0])
@pytest.mark.parametrize("field", list(FLAGS), ids=[".".join(k) for k in FLAGS])
def test_no_boolean_the_config_reads_takes_anything_but_a_boolean(
    field: tuple[str, str], value: object
) -> None:
    raw = _set(_minimal(), FLAGS[field], value)
    with pytest.raises(ConfigError, match=field[1]):
        EngineConfig.with_defaults(raw)
