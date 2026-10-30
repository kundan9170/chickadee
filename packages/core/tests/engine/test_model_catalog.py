"""The model catalog — `[models.<name>]`, a `[rungs]` table naming one per rung, and
prices keyed by model.

A rung used to *be* an endpoint, so there was nothing to switch a rung to, and a
price was keyed by rung though it is the provider's, for a model. What the catalog has to
hold: a rung runs on the model it names and on no other; one model may serve several rungs; a
model no rung names is a switch target and loads; a rung that names nothing the catalog has,
and a price for a model it lacks, each refuse by name; every model name is one each layer can
address; and every number sized off a window is sized off the model the rung names.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from chickadee.budget import BudgetCaps, CostLedger, PriceTable
from chickadee.config import ENV_READS_AS_VALUE, MODEL_NAME, EngineConfig
from chickadee.errors import ConfigError
from chickadee.limits import ResolvedLimit
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema.adapter import Usage
from chickadee.state.layers import Layer, _from_env  # pyright: ignore[reportPrivateUsage]
from chickadee.state.resolve import resolve


def _model(**over: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        # Large enough for every role to fit one request.
        "context_window": 65_536,
        "max_output_tokens": 16_384,
    }
    return {**fields, **over}


def _catalog(**extra: Any) -> dict[str, Any]:
    """Two models and the three default rungs: `low-param` (the worker's) on `local`, the
    upper two on `frontier` — a catalog where one model serves several rungs and neither is
    named like a rung."""
    return {
        "models": {"local": _model(), "frontier": _model(context_window=200_000)},
        "rungs": {name: "frontier" for name in DEFAULT_RUNGS} | {"low-param": "local"},
        "spend": {"hard_cost_usd": 5.0},
        **extra,
    }


def _limit(cfg: EngineConfig, role: str, name: str) -> ResolvedLimit:
    return next(r for r in cfg.resolved_limits if r.role == role and r.name == name)


def test_models_and_rungs_load_from_a_file(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "config.toml"
    f.write_text(
        """
[models.frontier]
base_url = "https://api.example.test/v1"
model_id = "f-1"
context_window = 200000
max_output_tokens = 16384

[models.local]
base_url = "http://localhost:8000/v1"
model_id = "l-1"
context_window = 65536
max_output_tokens = 16384

[rungs]
high-param = "frontier"
mid-param = "frontier"
low-param = "local"

[spend]
hard_cost_usd = 5.0
"""
    )
    cfg = EngineConfig.from_toml(f)
    assert set(cfg.models) == {"frontier", "local"}
    assert cfg.rungs["low-param"] == "local" and cfg.rungs["high-param"] == "frontier"
    assert cfg.rung_for("worker").model_id == "l-1", "the worker's rung is low-param"
    assert cfg.rung_for("zeus").model_id == "f-1"


def test_one_model_serves_two_rungs_and_both_resolve_to_its_endpoint() -> None:
    cfg = EngineConfig.with_defaults(_catalog())
    assert cfg.rungs["high-param"] == cfg.rungs["mid-param"] == "frontier"
    # `zeus` runs on high-param and `locate` on mid-param: two rungs, one entry.
    assert cfg.rung_for("zeus") is cfg.rung_for("locate") is cfg.models["frontier"]


def test_a_role_on_an_undeclared_rung_refuses_naming_the_role_rung_and_declared() -> None:
    """A rung is required when a role resolves to it. The first role on
    `mid-param` is `locate`."""
    body = _catalog()
    del body["rungs"]["mid-param"]
    with pytest.raises(ConfigError, match="role 'locate' runs on rung 'mid-param'") as err:
        EngineConfig.with_defaults(body)
    assert "the declared rungs are ['high-param', 'low-param']" in str(err.value)


def test_a_rung_naming_a_model_the_catalog_lacks_refuses_naming_both() -> None:
    body = _catalog()
    body["rungs"]["low-param"] = "gpu5"
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(body)
    message = str(err.value)
    assert "'gpu5' (named by low-param)" in message
    assert "['frontier', 'local']" in message


def test_a_missing_model_behind_every_rung_is_named_once() -> None:
    """The likeliest first config: every rung wired to one model whose table is still
    commented out. Grouped by model, the refusal names it once, with the rungs behind it —
    not six clauses that differ by one word."""
    body = _catalog()
    body["models"] = {}
    body["rungs"] = {name: "frontier" for name in DEFAULT_RUNGS}
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(body)
    message = str(err.value)
    assert message.count("'frontier'") == 1
    assert f"'frontier' (named by {', '.join(DEFAULT_RUNGS)})" in message


@pytest.mark.parametrize("name", ["GPT5", "a.b", "", "1st", "-x", "true", "nan", "inf"])
def test_a_model_name_no_layer_could_address_refuses(name: str) -> None:
    """`MODEL_NAME`: the environment layer lowercases every variable name and reads `true`,
    `nan` and `inf` as values, so each of these could never be overridden or pointed at from
    it; a dot reads as a nested key; an empty name names nothing."""
    body = _catalog()
    body["models"][name] = _model()
    with pytest.raises(ConfigError, match="cannot be used") as err:
        EngineConfig.with_defaults(body)
    assert repr(name) in str(err.value)


@pytest.mark.parametrize("name", ["local-coder", "gpt_5", "m", "qwen3-coder-30b"])
def test_a_model_name_the_rule_admits_loads(name: str) -> None:
    body = _catalog()
    body["models"][name] = _model()
    body["rungs"]["low-param"] = name
    assert EngineConfig.with_defaults(body).rungs["low-param"] == name


def test_the_reserved_names_are_exactly_what_the_environment_layer_reads_as_values() -> None:
    """Ties `ENV_READS_AS_VALUE` to the reader it exists for: each reserved name comes back
    from `CHICKADEE_RUNGS__CODER` as something other than itself, and names the rule admits
    come back unchanged — so the two cannot drift apart."""
    for name in ENV_READS_AS_VALUE:
        assert _from_env({"CHICKADEE_RUNGS__LOW-PARAM": name})["rungs"]["low-param"] != name
    for name in ("local-coder", "gpt_5", "infra", "nano", "falsey", "e5"):
        assert MODEL_NAME.fullmatch(name) and name not in ENV_READS_AS_VALUE
        assert _from_env({"CHICKADEE_RUNGS__LOW-PARAM": name})["rungs"]["low-param"] == name


def test_the_environment_overrides_a_models_field_and_moves_a_rung() -> None:
    """Every admitted name is addressable from the environment: a field of a hyphenated model,
    and a rung moved to another model, through `env 'NAME=value'` (a shell cannot `export` a
    name with a hyphen, and the default rungs' names have one)."""
    environ = {
        "CHICKADEE_MODELS__LOCAL-CODER__MODEL_ID": "overridden",
        "CHICKADEE_RUNGS__MID-PARAM": "local-coder",
    }
    body = _catalog()
    body["models"]["local-coder"] = _model()
    engine = resolve(
        [Layer("defaults", None, body), Layer("environment", None, _from_env(environ))]
    ).engine
    assert engine.models["local-coder"].model_id == "overridden"
    assert engine.rungs["mid-param"] == "local-coder"
    assert set(engine.models) == {"frontier", "local", "local-coder"}, "no second table"


def test_an_old_shape_file_is_told_the_layout_changed(tmp_path: pathlib.Path) -> None:
    """A `[rungs.<name>]` endpoint table is the layout in older versions. It is refused as that,
    with
    what to write instead — not as pydantic's *Input should be a valid string*."""
    f = tmp_path / "config.toml"
    f.write_text(
        "".join(
            f'[rungs.{name}]\nbase_url = "u"\nmodel_id = "m"\n'
            "context_window = 65536\nmax_output_tokens = 16384\n"
            for name in DEFAULT_RUNGS
        )
        + "[spend]\nhard_cost_usd = 5.0\n"
    )
    with pytest.raises(ConfigError) as err:
        EngineConfig.from_toml(f)
    message = str(err.value)
    assert "[rungs.high-param]" in message and "the older layout" in message
    assert "[models.<name>]" in message
    assert "valid string" not in message and "errors.pydantic.dev" not in message


def test_a_half_migrated_config_names_only_the_table_left_behind() -> None:
    body = _catalog()
    body["rungs"]["low-param"] = _model()
    with pytest.raises(ConfigError, match=r"^\[rungs\.low-param\] holds an endpoint") as err:
        EngineConfig.with_defaults(body)
    assert "[rungs.high-param]" not in str(err.value)


def test_a_rung_the_environment_set_to_a_number_is_refused_in_words() -> None:
    """`CHICKADEE_RUNGS__LOW-PARAM=5` reaches the model as the integer 5 (`state/layers.py`)."""
    layers = [
        Layer("defaults", None, _catalog()),
        Layer("environment", None, _from_env({"CHICKADEE_RUNGS__LOW-PARAM": "5"})),
    ]
    with pytest.raises(ConfigError, match=r"rungs\.low-param is 5 \(int\)") as err:
        resolve(layers)
    assert "errors.pydantic.dev" not in str(err.value)


def test_a_price_for_a_model_the_catalog_lacks_refuses_naming_it() -> None:
    body = _catalog(prices={"gpu5": {"price_in": 1.0, "price_out": 2.0}})
    with pytest.raises(ConfigError, match=r"\['gpu5'\], which the catalog does not have"):
        EngineConfig.with_defaults(body)


def test_a_price_is_keyed_by_the_model_never_the_rung() -> None:
    """`low-param` is a rung and the catalog's model is `local`, so a row under `low-param`
    prices nothing the catalog has. A rung can run on two models in one session;
    a price keyed by the rung could not say which."""
    body = _catalog(prices={"low-param": {"price_in": 1.0, "price_out": 2.0}})
    with pytest.raises(ConfigError, match=r"\['low-param'\], which the catalog does not have"):
        EngineConfig.with_defaults(body)
    body["prices"] = {"local": {"price_in": 1.0, "price_out": 2.0}}
    assert EngineConfig.with_defaults(body).prices["local"].price_in == 1.0


def test_a_catalog_model_no_rung_names_loads_even_where_no_role_could_fit_on_it() -> None:
    """A switch target, not an error. It is checked when it is switched to, against
    the roles of the rung that would run on it, so a small model listed beside the rungs' own
    does not refuse the config that lists it — and it may be priced ahead of the switch."""
    body = _catalog()
    body["models"]["tiny"] = _model(context_window=4_096, max_output_tokens=1_024)
    body["prices"] = {"tiny": {"price_in": 0.1, "price_out": 0.2}}
    cfg = EngineConfig.with_defaults(body)
    assert "tiny" in cfg.models and "tiny" not in cfg.rungs.values()
    assert "tiny" in cfg.prices


@pytest.mark.parametrize("key", ["base_url", "model_id"])
def test_a_model_naming_no_endpoint_refuses_naming_the_key(key: str) -> None:
    body = _catalog()
    del body["models"]["local"][key]
    with pytest.raises(ConfigError, match=rf"models\.local\.{key}"):
        EngineConfig.with_defaults(body)


def test_limits_resolve_against_the_model_each_rung_names() -> None:
    """Sized by the model, not by an endpoint of the rung's own: one
    fraction is a different number on rungs whose models have different windows, and the same
    number on two rungs that share a model."""
    body = _catalog(
        limits={"worker": {"design_layer": 0.3}, "zeus": {"max_cumulative_result_tokens": 1}}
    )
    cfg = EngineConfig.with_defaults(body)
    packed = _limit(cfg, "worker", "worker.design_layer")
    assert (packed.rung, packed.tokens_or_value) == ("low-param", 19_660), "0.3 of 65,536"
    whole = _limit(cfg, "zeus", "zeus.max_cumulative_result_tokens")
    assert (whole.rung, whole.tokens_or_value) == ("high-param", 200_000), "frontier's window"

    trigger = {
        r.role: r.tokens_or_value
        for r in cfg.resolved_limits
        if r.name == "all.compact_trigger"
    }
    assert trigger["zeus"] == trigger["athena"] == trigger["locate"], "all on frontier"
    assert trigger["zeus"] > trigger["worker"], "the worker alone on local"

    # A role moved to another rung follows the model *that* rung names.
    moved = EngineConfig.with_defaults(
        {
            **body,
            "rungs": {**body["rungs"], "mid-param": "local"},
            "roles": {"worker": "high-param"},
        }
    )
    got = _limit(moved, "worker", "worker.design_layer")
    assert (got.rung, got.tokens_or_value) == ("high-param", 60_000), "0.3 of 200,000"


def test_the_fit_check_reads_the_model_a_rung_names() -> None:
    """A role that cannot fit is refused by rung, and moving the rung to another model is
    what fixes it: the window is the model's, not a property of the rung's name."""
    body = _catalog()
    body["models"]["tiny"] = _model(context_window=28_000, max_output_tokens=16_000)
    body["rungs"]["low-param"] = "tiny"
    with pytest.raises(ConfigError, match="cannot fit one request on rung 'low-param'"):
        EngineConfig.with_defaults(body)
    body["rungs"]["low-param"] = "local"
    EngineConfig.with_defaults(body)


def test_a_call_is_charged_at_the_row_of_the_model_that_served_it() -> None:
    """The rung a call ran on may have run on two models in one session, so the
    ledger is told the model, and the same usage costs what each model's row says. A rung's
    name is not a model's: it has no row."""
    prices = {
        "frontier": {"price_in": 3.0, "price_out": 15.0},
        "local": {"price_in": 0.1, "price_out": 0.2},
    }
    cfg = EngineConfig.with_defaults(_catalog(prices=prices))
    ledger = CostLedger(
        PriceTable.from_config(cfg.prices),
        BudgetCaps.from_config(cfg.spend),
        rungs=DEFAULT_RUNGS,
    )
    used = Usage(prompt_tokens=1_000, completion_tokens=1_000)
    assert ledger.for_rung("high-param").charge_usage(
        "", used, cfg.rungs["high-param"]
    ) == pytest.approx(0.018)
    assert ledger.for_rung("high-param").charge_usage("", used, "local") == pytest.approx(
        0.0003
    )
    assert ledger.spend("session").cost_usd == pytest.approx(0.0183)
    with pytest.raises(ConfigError, match="no price row for model 'zeus'"):
        ledger.for_rung("high-param").charge_usage("", used, "zeus")
