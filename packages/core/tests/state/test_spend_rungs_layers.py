"""`[spend.rungs]` through the layered load: a cap per rung is the person's
engine config, named and refused the way every other engine key is.

The model's own check (`SpendPolicy._rung_caps_are_usable`) is tested beside the budget's
config. These are the layers' half: the per-layer check that names an unknown rung, its file
and its layer before any merge; a repository that may not set a cap at all; a retired rung
refused naming the decision; and the environment layer.
"""

from __future__ import annotations

import pathlib

import pytest
from repos import layer_roots

from chickadee.errors import ConfigError
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import load_layers
from chickadee.state.merge import UnknownKey
from chickadee.state.resolve import EngineOnlyKey, resolve
from chickadee.state.roots import Roots

ENGINE = (
    "".join(
        f"[models.{name}-model]\n"
        'base_url = "http://localhost:8000/v1"\n'
        f'model_id = "{name}-model"\n'
        "context_window = 65536\n"
        "max_output_tokens = 16384\n\n"
        for name in DEFAULT_RUNGS
    )
    + "[rungs]\n"
    + "".join(f'{name} = "{name}-model"\n' for name in DEFAULT_RUNGS)
    + "\n[spend]\nhard_cost_usd = 5.0\n"
)
"""A complete engine config with a finite total cap; each test adds its `[spend.rungs]`."""


def _engine(roots: Roots, rungs: str) -> None:
    (roots.user / "config.toml").write_text(
        ENGINE + "\n[spend.rungs]\n" + rungs, encoding="utf-8"
    )


def test_the_persons_rung_caps_load_through_the_layers(tmp_path: pathlib.Path) -> None:
    roots = layer_roots(tmp_path)
    _engine(roots, "high-param = 2.0\nlow-param = 1.5\n")

    spend = resolve(load_layers(roots, environ={})).engine.spend

    assert spend.rungs == {"high-param": 2.0, "low-param": 1.5}
    assert spend.hard_cost_usd == 5.0, "beneath the total, never instead of it"


def test_a_malformed_rung_name_refuses_naming_the_key_and_its_layer(
    tmp_path: pathlib.Path,
) -> None:
    """A name the rung rule refuses is refused before the merge, so the message
    names the file and the layer, not a cap that silently capped nothing."""
    roots = layer_roots(tmp_path)
    _engine(roots, "High-param = 2.0\n")

    with pytest.raises(UnknownKey) as err:
        resolve(load_layers(roots, environ={}))

    message = str(err.value)
    assert "invalid rung name `spend.rungs.High-param`" in message
    assert "layer: user" in message
    assert "config.toml" in message


def test_a_cap_on_an_undeclared_rung_refuses_naming_the_declared(
    tmp_path: pathlib.Path,
) -> None:
    """A well-formed name is checked against `[rungs]` once the layers merge,
    since a layer alone cannot know which rungs the merge declares."""
    roots = layer_roots(tmp_path)
    _engine(roots, "hgh-param = 2.0\n")

    with pytest.raises(ConfigError) as err:
        resolve(load_layers(roots, environ={}))

    assert "[spend.rungs] names ['hgh-param'], which [rungs] does not declare" in str(err.value)


def test_a_retired_rung_is_refused_naming_the_decision(tmp_path: pathlib.Path) -> None:
    """`mid` is not a typo: it passes the per-layer check and is refused naming D131, as it is
    in `[rungs]`."""
    roots = layer_roots(tmp_path)
    _engine(roots, "mid = 2.0\n")

    with pytest.raises(ValueError, match="D131"):
        resolve(load_layers(roots, environ={}))


def test_a_repository_cannot_set_a_rung_cap(tmp_path: pathlib.Path) -> None:
    """A cap is the person's money. A clone that could set one could also set one
    so high that it is no cap at all."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text(ENGINE, encoding="utf-8")
    (roots.project / "config.toml").write_text(
        "[spend.rungs]\nlow-param = 1000.0\n", encoding="utf-8"
    )

    with pytest.raises(EngineOnlyKey) as err:
        resolve(load_layers(roots, environ={}))

    assert "spend" in str(err.value)
    assert "layer: project" in str(err.value)


def test_the_environment_sets_a_rung_cap_over_the_engine_config(tmp_path: pathlib.Path) -> None:
    roots = layer_roots(tmp_path)
    _engine(roots, "low-param = 1.5\n")

    spend = resolve(
        load_layers(roots, environ={"CHICKADEE_SPEND__RUNGS__LOW-PARAM": "3.0"})
    ).engine.spend

    assert spend.rungs == {"low-param": 3.0}
