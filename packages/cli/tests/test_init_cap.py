"""Init writes repository facts only; the cap is the engine config's, and the report states
the one in force (IN-11 as amended). What init writes loads.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import tomllib
from decimal import Decimal

import pytest

from chickadee.config import EngineConfig
from chickadee.errors import ConfigError
from chickadee.init import publish
from chickadee.init.orchestrate import init, render_config
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import Layer, load_layers
from chickadee.state.resolve import check_layer, resolve
from chickadee.state.roots import roots
from chickadee_cli.init import render

_MODEL = (
    'base_url = "http://localhost:8000/v1"\nmodel_id = "m"\n'
    "context_window = 65536\nmax_output_tokens = 16384\n"
)


def _engine_config(cap: float | None = 7.5) -> str:
    """The person's engine config: model `m` serving every rung but the worker's, which runs on
    `w` (the critic may not share the worker's model), and the cap when given."""
    rungs = "".join(f'{n} = "{"w" if n == "low-param" else "m"}"\n' for n in DEFAULT_RUNGS)
    catalog = f"[models.m]\n{_MODEL}\n[models.w]\n{_MODEL}\n[rungs]\n" + rungs
    return catalog + ("" if cap is None else f"\n[spend]\nhard_cost_usd = {cap}\n")


@pytest.fixture
def engine_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Where the engine config goes, instead of this machine's: `$XDG_CONFIG_HOME/chickadee`,
    empty, with no `CHICKADEE_*` variable left to set a cap above it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in [n for n in os.environ if n.startswith("CHICKADEE_")]:
        monkeypatch.delenv(name)
    home = tmp_path / "xdg" / "chickadee"
    home.mkdir(parents=True)
    return home


def _written(repo: pathlib.Path) -> pathlib.Path:
    return publish.chickadee_dir(repo) / publish.CONFIG_NAME


def test_inits_file_holds_repository_facts_and_loads_as_the_repositorys(
    repo: pathlib.Path, engine_home: pathlib.Path
) -> None:
    """No cap, no rung, no model — not even commented, since a person who uncomments
    one gets a refused load. What is left passes the repository layer's check."""
    init(repo)
    path = _written(repo)
    text = path.read_text()
    assert set(tomllib.loads(text)) == {"profiles"}
    for engine_only in (
        "hard_cost_usd",
        "[spend]",
        "[models",
        "[rungs",
        "base_url",
        "model_id",
    ):
        assert engine_only not in text
    check_layer(Layer("project", path, tomllib.loads(text)))


def test_the_config_is_valid_toml_with_no_profiles_detected() -> None:
    """An empty repository still initialises; a template that only parses when something
    was found would fail on the first repository anyone creates."""
    assert tomllib.loads(render_config(()))["profiles"] == "auto"


def test_the_report_states_the_engine_configs_cap(
    repo: pathlib.Path, engine_home: pathlib.Path
) -> None:
    """A cap the person only discovers by hitting it is a cap they never agreed to, so the
    report states it — read from where it is set, since init no longer writes one."""
    (engine_home / "config.toml").write_text(_engine_config(7.5))
    report = init(repo)
    assert report.hard_cost_cap_usd == Decimal("7.5")
    assert not any("engine config" in w for w in report.warnings)


def test_the_environments_cap_is_the_one_reported(
    repo: pathlib.Path, engine_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap in force, not the file's: the environment layer is above it."""
    (engine_home / "config.toml").write_text(_engine_config(7.5))
    monkeypatch.setenv("CHICKADEE_SPEND__HARD_COST_USD", "2.5")
    assert init(repo).hard_cost_cap_usd == Decimal("2.5")


def test_with_no_engine_config_the_report_has_no_cap_and_names_setup(
    repo: pathlib.Path, engine_home: pathlib.Path
) -> None:
    """IN-11's finite seed moved to `chickadee setup`. Init states that nothing is
    set and where it would be, rather than printing a cap no one set."""
    report = init(repo)
    assert report.hard_cost_cap_usd is None
    [note] = [w for w in report.warnings if "engine config" in w]
    assert str(engine_home / "config.toml") in note
    assert "chickadee setup" in note


def test_the_command_prints_no_cap_as_none_never_as_a_number(
    repo: pathlib.Path, engine_home: pathlib.Path
) -> None:
    """`hard cap $None` reads as a figure to anyone skimming the output."""
    text = render(init(repo))
    assert "hard cap    none set (see note)" in text
    assert "$None" not in text
    (engine_home / "config.toml").write_text(_engine_config(7.5))
    assert "hard cap    $7.5" in render(init(repo))


@pytest.mark.parametrize("cap", [float("inf"), 0.0, -1.5])
def test_a_cap_no_run_would_use_is_reported_as_none(
    repo: pathlib.Path, engine_home: pathlib.Path, cap: float
) -> None:
    """`inf` is valid TOML and no cap at all, and `0` refuses every load. The
    report printed `hard cap $Infinity` and `hard cap $0`, as if either were in force."""
    (engine_home / "config.toml").write_text(_engine_config(cap))
    report = init(repo)
    assert report.hard_cost_cap_usd is None
    [note] = [w for w in report.warnings if "hard_cost_usd" in w]
    assert f"hard_cost_usd = {cap!r}" in note
    assert "finite and positive" in note
    assert "hard cap    none set (see note)" in render(report)


def test_an_engine_config_without_a_cap_is_named(
    repo: pathlib.Path, engine_home: pathlib.Path
) -> None:
    (engine_home / "config.toml").write_text(_engine_config(None))
    report = init(repo)
    assert report.hard_cost_cap_usd is None
    assert any("sets no [spend] hard_cost_usd" in w for w in report.warnings)


def test_what_init_writes_loads_beside_an_engine_config(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """D126 item 5 on the file init put on disk, with D127's split: alone it refuses, naming
    the tables to write and the file they belong in; beside an engine config it loads
    through the layered loader — the layers a run in this repository discovers, with the
    user layer pointed at a test directory so this machine's config cannot leak in — and
    through `from_toml` of the two files as one, to one config."""
    init(repo)
    user = tmp_path / "user"
    user.mkdir()
    scopes = dataclasses.replace(roots(repo), user=user)

    with pytest.raises(ConfigError, match=re.escape("[models.<name>]")) as err:
        resolve(load_layers(scopes, environ={}))
    assert "~/.config/chickadee/config.toml" in str(err.value)

    (user / "config.toml").write_text(_engine_config(7.5))
    layered = resolve(load_layers(scopes, environ={})).engine
    whole = tmp_path / "whole.toml"
    whole.write_text(_written(repo).read_text() + "\n" + _engine_config(7.5))
    assert layered == EngineConfig.from_toml(whole)
    assert layered.profiles == "auto"
    assert layered.spend.hard_cost_usd == 7.5
