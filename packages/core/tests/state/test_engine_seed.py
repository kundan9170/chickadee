"""The engine config `chickadee setup` seeds.

The seed is the one place a finite cap is written for a person: `init` writes none,
and a run refuses to load without one. Its model tables are commented, because the
models are the person's to choose, so the test fills them in exactly as written and loads the
result the way a run would, beside a repository config holding only `profiles`.
"""

from __future__ import annotations

import math
import re
import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

from chickadee.engine import SeedReport, seed_engine_config
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import Layer
from chickadee.state.resolve import resolve
from chickadee.state.seed import SEED_HARD_COST_USD, render

_EXAMPLE = re.compile(r"^# (?=\[|[a-z_-]+ = )")
"""An example line, commented: a table header or a `key = value`. Prose is `#:`."""


def _filled_in(text: str) -> str:
    """The seed with its example tables uncommented, as a person would."""
    return "\n".join(_EXAMPLE.sub("", line) for line in text.splitlines()) + "\n"


def test_the_seed_holds_a_finite_positive_cap_and_warnings() -> None:
    spend = tomllib.loads(render())["spend"]
    cap = spend["hard_cost_usd"]
    assert math.isfinite(cap) and cap > 0
    assert Decimal(str(cap)) == SEED_HARD_COST_USD
    fractions = spend["warning_fractions"]
    assert fractions and all(0 < f < 1 for f in fractions)


def test_the_seed_as_written_names_no_model() -> None:
    """The models are the person's choice: nothing in the seed picks one for them."""
    body = tomllib.loads(render())
    assert "models" not in body and "rungs" not in body


def test_filled_in_it_loads_beside_a_repository_config_of_profiles_only() -> None:
    body = tomllib.loads(_filled_in(render()))
    engine = resolve(
        [
            Layer("defaults", None, {}),
            Layer("user", Path("config.toml"), body),
            Layer("project", Path(".chickadee/config.toml"), {"profiles": "auto"}),
            Layer("environment", None, {}),
        ]
    ).engine
    assert set(engine.rungs) == set(DEFAULT_RUNGS)
    assert all(engine.rungs[r] in engine.models for r in DEFAULT_RUNGS)
    assert engine.spend.hard_cost_usd == float(SEED_HARD_COST_USD)


def test_setup_writes_the_seed_where_none_is(tmp_path: Path) -> None:
    path = tmp_path / "chickadee" / "config.toml"
    report = seed_engine_config(path)
    assert report == SeedReport(path=path, written=True, hard_cost_cap_usd=SEED_HARD_COST_USD)
    assert path.read_text(encoding="utf-8") == render()


def test_an_existing_engine_config_is_left_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    mine = b"# mine\n[spend]\nhard_cost_usd = 12.5\n"
    path.write_bytes(mine)
    report = seed_engine_config(path)
    assert not report.written
    assert path.read_bytes() == mine
    assert report.hard_cost_cap_usd == Decimal("12.5"), "the cap in force is the person's"


def test_an_existing_one_with_no_cap_is_reported_as_such(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b"[sandbox]\n")
    report = seed_engine_config(path)
    assert not report.written and report.hard_cost_cap_usd is None
    assert report.note is not None and "hard_cost_usd" in report.note


def test_the_environment_overrides_the_files_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHICKADEE_SPEND__HARD_COST_USD", "2.5")
    report = seed_engine_config(tmp_path / "config.toml")
    assert report.written and report.hard_cost_cap_usd == Decimal("2.5")


def test_the_default_path_is_the_engine_config_under_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    report = seed_engine_config()
    assert report.path == tmp_path / "chickadee" / "config.toml" and report.written


def test_an_existing_one_that_does_not_parse_is_reported_not_raised(tmp_path: Path) -> None:
    """Setup reads the file only to say which cap is in force; a file it cannot read is the
    person's to fix, and the report says so rather than a traceback."""
    path = tmp_path / "config.toml"
    path.write_bytes(b"[spend\n")
    report = seed_engine_config(path)
    assert not report.written and path.read_bytes() == b"[spend\n"
    assert report.hard_cost_cap_usd is None
    assert report.note is not None and "could not be read" in report.note
