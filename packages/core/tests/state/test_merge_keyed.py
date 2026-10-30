"""Layers, and why arrays of tables merge by identity key."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping
from typing import Any

import pytest
from repos import layer_roots

from chickadee.errors import ConfigError
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import load_layers
from chickadee.state.merge import MergeError, merge_arrays
from chickadee.state.resolve import Resolved, resolve
from chickadee.state.roots import Roots


def _complete() -> dict[str, Any]:
    """The smallest config that validates: a catalog of two models, the worker's rung on the
    second and every other rung on the first (the critic may not share the worker's model), and
    a finite cap. The layers under test set what they test on top of it, in
    `EngineConfig`'s names. The model names use underscores so the variables below can
    be set from any shell: a hyphenated name is addressable too, but only through
    `env 'NAME=value'`, since a shell cannot `export` a name with a hyphen."""
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 65_536,
        "max_output_tokens": 16_384,
    }
    return {
        "models": {"main_model": dict(model), "other_model": dict(model)},
        "rungs": {name: "main_model" for name in DEFAULT_RUNGS} | {"low-param": "other_model"},
        "spend": {"hard_cost_usd": 5.0},
    }


def _load(roots: Roots, environ: Mapping[str, str]) -> Resolved:
    return resolve(load_layers(roots, defaults=_complete(), environ=environ))


def test_same_hook_in_two_layers_produces_one_hook() -> None:
    """The reason keyed beats concatenated. Concatenation would run it TWICE. A hook's
    identity is its `(event, name)`."""
    lower = [{"event": "before_tool", "name": "g", "command": ["g.sh"], "on_error": "allow"}]
    higher = [{"event": "before_tool", "name": "g", "command": ["g.sh"], "on_error": "deny"}]
    out = merge_arrays(lower, higher, path="hooks")
    assert len(out) == 1
    assert out[0]["on_error"] == "deny"


def test_a_new_hook_is_appended_not_replacing() -> None:
    """And the reason keyed beats replacement: the lower layer's entry survives."""
    lower = [{"event": "before_tool", "name": "g", "command": ["g.sh"]}]
    higher = [{"event": "episode_ended", "name": "n", "command": ["n.sh"]}]
    out = merge_arrays(lower, higher, path="hooks")
    assert [e["command"] for e in out] == [["g.sh"], ["n.sh"]]


def test_two_hooks_at_one_point_are_two_hooks() -> None:
    """The name is part of the key: a second hook at the same point is added, not merged."""
    lower = [{"event": "before_tool", "name": "g", "command": ["g.sh"]}]
    higher = [{"event": "before_tool", "name": "h", "command": ["h.sh"]}]
    assert [e["name"] for e in merge_arrays(lower, higher, path="hooks")] == ["g", "h"]


def test_declaration_order_is_preserved() -> None:
    """Hooks fire in order, so a merge that reordered them would change behaviour while
    claiming only to combine configuration."""
    lower = [
        {"event": "after_tool", "name": "a", "command": ["1"]},
        {"event": "after_tool", "name": "b", "command": ["2"]},
    ]
    higher = [{"event": "after_tool", "name": "b", "command": ["2"], "on_error": "deny"}]
    assert [e["name"] for e in merge_arrays(lower, higher, path="hooks")] == ["a", "b"]


def test_array_with_no_declared_identity_refuses() -> None:
    with pytest.raises(MergeError, match="identity key"):
        merge_arrays([{"x": 1}], [{"x": 2}], path="mystery")


def test_project_beats_user(tmp_path: pathlib.Path) -> None:
    """The project knows facts the person does not."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text('profiles = "python"\n', encoding="utf-8")
    (roots.project / "config.toml").write_text('profiles = "rust"\n', encoding="utf-8")
    out = _load(roots, environ={})
    assert out.config["profiles"] == "rust"
    assert out.engine.profiles == "rust"


def test_environment_beats_project(tmp_path: pathlib.Path) -> None:
    roots = layer_roots(tmp_path)
    (roots.project / "config.toml").write_text('profiles = "rust"\n', encoding="utf-8")
    out = _load(roots, environ={"CHICKADEE_PROFILES": "go"})
    assert out.engine.profiles == "go"


def test_env_double_underscore_nests_and_single_stays_in_the_key(
    tmp_path: pathlib.Path,
) -> None:
    """Real keys contain single underscores — `hard_cost_usd`, `model_id` — so a separator
    that collides with them is a separator that cannot be used. Double nests, to any depth
    the model has, a named model's table included — and a rung's model is one leaf of
    `[rungs]`, which a higher layer sets without replacing the rest."""
    out = _load(
        layer_roots(tmp_path),
        environ={
            "CHICKADEE_SPEND__HARD_COST_USD": "7.5",
            "CHICKADEE_MODELS__MAIN_MODEL__MODEL_ID": "local-coder",
            # A default rung's name has a hyphen: a shell cannot `export` it, `env` can.
            "CHICKADEE_RUNGS__HIGH-PARAM": "other_model",
        },
    )
    assert out.config["spend"]["hard_cost_usd"] == 7.5
    assert out.engine.models["main_model"].model_id == "local-coder"
    other = out.engine.models["other_model"]
    assert other.model_id == "m", "a table merges, it does not replace"
    assert out.engine.rungs["high-param"] == "other_model"
    assert out.engine.rungs["mid-param"] == "main_model", "one rung moved, the rest did not"


def test_env_values_are_typed_not_left_as_strings(tmp_path: pathlib.Path) -> None:
    """A cap read as the string "5.0" compares wrongly against a float and fails open."""
    out = _load(
        layer_roots(tmp_path),
        environ={
            "CHICKADEE_SPEND__HARD_COST_USD": "5.0",
            "CHICKADEE_LIMITS__WORKER__MAX_TOOL_CALLS": "3",
        },
    )
    assert out.config["spend"]["hard_cost_usd"] == 5.0
    # The cap's home is the register since `T-3.121`; `[tools]` holds no numbers.
    assert isinstance(out.config["limits"]["worker.max_tool_calls"], int)
    worker_cap = next(
        rl
        for rl in out.engine.resolved_limits
        if (rl.role, rl.name) == ("worker", "worker.max_tool_calls")
    )
    assert worker_cap.tokens_or_value == 3


def test_a_repositorys_deny_add_adds_to_the_persons_never_replaces_it(
    tmp_path: pathlib.Path,
) -> None:
    """§4 rule 6. Merged as a scalar, the clone's list erased the person's, and a file
    they had denied became readable — with nothing for the trust digest to ask about."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text(
        '[secrets]\ndeny_add = ["*.kdbx", "vault/**"]\n', encoding="utf-8"
    )
    (roots.project / "config.toml").write_text(
        '[secrets]\ndeny_add = ["*.bak", "*.kdbx"]\n', encoding="utf-8"
    )
    secrets = _load(roots, environ={}).engine.secrets
    assert secrets.deny_add == ("*.kdbx", "vault/**", "*.bak"), "layer order, each once"
    unaccepted = secrets.denylist(accepted=False)
    assert not unaccepted.readable("vault.kdbx")
    assert not unaccepted.readable("notes.bak")


def test_a_deny_add_that_is_not_a_list_refuses_in_its_own_layer(
    tmp_path: pathlib.Path,
) -> None:
    """ST-11 again. `deny_add = "*.kdbx"` alone is refused by the model; under a repository's
    list it never reached the model, and the person's pattern vanished from a load that
    succeeded. Refused naming their file, since that is where the fix is."""
    roots = layer_roots(tmp_path)
    user = roots.user / "config.toml"
    user.write_text('[secrets]\ndeny_add = "*.kdbx"\n', encoding="utf-8")
    (roots.project / "config.toml").write_text(
        '[secrets]\ndeny_add = ["*.bak"]\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=rf"{re.escape(str(user))}: .*layer: user"):
        _load(roots, environ={})


def test_the_accumulated_list_names_every_layer_that_supplied_it(
    tmp_path: pathlib.Path,
) -> None:
    """The provenance is the layer that supplied each value. The union came from both, so
    the row says so — naming only the repository would credit it with the person's
    additions, and read as unchanged when theirs changed."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text(
        '[secrets]\ndeny_add = ["*.kdbx"]\n', encoding="utf-8"
    )
    (roots.project / "config.toml").write_text(
        '[secrets]\ndeny_add = ["*.bak"]\n', encoding="utf-8"
    )
    rows = {row.key: row.layer for row in _load(roots, environ={}).provenance}
    assert rows["secrets.deny_add"] == "user+project"
    assert rows["spend.hard_cost_usd"] == "defaults", "a key that overrides keeps one layer"


def test_a_limit_overrides_across_layers_however_either_spelled_it(
    tmp_path: pathlib.Path,
) -> None:
    """Two spellings, one key. Merged as written, the engine config's
    dotted key and the environment's nested one were "set twice" — a refusal no one could
    act on, since each file set it once. Each layer is brought to one spelling first."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text(
        '[limits]\n"worker.max_tool_calls" = 60\n', encoding="utf-8"
    )
    out = _load(roots, environ={"CHICKADEE_LIMITS__WORKER__MAX_TOOL_CALLS": "70"})
    assert out.config["limits"] == {"worker.max_tool_calls": 70}
    rows = [r for r in out.engine.resolved_limits if r.name == "worker.max_tool_calls"]
    assert [(r.tokens_or_value, r.source) for r in rows] == [(70, "config")]


def test_one_file_spelling_a_limit_both_ways_refuses_naming_it(tmp_path: pathlib.Path) -> None:
    """Within one file TOML allows both spellings, and which won would depend on table order."""
    roots = layer_roots(tmp_path)
    path = roots.user / "config.toml"
    path.write_text(
        '[limits]\n"worker.max_tool_calls" = 60\n\n[limits.worker]\nmax_tool_calls = 70\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=rf"{re.escape(str(path))}: .*set twice"):
        _load(roots, environ={})
