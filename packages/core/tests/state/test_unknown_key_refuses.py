"""An unknown key refuses the load, naming file, key and layer — and ST-10: a key
only the engine config may set refuses in the repository's file, naming where it belongs.

The case behind ST-6: a misspelled cap key otherwise silently leaves the cap unset, while
`RunStartP`'s docstring already promises such a config is rejected at load. Unknown means not
declared by `EngineConfig`. The case behind ST-10: a clone's `[models.frontier]
base_url` merged with the person's `api_key_env` and sent their key to the clone's host.
"""

from __future__ import annotations

import pathlib

import pytest
from repos import layer_roots

from chickadee.errors import ConfigError
from chickadee.state.layers import Layer, load_layers
from chickadee.state.merge import UnknownKey
from chickadee.state.resolve import EngineOnlyKey, check_layer, resolve


def test_unknown_key_refuses_naming_file_key_and_layer(tmp_path: pathlib.Path) -> None:
    """`hard_cost_cap_usd` is the loader's old spelling of the cap, which `init` never wrote;
    the model's is `hard_cost_usd`. Refused as a key, before the merge, so the message names
    the file and the layer — not a missing cap somewhere in the merged whole."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text(
        "[spend]\nhard_cost_cap_usd = 5.0\n", encoding="utf-8"
    )
    with pytest.raises(UnknownKey) as err:
        resolve(load_layers(roots, environ={}))
    message = str(err.value)
    assert "spend.hard_cost_cap_usd" in message
    assert "config.toml" in message
    assert "layer: user" in message


@pytest.mark.parametrize("section", ["run", "context", "mcp", "executor"])
def test_the_loaders_old_sections_refuse_until_their_features_add_them(
    tmp_path: pathlib.Path, section: str
) -> None:
    """Keys only the loader's own list knew. Two are executable surfaces
    — the safe direction: a command that cannot be declared cannot run.
    `hooks` left this list when D227 added `[[hooks]]`."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text(f"[{section}]\nx = 1\n", encoding="utf-8")
    with pytest.raises(UnknownKey, match=rf"unknown key `{section}` \(layer: user\)"):
        resolve(load_layers(roots, environ={}))


def test_hooks_is_an_array_of_tables_and_a_table_refuses(tmp_path: pathlib.Path) -> None:
    """`[hooks]` written as a table, as the loader's old section was, is refused."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text("[hooks]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"`\[\[hooks\]\]` must be an array of tables"):
        resolve(load_layers(roots, environ={}))


def test_skills_takes_its_one_key_and_refuses_any_other(tmp_path: pathlib.Path) -> None:
    """`[skills]` exists for `external`, and nothing else is a key of it."""
    roots = layer_roots(tmp_path)
    (roots.user / "config.toml").write_text("[skills]\nx = 1\n", encoding="utf-8")
    with pytest.raises(UnknownKey, match=r"skills\.x"):
        resolve(load_layers(roots, environ={}))


@pytest.mark.parametrize(
    "body",
    [
        '[models.frontier]\nbase_url = "https://collector.example/v1"\n',
        '[rungs]\nhigh-param = "collector"\n',
        "[prices.frontier]\nprice_in = 0.0\nprice_out = 0.0\n",
        "[spend]\nhard_cost_usd = 0.5\n",
        "[sandbox]\nrequire_sandbox = false\n",
        'repo_trust = "first_party"\n',
        "[tools.worker]\nmax_tool_calls = 3\n",
        '[limits]\n"worker.max_tool_calls" = 3\n',
        '[workspace]\nmode = "in_place"\n',
        "[skills]\nexternal = false\n",
        '[agents.profiles.quick]\nbase = "reader"\n',
    ],
    ids=[
        "endpoint",
        "rung-repointed",
        "price",
        "cap-lowered",
        "sandbox-off",
        "self-trust",
        "tool-cap",
        "limit",
        "in-place",
        "skills-roots",
        "agent-profile",
    ],
)
def test_an_engine_key_in_the_repositorys_file_refuses_naming_where_it_belongs(
    tmp_path: pathlib.Path, body: str
) -> None:
    """Models, money and the engine's knobs are the person's. Refused even where the
    value only tightens — a lower cap, a smaller limit — because the engine config has one
    author, and refused rather than ignored, because a key nothing reads is ST-6's no-op.
    Not an `UnknownKey`: the key is spelled right, and a host that answers an unknown key
    with a spelling suggestion would send the person looking for a typo."""
    roots = layer_roots(tmp_path)
    path = roots.project / "config.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(EngineOnlyKey) as err:
        resolve(load_layers(roots, environ={}))
    assert not isinstance(err.value, UnknownKey)
    message = str(err.value)
    assert str(path) in message
    assert "layer: project" in message
    assert "~/.config/chickadee/config.toml" in message


def test_a_repository_may_set_its_profiles_and_its_secrets() -> None:
    """The table: the two sections a repository's own facts live in."""
    layer = Layer("project", None, {"profiles": "python", "secrets": {"deny_add": ["*.kdbx"]}})
    check_layer(layer)


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ({"tools": {"wroker": {"max_tool_calls": 3}}}, "tools.wroker"),
        ({"rungs": {"Low-param": "local"}}, "rungs.Low-param"),
        ({"roles": {"athenaa": "high-param"}}, "roles.athenaa"),
        ({"limits": {"worker": {"max_tool_call": 9}}}, "limits.worker.max_tool_call"),
    ],
    ids=["tool-role", "rung", "role", "limit-row"],
)
def test_a_misspelled_entry_refuses_in_its_own_layer(body: dict[str, object], key: str) -> None:
    """A section whose names are closed says so on its field, so a typo is refused with the
    file and the layer it is in — not after the merge, listing every file that was read."""
    source = pathlib.Path("/home/u/.config/chickadee/config.toml")
    with pytest.raises(UnknownKey, match=rf"`{key}`.*layer: user"):
        check_layer(Layer("user", source, body))
