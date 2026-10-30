"""`EngineConfig` is the one config vocabulary, and what `init` writes loads — beside
the person's engine config, since put models and money there.

The layered loader kept a closed key set of its own, `DECLARED`, in names nothing wrote. It
refused every `rungs.<name>` table and the `[spend]` table `chickadee init` seeds, and
`EngineConfig` refused init's file too. Nothing broke only because no production path loaded
a config file yet; these are the properties the first one inherits.
"""

from __future__ import annotations

import pathlib
import re
import tomllib
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict
from repos import layer_roots

import chickadee.state.resolve as loader
from chickadee.config import REPOSITORY, EngineConfig, Names
from chickadee.errors import ConfigError
from chickadee.init.orchestrate import render_config
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import Layer, load_layers
from chickadee.state.merge import UnknownKey
from chickadee.state.resolve import EngineOnlyKey, check_layer, resolve

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
"""A complete engine config: a model per rung, the `[rungs]` table naming them, and a finite
cap."""


def test_what_init_writes_loads_beside_an_engine_config_through_both_paths(
    tmp_path: pathlib.Path,
) -> None:
    """D126 item 5, under D127. init's file holds repository facts only, so it passes the
    project layer's scope check; over the person's engine config it loads through
    `resolve()`, and the two files as one load through `from_toml` to an equal config — one
    path, so they cannot disagree."""
    roots = layer_roots(tmp_path)
    repository = roots.project / "config.toml"
    repository.write_text(render_config(()), encoding="utf-8")
    seeded = tomllib.loads(repository.read_text(encoding="utf-8"))
    assert set(seeded) == {"profiles"}, "no cap, no rungs: those are the person's"
    check_layer(Layer("project", repository, seeded))
    (roots.user / "config.toml").write_text(ENGINE, encoding="utf-8")

    layered = resolve(load_layers(roots, environ={})).engine
    one = tmp_path / "one.toml"
    one.write_text(render_config(()) + "\n" + ENGINE, encoding="utf-8")
    assert layered == EngineConfig.from_toml(one)
    assert layered.profiles == "auto"


def test_with_no_engine_config_the_load_names_the_tables_and_the_file(
    tmp_path: pathlib.Path,
) -> None:
    """Refused, and in words a person can act on: the tables to write — the model catalog
    and the rung table that names into it — and the file they go in, the engine
    config and never the repository's, not pydantic's *Field required*."""
    roots = layer_roots(tmp_path)
    repository = roots.project / "config.toml"
    repository.write_text(render_config(()), encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape("[models.<name>]")) as layered:
        resolve(load_layers(roots, environ={}))
    with pytest.raises(ConfigError, match=re.escape("[rungs]")):
        EngineConfig.from_toml(repository)
    message = str(layered.value)
    assert str(repository) in message
    assert "~/.config/chickadee/config.toml" in message
    assert "Field required" not in message
    assert "errors.pydantic.dev" not in message, "the reason, not pydantic's wrapper"


def test_a_model_table_passes_the_per_layer_check() -> None:
    """`DECLARED` gave `rungs` an empty key set, so every rung name was unknown. A named
    model's entries are checked against the model, keys and all — through an optional
    nested model too, or a typo there would surface as a validation error naming no file —
    and `[rungs]` against the rung name rule, its values being model names
    for the merged validation to look up."""
    coder = {"base_url": "http://localhost:8000/v1", "model_id": "m"}
    check_layer(
        Layer("user", None, {"models": {"local-coder": coder}, "rungs": {"low-param": "x"}})
    )
    with pytest.raises(UnknownKey, match=r"`models\.local-coder\.model_idd`.*layer: user"):
        check_layer(Layer("user", None, {"models": {"local-coder": {"model_idd": "m"}}}))
    reasoning = {"local-coder": {"reasoning": {"levle": "high"}}}
    with pytest.raises(UnknownKey, match=r"`models\.local-coder\.reasoning\.levle`"):
        check_layer(Layer("user", None, {"models": reasoning}))


def _every_key(
    schema: dict[str, Any], defs: dict[str, Any], entries: dict[str, str] | None = None
) -> dict[str, Any]:
    """A body setting every key a JSON schema declares, at every depth it gives a table
    shape. Built from pydantic's own schema of the model rather than from the model's
    annotations, so it is not the loader's derivation checking itself. A named section's
    entry is called by one of its closed names, `entries`, else `any_name`."""
    names = entries or {}
    return {
        name: _value(prop, defs, names.get(name, "any_name"))
        for name, prop in schema.get("properties", {}).items()
    }


def _value(prop: dict[str, Any], defs: dict[str, Any], entry: str) -> Any:
    for option in prop.get("anyOf", [prop]):
        if "$ref" in option:
            return _every_key(defs[option["$ref"].rsplit("/", 1)[-1]], defs)
        members = option.get("additionalProperties")
        if isinstance(members, dict) and "$ref" in members:
            ref: str = members["$ref"]
            return {entry: _every_key(defs[ref.rsplit("/", 1)[-1]], defs)}
    return 0


def test_every_key_the_model_declares_passes_with_no_list_in_the_loader() -> None:
    """The check derives from `EngineConfig`, so a section is a key the day the model has
    it: `limits` is accepted though the loader never heard of it."""
    schema = EngineConfig.model_json_schema()
    entries = {
        # A closed set's first name; an open one any name its rule accepts.
        name: min(mark.allowed) if mark.allowed else "any-rung"
        for name, field in EngineConfig.model_fields.items()
        for mark in field.metadata
        if isinstance(mark, Names)
    }
    body = _every_key(schema, schema["$defs"], entries)
    assert {"limits", "profiles", "models", "rungs", "spend", "secrets"} <= set(body)
    assert set(body["models"]["any_name"]) >= {"base_url", "context_window", "reasoning"}
    check_layer(Layer("user", None, body))
    assert not hasattr(loader, "DECLARED")


class _LaterSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    knob: int = 1


class _EngineConfigWithALaterSection(EngineConfig):
    later: _LaterSection = _LaterSection()
    later_fact: Annotated[_LaterSection, REPOSITORY] = _LaterSection()


def test_a_section_a_later_task_adds_needs_no_edit_to_the_loader() -> None:
    """A key joins the model with its feature, and the loader follows. Until
    it joins, the loader refuses it — a key nothing reads is ST-6's silent no-op."""
    later = Layer("user", None, {"later": {"knob": 2}})
    misspelled = Layer("user", None, {"later": {"knb": 2}})
    check_layer(later, _EngineConfigWithALaterSection)
    with pytest.raises(UnknownKey, match=r"`later\.knb`"):
        check_layer(misspelled, _EngineConfigWithALaterSection)
    with pytest.raises(UnknownKey, match="`later`"):
        check_layer(later)


def test_unsandboxed_escapes_are_the_persons_to_allow_and_never_a_repositorys() -> None:
    """`sandbox.allow_unsandboxed_escapes` is engine config, default false. A
    repository's `config.toml` carrying it refuses at load: a clone must not choose, for
    everyone who opens it, to run its own model-chosen commands unprotected."""
    assert EngineConfig.model_fields["sandbox"].default.allow_unsandboxed_escapes is False
    check_layer(Layer("user", None, {"sandbox": {"allow_unsandboxed_escapes": True}}))
    with pytest.raises(EngineOnlyKey, match="`sandbox`"):
        check_layer(Layer("project", None, {"sandbox": {"allow_unsandboxed_escapes": True}}))


def test_an_unsandboxed_shell_is_the_persons_to_allow_and_never_a_repositorys() -> None:
    """D137 decision 3, per role since item 8: `sandbox.allow_unsandboxed_shell` is engine
    config, each role off by default. A repository's `config.toml` carrying it refuses at load:
    a clone must not lift, for everyone who opens it, the confirmation that stands in for a
    missing sandbox."""
    lift = EngineConfig.model_fields["sandbox"].default.allow_unsandboxed_shell
    assert (lift.zeus, lift.worker, lift.athena) == (False, False, False)
    check_layer(Layer("user", None, {"sandbox": {"allow_unsandboxed_shell": {"zeus": True}}}))
    with pytest.raises(EngineOnlyKey, match="`sandbox`"):
        check_layer(
            Layer("project", None, {"sandbox": {"allow_unsandboxed_shell": {"zeus": True}}})
        )


def test_a_later_section_is_the_engine_configs_unless_its_decision_says_otherwise() -> None:
    """Engine-only is the default, so a section a later task adds cannot be set
    by a clone until the model marks it `REPOSITORY` — the safe direction to forget in."""
    with pytest.raises(EngineOnlyKey, match="`later`"):
        check_layer(
            Layer("project", None, {"later": {"knob": 2}}), _EngineConfigWithALaterSection
        )
    fact = Layer("project", None, {"later_fact": {"knob": 2}})
    check_layer(fact, _EngineConfigWithALaterSection)
