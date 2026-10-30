"""EngineConfig load and rejection paths.

The rejections are the deliverable. Each of these would otherwise surface mid-run, where
an unbounded loop or an unpayable bill is already happening.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from pydantic import ValidationError

from chickadee.config import EngineConfig, PriceRow, RungEndpoint
from chickadee.errors import ConfigError
from chickadee.roles import DEFAULT_RUNGS, ROLES
from chickadee.secrets.denylist import Denylist
from chickadee.state.layers import Layer
from chickadee.state.merge import UnknownKey
from chickadee.state.resolve import EngineOnlyKey, check_layer

_MODEL_TOML = 'base_url="u"\nmodel_id="m"\ncontext_window=65536\nmax_output_tokens=16384\n'


def _catalog_toml() -> str:
    """Model `m` serving every rung but the worker's, which runs on `w` (the critic may not
    share the worker's model): the smallest engine config that loads, less its cap."""
    return f"[models.m]\n{_MODEL_TOML}[models.w]\n{_MODEL_TOML}[rungs]\n" + "".join(
        f'{name} = "{"w" if name == "low-param" else "m"}"\n' for name in DEFAULT_RUNGS
    )


def _model(raw: dict[str, Any], rung: str) -> dict[str, Any]:
    """The catalog entry a rung runs on: what a test edits to change that rung's endpoint."""
    return raw["models"][raw["rungs"][rung]]


def test_minimal_config_loads(raw: dict[str, Any]) -> None:
    cfg = EngineConfig.with_defaults(raw)
    assert set(cfg.rungs) == set(DEFAULT_RUNGS)
    assert all(model in cfg.models for model in cfg.rungs.values())
    assert cfg.spend.hard_cost_usd == 5.0


def test_missing_hard_cap_is_rejected(raw: dict[str, Any]) -> None:
    """D42 and G0 item 4. `08-budget.md` §4 still calls this optional for interactive runs;
    that wording predates D42, and plan §0 rule 6 makes the newer decision authoritative."""
    raw["spend"] = {}
    with pytest.raises(ConfigError):
        EngineConfig.with_defaults(raw)


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_non_positive_cap_is_rejected(raw: dict[str, Any], cap: float) -> None:
    raw["spend"] = {"hard_cost_usd": cap}
    with pytest.raises(ConfigError):
        EngineConfig.with_defaults(raw)


@pytest.mark.parametrize("cap", ["inf", "nan"])
def test_a_cap_that_is_not_finite_refuses_the_load(tmp_path: pathlib.Path, cap: str) -> None:
    """Finite. TOML spells `inf` plainly, and `gt=0` alone passed it, so the file
    loaded and the run had no cap at all."""
    f = tmp_path / "c.toml"
    f.write_text(_catalog_toml() + f"[spend]\nhard_cost_usd = {cap}\n")
    with pytest.raises(ConfigError, match=r"spend\.hard_cost_usd"):
        EngineConfig.from_toml(f)


def test_model_without_context_window_is_rejected(raw: dict[str, Any]) -> None:
    """Without it nothing can tell whether a prompt fits — a correctness gap. Declared
    per model since, so the refusal names the catalog entry."""
    del _model(raw, "low-param")["context_window"]
    with pytest.raises(ConfigError, match=r"models\.low-param-model\.context_window"):
        EngineConfig.with_defaults(raw)


def test_model_without_max_output_tokens_is_rejected(raw: dict[str, Any]) -> None:
    del _model(raw, "low-param")["max_output_tokens"]
    with pytest.raises(ConfigError, match=r"models\.low-param-model\.max_output_tokens"):
        EngineConfig.with_defaults(raw)


def test_output_larger_than_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no room for a prompt"):
        RungEndpoint(base_url="u", model_id="m", context_window=4096, max_output_tokens=4096)


def test_missing_rung_is_rejected(raw: dict[str, Any]) -> None:
    """Naming the role that runs on it, the entry to add and the rungs that
    are declared, not only the rung."""
    del raw["rungs"]["high-param"]
    with pytest.raises(ConfigError, match=r"role 'zeus' runs on rung 'high-param'") as err:
        EngineConfig.with_defaults(raw)
    message = str(err.value)
    assert 'high-param = "<model>"' in message
    assert "the declared rungs are ['mid-param', 'low-param']" in message


def test_role_pointing_at_unknown_rung_is_rejected(raw: dict[str, Any]) -> None:
    raw["roles"] = {"locate": "nonexistent"}
    with pytest.raises(
        ConfigError, match=r"rung 'nonexistent', which \[rungs\] does not declare"
    ):
        EngineConfig.with_defaults(raw)


def test_roles_are_assignments_not_endpoints(raw: dict[str, Any]) -> None:
    """Moving a role onto another rung — and so onto that rung's model — stays a config
    edit."""
    raw["roles"] = {"locate": "high-param", "athena": "high-param"}
    cfg = EngineConfig.with_defaults(raw)
    assert cfg.assignment["locate"] == "high-param"
    assert cfg.rung_for("locate") is cfg.models[cfg.rungs["high-param"]]
    coder = cfg.models[cfg.rungs["low-param"]]
    assert cfg.rung_for("worker") is coder, "the cast's rung when not overridden"


def test_every_rung_and_no_tools_table_loads_with_every_role_capped(
    raw: dict[str, Any],
) -> None:
    """D48 by D122: the caps are the register's rows, resolved at load, one per tool-using
    role. `[tools]` holds none."""
    cfg = EngineConfig.with_defaults(raw)
    assert cfg.tools == {}
    caps = {
        rl.role: rl.tokens_or_value
        for rl in cfg.resolved_limits
        if rl.name == f"{rl.role}.max_tool_calls"
    }
    for r in ROLES:
        if r.tools != "none":
            assert caps[r.name] == r.max_tool_calls


def test_model_fields_default_and_validate(raw: dict[str, Any]) -> None:
    cfg = EngineConfig.with_defaults(raw)
    coder = cfg.rung_for("worker")
    assert coder.input_limit is None and coder.temperature is None
    assert coder.supports_temperature and coder.native_tools
    model = raw["rungs"]["low-param"]
    base = dict(raw["models"][model])
    raw["models"][model] = {**base, "input_limit": 60_000, "temperature": 0.2}
    assert EngineConfig.with_defaults(raw).rung_for("worker").input_limit == 60_000
    raw["models"][model] = {**base, "input_limit": base["context_window"] + 1}
    with pytest.raises(ConfigError, match="input_limit"):
        EngineConfig.with_defaults(raw)


def test_supports_images_defaults_true_and_loads(raw: dict[str, Any]) -> None:
    """A model can see unless its entry says it cannot."""
    assert EngineConfig.with_defaults(raw).rung_for("zeus").supports_images
    model = raw["rungs"]["high-param"]
    raw["models"][model] = {**raw["models"][model], "supports_images": False}
    assert not EngineConfig.with_defaults(raw).rung_for("zeus").supports_images


@pytest.mark.parametrize("value", ["false", "no", 0, 1])
def test_supports_images_that_is_not_a_boolean_is_refused(
    raw: dict[str, Any], value: object
) -> None:
    """A quoted `"false"` read loosely would load as a model that can see."""
    model = raw["rungs"]["high-param"]
    raw["models"][model] = {**raw["models"][model], "supports_images": value}
    with pytest.raises(ConfigError, match="supports_images"):
        EngineConfig.with_defaults(raw)


def test_price_cache_write_accepts_none_and_a_rate(raw: dict[str, Any]) -> None:
    model = raw["rungs"]["low-param"]
    raw["prices"] = {model: {"price_in": 1.0, "price_out": 2.0}}
    assert EngineConfig.with_defaults(raw).prices[model].price_cache_write is None
    raw["prices"] = {model: {"price_in": 1.0, "price_out": 2.0, "price_cache_write": 1.25}}
    assert EngineConfig.with_defaults(raw).prices[model].price_cache_write == 1.25


def test_a_model_a_role_runs_on_priced_zero_is_a_notice(raw: dict[str, Any]) -> None:
    """T-3.159 (a REPL run): every price was 0, so spend read $0 and the cap could never trip.
    A notice, not a refusal: a free model is a real case."""
    model = raw["rungs"]["low-param"]
    raw["prices"] = {model: {"price_in": 0, "price_out": 0}}
    [said] = [n for n in EngineConfig.with_defaults(raw).notices if "prices" in n]
    assert said.startswith(f"[prices.{model!r}] is 0 in and out") and "cap can never" in said
    raw["prices"] = {model: {"price_in": 0, "price_out": 0.5}}
    assert not [n for n in EngineConfig.with_defaults(raw).notices if "prices" in n]


def test_price_table_naming_a_model_the_catalog_lacks_is_rejected(raw: dict[str, Any]) -> None:
    raw["prices"] = {"gpu5": {"price_in": 1.0, "price_out": 2.0}}
    with pytest.raises(ConfigError, match=r"\['gpu5'\], which the catalog does not have"):
        EngineConfig.with_defaults(raw)


def test_unknown_cache_price_is_none_not_zero() -> None:
    """B-4 prices unknown cache pessimistically; a 0 would claim a discount we cannot
    demonstrate, straight into the headline cost number."""
    assert PriceRow(price_in=1.0, price_out=2.0).price_cached_in is None


@pytest.mark.parametrize(
    "fractions",
    [(0.0,), (1.0,), (1.2,), (0.8, 0.5), (0.8, 0.8)],
    ids=["zero", "one", "above-one", "decreasing", "equal"],
)
def test_bad_warning_fractions_rejected(
    raw: dict[str, Any], fractions: tuple[float, ...]
) -> None:
    raw["spend"] = {"hard_cost_usd": 5.0, "warning_fractions": list(fractions)}
    with pytest.raises(ConfigError):
        EngineConfig.with_defaults(raw)


def test_good_warning_fractions_accepted(raw: dict[str, Any]) -> None:
    raw["spend"] = {"hard_cost_usd": 5.0, "warning_fractions": [0.5, 0.8, 0.95]}
    assert EngineConfig.with_defaults(raw).spend.warning_fractions == (0.5, 0.8, 0.95)


def test_unknown_field_is_rejected(raw: dict[str, Any]) -> None:
    raw["verify_tier"] = "deep"
    with pytest.raises(ConfigError):
        EngineConfig.with_defaults(raw)


def test_sandbox_defaults_to_requiring_one(raw: dict[str, Any]) -> None:
    """Defaulting this False would make the safest-looking config the least safe."""
    cfg = EngineConfig.with_defaults(raw)
    assert cfg.sandbox.require_sandbox is True
    assert cfg.sandbox.backend == "auto"


def test_workspace_defaults_to_in_place(raw: dict[str, Any]) -> None:
    """D230 decision 3. Was D39: in-place was an explicit choice, never a default."""
    cfg = EngineConfig.with_defaults(raw)
    assert cfg.workspace.mode == "in_place"
    assert cfg.workspace.require_lease is True


def test_repo_trust_defaults_to_first_party(raw: dict[str, Any]) -> None:
    assert EngineConfig.with_defaults(raw).repo_trust == "first_party"


def test_reasoning_and_json_mode_are_per_model(raw: dict[str, Any]) -> None:
    """Config, never chosen by a model mid-run. A property of the catalog entry since
    D128, so the rung that runs it inherits both and a rung on another model does not."""
    _model(raw, "high-param")["reasoning"] = {"level": "high"}
    _model(raw, "high-param")["json_mode"] = True
    cfg = EngineConfig.with_defaults(raw)
    planner = cfg.models[cfg.rungs["high-param"]]
    assert planner.reasoning is not None
    assert planner.reasoning.level == "high"
    assert planner.json_mode is True
    assert cfg.models[cfg.rungs["low-param"]].json_mode is False, (
        "code models never use JSON mode"
    )


def test_config_is_frozen(raw: dict[str, Any]) -> None:
    """Profiles freeze at run start; the same reasoning applies to the config a run began
    with — a mid-run edit must not change an in-flight run."""
    cfg = EngineConfig.with_defaults(raw)
    with pytest.raises(ValidationError):
        cfg.repo_trust = "third_party"  # type: ignore[misc]


def test_toml_roundtrip(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "chickadee.toml"
    f.write_text(
        """
[models.local-scout]
base_url = "http://localhost:8000/v1"
model_id = "scout-m"
context_window = 32768
max_output_tokens = 2048

[models.local-coder]
base_url = "http://localhost:8000/v1"
model_id = "coder-m"
context_window = 65536
max_output_tokens = 16384

[models.frontier]
base_url = "https://api.example.test/v1"
model_id = "frontier-m"
api_key_env = "CHICKADEE_FRONTIER_KEY"
context_window = 200000
max_output_tokens = 16384

[models.other-family]
base_url = "https://api.other.test/v1"
model_id = "other-family-m"
context_window = 128000
max_output_tokens = 8192

[models.spare]
base_url = "http://localhost:8001/v1"
model_id = "spare-m"
context_window = 65536
max_output_tokens = 16384

[rungs]
high-param = "frontier"
mid-param = "local-scout"
low-param = "local-coder"
cross-family = "other-family"

[roles]
critic = "cross-family"

[prices.frontier]
price_in = 3.0
price_out = 15.0

[spend]
hard_cost_usd = 12.5

[limits]
"worker.max_tool_calls" = 8
"""
    )
    cfg = EngineConfig.from_toml(f)
    assert cfg.models["frontier"].api_key_env == "CHICKADEE_FRONTIER_KEY"
    assert cfg.rungs["high-param"] == "frontier"
    assert cfg.rung_for("athena") is cfg.rung_for("zeus") is cfg.models["frontier"]
    assert cfg.rung_for("critic") is cfg.models["other-family"], "a rung of the person's own"
    assert cfg.notices == (), "every declared rung has a role on it"
    assert "spare" in cfg.models and "spare" not in cfg.rungs.values()
    assert cfg.prices["frontier"].price_out == 15.0
    assert cfg.spend.hard_cost_usd == 12.5
    worker_cap = next(
        rl
        for rl in cfg.resolved_limits
        if (rl.role, rl.name) == ("worker", "worker.max_tool_calls")
    )
    assert (worker_cap.tokens_or_value, worker_cap.source) == (8, "config")


def test_toml_with_an_unknown_key_is_a_config_error_naming_it(tmp_path: pathlib.Path) -> None:
    """ST-6's refusal is `state`'s `UnknownKey`; `from_toml` still promises one exception
    type. `hard_cost_cap_usd` is the layered loader's old spelling of the cap."""
    f = tmp_path / "c.toml"
    f.write_text("[spend]\nhard_cost_cap_usd = 5.0\n")
    with pytest.raises(ConfigError, match=r"c\.toml: unknown key `spend\.hard_cost_cap_usd`"):
        EngineConfig.from_toml(f)


def test_a_missing_required_table_is_named_not_reported_as_a_field(
    raw: dict[str, Any],
) -> None:
    """D126 item 5, in D128's shape: the tables a person has to write — the catalog and the
    rung table that names into it — not pydantic's *Field required*."""
    del raw["models"], raw["rungs"], raw["spend"]
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(raw)
    message = str(err.value)
    assert "[models.<name>]" in message and "[rungs]" in message
    assert all(name in message for name in DEFAULT_RUNGS), "the rungs the cast runs on"
    assert "[spend]" in message and "hard_cost_usd" in message
    assert "Field required" not in message


@pytest.mark.parametrize(
    ("missing", "named", "absent"),
    [
        ("spend", "[spend]", ("[models.<name>]", "[rungs]")),
        ("models", "[models.<name>]", ("[rungs]", "[spend]")),
        ("rungs", "[rungs]", ("[models.<name>]", "[spend]")),
    ],
)
def test_one_missing_table_is_named_alone(
    raw: dict[str, Any], missing: str, named: str, absent: tuple[str, ...]
) -> None:
    """The sentence reads right whichever table is missing: it said "Both belong" when
    only one did."""
    del raw[missing]
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(raw)
    message = str(err.value)
    assert named in message
    assert not any(other in message for other in absent)
    assert "Both" not in message
    assert "~/.config/chickadee/config.toml" in message


@pytest.mark.parametrize("tools", [5, False, "worker"])
def test_a_tools_value_that_is_not_a_table_is_refused_not_filled(
    raw: dict[str, Any], tools: object
) -> None:
    """Read as "no table", `tools = false` was filled with the cast's caps and loaded as if
    nothing were wrong; `tools = 5` raised `TypeError`, and a string `ValueError` — neither
    the `ConfigError` a load promises."""
    raw["tools"] = tools
    with pytest.raises(ConfigError, match="tools"):
        EngineConfig.with_defaults(raw)


def test_a_misspelt_rung_loads_and_is_reported_with_the_name_it_meant(
    raw: dict[str, Any],
) -> None:
    """A well-formed `[rungs]` key declares a rung, so a misspelt
    `low-parm = "…"` beside a complete table loads; no role runs
    on it, and the notice names the rung it was likely meant to change."""
    raw["rungs"]["low-parm"] = raw["rungs"]["low-param"]
    cfg = EngineConfig.with_defaults(raw)
    assert cfg.notices == (
        "rung 'low-parm' is declared and no role runs on it; did you mean 'low-param'?",
    )


def test_a_refusal_is_the_validators_sentence_not_pydantics_wrapper(
    raw: dict[str, Any],
) -> None:
    """A person reads this: what to change, not an error count and a documentation URL."""
    del raw["rungs"]["high-param"]
    with pytest.raises(ConfigError) as err:
        EngineConfig.with_defaults(raw, source="/home/u/.config/chickadee/config.toml")
    message = str(err.value)
    assert message.startswith("/home/u/.config/chickadee/config.toml: role 'zeus' runs on")
    assert "errors.pydantic.dev" not in message
    assert "validation error" not in message


@pytest.mark.parametrize(
    "content", [b"[spend\nhard_cost_usd = 1.0\n", b"\xff\xfe not utf-8"], ids=["toml", "bytes"]
)
def test_a_file_that_cannot_be_parsed_is_a_config_error_naming_it(
    tmp_path: pathlib.Path, content: bytes
) -> None:
    """`from_toml`'s one exception type covers the parse too, not only the validation."""
    f = tmp_path / "c.toml"
    f.write_bytes(content)
    with pytest.raises(ConfigError, match=r"c\.toml: "):
        EngineConfig.from_toml(f)


def test_profiles_defaults_to_detection_and_takes_a_name(raw: dict[str, Any]) -> None:
    """`24-init.md` step 4 seeds `"auto"`; the state tests' layers set a name."""
    assert EngineConfig.with_defaults(raw).profiles == "auto"
    raw["profiles"] = "python"
    assert EngineConfig.with_defaults(raw).profiles == "python"
    raw["profiles"] = ""
    with pytest.raises(ConfigError):
        EngineConfig.with_defaults(raw)


# --- [secrets]: the deny list a person edits ---


def test_the_secrets_block_round_trips(raw: dict[str, Any]) -> None:
    raw["secrets"] = {
        "deny_add": ["*.kdbx"],
        "deny_remove": [".npmrc"],
        "allow_secret_read": False,
    }
    cfg = EngineConfig.with_defaults(raw)
    assert cfg.secrets.deny_add == ("*.kdbx",)
    assert cfg.secrets.deny_remove == (".npmrc",)
    listed = cfg.secrets.denylist(accepted=True)
    assert listed.readable(".npmrc") and not listed.readable("vault.kdbx")
    assert not listed.readable(".env"), "a removal is per entry"
    assert listed.removed == (".npmrc",)
    again = EngineConfig.with_defaults({**raw, "secrets": cfg.secrets.model_dump()})
    assert again.secrets == cfg.secrets


def test_no_secrets_block_is_the_built_in_list(raw: dict[str, Any]) -> None:
    listed = EngineConfig.with_defaults(raw).secrets.denylist(accepted=True)
    assert listed.patterns == Denylist().patterns and listed.removed == ()
    assert not listed.allow_secret_read


def test_an_unknown_secrets_key_refuses_with_its_path(raw: dict[str, Any]) -> None:
    raw["secrets"] = {"deny_rmove": [".npmrc"]}
    with pytest.raises(ConfigError, match=r"secrets\.deny_rmove"):
        EngineConfig.with_defaults(raw)


def test_a_removal_naming_no_built_in_refuses_the_load(raw: dict[str, Any]) -> None:
    """At load, not at the first spawn: `.nmprc` removes nothing and must not read as a
    removal that worked."""
    raw["secrets"] = {"deny_remove": [".nmprc"]}
    with pytest.raises(ConfigError, match=r"\[secrets\].*\.nmprc"):
        EngineConfig.with_defaults(raw)


def test_an_unaccepted_config_keeps_its_additions_and_drops_its_shrinking_half(
    raw: dict[str, Any],
) -> None:
    """`accepted=False` is a clone whose owner has not read the trust digest: the built-ins
    stand, the additions apply, and neither the removal nor the flag has any effect."""
    raw["secrets"] = {
        "deny_add": ["*.kdbx"],
        "deny_remove": [".env"],
        "allow_secret_read": True,
    }
    inert = EngineConfig.with_defaults(raw).secrets.denylist(accepted=False)
    assert not inert.readable(".env") and not inert.readable("vault.kdbx")
    assert not inert.allow_secret_read and inert.removed == ()
    live = EngineConfig.with_defaults(raw).secrets.denylist(accepted=True)
    assert live.allow_secret_read


# -- sandbox.backend ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "require"), [("auto", True), ("native", True), ("none", False)]
)
def test_each_usable_backend_loads(raw: dict[str, Any], backend: str, require: bool) -> None:
    raw["sandbox"] = {"backend": backend, "require_sandbox": require}
    assert EngineConfig.with_defaults(raw).sandbox.backend == backend


def test_none_with_require_sandbox_is_refused_naming_both_keys(raw: dict[str, Any]) -> None:
    """Every command would refuse to run: a session that can do nothing is worse than a load
    that says why."""
    raw["sandbox"] = {"backend": "none"}
    with pytest.raises(ConfigError, match=r"sandbox\.backend.*sandbox\.require_sandbox"):
        EngineConfig.with_defaults(raw)


def test_container_is_refused_until_it_can_enforce(raw: dict[str, Any]) -> None:
    """The container backend is a sketch: selected, it would be reported as an
    enforcing sandbox it cannot be. Refused at load, naming why."""
    raw["sandbox"] = {"backend": "container"}
    with pytest.raises(ConfigError, match="container backend is a sketch"):
        EngineConfig.with_defaults(raw)


def test_a_repository_layer_carrying_sandbox_is_still_refused() -> None:
    check_layer(Layer("user", None, {"sandbox": {"backend": "native"}}))
    with pytest.raises(EngineOnlyKey, match="`sandbox`"):
        check_layer(Layer("project", None, {"sandbox": {"backend": "native"}}))


def test_the_workers_edit_choices_load_from_a_file(tmp_path: pathlib.Path) -> None:
    """The three keys are `[tools.worker]`'s, and a model's section
    replaces them key by key; capabilities stay on the model."""
    f = tmp_path / "c.toml"
    f.write_text(
        _catalog_toml().replace(
            "[models.w]\n", "[models.w]\ngrammar_tools = true\n"
        )  # the worker's model, `w`, takes a grammar tool
        + """
[spend]
hard_cost_usd = 1.0

[tools.worker]
enabled = ["read", "apply_patch", "replace_lines"]
line_anchors = "number"
edit_check = "report"
constrain = ["apply_patch"]

[tools.worker.models.m]
line_anchors = "number"
constrain = []
"""
    )
    cfg = EngineConfig.from_toml(f)
    policy = cfg.tool_policy("worker")
    assert (policy.line_anchors, policy.edit_check, policy.constrain) == (
        "number",
        "report",
        ("apply_patch",),
    )


def test_an_edit_choice_nothing_reads_is_refused_from_a_file(tmp_path: pathlib.Path) -> None:
    """`edit_check` is read by an edit tool; a worker offered none has nothing that reads it."""
    f = tmp_path / "c.toml"
    f.write_text(
        _catalog_toml()
        + "[spend]\nhard_cost_usd = 1.0\n"
        + '[tools.worker]\nenabled = ["read"]\nedit_check = "off"\n'
    )
    with pytest.raises(ConfigError, match=r"tools\.worker: edit_check is set"):
        EngineConfig.from_toml(f)


def test_the_shell_lift_is_a_table_per_role_and_the_old_boolean_is_refused(
    raw: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    """D199 item 8 moved this answer: `allow_unsandboxed_shell = true` lifted Zeus's floor; now
    it is refused naming the table, never read as `zeus = true`."""
    f = tmp_path / "c.toml"
    f.write_text(
        _catalog_toml()
        + "[spend]\nhard_cost_usd = 1.0\n[sandbox.allow_unsandboxed_shell]\nworker = true\n"
    )
    lift = EngineConfig.from_toml(f).sandbox.allow_unsandboxed_shell
    assert (lift.of("zeus"), lift.of("worker"), lift.of("athena")) == (False, True, False)
    f.write_text(
        _catalog_toml()
        + "[spend]\nhard_cost_usd = 1.0\n[sandbox]\nallow_unsandboxed_shell = true\n"
    )
    with pytest.raises(
        ConfigError, match=r"allow_unsandboxed_shell is a table, one key per role"
    ):
        EngineConfig.from_toml(f)
    with pytest.raises(
        ConfigError, match=r"allow_unsandboxed_shell is a table, one key per role"
    ):
        EngineConfig.with_defaults({**raw, "sandbox": {"allow_unsandboxed_shell": True}})


def test_never_ask_is_off_unless_the_person_sets_it_and_never_a_repositorys(
    raw: dict[str, Any],
) -> None:
    """D199 item 8b: the person's alone, as all of `[sandbox]` is."""
    assert EngineConfig.with_defaults(raw).sandbox.never_ask is False
    assert EngineConfig.with_defaults({**raw, "sandbox": {"never_ask": True}}).sandbox.never_ask
    check_layer(Layer("user", None, {"sandbox": {"never_ask": True}}))
    with pytest.raises(EngineOnlyKey, match="`sandbox`"):
        check_layer(Layer("project", None, {"sandbox": {"never_ask": True}}))


_RETIRED_NETWORK = (
    r"`network(\.require_host_enforcement)?` is no longer read.*nothing to enforce"
)


def test_host_enforcement_is_refused(raw: dict[str, Any], tmp_path: pathlib.Path) -> None:
    """D199 moved this answer: `[network] require_host_enforcement` loaded, and
    now no backend has a host to enforce, so a file or a config built in code carrying it is
    refused naming why, never loaded as if it did something."""
    f = tmp_path / "c.toml"
    f.write_text(
        _catalog_toml()
        + "[spend]\nhard_cost_usd = 1.0\n[network]\nrequire_host_enforcement = true\n"
    )
    with pytest.raises(ConfigError, match=_RETIRED_NETWORK):
        EngineConfig.from_toml(f)
    with pytest.raises(ConfigError, match=_RETIRED_NETWORK):
        EngineConfig.with_defaults({**raw, "network": {"require_host_enforcement": False}})
    for layer in ("user", "project"):
        with pytest.raises(UnknownKey, match="nothing to enforce"):
            check_layer(Layer(layer, None, {"network": {"require_host_enforcement": True}}))
