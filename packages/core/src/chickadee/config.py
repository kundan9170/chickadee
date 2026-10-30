"""EngineConfig — every bound the system promises, expressed where it can be checked.

The deliverable here is the **rejection paths**, not the fields. Invariant 11 says all loops
and spend are bounded; a bound that has no mechanism behind it is not a bound, and a
bound that is not expressible in config is not a bound. So each of these raises at load
rather than surfacing mid-run:

- a model without `context_window` / `max_output_tokens` — nothing can tell whether a
  prompt fits, which is a correctness gap, not a cost one;
- a number in `[tools.<role>]`, which nothing reads: a role's caps are `[limits]` rows, and
every tool-using role has one there, so none can be left unbounded;

- a config without a finite hard spend cap;
- a role resolving to a rung `[rungs]` does not declare, and two roles a role's
  `distinct_from` pairs resolving to one model;
- a rung naming no model, or a model the catalog lacks, and a price for a model the catalog
  lacks; a model name no layer could address (`MODEL_NAME`); a `[rungs.<name>]`
  endpoint table, the layout in older versions, refused as that rather than as a type error;
- warning fractions that are not strictly increasing inside (0, 1);
- a `limits.` key naming no row of the register, or a role that cannot fit one request on
  the model its rung runs;
- a rung name the rule refuses (`roles.rung_name_error`), and a former rung name, refused
  naming its successor (`roles.FORMER_RUNGS`).

A declared rung no role runs on loads, and is a notice (`EngineConfig.notices`), not a refusal:
a person may keep a slot for an experiment.

Which file may set each section is part of the vocabulary too: the engine config is
the person's, and a section a repository's config may set is marked `REPOSITORY` here.

**Models and rungs.** `[models.<name>]` is the catalog — what a `[rungs.<name>]`
table held before, one entry per model — and `[rungs]` declares the rungs, each naming the
catalog model it runs on. Both are the engine config's, so a clone cannot repoint a rung or the
endpoint behind it. Which rung each role runs on is answered once, here, as `assignment`.

Discovering any of these mid-run means discovering them as an unbounded loop or an
unpayable bill.
"""

from __future__ import annotations

import functools
import math
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    Strict,
    ValidationError,
    field_validator,
    model_validator,
)

from chickadee.adapters.base import BOX_DESCRIPTIONS, tool_descriptions
from chickadee.approval.commands import (
    CommandRow,
    CommandTable,
    CommandTableError,
    parse_row,
    shipped,
)
from chickadee.client.errors import AMBIENT_HEADERS_ENV
from chickadee.context.request import est_tools
from chickadee.context.terms import Vision
from chickadee.edits.lines import ANCHOR_STYLES
from chickadee.errors import ConfigError
from chickadee.hooks.spec import HookSpec
from chickadee.limits import (
    LIMITS,
    ResolvedLimit,
    check_fit,
    harness_limits,
    resolve,
    tokens_for_chars,
)
from chickadee.roles import (
    DEFAULT_RUNGS,
    FORMER_RUNGS,
    ROLES,
    assign,
    former_rung_error,
    role,
    rung_name_error,
)
from chickadee.schema import ReasoningSpec
from chickadee.schema.isolation import WorkspaceMode
from chickadee.secrets.denylist import Denylist
from chickadee.state.resolve import PROJECT_LAYER
from chickadee.suggest import nearest
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.meta import shown
from chickadee.tools.policy import (
    EDIT_KEYS,
    Calling,
    Capabilities,
    Ceiling,
    Choices,
    EditCheck,
    FileRule,
    RolePolicy,
    Selection,
    choice_refusals,
    edit_keys_read,
)
from chickadee.tools.policy import resolve as resolve_policy
from chickadee.tools.spec import SURFACE
from chickadee.tools.validate import resident

WIRE = ConfigDict(extra="forbid", frozen=True)


def _not_a_boolean(value: object) -> object:
    """TOML's `true` is not 1.0. Pydantic's default mode reads a boolean as a number, so a slip
    in the person's own config became a $1.00 cap, a price of 1.0 or a one-second timeout, and
    loaded without a word. Refused before the number is read, so the message is
    about the boolean, never about the 1.0 or 0.0 it would have become. Only a boolean: an int
    and a numeric string load as they always have."""
    if isinstance(value, bool):
        raise ValueError(
            f"{str(value).lower()} is a boolean, not a number; write one such as 5 or 2.5"
        )
    return value


Number = Annotated[float, BeforeValidator(_not_a_boolean)]
"""Every float the engine config reads: a cap, a price, a timeout, a sampling
value, a weight. `test_budget_config.py` sweeps the config's models and fails a float field
that is not one."""

Count = Annotated[int, BeforeValidator(_not_a_boolean)]
"""Every int the engine config reads: a window, a token or byte bound, a call cap,
an attempt count. `true` was a 1-token window. Swept as `Number` is. `[limits]` refuses a
boolean through its own register check (`limits.py`), which runs before any model."""

Flag = Annotated[bool, Strict()]
"""Every boolean the engine config reads: TOML's `true` or `false` and nothing else,
as `Number` and `Count` are strict about their kinds. Pydantic's default mode reads `"yes"`,
`"no"`, `1` and `0` as booleans, so a quoted `require_sandbox = "false"` loaded as an
unsandboxed run and a `supports_images = 0` as a blind model, without a word. The environment
layer turns `true` and `false` into booleans before this reads them (`state.layers._coerce`).
`test_config.py` sweeps the config's models and fails a boolean field that is not one."""

# The roles and their call caps are `roles.ROLES`.
ROLE_NAMES: tuple[str, ...] = tuple(r.name for r in ROLES)


@dataclass(frozen=True)
class RepositoryMaySet:
    """Marks an `EngineConfig` section a repository's `.chickadee/config.toml` may set.
    Unmarked is the default, and means the engine config's alone — the person's.
    On the field, so a section's scope is declared where it is."""


REPOSITORY = RepositoryMaySet()


@dataclass(frozen=True)
class Names:
    """A `dict` section's keys, named on the field: a closed set, or an open one any name a
    `rule` accepts may join. The per-layer check reads it, so a misspelled or malformed entry
    is refused by file and layer rather than after the merge, when neither is known any more;
    the validators below still refuse it for a config built in code."""

    allowed: frozenset[str]
    what: str
    retired: frozenset[str] = frozenset()
    """Names a decision retired. Not a typo, so the per-layer check lets them through without
    listing them as known, and the model refuses each naming the decision that retired it."""
    rule: Callable[[str], str | None] | None = None
    """The open form: a key is known when this returns None, and refused with what it
    returns otherwise. `allowed` is then empty."""


RETIRED_ROLES: Mapping[str, str] = {
    "elaborator": (
        "a role that was retired: elaboration is a request on Athena's session, not a "
        "role of its own with a rung or a cap"
    ),
}
"""Names a config written before a decision may still carry. They pass the per-layer check —
unlike a misspelling, they are not a typo to point at, and a typo's list of known names never
offers them — and `_refuse_retired` refuses each one naming the decision that retired it and
what replaced it, before any field is validated. Former rung names are `roles.FORMER_RUNGS`."""

RUNG_KEYS = Names(frozenset(), "rung", frozenset(FORMER_RUNGS), rule=rung_name_error)
ROLE_KEYS = Names(frozenset(ROLE_NAMES), "role", frozenset(RETIRED_ROLES))
LIMIT_KEYS = Names(frozenset(lim.name for lim in LIMITS), "limit")
ENGINE_CONFIG_PATH = "~/.config/chickadee/config.toml"
"""Where the engine config lives by default (`$XDG_CONFIG_HOME/chickadee/` when set), for
the refusals that tell a person which file to edit."""

MODEL_NAME = re.compile(r"[a-z][a-z0-9_-]*")
"""A catalog name: lower-case letters, digits, `_` and `-`, starting with a letter.
Every layer must be able to name a model, and the environment layer lowercases each part of
a variable name (`state/layers.py`): `[models.GPT5]` could never be overridden from it, and
`CHICKADEE_MODELS__GPT5__MODEL_ID` would build a second, incomplete `gpt5` table. A dot would
read as a nested key in the provenance record, and an empty name names nothing."""

_HOST = re.compile(r"(?i)(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")
"""A DNS host name with a dot in it (`[web.fetch] docs_domains`): no scheme, port or path."""

ENV_READS_AS_VALUE = frozenset({"true", "false", "inf", "infinity", "nan"})
"""Names `MODEL_NAME` admits that the environment layer reads as a boolean or a number, so
`CHICKADEE_RUNGS__CODER=nan` could not point a rung at a model called `nan`."""


class RungEndpoint(BaseModel):
    """One model surface: a `[models.<name>]` catalog entry, which the rungs that name
    it run on. Model identity is config; experiments pin it. The type keeps the name it
    had when a rung *was* its endpoint — it is still the endpoint a rung runs on."""

    model_config = WIRE

    provider: str = "local"
    base_url: str
    model_id: str
    api_key_env: str | None = None
    first_token_s: Number = Field(default=300.0, gt=0, allow_inf_nan=False)
    """The longest wait from sending a request to its reply's first streamed
    event, a server that never starts. Past it the attempt is retried. Five minutes by default,
    the owner's call (2026-10-20) over two: a local server loading a model, or a long
    prompt's prefill, can take longer than two before its first token."""
    token_gap_s: Number = Field(default=60.0, gt=0, allow_inf_nan=False)
    """The longest gap between two events once a reply has started, a server
    that stalls mid-reply. A reply that keeps arriving is never cut off, however long."""

    context_window: Count = Field(gt=0)
    """Required. Without it nothing in the system can tell whether a prompt fits."""
    max_output_tokens: Count = Field(gt=0)
    """Required, for the same reason."""

    reasoning: ReasoningSpec | None = None
    """Per-model (once per rung, now per model), provider-neutral, and
    never a retry lever — escalating reasoning on retry would confound the failing-set-
    stability signal the classifier reads."""
    input_limit: Count | None = None
    """The provider's separate input cap when it has one. `window_basis` is
    `input_limit` iff set; `None` → `context_limit`, and the response reserve does the
    extra work it then needs to. Never above `context_window`."""
    temperature: Number | None = None
    """The "rung default". `None` → the provider's default. A role's `Sampling`
    may override it; the `d5` ladder moves it per attempt."""
    supports_temperature: Flag = True
    """`False` → the parameter is never sent, and the temperature ladder is journaled as *not
    applied* on a rung running this model rather than 400ing."""
    supports_images: Flag = True
    """Whether this model can see an image. Per model, so every rung
    naming it sees alike; a rung never carries its own. `False` → `read` answers an image at
    `depth="full"` `NOT_SUPPORTED` on that rung, and the client refuses one that reaches it.
    Read through `EngineConfig.vision_for`, never by hand."""
    supports_documents: Flag = False
    """Whether this model reads a PDF itself, sent as a document part (Anthropic's
    `document` block, OpenAI Chat Completions' `file` part). Off by default, as `forced_tools`
    is: a local server that ignores a part it does not know says nothing, and a model must not
    be believed to have read what it was never shown. `False` → a PDF reaches it as its text
    layer, labelled so. Read through `EngineConfig.vision_for`, never by hand."""
    native_tools: Flag = True
    """The provider takes a `tools` array. Every role with tools calls them natively,
    so a model one runs on without it is refused at load."""
    grammar_tools: Flag = False
    """The provider takes a tool decoded under a grammar (OpenAI's Chat
    Completions custom tool). A capability only: a tool is constrained when a
    role's policy names it (`RolePolicy.constrain`), and naming one on a model without this
    is refused. Off by default: a local server's grammar constrains a whole reply, never one
    tool."""
    forced_tools: Flag = False
    """The provider enforces a forced tool call (`tool_choice: "required"`, Anthropic's
    `{"type": "any"}`): a reply that is not a call to an offered tool cannot come back. A
    capability only: the worker's `force_tool_call` asks for it. Off by default, because a
    server that ignores the field says nothing, and an unenforced force must not be believed."""
    json_mode: Flag = False
    """Whether this model may use the provider's JSON mode for structured non-code
    output. Not a switch for its calls: the code ban and the structured-output default
    are decided by what a call produces, never by this flag alone — since one
    model may serve a code rung and a structured rung, and only the structured calls may
    use it."""

    @model_validator(mode="after")
    def _output_fits_window(self) -> RungEndpoint:
        if self.max_output_tokens >= self.context_window:
            raise ConfigError(
                f"max_output_tokens ({self.max_output_tokens}) leaves no room for a prompt "
                f"in a {self.context_window}-token window"
            )
        if self.input_limit is not None and not 0 < self.input_limit <= self.context_window:
            raise ConfigError(
                f"input_limit ({self.input_limit}) must be positive and at most the "
                f"context_window ({self.context_window})"
            )
        return self


class PriceRow(BaseModel):
    """USD per **million** tokens, every column — the unit a provider publishes, so
    `price_in = 3.0` is copied from the price page as written; `budget/prices.py` divides
    once. Keyed by catalog model, not by rung: a price is the provider's, for a model.
    Local models are priced at a conservative hosted-market equivalent for headline
    economics; measured local TCO is reported separately."""

    model_config = WIRE

    price_in: Number = Field(ge=0)
    price_out: Number = Field(ge=0)
    price_cached_in: Number | None = None
    """None means the provider does not price cached input separately. Unknown cache is
    priced pessimistically rather than assuming the discount."""
    cache_affinity: Flag = False
    cache_ttl_s: Count = Field(default=0, ge=0)
    """How long a prefix stays warm, which is what session compaction reads to
    decide whether compacting is free or throws away a paid-for cache."""
    price_cache_write: Number | None = Field(default=None, ge=0)
    """Anthropic-style cache creation, 1.25x or 2x input. `None` → the provider does
    not price writes separately; they bill at `price_in` and the handoff names the
    undercount while the provider reports writes."""


class FileChoice(BaseModel):
    """`files = { scope =..., tests =... }`: a `FileRule`."""

    model_config = WIRE

    scope: Literal["packet", "graph_visible", "repository"]
    tests: Literal["hidden", "names_only", "visible"]


class ToolSelection(BaseModel):
    """One choice of a role's tools, every key optional: what it names replaces the role's
    default, what it omits is the default's."""

    model_config = WIRE

    enabled: tuple[str, ...] | None = None
    """Registry names, in the order the model is shown them."""
    shown_as: dict[str, str] | None = None
    """Registry name -> the name the model sees."""
    files: FileChoice | None = None
    line_anchors: str | None = None
    """An `ANCHOR_STYLES` name, for a worker offered a line tool."""
    edit_check: EditCheck | None = None
    """What a syntax error an edit call made does."""
    constrain: tuple[str, ...] | None = None
    """Offered tools decoded under their grammar."""
    force_tool_call: Flag | None = None
    """The worker's turns are asked to be tool calls, on a model with `forced_tools`."""
    calling: Calling | None = None
    """`"python"` turns on the Python arm for the worker, which calls its tools
    by Python blocks instead of native calls. The worker's alone; `"native"` by default."""

    def chosen(self) -> bool:
        return (
            self.enabled is not None
            or self.shown_as is not None
            or self.files is not None
            or any(getattr(self, k) is not None for k in EDIT_KEYS)
        )

    def selection(self, models: Mapping[str, Selection] | None = None) -> Selection:
        files = self.files
        return Selection(
            enabled=self.enabled,
            shown_as=self.shown_as,
            files=FileRule(files.scope, files.tests) if files is not None else None,
            models=models or {},
            line_anchors=self.line_anchors,
            edit_check=self.edit_check,
            constrain=self.constrain,
            force_tool_call=self.force_tool_call,
            calling=self.calling,
        )


class ToolPolicy(ToolSelection):
    """`[tools.<role>]`: which tools a role gets, and nothing else, and per
    catalog model in `models."<model>"`. Each resolved policy passes the
    dispatcher's checks (`tools/policy.py` `resolve`), within the role's ceiling in `ROLES`.

    It held three numbers once, and nothing read them: a role's tool-call cap binds from the
    limits register, a result's size from the tool that produced it,
    and repeated calls from the role's rules. A file that sets one
    is refused naming where it lives (`retired_key`), never loaded as if it did something.
    """

    models: dict[str, ToolSelection] = {}

    def chosen(self) -> bool:
        return super().chosen() or bool(self.models)


_BINDS: Mapping[str, str] = {
    "critic": "the critic task",
}
"""The task that moves each role's loop onto the dispatcher, and so first reads its
`[tools.<role>]` selection. Until then a selection would load and change nothing, which
the state rules forbid: it is refused naming the task. Each task deletes its row."""

_UNREAD_KEYS: Mapping[str, Mapping[str, str]] = {
    "athena": {
        "files": (
            "Athena reads through her design's Reader, over the run's whole worktree with "
            "tests visible; no port narrows it by the policy's rule, so a rule would change "
            "nothing"
        ),
    },
    "zeus": {
        "files": (
            "Zeus reads through its session's Reader, over the whole repository with tests "
            "visible; no port narrows it, so a narrower rule would change nothing"
        ),
    },
    "worker": {
        "files": (
            "the worker reads through its packet port, scoped by the packer's view of the "
            "packet; no port narrows it by the policy's rule, so a rule would "
            "change nothing until the packet port reads it"
        ),
    },
    "locate": {
        "files": (
            "the scout reads through its session's Reader and Search, over the whole "
            "repository; no port narrows them by the policy's rule, so a rule would change "
            "nothing"
        ),
    },
}
"""Keys of a bound role's selection its loop still does not read, each with why: refused when
set, here or in a `models` section."""


_RETIRED_TOOL_KEYS: dict[str, str] = {
    "max_tool_calls": ('a role\'s tool-call cap is `[limits] "{role}.max_tool_calls"`'),
    "max_result_bytes": (
        "a result is bounded by the tool that produces it, in `[limits]` rows such as "
        '`"zeus.inline_bytes"` and `"worker.result_max"`'
    ),
    "dedup": "repeated calls are handled by the role's rules, not by config",
}

_REBUILT_TOOL_KEYS: dict[str, str] = {
    "tool_protocol": (
        "every role calls its tools natively since the rebuild, and the "
        "prompt protocols retired with the packet loop"
    ),
    "reply_format": (
        "every edit is an edit-tool call since the rebuild: no patch is read from a reply"
    ),
}
"""Keys that were retired, refused with its reason rather than as unknown."""

_RETIRED_TABLES: dict[str, str] = {
    "network": (
        "a command asks for the network with `network: true` and the person's approval opens "
        "it; no backend filters by host, so `require_host_enforcement` had nothing to enforce"
    ),
}
"""Top-level tables a decision retired whole, refused with its reason, the table or any key
in it."""


def retired_key(path: str) -> str | None:
    """Why a key the config no longer reads is refused, or `None` for any other key. `path`
    is the key's dotted spelling (`tools.worker.max_tool_calls`). The one home of these
    reasons: the layer check (`state.resolve`) and the model both ask it, so a file and a
    config built in code hear the same words (no key loads without
    effect)."""
    parts = path.split(".")
    table = _RETIRED_TABLES.get(parts[0])
    if table is not None:
        return f"`{path}` is no longer read: {table}. Remove it"
    if len(parts) < 3 or parts[0] != "tools":
        return None
    key = parts[-1]
    rebuilt = _REBUILT_TOOL_KEYS.get(key)
    if rebuilt is not None:
        # On a role's table or on one of its `models."<model>"` sections alike.
        return f"`{path}` is no longer read: {rebuilt}. Remove it"
    if len(parts) != 3 or key not in _RETIRED_TOOL_KEYS:
        return None
    name = parts[1]
    reason = _RETIRED_TOOL_KEYS[key].format(role=name)
    return f"`{path}` is no longer read: {reason}. `[tools.{name}]` chooses tools only"


_OWN_SPECS = {s.name: s for s in SURFACE}
"""Specs of tools a role's default lists before they register (`tools/policy.py`): Zeus's
surface, so a choice of them is checked against a ceiling already."""


def _selection(chosen: ToolPolicy) -> Selection:
    models = {m: sel.selection() for m, sel in chosen.models.items()}
    return chosen.selection(models)


def _tool_refusals(
    name: str, default: RolePolicy, ceiling: Ceiling, chosen: ToolPolicy, model: str | None
) -> tuple[str, ...]:
    _, refusals = resolve_policy(
        default, ceiling, _selection(chosen), model=model, known=BUILTINS, own=_OWN_SPECS
    )
    return _placed(name, model, refusals)


def _where(name: str, model: str | None) -> str:
    """A role's table, or one of its `models` sections, as a file spells it."""
    return name + (f'.models."{model}"' if model is not None else "")


def _placed(name: str, model: str | None, refusals: Iterable[str]) -> tuple[str, ...]:
    """Refusals `tools/policy` words for a role, re-worded for the table they came from."""
    return tuple(
        f"tools.{_where(name, model)}: {r.removeprefix(f'{name}: ')}" for r in refusals
    )


_EDIT_ROLES = frozenset({"worker"})
"""The roles whose loop reads the edit choices: the worker writes code; no other
role edits, so on another role's table the keys would load and change nothing."""


_CHOICES = Choices(
    styles=frozenset(ANCHOR_STYLES),
    refused_styles={
        "hash": "the task worker's read shows line numbers, never hash tags, so a hash anchor "
        "could not be copied from what it reads; use 'number'"
    },
)
"""What the edit choices may name: the registry, so a new style is a choice the
moment it registers, less a style the worker's reads do not show."""


def _capabilities(endpoint: RungEndpoint) -> Capabilities:
    return Capabilities(grammar_tools=endpoint.grammar_tools)


def _refuse_retired_keys(table: Mapping[str, object]) -> None:
    """`retired_key` over the raw mapping, for a config built in code, which meets no layer
    check."""
    for name, entry in table.items():
        if name in _RETIRED_TABLES:
            keys = list(
                cast("Mapping[str, object]", entry) if isinstance(entry, Mapping) else ()
            )
            path = f"{name}.{keys[0]}" if keys else name
            raise ConfigError(retired_key(path) or name)
    tools = table.get("tools")
    if not isinstance(tools, Mapping):
        return
    for name, entry in cast("Mapping[str, object]", tools).items():
        if not isinstance(entry, Mapping):
            continue
        for key in cast("Mapping[str, object]", entry):
            reason = retired_key(f"tools.{name}.{key}")
            if reason is not None:
                raise ConfigError(reason)
        models = cast("Mapping[str, object]", entry).get("models")
        if not isinstance(models, Mapping):
            continue
        for model, section in cast("Mapping[str, object]", models).items():
            if not isinstance(section, Mapping):
                continue
            for key in cast("Mapping[str, object]", section):
                reason = retired_key(f"tools.{name}.models.{model}.{key}")
                if reason is not None:
                    raise ConfigError(reason)


def _layer_file(layer: str) -> str:
    """Where a layer's rows were written, as a refusal names it."""
    return {
        PROJECT_LAYER: "the repository's.chickadee/config.toml",
        "user": f"the engine config, {ENGINE_CONFIG_PATH}",
    }.get(layer, f"the {layer} layer")


def is_cap(value: float) -> bool:
    """A cap is a finite amount above zero: the one predicate the rung caps' validator and
    `PacketEngine.set_cap` ask. `hard_cost_usd`'s `Field(gt=0,
    allow_inf_nan=False)` states the same rule to pydantic. A boolean is a separate question,
    each caller's own."""
    return math.isfinite(value) and value > 0


class SpendPolicy(BaseModel):
    model_config = WIRE

    hard_cost_usd: Number = Field(gt=0, allow_inf_nan=False)
    """Required and finite. Older wording called this optional for
    interactive runs; the cap is now required and enforced. Finite is its own
    constraint: `gt=0` passes `inf`, which TOML spells plainly and which is no cap."""
    warning_fractions: tuple[Number, ...] = (0.8,)
    packet_warning_cost_usd: Number | None = None
    """A diagnostic only. It never substitutes for the run cap — a per-packet warning
    cannot stop a run that is bleeding across many cheap packets."""
    rungs: Annotated[dict[str, Number], RUNG_KEYS] = {}
    """A cap per rung, beneath `hard_cost_usd`, never instead of it. A rung not named
    has no rung cap. The person's engine config only: a repository cannot set one.
    Each names a rung `[rungs]` declares, which `EngineConfig` checks: this table cannot see
    `[rungs]`."""

    @model_validator(mode="after")
    def _rung_caps_are_usable(self) -> SpendPolicy:
        _refuse_former(self.rungs, "[spend.rungs]")
        for rung, cap in self.rungs.items():
            if not is_cap(cap):
                raise ConfigError(
                    f"[spend.rungs] {rung} = {cap}: a rung's cap is finite and above zero "
                )
        return self

    @model_validator(mode="after")
    def _fractions_are_usable(self) -> SpendPolicy:
        prev = 0.0
        for f in self.warning_fractions:
            if not 0.0 < f < 1.0:
                raise ConfigError(f"warning fraction {f} is not inside (0, 1)")
            if f <= prev:
                raise ConfigError(
                    f"warning fractions must strictly increase, got {self.warning_fractions}"
                )
            prev = f
        return self


ShellRole = Literal["zeus", "worker", "athena"]
"""A role that holds a shell."""

SHELL_ROLES: Final[tuple[ShellRole, ...]] = ("zeus", "worker", "athena")


class ShellLift(BaseModel):
    """`[sandbox.allow_unsandboxed_shell]`: per role, whether its shell runs a command unasked
    where no sandbox enforces anything. Each off by default."""

    model_config = WIRE

    zeus: Flag = False
    worker: Flag = False
    athena: Flag = False

    def of(self, role: ShellRole) -> bool:
        """The role's key: each field is named for its role (`SHELL_ROLES`)."""
        lifted: bool = getattr(self, role)
        return lifted


class SandboxPolicy(BaseModel):
    model_config = WIRE

    backend: Literal["auto", "native", "container", "none"] = "auto"
    require_sandbox: Flag = True
    """When no enforcing backend exists, abort rather than run at `sandbox=none`.
    Defaulting this to False would make the safest-looking config the least safe one."""
    allow_unsandboxed_escapes: Flag = False
    """Whether the scout's `request_exec` may run a command where no sandbox backend
    enforces DISCOVERY. The command is a model's, chosen after reading the repository, and
    nobody confirms it, so without an enforcing backend it is refused unless the person
    opted in here. The person's alone: `sandbox` is engine config, so a repository's
    `config.toml` carrying it refuses at load. It is not `require_sandbox`, which governs
    every runner; turning that off to run tests on a sandbox-less machine does not agree to
    run model-chosen commands unprotected. Read by `zeus.tools.navigate.scout_locator`."""
    allow_unsandboxed_shell: ShellLift = ShellLift()
    """Whether each role's `shell` runs a command unasked where no sandbox
    backend enforces anything. Off, every command there is T2 and confirmed before it spawns,
    because nothing makes it sealed; on, an undeclared command is T0 again, and the network,
    the command table, a git-state or outside-the-repository command and every
    `deny` still raise or refuse it. One key per role, so lifting Zeus's asking never lifts a
    cheaper model's. The person's alone, as `allow_unsandboxed_escapes` is. Read by the
    engine's composition (Zeus's), the episode's (the box roles') and the run header."""
    allow_unsandboxed_python: Flag = False
    """Whether the worker's Python blocks
    (`calling = "python"`) run in a plain child process where no sandbox backend enforces
    anything. Off, such a block is not run, and its refusal names this key. `require_sandbox`
    wins: with it set, no block runs unsandboxed, as no shell does. The person's alone, as
    every `[sandbox]` key is."""
    never_ask: Flag = False
    """No `shell` command is asked, for any role, attended or not: what would be
    asked runs, approved as a preference, one `action_granted` line each. The person's `deny`
    rules and a secret path still refuse; other approvals (a mutating MCP call) still ask.
    Read by the session's grant check (`approval.grants.GrantingPort`) and the run header."""

    @model_validator(mode="before")
    @classmethod
    def _lift_is_a_table(cls, data: Any) -> Any:
        """The old boolean is refused naming the table, never read as `zeus = true` (a moved
        answer said rather than guessed)."""
        if not isinstance(data, Mapping):
            return data
        table = cast("Mapping[str, object]", data)
        if isinstance(table.get("allow_unsandboxed_shell"), bool):
            raise ConfigError(
                "sandbox.allow_unsandboxed_shell is a table, one key per role: "
                "write [sandbox.allow_unsandboxed_shell] with zeus, worker and athena, each "
                "true or false"
            )
        return table

    @model_validator(mode="after")
    def _backend_is_usable(self) -> SandboxPolicy:
        """`backend` is read by every engine spawn site and the run header, so a value
        that could not run is refused here, naming why, rather than met at the first command."""
        if self.backend == "container":
            # Selected, the sketch in `isolation/container.py` would be reported as an
            # enforcing sandbox it cannot be: no image, no mounts beyond the workspace, no
            # `--user`, no environment, no cancellation.
            raise ConfigError(
                'sandbox.backend = "container" is not usable yet: the container backend is a '
                "sketch that cannot enforce what it would be reported as enforcing. "
                'Use "auto" or "native"'
            )
        if self.backend == "none" and self.require_sandbox:
            raise ConfigError(
                'sandbox.backend = "none" with sandbox.require_sandbox = true would refuse '
                "every command: set require_sandbox = false to run unwrapped on purpose, or "
                "choose another backend"
            )
        return self


class SecretsPolicy(BaseModel):
    """`[secrets]` — the deny list a person edits.

    ```toml
    [secrets]
    deny_add = ["*.kdbx", "vault/**"] # tighten: always free
    deny_remove = [".npmrc"] # a built-in, by name; the rest stay in force
    allow_secret_read = false # Layer 1 off whole; loud in header + handoff
    ```

    Until this block, both knobs were *described rather than built*: `allow_secret_read`
    was a dataclass default and nothing built a list from config, so the only deny list any
    run used was the built-in one. The two that **shrink** the list — `deny_remove` and
    `allow_secret_read = true` — are executable config:
    shipped in a repository, they are inert on a clone until its owner accepts the trust
    digest, which is why `denylist` asks whether they were accepted. `deny_add` is not in
    the digest: it can only tighten, and an inert addition would leave a fresh clone
    *looser* than the project asked for — the wrong direction for "inert".
    """

    model_config = WIRE

    deny_add: tuple[str, ...] = ()
    deny_remove: tuple[str, ...] = ()
    allow_secret_read: Flag = False

    @model_validator(mode="after")
    def _the_list_resolves(self) -> SecretsPolicy:
        """Every refusal the deny list has, at load: a removal naming no built-in, a pattern
        both added and removed, a blank pattern. Found when the file is read, not when the
        first spawn resolves its mask."""
        try:
            self.denylist(accepted=True)
        except ValueError as exc:
            raise ConfigError(f"[secrets]: {exc}") from exc
        return self

    def denylist(self, *, accepted: bool) -> Denylist:
        """The resolved list. `accepted` is `state.trust.is_active` for this config: when the
        clone's owner has not accepted the executable subset, the shrinking half is inert —
        the built-ins stand, the additions apply — and nothing a repository shipped has
        weakened the list on a machine whose owner never read it. No default: a caller that
        has not asked the trust question has to say so."""
        if not accepted:
            return Denylist.configure(add=self.deny_add)
        return Denylist.configure(
            add=self.deny_add,
            remove=self.deny_remove,
            allow_secret_read=self.allow_secret_read,
        )


class CommandRowConfig(BaseModel):
    """One `[[commands]]` row, as a layer wrote it:

    ```toml
    [[commands]]
    prefix = ["deploy.sh"] # a command's name, without its directory
    class = "push"
    ```

    `layer` is where the loader found it (`state/resolve`), never written in a file: the
    repository's rows are executable config, inert until accepted. A row
    in a config built in code is the person's."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    prefix: tuple[str, ...]
    cls: str = Field(alias="class")
    destination: str = "none"
    layer: str = "user"

    def row(self) -> CommandRow:
        """The row the table holds, refused as `approval.commands.parse_row` refuses one."""
        return parse_row(
            {"prefix": list(self.prefix), "class": self.cls, "destination": self.destination}
        )

    @property
    def from_repository(self) -> bool:
        return self.layer == PROJECT_LAYER


class PlanningPolicy(BaseModel):
    """The deterministic pre-model route. Runs before any model call (invariant 10)."""

    model_config = WIRE

    hard_triggers: tuple[str, ...] = (
        "vague_goal",
        "cross_module",
        "public_contract",
        "external_action",
        "oversized",
        "risky",
    )
    score_weights: dict[str, Number] = {}
    threshold: Number = Field(default=0.5, ge=0.0, le=1.0)


class ExecutionDifficultyPolicy(BaseModel):
    model_config = WIRE

    score_weights: dict[str, Number] = {}
    frontier_threshold: Number = Field(default=0.8, ge=0.0, le=1.0)
    starting_candidates: Count = Field(default=1, gt=0)


class RescuePolicy(BaseModel):
    model_config = WIRE

    enrich_retry_attempts: Count = Field(default=2, gt=0)
    """The reduced allowance a rescue-origin packet re-enters with."""
    max_replans: Count = Field(default=2, ge=0)


class WorkspacePolicy(BaseModel):
    model_config = WIRE

    mode: WorkspaceMode = "in_place"
    """The work runs in the person's
    checkout, as every in-place harness does, and turn snapshots are its undo point. A worktree
    is the person's choice (`mode = "worktree"`, or asked for in the conversation). A `coder`
    call's own `workspace` wins, then where its design's folder is, then the worktree the
    session already writes (`delegation.chosen_workspace`)."""
    require_lease: Flag = True


class InteractionPolicy(BaseModel):
    """`[interaction]` — how long a person is waited for. The person's engine config: a
    repository cannot set it. One knob for every frontend, so the CLI, a served client and an
    unattended host reach the same terminals."""

    model_config = WIRE

    question_timeout_s: Number = Field(default=900, gt=0, allow_inf_nan=False)
    """A blocking question unanswered this long resolves to its default, as
    `timeout_default`, so a frontend that went away cannot hang a session. Read by
    `RemoteInteraction`."""
    max_questions_per_run: Count = Field(default=3, ge=0)
    """The blocking questions one run may put to a person; past it, a question is
    answered by its default and the log says why. Read by the session's `RecordingPort`."""
    auto_title: Flag = True
    """The `title` role names a session after its first turn, one short call on its
    rung; false makes no call, and the session keeps its first line as its title
    until the person renames it."""


class UndoPolicy(BaseModel):
    """`[undo]` — how long turn snapshots live and what they keep. The
    person's alone: `undo` is engine config, so a repository's `config.toml` carrying it refuses
    at load. `on_your_branch` is read by the undo."""

    model_config = WIRE

    keep_days: Count = Field(default=7, ge=0)
    """"deleting like in a week". A snapshot younger than this is
    never removed, and an older one only when it is outside `keep_turns` too."""
    keep_turns: Count = Field(default=50, ge=0)
    """"always keep last 50 or so turns". The session's newest
    this many snapshotted turns are kept whatever their age."""
    untracked_max_bytes: Count = Field(default=2 * 1024 * 1024, ge=0)
    """"keep default 2MB", OpenCode's bound. A turn snapshot skips
    an untracked file larger than this and names it; tracked files are always kept."""
    on_your_branch: Literal["restore", "commit"] = "restore"
    """Both, as a config choice. After `apply` or `squash`,
    an undo puts the files back uncommitted, or also commits
    `Undo turns N-M` (`commit`). A conflict, or a per-command choice, falls back to `restore`;
    in place nothing is ever committed. Read by `undo.py`."""


_TEMPLATES = Path(__file__).parent / "adapters"


@functools.cache
def _template_tokens(family: str, name: str, work_dir: str) -> int:
    """A role's prompt template, measured with the token estimator: the role's own file, which
    is
    what its request sends; the worker's tools are measured beside it (`_worker_tools_tokens`).
    The family preamble and protocol the packet loop's prompts began with retired with it:
    counting them measured text no request sends. A template that does not
    exist yet (`roles.UNWRITTEN_PROMPTS`) measures 0; `fits()` still gates each call once it
    does. Measured as rendered, `work_dir` named (`work.with_work_dir`), so a long configured
    folder is counted where the request sends it. Cached: the files are
    package data and do not change under a running process."""
    from chickadee.work import with_work_dir

    path = _TEMPLATES / family / "templates" / name
    if not path.is_file():
        return 0
    return tokens_for_chars(len(with_work_dir(path.read_text(encoding="utf-8"), work_dir)))


def _worker_tools_tokens(policy: RolePolicy) -> int:
    """The worker's tools as its request sends them: each enabled built-in's native schema and
    description (`est_tools`, the one measure of them), as `box/worker.WorkerRun` shows them:
    each spec as the policy resolves it, so a withheld argument is not counted."""
    specs = tuple(
        policy.resolved(t.spec) for n in policy.enabled if (t := BUILTINS.get(n)) is not None
    )
    return est_tools(shown(resident(specs, tool_descriptions(specs, *BOX_DESCRIPTIONS))))


def flatten_limits(section: Mapping[str, Any], prefix: str = "limits") -> dict[str, Any]:
    """`[limits.worker] max_tool_calls = 50` and `[limits] "worker.max_tool_calls" = 50` are
    one key; `resolve()` reads the dotted spelling. A key spelled both ways is refused: TOML
    allows it, and which value won would depend on the order of the tables."""
    out: dict[str, Any] = {}
    for k, v in section.items():
        key = f"{prefix}.{k}"
        parts = (
            flatten_limits(v, key)  # pyright: ignore[reportUnknownArgumentType]
            if isinstance(v, Mapping)
            else {key: v}
        )
        for flat, value in parts.items():
            if flat in out:
                raise ConfigError(f"{flat} is set twice under [limits]; set it once")
            out[flat] = value
    return out


class SkillsPolicy(BaseModel):
    """`[skills]`. The person's alone: the compatibility roots include
    their own home directory's, so a repository's `config.toml` carrying it refuses at load."""

    model_config = WIRE

    external: Flag = True
    """Index `.claude/skills/` and `.agents/skills/` under the repository and `$HOME`,
    read-only and marked `external`. `false` indexes only `.chickadee/` and the user's own."""


class AgentProfileConfig(BaseModel):
    """`[agents.profiles.<name>]` a person's own profile,
    a narrowing of a built-in one. Each key it sets may only take something away from its
    `base`: fewer tools, a lower limit. `rung` moves it to another declared rung."""

    model_config = WIRE

    base: Literal["reader", "reviewer", "researcher"]
    rung: str | None = None
    tools: tuple[str, ...] | None = None
    """Registry names, each one its base's policy offers."""
    max_tool_calls: Annotated[Count, Field(gt=0)] | None = None
    max_cumulative_result_tokens: Annotated[Number, Field(gt=0, le=1)] | None = None
    """A window fraction, on the rung the profile runs on."""
    seconds: Annotated[Count, Field(gt=0)] | None = None
    max_cost_cents: Annotated[Count, Field(gt=0)] | None = None
    report_reasks: Annotated[Count, Field(ge=0)] | None = None


class AgentsPolicy(BaseModel):
    """`[agents]`. The person's alone: a profile decides what an agent may touch, so a
    repository's `config.toml` carrying it refuses at load."""

    model_config = WIRE

    profiles: dict[str, AgentProfileConfig] = {}


class TemplatesPolicy(BaseModel):
    """`[templates]`: a hand-off template file for each kind, so an eval
    run can pick one without a code change. The second place `handoffs.resolve` looks: a
    repository's `.chickadee/templates/<kind>.md` comes first, and the shipped template last.
    A `~` is the person's home; a relative path is read from the repository's root. A file
    named and missing falls through to the shipped one with a warning."""

    model_config = WIRE

    design: str | None = None
    plan: str | None = None
    brief: str | None = None

    def configured(self) -> dict[str, Path]:
        """Each kind this table names, its path with `~` expanded."""
        named = {"design": self.design, "plan": self.plan, "brief": self.brief}
        return {kind: Path(path).expanduser() for kind, path in named.items() if path}


class WorkPolicy(BaseModel):
    """`[work]` (B7): where a design's work folder goes and whether it is
    committed. The person's alone: a repository's `config.toml` carrying it refuses at load."""

    model_config = WIRE

    dir: str = Field(default="docs/chickadee", min_length=1)
    """The work folders' parent, from the repository root (decision 4). Relative and inside the
    repository: `work.dir_error` refuses anything else at load."""
    commit: Flag = True
    """B7: `true` commits the folder with the work; `false` keeps it out of the repository."""

    @field_validator("dir")
    @classmethod
    def _inside(cls, value: str) -> str:
        from chickadee.work import dir_error

        why = dir_error(value)
        if why is not None:
            raise ValueError(why)
        return value


class WebFetchPolicy(BaseModel):
    """`[web.fetch]`."""

    model_config = WIRE

    engine: Literal["auto", "http", "lightpanda"] = "auto"
    """`auto`: `lightpanda` when it is installed and a sandbox seals it, else `http`. `http`
    reads a page as its server sent it; `lightpanda` runs its scripts first. Naming
    `lightpanda` when it is not installed, or when `sandbox.require_sandbox` is set and no
    sandbox seals, refuses the session as it is composed, saying which: said once, before any
    round, never inside one."""
    approval: Literal["never", "per_domain"] = "never"
    """`never`: a fetch is a read of a URL that appeared, approved by nobody. `per_domain`: the
    person is asked the first time a session reaches each host (`external/hosts.py`); with
    nobody to ask, the host is refused."""

    docs_domains: tuple[str, ...] = ()
    """Hosts, beyond the shipped `external.urls.DOCS_DOMAINS`, on which `web_fetch` may fetch a
    URL nobody gave it, with no query string: each a host whose pages its owner writes. A host
    that serves pages its users write lets a guess land on a page an attacker wrote."""

    @field_validator("docs_domains")
    @classmethod
    def _hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for host in value:
            if not _HOST.fullmatch(host):
                raise ValueError(f"{host!r} is not a host name: write it as docs.example.com")
        return tuple(host.lower() for host in value)


class WebSearchPolicy(BaseModel):
    """`[web.search]`: the general engine `web_search`'s `web` source asks,
    on the person's own key, and the person's GitHub token. Each key is named by the
    environment variable that holds it, as a rung's is, and read as the session is composed:
    never written to a config file, never journaled."""

    model_config = WIRE

    engine: Literal["tavily", "brave", "exa", "mojeek", "searxng"] | None = None
    """None: `web` is Exa's keyless trial. `searxng` is the person's own instance, at
    `searxng_url`."""
    api_key_env: str | None = Field(default=None, min_length=1)
    """The variable holding the engine's key; every engine but `searxng` needs one."""
    searxng_url: str | None = Field(default=None, min_length=1)
    """The person's SearXNG, `https://search.example.org`: the one host a search may reach on
    their own network."""
    github_token_env: str | None = Field(default=None, min_length=1)
    """The variable holding the person's GitHub token, for the `github` source's higher
    limit. Sent to GitHub only."""

    @model_validator(mode="after")
    def _complete(self) -> WebSearchPolicy:
        if self.engine == "searxng" and self.searxng_url is None:
            raise ValueError('[web.search] engine = "searxng" needs searxng_url')
        if self.engine not in (None, "searxng") and self.api_key_env is None:
            raise ValueError(
                f"[web.search] engine = {self.engine!r} needs api_key_env, the variable that "
                "holds its key"
            )
        if self.searxng_url is not None and not self.searxng_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("[web.search] searxng_url is an http or https URL")
        return self

    def key_variables(self) -> frozenset[str]:
        """The variables that hold this table's keys: withheld from every command."""
        named = (self.api_key_env, self.github_token_env)
        return frozenset(n for n in named if n)


class WebPolicy(BaseModel):
    """`[web]`. The person's alone: what a researcher may fetch from the network is not
    a repository's to widen, so a repository's `config.toml` carrying it refuses at load."""

    model_config = WIRE

    fetch: WebFetchPolicy = WebFetchPolicy()
    search: WebSearchPolicy = WebSearchPolicy()


class EngineConfig(BaseModel):
    """The one config vocabulary: a key exists exactly when this model, or a model
    nested in it, declares it. `state/resolve.py` checks every layer against it and
    validates the merge into it; `from_toml` is the one-layer case of that path."""

    model_config = WIRE

    profiles: Annotated[str, REPOSITORY] = Field(default="auto", min_length=1)
    """`init` seeds `"auto"`: the per-file detection, which
    may activate several profiles and never picks a repository-wide winner. Any other value
    names one profile. A key because `init` writes it and it is a real knob;
    no run selects profiles from config yet — detection is what runs — so a name set here
    changes nothing until the task that loads profiles for a run reads it."""
    models: dict[str, RungEndpoint]
    """The catalog: every model the person can run, by name — the endpoint, the windows
    and the capabilities a `[rungs.<name>]` table held before. The names are the
    person's to choose, so nothing closes them the way `rungs` closes its own; `MODEL_NAME`
    only keeps each one addressable from every layer. One model may
    serve several rungs, and a model no rung names is a switch target, not an
    error."""
    rungs: Annotated[dict[str, str], RUNG_KEYS]
    """Rung → the catalog model it runs on, by name: every key declares a rung, and a
    name the catalog lacks refuses the load. A rung a role resolves to must
    be declared; one no role resolves to loads, as a notice."""
    roles: Annotated[dict[str, str], ROLE_KEYS] = {}
    """Role → rung name, over the role's `default_rung`. Keys are the cast's names and nothing
    else: moving `locate` onto Athena's model stays a config edit
    (`roles = {locate = "high-param"}`), `roles = {reader = "high-param"}` is a refused load,
    and a retired name (`RETIRED_ROLES`) is refused naming its decision. Read through
    `assignment`, never directly."""
    prices: dict[str, PriceRow] = {}
    """Keyed by catalog model, not by rung: a price is the provider's, for a model, and
    a rung can run on two models in one session. A model the catalog lacks refuses the load;
    a catalog model with no row loads, and a call charged to it raises."""
    tools: Annotated[dict[str, ToolPolicy], ROLE_KEYS] = {}
    spend: SpendPolicy
    sandbox: SandboxPolicy = SandboxPolicy()
    secrets: Annotated[SecretsPolicy, REPOSITORY] = SecretsPolicy()
    planning: PlanningPolicy = PlanningPolicy()
    difficulty: ExecutionDifficultyPolicy = ExecutionDifficultyPolicy()
    rescue: RescuePolicy = RescuePolicy()
    workspace: WorkspacePolicy = WorkspacePolicy()
    undo: UndoPolicy = UndoPolicy()
    interaction: InteractionPolicy = InteractionPolicy()
    skills: SkillsPolicy = SkillsPolicy()
    agents: AgentsPolicy = AgentsPolicy()
    templates: TemplatesPolicy = TemplatesPolicy()
    work: WorkPolicy = WorkPolicy()
    web: WebPolicy = WebPolicy()
    commands: Annotated[tuple[CommandRowConfig, ...], REPOSITORY] = ()
    """`[[commands]]`: every layer's rows, each with its layer (`state/resolve`
    keeps them all, as it accumulates `secrets.deny_add`). Read through `command_table`."""
    hooks: Annotated[tuple[HookSpec, ...], REPOSITORY] = ()
    """`[[hooks]]`: merged across layers by `(event, name)`, each
    entry marked with the layer that set its `use` or `command` (`state/resolve`). A
    repository's script is inert until accepted; bound per episode and per session
    (`hooks/bind.py`)."""
    repo_trust: Literal["first_party", "third_party"] = "first_party"
    """`third_party` tightens injection policy. It is not the prototype's mode, and
    the alpha gate is what makes it usable, not this flag alone."""
    limits: Annotated[dict[str, Any], LIMIT_KEYS] = {}
    """Overrides of the register's defaults, as `[limits.worker]
    max_tool_calls = 50` or `"worker.max_tool_calls" = 50` — both spell the key
    `limits.worker.max_tool_calls`. Resolved once, here, at load."""

    _resolved_limits: tuple[ResolvedLimit, ...] = PrivateAttr(default=())
    _assignment: Mapping[str, str] = PrivateAttr(default_factory=dict[str, str])
    _notices: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="before")
    @classmethod
    def _required_tables_are_named(cls, data: Any) -> Any:
        """`models`, `rungs` and `spend` have no default, and pydantic reports their absence as
        *Field required* — a field, where a person editing TOML needs the table to write. A
        person who has not yet written an engine config has none of them, because `init`
        cannot know the endpoints, so this is the refusal most people meet first."""
        if not isinstance(data, Mapping):
            return data
        table = cast("Mapping[str, object]", data)
        _refuse_retired_names(table)
        _refuse_retired_keys(table)
        _refuse_misshapen_rungs(table.get("rungs"))
        missing: list[str] = []
        if "models" not in table:
            missing.append(
                "its model catalog — fill in [models.<name>], one table per model, each "
                "with its base_url, model_id, context_window and max_output_tokens"
            )
        if "rungs" not in table:
            missing.append(
                "its [rungs] table — name the catalog model each rung runs on; the cast runs "
                "on " + ", ".join(DEFAULT_RUNGS) + " unless [roles] moves it"
            )
        if "spend" not in table:
            missing.append("its [spend] table, with a finite hard_cost_usd")
        if missing:
            raise ConfigError(
                "the config is missing "
                + "; and ".join(missing)
                + ". Models and the spend cap belong in the engine config, "
                f"{ENGINE_CONFIG_PATH} — never in a repository's"
            )
        return table

    @model_validator(mode="after")
    def _model_names_are_usable(self) -> EngineConfig:
        """`MODEL_NAME`, at load: a name no layer above the file could address would make an
        override silently build a second model instead of changing this one."""
        bad = sorted(
            name
            for name in self.models
            if not MODEL_NAME.fullmatch(name) or name in ENV_READS_AS_VALUE
        )
        if bad:
            raise ConfigError(
                f"model name(s) {bad} cannot be used: a catalog name is lower-case letters, "
                "digits, '_' and '-', starting with a letter, and not one of "
                f"{sorted(ENV_READS_AS_VALUE)}. The environment layer lowercases variable "
                "names and reads those as a boolean or a number, so any other name could not "
                "be overridden from it"
            )
        return self

    @model_validator(mode="after")
    def _rungs_resolve(self) -> EngineConfig:
        """At load: the declared rungs, each role's rung (`assignment`, the one answer),
        and what a person must hear about them. Refused naming the rung, the role or the
        catalog — not by the first call to find a role with nothing behind it. The later
        validators read `assignment` and index `self.models` by what `rungs` names, and rely
        on this one having run."""
        bad = [why for name in self.rungs if (why := rung_name_error(name)) is not None]
        if bad:
            raise ConfigError(f"config's [rungs] table: {'; '.join(bad)}")
        # Grouped by model: with the catalog's tables commented out, the likeliest first
        # config, one model behind every rung would otherwise be named once per rung.
        lacking: dict[str, list[str]] = {}
        for rung, model in self.rungs.items():
            if model not in self.models:
                lacking.setdefault(model, []).append(rung)
        if lacking:
            named = "; ".join(f"{m!r} (named by {', '.join(rs)})" for m, rs in lacking.items())
            raise ConfigError(
                f"[rungs] names model(s) the catalog does not have: {named}. The catalog is "
                f"{sorted(self.models)} — add a [models.<name>] table for each, or name one of "
                "these"
            )
        assignment = assign(self.roles)
        declared = list(self.rungs)
        for name, rung in assignment.items():
            if rung not in self.rungs:
                raise ConfigError(
                    f"role {name!r} runs on rung {rung!r}, which [rungs] does not declare; the "
                    f'declared rungs are {declared}. Declare it ({rung} = "<model>"), or '
                    "point the role at a declared rung under [roles]"
                )
        for r in ROLES:
            for other in r.distinct_from:
                mine, theirs = assignment[r.name], assignment[other]
                if self.rungs[mine] == self.rungs[theirs]:
                    raise ConfigError(
                        f"roles {r.name!r} and {other!r} must run on different models; "
                        f"both resolve to {self.rungs[mine]!r} "
                        f"({r.name} on rung {mine!r}, {other} on {theirs!r})"
                    )
        undeclared = sorted(set(self.spend.rungs) - set(self.rungs))
        if undeclared:
            raise ConfigError(
                f"[spend.rungs] names {undeclared}, which [rungs] does not declare; the "
                f"declared rungs are {declared}. A cap on an undeclared rung caps nothing"
            )
        resolved = set(assignment.values())
        unpriced = sorted(
            {
                model
                for r in resolved
                if (model := self.rungs.get(r)) is not None
                and (row := self.prices.get(model)) is not None
                and row.price_in == 0
                and row.price_out == 0
            }
        )
        self._notices = (
            *(_unused(r, resolved) for r in declared if r not in resolved),
            *(_unpriced(m, self.spend.hard_cost_usd) for m in unpriced),
        )
        self._assignment = assignment
        return self

    @model_validator(mode="after")
    def _tool_roles_call_natively(self) -> EngineConfig:
        """At load: every role with tools calls them natively, so a model one runs on
        with `native_tools = false` could not call a tool at all. Refused naming the model and
        the roles, before any of them is asked something it could only guess at."""
        lacking: dict[str, list[str]] = {}
        for r in ROLES:
            if r.tool_defaults is None:
                continue
            model = self.rungs[self._assignment[r.name]]
            if not self.models[model].native_tools:
                lacking.setdefault(model, []).append(r.name)
        if lacking:
            named = "; ".join(f"{m!r} (for {', '.join(rs)})" for m, rs in lacking.items())
            raise ConfigError(
                f"model(s) {named} are configured native_tools = false, and every role with "
                "tools calls them natively since the rebuild: no prompt "
                "protocol reads a call from text any more. Serve the model with its tool-call "
                "parser and set native_tools = true, or run these roles on another model"
            )
        return self

    @model_validator(mode="after")
    def _tool_selections_resolve(self) -> EngineConfig:
        """At load: every `[tools.<role>]` selection,
        and each of its `models."<model>"` sections, resolves within the role's ceiling. A
        selection for a role whose loop does not read one yet is refused after that, naming
        the task that binds it (`_BINDS`)."""
        problems: list[str] = []
        cast_names = {r.name for r in ROLES}
        for name, chosen in self.tools.items():
            if name not in cast_names:
                continue  # refused as "not roles", with the cast, by `ROLE_KEYS`
            r = role(name)
            if r.tool_defaults is None:
                problems.append(f"tools.{name}: the role has no tools to choose")
                continue
            unknown = sorted(set(chosen.models) - set(self.models))
            if unknown:
                problems.append(
                    f"tools.{name}.models names {unknown}, which the catalog does not have"
                )
            for model in (None, *sorted(set(chosen.models) & set(self.models))):
                problems.extend(
                    _tool_refusals(name, r.tool_defaults, r.tool_ceiling, chosen, model)
                )
            problems.extend(self._edit_refusals(name, chosen))
        if problems:
            raise ConfigError("; ".join(problems))
        for name, chosen in self.tools.items():
            for key, why in _UNREAD_KEYS.get(name, {}).items():
                arms = (chosen, *chosen.models.values())
                if any(getattr(arm, key) is not None for arm in arms):
                    raise ConfigError(f"`tools.{name}.{key}` is not read: {why}")
            task = _BINDS.get(name)
            if chosen.chosen() and task is not None:
                raise ConfigError(
                    f"`[tools.{name}]` is not read yet: the {name} role's tools move onto the "
                    f"dispatcher in `{task}`, and a selection before then would change nothing "
                )
        return self

    def _edit_refusals(self, name: str, chosen: ToolPolicy) -> list[str]:
        """The load rule over a role's edit choices. Each `models."<name>"` section is
        judged merged over the role's table, against that model's capabilities; the role's
        table is judged against the model its rung runs now, unless that model has its own
        section (the view that then runs). A key in the role's table is read when any view
        reads it, so a default beside a per-model tool that reads it is not refused."""
        arms = {None: chosen, **chosen.models}
        if name not in _EDIT_ROLES:
            return [
                f"tools.{_where(name, model)}.{key} is not read: only the worker's edits and "
                "how it calls its tools are chosen on its tool table"
                for model, arm in arms.items()
                for key in EDIT_KEYS
                if getattr(arm, key) is not None
            ]
        r = role(name)
        assert r.tool_defaults is not None  # refused above for a role with no tools
        selection = _selection(chosen)
        sections = sorted(set(chosen.models) & set(self.models))
        current = self.rungs[self.rung_name_for(name)]
        problems: list[str] = []
        read_by_sections: set[str] = set()
        for model in sections:
            policy, _ = resolve_policy(
                r.tool_defaults, r.tool_ceiling, selection, model=model, known=BUILTINS
            )
            own = selection.models[model].chosen_edit_keys()
            # A key the section sets itself replaces the table's there: not a read of it.
            read_by_sections |= edit_keys_read(policy, BUILTINS) - own
            refused = choice_refusals(
                policy,
                own,
                BUILTINS,
                _CHOICES,
                _capabilities(self.models[model]),
            )
            problems.extend(_placed(name, model, refused))
        policy, _ = resolve_policy(
            r.tool_defaults, r.tool_ceiling, selection, model=None, known=BUILTINS
        )
        refused = choice_refusals(
            policy,
            selection.chosen_edit_keys(),
            BUILTINS,
            _CHOICES,
            None if current in sections else _capabilities(self.models[current]),
            read_elsewhere=frozenset(read_by_sections),
        )
        problems.extend(_placed(name, None, refused))
        return problems

    def command_table(self, *, accepted: bool) -> CommandTable:
        """The one `[[commands]]` table: the person's
        rows over the shipped table by prefix, then the repository's when `accepted`, which
        win a prefix both name. No row lowers a shipped row. `accepted` is the trust decision
        for this config, as `SecretsPolicy.denylist` takes it: unaccepted, the repository's
        rows are inert and the person's stand alone."""
        person = tuple(c.row() for c in self.commands if not c.from_repository)
        layers: list[tuple[str, tuple[CommandRow, ...]]] = [("user", person)]
        if accepted:
            repository = tuple(c.row() for c in self.commands if c.from_repository)
            layers.append((PROJECT_LAYER, repository))
        return shipped().layered(layers)

    @model_validator(mode="after")
    def _command_rows_resolve(self) -> EngineConfig:
        """At load: every row parses and none lowers a shipped row, accepted or not,
        refused naming the row's file, so a row that could never apply is found now."""
        for c in self.commands:
            try:
                c.row()
            except CommandTableError as exc:
                raise ConfigError(f"{_layer_file(c.layer)}: {exc}") from exc
        try:
            self.command_table(accepted=True)
        except CommandTableError as exc:
            raise ConfigError(str(exc)) from exc
        return self

    def tool_policy(self, name: str) -> RolePolicy:
        """The role's policy for the model its rung runs now: how a loop
        reads `[tools.<role>]`. A role with no tools is a harness bug to ask about."""
        r = role(name)
        if r.tool_defaults is None:
            raise ConfigError(f"role {name!r} has no tools")
        chosen = self.tools.get(name)
        model = self.rungs.get(self.rung_name_for(name))
        policy, _ = resolve_policy(
            r.tool_defaults,
            r.tool_ceiling,
            _selection(chosen) if chosen is not None else None,
            model=model,
            known=BUILTINS,
            own=_OWN_SPECS,
        )
        return policy

    @model_validator(mode="after")
    def _roles_resolve(self) -> EngineConfig:
        unknown = sorted(set(self.roles) - set(ROLE_NAMES))
        if unknown:
            raise ConfigError(
                f"roles table names {unknown}, which are not roles; the cast is "
                f"{list(ROLE_NAMES)} — a role is added by a decision, "
                "never by configuration"
            )
        unknown_tools = sorted(set(self.tools) - set(ROLE_NAMES))
        if unknown_tools:
            raise ConfigError(
                f"tools table names {unknown_tools}, which are not roles; the cast is "
                f"{list(ROLE_NAMES)}"
            )
        return self

    @model_validator(mode="after")
    def _limits_resolve_and_fit(self) -> EngineConfig:
        """At load: every row becomes a number on its role's rung —
        sized by the model that rung runs — and a role whose floors cannot fit that
        model's window is refused by name, not on the first request of an afternoon's session.
        A catalog model no rung names is nobody's window here; it is checked if it is ever
        switched to."""
        endpoints = self._rung_endpoints()
        assignment = self.assignment
        resolved = resolve(
            LIMITS,
            ROLES,
            endpoints,
            flatten_limits(self.limits),
            assignment=assignment,
        )
        for r in ROLES:
            template = _template_tokens(r.family, r.template, self.work.dir)
            if r.name == "worker":
                template += _worker_tools_tokens(self.tool_policy(r.name))
            check_fit(
                r,
                endpoints[assignment[r.name]],
                resolved,
                template,
                rung_name=assignment[r.name],
            )
        self._resolved_limits = resolved
        return self

    def harness_limit(self, name: str) -> int:
        """A harness-wide row of the limits register (`limits.HARNESS_OWNERS`),
        as this config sets it or its default."""
        return harness_limits(LIMITS, flatten_limits(self.limits))[name]

    @property
    def resolved_limits(self) -> tuple[ResolvedLimit, ...]:
        """The table the session log's `limits` line records: what a
        report quotes as "the worker was allowed 32,768 tokens of packed context".

        Resolving a row is not enforcing it. Each row is enforced by the component
        the register names, and each of those reads its row in its own task; until one does, an
        override of its row is resolved and checked here
        but changes no behaviour. `source` says where the number came from, not that
        anything applies it — whoever writes this table to the journal must say which
        rows are read (invariant 20)."""
        return self._resolved_limits

    @property
    def assignment(self) -> Mapping[str, str]:
        """Role → the rung it runs on, for every cast role: `[roles]` over
        each role's default, computed once at load. The only answer: the limits, the run's
        ledger views and the engine all read it, and none recomputes it."""
        return self._assignment

    @property
    def rung_names(self) -> tuple[str, ...]:
        """The declared rungs, `[rungs]`' keys in order: handed to the
        budget's readers once, at composition, never re-listed."""
        return tuple(self.rungs)

    @property
    def notices(self) -> tuple[str, ...]:
        """What the load found worth saying and not worth refusing: each
        declared rung no role runs on, and each model a role runs on priced 0 in and out.
        The engine logs each once."""
        return self._notices

    def rung_name_for(self, name: str) -> str:
        """The rung a role runs on: `assignment[name]`, an unknown role refused by name."""
        return self.assignment[role(name).name]

    def rung_for(self, name: str) -> RungEndpoint:
        """The endpoint a role runs on: the catalog model its rung names."""
        return self.models[self.rungs[self.rung_name_for(name)]]

    @property
    def endpoint_keys(self) -> frozenset[str]:
        """The variables that hold the harness's model endpoints' keys (`api_key_env`), every
        catalog model's, the web search's (`[web.search]`), and the SDKs' header variables
        (`AMBIENT_HEADERS_ENV`): withheld from a command the person approved to run outside the
        sandbox, which otherwise gets their whole environment."""
        keys = {m.api_key_env for m in self.models.values() if m.api_key_env}
        # The web's search keys too: the researcher's engine and GitHub token.
        keys |= self.web.search.key_variables()
        # The SDKs' header variables can carry a gateway's credential too.
        return frozenset(keys | set(AMBIENT_HEADERS_ENV))

    def vision_for(self, rung: str) -> Vision:
        """What a reader on `rung` can see: the rung's name, and whether the
        model it names can see images and read PDFs. The one way a reader's sight is
        built, so two rungs naming one model see alike and moving a rung to another model
        moves its sight."""
        model = self.models[self.rungs[rung]]
        return Vision(
            rung,
            supports_images=model.supports_images,
            supports_documents=model.supports_documents,
        )

    def with_rung(self, rung: str, model: str) -> EngineConfig:
        """This config with `rung` moved to catalog model `model`: validated as a
        load is, so a switch meets every load check (the catalog, each role's fit to the new
        window, the roles kept apart, the tool selections) and none is restated. `ConfigError`
        naming why, and `self` is unchanged."""
        if rung not in self.rungs:
            raise ConfigError(
                f"rung {rung!r} is not declared; the declared rungs are {list(self.rungs)}"
            )
        if model not in self.models:
            raise ConfigError(
                f"model {model!r} is not in the catalog; the catalog is {sorted(self.models)}. "
                "A model is added by editing the engine config"
            )
        return self.rewired({**self.rungs, rung: model}, source=f"rung {rung} on model {model}")

    def rewired(
        self, rungs: Mapping[str, str], *, source: str = "the session's wiring"
    ) -> EngineConfig:
        """This config with its `[rungs]` replaced by `rungs`, validated as a load is: the one
        way a session's switched wiring becomes a config."""
        # Its fields as they are, nested models included, with `rungs` replaced: validating
        # them runs every load check again over the switched wiring.
        return EngineConfig.with_defaults({**dict(self), "rungs": dict(rungs)}, source=source)

    def _rung_endpoints(self) -> dict[str, RungEndpoint]:
        """Each rung's endpoint, the catalog entry it names. The limits register thinks in
        rungs, and a rung's window is its model's."""
        return {rung: self.models[model] for rung, model in self.rungs.items()}

    @model_validator(mode="after")
    def _prices_name_catalog_models(self) -> EngineConfig:
        unknown = sorted(set(self.prices) - set(self.models))
        if unknown:
            raise ConfigError(
                f"price table names model(s) {unknown}, which the catalog does not have; the "
                f"catalog is {sorted(self.models)}. A price is keyed by model, not by rung "
            )
        return self

    @model_validator(mode="after")
    def _agent_profiles_narrow(self) -> EngineConfig:
        """A person's profile never widens its base: checked at load, after
        the limits it is compared with are resolved."""
        from chickadee.agents.profiles import config_profiles

        config_profiles(self)
        return self

    # -- loading ------------------------------------------------------------

    @staticmethod
    def from_toml(path: Path | str) -> EngineConfig:
        """Load one file as the layered loader loads any: `state.resolve` over one layer,
        so the per-layer key check and the validation are the ones every config meets, not
        a second path that could drift from them.

        Every rejection surfaces as `ConfigError`: a file that cannot be read or parsed,
        pydantic's `ValidationError`, and the state layer's refusal (`UnknownKey`) alike, so
        an operator gets one exception type and does not have to know which part of the
        code refused their file.
        """
        # Here, not at module level: `state.resolve` validates into this module's
        # `EngineConfig`, and the two stay free of an import cycle only while one side
        # defers. Imported under its own name, because this module's `resolve` is the
        # limits register's.
        from chickadee.state.layers import Layer
        from chickadee.state.merge import MergeError
        from chickadee.state.resolve import resolve as resolve_layers

        source = Path(path)
        try:
            data = tomllib.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"{source}: {exc}") from exc
        try:
            return resolve_layers([Layer("file", source, data)]).engine
        except MergeError as exc:
            raise ConfigError(str(exc)) from exc

    @staticmethod
    def with_defaults(raw: Mapping[str, Any], *, source: str | None = None) -> EngineConfig:
        """Validate: the last step of every load (`state.resolve`, and through it
        `from_toml`), and the entry point for a config built in code. `source` says what was
        validated, for the message. The name is its callers'; every default now lives on a
        field or in the limits register, so nothing is filled here."""
        try:
            return EngineConfig.model_validate(raw)
        except ValidationError as exc:
            reason = _reason(exc)
            raise ConfigError(f"{source}: {reason}" if source else reason) from exc


def _refuse_retired(names: Iterable[str], retired: Mapping[str, str], where: str) -> None:
    """A retired name gets its own refusal: the decision that retired it and what replaced
    it, where an unknown name would only list the valid ones."""
    gone = [name for name in names if name in retired]
    if gone:
        raise ConfigError(
            "; ".join(f"{where} names {name!r}, {retired[name]}" for name in gone)
            + " — remove the entry"
        )


def _refuse_retired_names(table: Mapping[str, object]) -> None:
    """On the raw mapping, before any field is validated: a retired entry
    whose value is also wrong (`[tools.elaborator] max_tool_calls = 0`) hears why the entry is
    refused, not that its value is out of range."""
    for section, retired, where in (
        ("roles", RETIRED_ROLES, "roles table"),
        ("tools", RETIRED_ROLES, "tools table"),
    ):
        entries = table.get(section)
        if isinstance(entries, Mapping):
            _refuse_retired(cast("Mapping[str, object]", entries), retired, where)
    rungs = table.get("rungs")
    if isinstance(rungs, Mapping):
        _refuse_former(cast("Mapping[str, object]", rungs), "[rungs]")
    roles = table.get("roles")
    if isinstance(roles, Mapping):
        named = cast("Mapping[str, object]", roles)
        _refuse_former((rung for rung in named.values() if isinstance(rung, str)), "[roles]")


def _refuse_former(names: Iterable[str], where: str) -> None:
    """A former rung name is never declared, pointed at or capped again, so
    an old line naming it means one thing."""
    for name in names:
        why = former_rung_error(name)
        if why is not None:
            raise ConfigError(f"{where}: {why} — rename or remove the entry")


def _unpriced(model: str, cap: float) -> str:
    """A notice for a model a role runs on, priced 0 in and out: every call costs
    nothing, so the run's cap never trips and the cost figures read $0. Not a refusal: a free
    model is a real case; the person is told, once."""
    return (
        f"[prices.{model!r}] is 0 in and out: spend reads $0 and the ${cap:g} cap can never "
        "stop a run. Price a local model at a hosted-market equivalent to make the cap "
        "and the cost figures mean something"
    )


def _unused(rung: str, resolved: Iterable[str]) -> str:
    """The notice for a declared rung nothing runs on, with the nearest resolved
    rung when one is close: a misspelled default is refused anyway as well, and this
    says which line was meant."""
    text = f"rung {rung!r} is declared and no role runs on it"
    near = nearest(rung, resolved)
    return f"{text}; did you mean {near!r}?" if near is not None else text


def _refuse_misshapen_rungs(rungs: object) -> None:
    """A `[rungs]` entry is a catalog model's name. Two other shapes reach here, and
    pydantic would answer each with *Input should be a valid string* and a documentation URL:
    an endpoint table under `[rungs.<name>]`, the layout in older versions, and a value the
    environment layer read as a number or a boolean (`CHICKADEE_RUNGS__CODER=5`). Each is
    refused in words that say what to write instead."""
    if not isinstance(rungs, Mapping):
        return
    entries = cast("Mapping[str, object]", rungs)
    tables = [name for name, value in entries.items() if isinstance(value, Mapping)]
    if tables:
        raise ConfigError(
            ", ".join(f"[rungs.{name}]" for name in tables)
            + (" holds" if len(tables) == 1 else " hold")
            + " an endpoint, the older layout: a rung now names a model. Move "
            "each table's fields into a [models.<name>] table, and name that model under "
            f'[rungs] — {tables[0]} = "<name>". One model may serve several rungs'
        )
    for name, value in entries.items():
        if not isinstance(value, str):
            raise ConfigError(
                f"rungs.{name} is {value!r} ({type(value).__name__}); a rung names a catalog "
                "model by its name, a string — a model name is never a number or a boolean"
            )


def _reason(exc: ValidationError) -> str:
    """A refusal in words a person can act on. When every error is a `ConfigError` one of
    this module's validators raised, those sentences are the whole reason; pydantic's
    wrapper around them — a count, the input echoed back, a documentation URL — is noise.
    Anything else keeps pydantic's text, which names the field."""
    reasons: list[str] = []
    for error in exc.errors():
        cause = error.get("ctx", {}).get("error")
        if not isinstance(cause, ConfigError):
            return str(exc)
        reasons.append(str(cause))
    return "; ".join(reasons) or str(exc)
