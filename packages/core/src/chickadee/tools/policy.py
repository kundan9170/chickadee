"""A role is a ceiling and a policy.

The **ceiling** is fixed in `ROLES` and is a trust boundary: what a role may ever mutate, the
highest tier it may reach, the ports it may be given, the widest files it may see. A config
chooses **within** it, never the ceiling itself.

The **policy** is which tools a role is offered, under which names, over which files, and the
pipeline its calls pass. The default is `ROLES`'; a config's `[tools.<role>]` selection, and
a `models."<model>"` section of it, replace keys of it. `resolve` is the one place a policy is
built, and it returns every refusal D164 decision 8 names rather than the first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from chickadee.edits.lines import ANCHOR_STYLES, AnchorStyle
from chickadee.errors import HarnessBug
from chickadee.schema.authorization import TIER_ORDER, Tier
from chickadee.tools.registry import Registry
from chickadee.tools.rules import Pipeline
from chickadee.tools.spec import ANCHOR_ARG, ToolSpec
from chickadee.tools.tool import Port, Tool

__all__ = [
    "EDIT_KEYS",
    "Capabilities",
    "Ceiling",
    "Choices",
    "EditCheck",
    "FileRule",
    "RolePolicy",
    "Selection",
    "choice_refusals",
    "edit_keys_read",
    "resolve",
    "within",
]

Mutates = Literal["nothing", "environment", "source"]
_MUTATES: tuple[Mutates, ...] = ("nothing", "environment", "source")

FileScope = Literal["packet", "graph_visible", "repository"]
_SCOPES: tuple[FileScope, ...] = ("packet", "graph_visible", "repository")

TestView = Literal["hidden", "names_only", "visible"]
_TESTS: tuple[TestView, ...] = ("hidden", "names_only", "visible")


@dataclass(frozen=True)
class FileRule:
    """What a role's file reads may reach. `scope` widens from a packet's own paths to the
    whole repository; `tests` from hidden to visible."""

    scope: FileScope
    tests: TestView

    def wider_than(self, other: FileRule) -> bool:
        return _SCOPES.index(self.scope) > _SCOPES.index(other.scope) or _TESTS.index(
            self.tests
        ) > _TESTS.index(other.tests)


@dataclass(frozen=True)
class Ceiling:
    """Fixed in `ROLES`: a trust boundary, never configuration."""

    mutates: Mutates
    tier: Tier
    ports: frozenset[Port]
    widest_files: FileRule


@dataclass(frozen=True)
class RolePolicy:
    """Which tools a role gets, and how its calls pass."""

    role: str
    enabled: tuple[str, ...]
    """Registry names, in the order the model is shown them."""
    shown_as: Mapping[str, str]
    """Registry name -> the name the model sees. Dispatch, records and the manifest use the
    registry name; only the rendered schema and the parsed call use the shown one."""
    files: FileRule
    graph_depth: int | None
    pipeline: Pipeline
    line_anchors: str = "number"
    """How a line tool's anchor is written: an `ANCHOR_STYLES` name. Read only when a
    tool with `anchor = "line"` is offered (`worker_tools.line_style`)."""
    constrain: tuple[str, ...] = ()
    """Registry names of offered tools decoded under their spec's grammar, on a
    model with `grammar_tools`. Empty by default, so nothing is
    constrained unless a config names it."""
    edit_check: EditCheck = "reject"
    """What a syntax error an edit call made does: `reject` undoes the call, `report`
    keeps it and says so, `off` judges nothing."""
    force_tool_call: bool = False
    """Each of the worker's turns is asked to be a tool call (`ChatRequest.force_tool`),
    but a wrap-up turn. The client forces it only where its provider enforces it."""
    calling: Calling = "native"
    """How the worker calls its tools. `python` is D225's arm (`pycall.py`):
    no native tool is sent, and each reply's Python blocks are its calls."""
    withheld: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    """Registry name -> arguments this role is never shown, because its tool refuses them
    here: the box shell's `effects`, which its profile refuses. Withheld from the
    schema the model sees and the one its call is validated against, so passing one is
    `INVALID_ARGUMENTS` naming the arguments it has."""

    @property
    def anchor_style(self) -> AnchorStyle:
        return ANCHOR_STYLES[self.line_anchors]

    def resolved(self, spec: ToolSpec) -> ToolSpec:
        """`spec` as this policy offers it: each anchor argument (`ANCHOR_ARG`) given the
        style's schema, its description kept, and its `grammar` kept only when the policy
        constrains it (`constrain`), so a resolved spec's grammar means "decoded under this".
        What the model is shown and what its call is validated against; a spec neither touches
        is returned as it is."""
        anchored = self._without(self._anchored(spec))
        if anchored.grammar is not None and spec.name not in self.constrain:
            return replace(anchored, grammar=None)
        return anchored

    def _without(self, spec: ToolSpec) -> ToolSpec:
        names = self.withheld.get(spec.name, ())
        props = spec.schema.get("properties")
        if not names or not isinstance(props, Mapping):
            return spec
        typed = cast("Mapping[str, Mapping[str, object]]", props)
        schema = dict(spec.schema)
        schema["properties"] = {k: v for k, v in typed.items() if k not in names}
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [r for r in cast("list[str]", required) if r not in names]
        return replace(spec, schema=schema)

    def _anchored(self, spec: ToolSpec) -> ToolSpec:
        props = spec.schema.get("properties")
        if not isinstance(props, Mapping):
            return spec
        typed = cast("Mapping[str, Mapping[str, object]]", props)
        kinds = {k: v[ANCHOR_ARG] for k, v in typed.items() if ANCHOR_ARG in v}
        if not kinds:
            return spec
        out: dict[str, object] = {}
        for key, prop in typed.items():
            kind = kinds.get(key)
            if kind is None:
                out[key] = prop
                continue
            if kind not in ("line", "after"):
                raise HarnessBug(f"{spec.name}.{key}: {ANCHOR_ARG} is {kind!r}")
            schema = self.anchor_style.argument_schema(kind)
            description = prop.get("description")
            out[key] = schema if description is None else {**schema, "description": description}
        return replace(spec, schema={**spec.schema, "properties": out})

    def edits_by_tool(self, registry: Registry) -> bool:
        """Whether the role edits through tools: it offers a tool of family `edit`.
        Derived, never configured, so nothing can disagree with it."""
        return any(s.family == "edit" for s in self._offered(registry))

    def anchors_lines(self, registry: Registry) -> bool:
        """Whether it offers a tool whose anchor is a line, so its views are written
        in `line_anchors`."""
        return any(s.anchor == "line" for s in self._offered(registry))

    def _offered(self, registry: Registry) -> list[ToolSpec]:
        # Unresolved specs, a name the registry lacks skipped: `resolve` refuses one at load,
        # and the load rule asks before that refusal is raised.
        return [t.spec for name in self.enabled if (t := registry.get(name)) is not None]

    def shown(self, name: str) -> str:
        return self.shown_as.get(name, name)

    @property
    def shown_names(self) -> tuple[str, ...]:
        """Every enabled tool under the name the model sees, in the order it is shown: what
        a parse admits and `## TOOLS` lists."""
        return tuple(self.shown(n) for n in self.enabled)

    def surface(self, registry: Registry) -> tuple[ToolSpec, ...]:
        """The enabled tools' specs, in order, under the names the model sees and with their
        anchors in the policy's style (`resolved`): what a role's prefix lists. A name the
        registry lacks is a harness bug (`resolve` refuses one at load)."""
        out: list[ToolSpec] = []
        for name in self.enabled:
            tool = registry.get(name)
            if tool is None:
                raise HarnessBug(
                    f"{self.role}'s policy enables {name!r}, which no module registers"
                )
            out.append(replace(self.resolved(tool.spec), name=self.shown(name)))
        return tuple(out)

    def without(self, *names: str) -> RolePolicy:
        """This policy with `names` no longer offered: a tool whose port this composition
        cannot give (the worker's `shell` where no backend enforces its seal)."""
        return replace(self, enabled=tuple(n for n in self.enabled if n not in names))

    def registry_name(self, shown: str) -> str | None:
        """The registry name a model's call names, or None when no enabled tool is shown so."""
        for name in self.enabled:
            if self.shown(name) == shown:
                return name
        return None


@dataclass(frozen=True)
class Selection:
    """A config's choice for one role, every key optional: what it names replaces the
    default's, what it omits is the default's. The last three are the
    worker's edit choices, checked by `choice_refusals`. Every role calls its tools
    natively and every edit is a tool call, so the protocol and the reply format
    are no longer choices."""

    enabled: tuple[str, ...] | None = None
    shown_as: Mapping[str, str] | None = None
    files: FileRule | None = None
    models: Mapping[str, Selection] = field(default_factory=dict[str, "Selection"])
    line_anchors: str | None = None
    edit_check: EditCheck | None = None
    constrain: tuple[str, ...] | None = None
    force_tool_call: bool | None = None
    calling: Calling | None = None

    def chosen_edit_keys(self) -> frozenset[str]:
        """The edit choices this selection sets itself, by key."""
        return frozenset(k for k in EDIT_KEYS if getattr(self, k) is not None)


EditCheck = Literal["reject", "report", "off"]
"""D172 the three answers to a syntax error an edit call made: a closed set, not a registry."""

Calling = Literal["native", "python"]
"""Native tool calls, or D225's Python blocks."""

EDIT_KEYS: tuple[str, ...] = (
    "line_anchors",
    "edit_check",
    "constrain",
    "force_tool_call",
    "calling",
)
"""The worker's choices on `[tools.worker]`, as the config spells
them."""


@dataclass(frozen=True)
class Choices:
    """What the edit choices may name, passed in so the registry is the config's: the anchor
    styles."""

    styles: frozenset[str]
    refused_styles: Mapping[str, str] = field(default_factory=dict[str, str])
    """A known style this role may not choose, and why: its reads do not show it, so
    an anchor in it could never be copied from a view."""


@dataclass(frozen=True)
class Capabilities:
    """A model's facts the choices are checked against: never refused for being
    unused, only a choice that needs one it lacks. `native_tools` is no choice's: an endpoint
    without it is refused by the config whatever a role chooses."""

    grammar_tools: bool


def within(tier: Tier | Literal["computed"], ceiling: Tier) -> bool:
    """Whether a spec's tier fits under a ceiling's. A fixed tier must not exceed it. A
    `computed` one (the shell, the coder) is decided per call by the router, and a call
    computed above the role's ceiling is refused then, never asked of a role with no person
    to ask (the worker's shell is T0, above it refused), so it fits any ceiling."""
    if tier == "computed":
        return True
    return TIER_ORDER.index(tier) <= TIER_ORDER.index(ceiling)


def resolve(
    default: RolePolicy,
    ceiling: Ceiling,
    selection: Selection | None,
    *,
    model: str | None,
    known: Registry,
    own: Mapping[str, ToolSpec] | None = None,
) -> tuple[RolePolicy, tuple[str, ...]]:
    """The role's policy under `selection`, for the model its rung runs (`model`), and every
    reason it is refused, each naming the key and the reason.

    A name is known when `known` registers it or the role's default lists it: Zeus's
    surface, the scout's own two and the worker's four register only as their loops move.
    `own` holds the specs of those it can check against
    the ceiling already (Zeus's `SURFACE`)."""
    chosen = _merged(selection, model)
    enabled = chosen.enabled if chosen.enabled is not None else default.enabled
    # A name the default shows a tool under applies only while that tool is enabled; a name
    # the selection gives is checked as given.
    shown_as = (
        dict(chosen.shown_as)
        if chosen.shown_as is not None
        else {k: v for k, v in default.shown_as.items() if k in enabled}
    )
    policy = replace(
        default,
        enabled=enabled,
        shown_as=shown_as,
        files=chosen.files if chosen.files is not None else default.files,
        line_anchors=_or(chosen.line_anchors, default.line_anchors),
        edit_check=_or(chosen.edit_check, default.edit_check),
        constrain=_or(chosen.constrain, default.constrain),
        force_tool_call=_or(chosen.force_tool_call, default.force_tool_call),
        calling=_or(chosen.calling, default.calling),
    )
    return policy, _refusals(policy, ceiling, known, own or {}, frozenset(default.enabled))


def _merged(selection: Selection | None, model: str | None) -> Selection:
    if selection is None:
        return Selection()
    per_model = selection.models.get(model) if model is not None else None
    if per_model is None:
        return selection
    return Selection(
        enabled=_or(per_model.enabled, selection.enabled),
        shown_as=_or(per_model.shown_as, selection.shown_as),
        files=_or(per_model.files, selection.files),
        line_anchors=_or(per_model.line_anchors, selection.line_anchors),
        edit_check=_or(per_model.edit_check, selection.edit_check),
        constrain=_or(per_model.constrain, selection.constrain),
        force_tool_call=_or(per_model.force_tool_call, selection.force_tool_call),
        calling=_or(per_model.calling, selection.calling),
    )


def _or[T](chosen: T | None, default: T) -> T:
    """A key the section sets, over the one it would otherwise have."""
    return chosen if chosen is not None else default


def choice_refusals(
    policy: RolePolicy,
    chosen: frozenset[str],
    known: Registry,
    choices: Choices,
    model: Capabilities | None,
    *,
    read_elsewhere: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """D172 the one load rule, over a resolved policy: every reason its edit choices are
    refused, each naming the key. Derived from what the plug-ins declare (the offered specs'
    `family`, `anchor` and `grammar`), so a new tool or style is checked with no change here.

    `chosen` is the keys set explicitly in this view (a default is never refused for being
    unread); `model` is the capabilities to check against, or None when this view's model is
    another view's to check; `read_elsewhere`, the keys another view of the same table reads
    (a role's own table holding a default for a per-model section)."""
    out: list[str] = []
    reads = edit_keys_read(policy, known)
    if policy.line_anchors not in choices.styles:
        out.append(
            f"line_anchors {policy.line_anchors!r} is not a known name; known: "
            f"{sorted(choices.styles)}"
        )
    elif policy.line_anchors in choices.refused_styles and policy.anchors_lines(known):
        why = choices.refused_styles[policy.line_anchors]
        out.append(f"line_anchors {policy.line_anchors!r} is refused: {why}")
    for key in sorted(chosen):
        if key not in reads and key not in read_elsewhere:
            out.append(f"{key} is set and nothing offered reads it ({_unread(key)})")
    for name in policy.constrain:
        tool = known.get(name)
        if name not in policy.enabled:
            out.append(f"constrain names {name!r}, which is not offered")
        elif tool is not None and tool.spec.grammar is None:
            out.append(f"constrain names {name!r}, whose spec carries no grammar")
    if model is not None and policy.constrain and not model.grammar_tools:
        out.append("constrain needs grammar_tools; the model lacks it")
    if policy.calling == "python" and policy.force_tool_call:
        # The Python arm sends no native tool, so there is none to force
        # a config asking for both says two things.
        out.append(
            "calling 'python' sends no native tool, so force_tool_call has nothing to force; "
            "set one of them"
        )
    return tuple(f"{policy.role}: {r}" for r in out)


def edit_keys_read(policy: RolePolicy, known: Registry) -> frozenset[str]:
    """The edit choices a policy's offered tools read, derived from their specs:
    `edit_check` with an edit tool, `line_anchors` with a line tool. `constrain` is always
    read, and has its own refusals; `force_tool_call` is always read, and never refused for the
    model's capability: a model without `forced_tools` runs unforced."""
    reads = {
        "line_anchors": policy.anchors_lines(known),
        "edit_check": policy.edits_by_tool(known),
        "constrain": True,
        "force_tool_call": True,
        "calling": True,
    }
    return frozenset(k for k, read in reads.items() if read)


def _unread(key: str) -> str:
    """Why `key` is read only sometimes: what must be offered for it to change anything."""
    return {
        "line_anchors": "it is read only when an offered tool has anchor 'line'",
        "edit_check": "it is read only when an offered tool has family 'edit'",
    }.get(key, "")


def _refusals(
    policy: RolePolicy,
    ceiling: Ceiling,
    known: Registry,
    own: Mapping[str, ToolSpec],
    defaults: frozenset[str],
) -> tuple[str, ...]:
    out: list[str] = []
    role = policy.role
    for name in policy.enabled:
        tool = known.get(name)
        spec = tool.spec if tool is not None else own.get(name)
        if spec is None:
            if name not in defaults:
                names = sorted({*known.names(), *defaults})
                out.append(f"{role}: {name!r} is not a tool; known: {names}")
            continue
        if _MUTATES.index(spec.mutates) > _MUTATES.index(ceiling.mutates):
            out.append(
                f"{role}: {name!r} mutates {spec.mutates}; the role may mutate at most "
                f"{ceiling.mutates}"
            )
        if not within(spec.tier, ceiling.tier):
            tier = spec.tier.value.upper() if isinstance(spec.tier, Tier) else spec.tier
            out.append(f"{role}: {name!r} is tier {tier}; the ceiling is {ceiling.tier.name}")
        if tool is not None:
            out.extend(_ports(role, tool, ceiling))
    for name in policy.shown_as:
        if name not in policy.enabled:
            out.append(f"{role}: shown_as names {name!r}, which is not enabled")
    shown = list(policy.shown_names)
    clashes = sorted({s for s in shown if shown.count(s) > 1})
    if clashes:
        out.append(f"{role}: two enabled tools are shown as {clashes}")
    if policy.files.wider_than(ceiling.widest_files):
        out.append(
            f"{role}: files {policy.files} is wider than the role may see "
            f"({ceiling.widest_files})"
        )
    if policy.pipeline.confirms and "user" not in ceiling.ports:
        out.append(f"{role}: the pipeline asks the person, and the role has no user port")
    return tuple(out)


def _ports(role: str, tool: Tool, ceiling: Ceiling) -> list[str]:
    outside = sorted(tool.needs - ceiling.ports)
    if not outside:
        return []
    return [f"{role}: {tool.spec.name!r} needs ports {outside} the role is never given"]
