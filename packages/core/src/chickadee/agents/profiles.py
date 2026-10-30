"""Profiles: what an agent may do (T-3.188 Design §8).

A built-in profile is a role in `ROLES`, `reader`, `reviewer` or `researcher`: its
ceiling, default tools, rung, template and limits rows are the role's, fixed in code. A
person's own profile, `[agents.profiles.<name>]` in their user config, is a narrowing of one: it
names a `base`, and may pick another `rung`, fewer `tools`, and lower limits. It never widens,
and a repository's config cannot declare one (`config.py`), because a profile decides what an
agent may touch.

`Profile` is the one value the spawn reads: the role it runs as, the tools it is offered (the
config's `[tools.<role>]` policy, narrowed), the rung it runs on, and its five limits resolved
to numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from chickadee.errors import ConfigError
from chickadee.limits import ResolvedLimit, limit, resolved_value, window

if TYPE_CHECKING:
    from chickadee.config import AgentProfileConfig, EngineConfig

__all__ = [
    "BUILTIN_PROFILES",
    "LIMIT_SUFFIXES",
    "Profile",
    "ProfileLimits",
    "agent_costs",
    "builtin",
    "config_profiles",
]

BUILTIN_PROFILES: Final = ("reader", "reviewer", "researcher")
"""The roles a profile may be built on (`roles.py`)."""

LIMIT_SUFFIXES: Final = (
    "max_tool_calls",
    "max_cumulative_result_tokens",
    "seconds",
    "max_cost_cents",
    "report_reasks",
)
"""Each profile's limits rows, `<role>.<suffix>` in the register (Design §7)."""


@dataclass(frozen=True)
class ProfileLimits:
    """One round's limits, as numbers: the register's rows resolved on the role's rung, or a
    person's profile's lower values."""

    max_tool_calls: int
    max_result_tokens: int
    seconds: int
    max_cost_cents: int
    report_reasks: int

    @staticmethod
    def of_role(resolved: tuple[ResolvedLimit, ...], role: str) -> ProfileLimits:
        def row(suffix: str) -> int:
            return resolved_value(resolved, role, f"{role}.{suffix}")

        return ProfileLimits(
            row("max_tool_calls"),
            row("max_cumulative_result_tokens"),
            row("seconds"),
            row("max_cost_cents"),
            row("report_reasks"),
        )

    def lowered(self, values: Mapping[str, int]) -> ProfileLimits:
        """These limits with `values` (by suffix) in their place. The config checked each is
        no higher than this; a key it does not name keeps this one's."""
        tokens = values.get("max_cumulative_result_tokens", self.max_result_tokens)
        return ProfileLimits(
            values.get("max_tool_calls", self.max_tool_calls),
            tokens,
            values.get("seconds", self.seconds),
            values.get("max_cost_cents", self.max_cost_cents),
            values.get("report_reasks", self.report_reasks),
        )


@dataclass(frozen=True)
class Profile:
    name: str
    role: str
    """The built-in role it runs as: its ceiling, template and rows."""
    tools: tuple[str, ...]
    """Registry names, in the order the model is shown them: the role's policy, narrowed."""
    rung: str
    """The rung its model is on."""
    limits: ProfileLimits

    def narrowed(
        self, names: tuple[str, ...] | None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The tools left after `names` narrows them, and the names that are not this
        profile's (which narrow nothing, and are said so). None narrows nothing."""
        if names is None:
            return self.tools, ()
        kept = tuple(t for t in self.tools if t in names)
        foreign = tuple(n for n in names if n not in self.tools)
        return kept, foreign


def builtin(
    role: str, tools: tuple[str, ...], rung: str, resolved: tuple[ResolvedLimit, ...]
) -> Profile:
    """A built-in profile: the role as the config resolves it."""
    return Profile(role, role, tools, rung, ProfileLimits.of_role(resolved, role))


def config_profiles(config: EngineConfig) -> dict[str, Profile]:
    """Every profile `config` gives: each built-in as it resolves its role (`[tools.<role>]`'s
    tools, its rung under `[roles]`, its rows under `[limits]`), then the person's own
    (`[agents.profiles]`), each a narrowing of one. `ConfigError` for one that would widen its
    base, or names a rung or tool it cannot have."""
    out = {
        name: builtin(
            name,
            config.tool_policy(name).enabled,
            config.rung_name_for(name),
            config.resolved_limits,
        )
        for name in BUILTIN_PROFILES
    }
    for name, spec in config.agents.profiles.items():
        out[name] = _person(config, name, spec, out[spec.base])
    return out


def _person(
    config: EngineConfig, name: str, spec: AgentProfileConfig, base: Profile
) -> Profile:
    where = f"[agents.profiles.{name}]"
    if name in BUILTIN_PROFILES:
        raise ConfigError(f"{where} names a built-in profile; give yours a name of its own")
    rung = spec.rung if spec.rung is not None else base.rung
    if rung not in config.rung_names:
        raise ConfigError(
            f"{where}: rung {rung!r} is not declared; the rungs are {sorted(config.rung_names)}"
        )
    tools = base.tools
    if spec.tools is not None:
        foreign = sorted(set(spec.tools) - set(base.tools))
        if foreign:
            raise ConfigError(
                f"{where}: {foreign} are not {spec.base}'s tools, and a profile only narrows "
                f"its base (D213); {spec.base} has {list(base.tools)}"
            )
        tools = tuple(t for t in base.tools if t in spec.tools)
    lowered = _lowered(config, where, spec, base, rung)
    return Profile(name, base.role, tools, rung, base.limits.lowered(lowered))


def _lowered(
    config: EngineConfig, where: str, spec: AgentProfileConfig, base: Profile, rung: str
) -> dict[str, int]:
    """The limits `spec` sets, each checked no higher than its base's. Result tokens are a
    window fraction, compared as one and sized on the profile's own rung."""
    from chickadee.config import flatten_limits

    values: dict[str, int] = {}
    own = base.limits
    for key, mine, theirs in (
        ("max_tool_calls", spec.max_tool_calls, own.max_tool_calls),
        ("seconds", spec.seconds, own.seconds),
        ("max_cost_cents", spec.max_cost_cents, own.max_cost_cents),
        ("report_reasks", spec.report_reasks, own.report_reasks),
    ):
        if mine is None:
            continue
        if mine > theirs:
            raise ConfigError(
                f"{where}: {key} = {mine} is above {spec.base}'s {theirs}; a profile only "
                "narrows its base (D213)"
            )
        values[key] = mine
    row = f"{base.role}.max_cumulative_result_tokens"
    configured = flatten_limits(config.limits).get(f"limits.{row}", limit(row).value)
    fraction = float(configured)
    if spec.max_cumulative_result_tokens is not None:
        if spec.max_cumulative_result_tokens > fraction:
            raise ConfigError(
                f"{where}: max_cumulative_result_tokens = "
                f"{spec.max_cumulative_result_tokens} is above {spec.base}'s {fraction}"
            )
        fraction = spec.max_cumulative_result_tokens
    if spec.rung is not None or spec.max_cumulative_result_tokens is not None:
        endpoint = config.models[config.rungs[rung]]
        values["max_cumulative_result_tokens"] = max(1, int(fraction * window(endpoint)))
    return values


def agent_costs(profiles: Mapping[str, Profile]) -> dict[str, str]:
    """What one round of each profile may spend, as `find_skill`'s cost hint says it."""
    return {
        name: (
            f"up to {p.limits.max_tool_calls} tool calls and {p.limits.max_cost_cents} cents "
            f"a round on {p.rung}"
        )
        for name, p in profiles.items()
    }
