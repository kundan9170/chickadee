"""What `chickadee hooks` shows: the points, and each configured entry over the
defaults with what binding it would do now, from the config alone, nothing run.

Each entry's standing is `bind.standing`, the one reading `bind` acts on: `bound`, `off`,
`inert` (a repository's script not accepted: `chickadee trust`) or `refused`, each with
`bind`'s own words, so the listing and a run cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from chickadee.errors import ConfigError
from chickadee.hooks.bind import HookConfig, Standing, entries, standing
from chickadee.hooks.points import POINTS
from chickadee.hooks.script import seals
from chickadee.hooks.spec import HookSpec
from chickadee.isolation.base import configured
from chickadee.state.layers import LayerReadError
from chickadee.state.load import load_host_config
from chickadee.state.merge import MergeError

__all__ = ["HookRow", "render", "rows", "shown"]


@dataclass(frozen=True)
class HookRow:
    spec: HookSpec
    standing: Standing


def rows(configured: HookConfig, *, sealed: bool) -> tuple[HookRow, ...]:
    """Each entry `bind` would see, in its order, and its standing (`bind.standing`).
    `sealed`: whether the configured backend seals a process (`script.seals`)."""
    return tuple(
        HookRow(spec, standing(spec, accepted=configured.accepted, sealed=sealed))
        for spec in entries(configured.specs)
    )


def render(shown: tuple[HookRow, ...], configured: HookConfig) -> str:
    lines = ["points:"]
    for point, spec in POINTS.items():
        takes = ", ".join(sorted(k.__name__ for k in spec.accepts))
        claude = f" (Claude Code: {spec.claude})" if spec.claude else ""
        lines.append(f"  {point.value}  level {spec.level}  takes {takes}{claude}")
    limits = configured.limits
    lines.append(
        f"bounds: {limits.max_continues} continue(s) a stop, {limits.max_retries} retry(ies) a "
        f"task, {limits.max_steering} steering a planning"
    )
    lines.append("hooks:" if shown else "hooks: none")
    for row in shown:
        spec = row.spec
        what = (
            f"use {spec.use}" if spec.use is not None else f"command {list(spec.command or ())}"
        )
        role = f" role {spec.role}" if spec.role else ""
        found = row.standing
        state = found.state if found.why is None else f"{found.state}: {found.why}"
        lines.append(f"  {spec.event.value} {spec.name}: {what}{role} [{spec.layer}] {state}")
    return "\n".join(lines)


def shown(cwd: Path, *, environ: Mapping[str, str] | None = None) -> str:
    """`chickadee hooks` for the repository containing `cwd`, read as `load_host_config` reads
    it, so the listing is the config a run binds. `NotARepository` outside one; anything else
    that stops the read is a `ConfigError` naming it, as `repository_trust` raises."""
    try:
        host = load_host_config(cwd, environ=environ)
    except (LayerReadError, MergeError) as exc:
        raise ConfigError(str(exc)) from exc
    config = host.engine
    hooks = HookConfig.of(config, accepted=host.trust.accepted)
    sealed = seals(configured(config.sandbox.backend))
    return render(rows(hooks, sealed=sealed), hooks)
