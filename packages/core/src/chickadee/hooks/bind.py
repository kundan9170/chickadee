"""A configured hook set, checked and fixed for an episode or a session.

`bind` refuses, with the reason, what can never run (an unknown built-in, a built-in at a point
it does not fire at or answering what that point does not take, a `with` it rejects, a script
with no sandbox backend), and skips, naming it, what may not run now: a repository's script
whose digest the person has not accepted. The built-in `DEFAULTS` are bound unless an entry
under the same `(event, name)` replaces or switches one off.

The bound set is written once, as a `hook` line of kind `bound` holding the entries as a blob
(`write_bound`), and a resumed episode binds from that line (`specs_of`), not from today's
config: it runs the hooks it started with, and says once if the configured set has changed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter

from chickadee.errors import ConfigError
from chickadee.hooks.builtins import DEFAULTS, bind_builtin, check_shared, default_set
from chickadee.hooks.fire import BoundHook, HookFn, HookLimits, HookSet, HookSink
from chickadee.hooks.spec import HookSpec
from chickadee.journal.projection import Placed
from chickadee.schema.journal import HookP
from chickadee.state.resolve import PROJECT_LAYER

if TYPE_CHECKING:
    from chickadee.config import EngineConfig

__all__ = [
    "BOUND",
    "HookConfig",
    "ScriptRunner",
    "Standing",
    "bind",
    "bind_once",
    "bind_session",
    "entries",
    "limits_of",
    "specs_of",
    "standing",
    "write_bound",
]

BOUND: Final = "bound"
"""The `hook` line kind a bound set is written as: its `event` and its `decision`."""

ScriptRunner = Callable[[HookSpec], tuple[HookFn, str]]
"""A script entry's run, and its digest (its entry and its file's bytes, `state/trust`)."""

_SPECS: Final = TypeAdapter(tuple[HookSpec, ...])
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookConfig:
    """What the config says of hooks, before any is bound: the entries, the bounds, and
    whether the repository's executable config was accepted."""

    specs: tuple[HookSpec, ...] = ()
    limits: HookLimits = field(default_factory=HookLimits)
    accepted: bool = False

    @classmethod
    def of(cls, config: EngineConfig, *, accepted: bool) -> HookConfig:
        """`config`'s `[[hooks]]` and its `hooks.*` limits; `accepted`, the clone's trust."""
        return cls(config.hooks, limits_of(config.harness_limit), accepted)


def limits_of(harness_limit: Callable[[str], int]) -> HookLimits:
    """The `hooks.*` rows of the limits register, as the config resolves them
    (`EngineConfig.harness_limit`)."""
    return HookLimits(
        max_continues=harness_limit("hooks.max_continues"),
        max_retries=harness_limit("hooks.max_retries"),
        max_steering=harness_limit("hooks.max_steering"),
        recheck_wait_s=harness_limit("hooks.recheck_wait_s"),
    )


@dataclass(frozen=True)
class Standing:
    """What binding does with one entry, and why: the one reading of an entry, which `bind`
    acts on and `chickadee hooks` shows (`hooks/show.py`), so the two cannot disagree."""

    state: Literal["bound", "off", "inert", "refused"]
    why: str | None = None


def standing(spec: HookSpec, *, accepted: bool, sealed: bool) -> Standing:
    """`spec`'s standing. `accepted`: whether the clone's owner accepted the repository's
    executable config, whose digest covers every script entry (`state/trust`). `sealed`:
    whether a sandbox backend enforces a script's seal (`hooks/script.seals`)."""
    if not spec.enabled:
        # Switched off is not bound: a script that is off needs no sandbox and no trust.
        return Standing("off")
    if spec.kind == "builtin":
        try:
            bind_builtin(spec)
        except ConfigError as exc:
            return Standing("refused", str(exc))
        return Standing("bound")
    # Acceptance first: a repository's script nobody accepted is inert wherever it is read,
    # sandbox or none, and never stops the clone from running.
    if spec.layer == PROJECT_LAYER and not accepted:
        return Standing("inert", "a repository's script, not accepted: `chickadee trust`")
    if not sealed:
        return Standing(
            "refused",
            f"hook {spec.name!r} is a script, and no sandbox backend is available to run it "
            "in; a script never runs unsandboxed (HK-2). Remove it, or run where bubblewrap "
            "or Seatbelt is installed",
        )
    return Standing("bound")


def bind(
    configured: Sequence[HookSpec],
    *,
    accepted: bool,
    scripts: ScriptRunner | None,
    limits: HookLimits | None = None,
) -> HookSet:
    """`configured` over the defaults, bound as each entry's `standing` says: a refusal
    raised, an inert entry skipped and named. `scripts` runs a script, None where no sandbox
    backend enforces a script's seal."""
    hooks: list[BoundHook] = []
    skipped: list[tuple[str, str]] = []
    for spec in entries(configured):
        found = standing(spec, accepted=accepted, sealed=scripts is not None)
        if found.state == "refused":
            raise ConfigError(found.why or f"hook {spec.name!r} cannot be bound")
        if found.state == "inert":
            skipped.append((spec.name, found.why or "inert"))
        if found.state != "bound":
            continue
        if spec.kind == "builtin":
            hooks.append(bind_builtin(spec))
        elif scripts is not None:
            run, digest = scripts(spec)
            hooks.append(BoundHook(spec, run, digest))
    check_shared(hooks)
    return HookSet(tuple(hooks), tuple(skipped), limits or HookLimits())


def entries(configured: Sequence[HookSpec]) -> tuple[HookSpec, ...]:
    """The defaults, each replaced in place by a configured entry of the same `(event, name)`,
    then every other configured entry in its order (ST-5's keyed merge, as the layers merge)."""
    by_key = {spec.key: spec for spec in configured}
    defaults = tuple(_over(d, by_key.get(d.key)) for d in DEFAULTS)
    named = {d.key for d in DEFAULTS}
    return (*defaults, *(s for s in configured if s.key not in named and _defines(s)))


def _over(default: HookSpec, entry: HookSpec | None) -> HookSpec:
    """A default as an entry of its `(event, name)` leaves it: replaced, or switched off."""
    if entry is None:
        return default
    return entry if _defines(entry) else default.model_copy(update={"enabled": entry.enabled})


def _defines(spec: HookSpec) -> bool:
    return spec.use is not None or spec.command is not None


def bind_once(
    sink: HookSink,
    configured: HookConfig,
    scripts: ScriptRunner | None,
    *,
    where: str,
) -> tuple[HookSet, tuple[HookSpec, ...] | None]:
    """A run's or a session's set (decision 14), and the entries still to be written as its
    `bound` line, or None once they are: from that line where there is one, so a resumed run
    or session runs the hooks it started with and says once in the log if the config has moved
    since; else from the config. The caller writes the line where its record allows
    (`write_bound`): a run's after its `run_start`. `where` names it in that log line."""
    stored = specs_of(sink.lines(), sink.read)
    specs, limits = stored if stored is not None else (configured.specs, configured.limits)
    bound = bind(specs, accepted=configured.accepted, scripts=scripts, limits=limits)
    if stored is None:
        return bound, specs
    if specs != configured.specs:
        _log.info("%s resumes with the hooks it bound, not today's config", where)
    return bound, None


def bind_session(
    sink: HookSink, configured: HookConfig, scripts: ScriptRunner | None
) -> HookSet:
    """A session's set, bound from today's config each time it is opened, so an edit to the
    config reaches the session's next open, as a person editing it expects. Its `bound` line is
    written when the set differs from the last one the log holds, or, with none, from the
    defaults alone: a session nobody configured hooks for has no line, and its log is as it
    was in older versions. An episode, the unit an eval arm reproduces, reads its set back
    instead
    (`bind_once`)."""
    bound = bind(
        configured.specs,
        accepted=configured.accepted,
        scripts=scripts,
        limits=configured.limits,
    )
    last = next(
        (p.payload.input_sha for p in reversed(sink.lines()) if p.payload.event == BOUND),
        default_set(configured.limits).digest,
    )
    if bound.digest != last:
        write_bound(sink, bound, configured.specs)
    return bound


def write_bound(sink: HookSink, bound: HookSet, specs: Sequence[HookSpec]) -> None:
    """The bound set, as the line a resume binds from."""
    body = json.dumps(
        {
            "specs": json.loads(_SPECS.dump_json(tuple(specs), by_alias=True)),
            "skipped": [list(s) for s in bound.skipped],
            "limits": vars(bound.limits),
        },
        sort_keys=True,
    ).encode("utf-8")
    sink.hook(
        HookP(
            event=BOUND,
            name="*",
            decision=BOUND,
            wall_ms=0,
            message=f"{len(bound.hooks)} bound, {len(bound.skipped)} skipped",
            text_sha=sink.blob(body),
            input_sha=bound.digest,
        ),
        task_id=None,
        round=None,
    )


def specs_of(
    lines: Sequence[Placed[HookP]], blob: Callable[[str], bytes]
) -> tuple[tuple[HookSpec, ...], HookLimits] | None:
    """The entries and bounds a run bound when it started, or None for a run written before
    D227, which had none to bind."""
    found = next((p.payload for p in lines if p.payload.event == BOUND), None)
    if found is None or found.text_sha is None:
        return None
    body = json.loads(blob(found.text_sha))
    return _SPECS.validate_python(body["specs"]), HookLimits(**body["limits"])
