"""A session's rung wiring: which catalog model each rung
calls, what a switch changed, and whether the session is pinned.

**Two session-log kinds, typed here beside their writers**, as the skill
pins are: `model_switched`, whose payload is a `ModelSwitch`, and `pinned`. The log checks only
that each is a JSON object; what a line may say is checked here, by its writer.

**The wiring is a fold, stored nowhere.** It is the engine config's `[rungs]`, moved by the
session's `model_switched` lines in order (`wired`), and `rungs_sha` is its digest.
`PacketEngine.models()` reports it as a `ModelWiring`, and `switch_model()` returns the
`ModelSwitch` its line records.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Final

from pydantic import TypeAdapter, with_config

from chickadee.context.session import Session, SessionState
from chickadee.context.store import SessionEvent
from chickadee.context.vocabulary import LINE, PayloadError, SessionKind, canonical_json
from chickadee.roles import rung_name_error
from chickadee.schema import sha256_hex

__all__ = [
    "MODEL_SWITCHED",
    "PINNED",
    "PINNED_EVAL",
    "CatalogModel",
    "ModelSwitch",
    "ModelWiring",
    "Payload",
    "Pinned",
    "pin_of",
    "record_pinned",
    "record_switch",
    "rungs_sha",
    "switches_of",
    "wired",
]

MODEL_SWITCHED: Final[SessionKind] = "model_switched"
PINNED: Final[SessionKind] = "pinned"

PINNED_EVAL: Final = "an eval run: its manifest pins its models and caps (EV-12)"
"""Why `create_session(pinned=True)` pins a session."""

Payload = Callable[[SessionEvent], bytes]
"""A line's payload bytes: the open session's `blob`, or a log read without its lock."""


@with_config(LINE)
@dataclass(frozen=True)
class ModelSwitch:
    """What `switch_model` changed: the payload `model_switched` journals."""

    KIND: ClassVar[SessionKind] = MODEL_SWITCHED
    rung: str
    from_model: str
    """A catalog name, `[models.<name>]`."""
    to_model: str
    model_id: str
    """`to_model`'s provider id, as its endpoint is called."""
    rungs_sha: str
    """The wiring after the switch (`rungs_sha`)."""


@with_config(LINE)
@dataclass(frozen=True)
class Pinned:
    """The session's models and caps are fixed: an eval run's, whose manifest pins
    them. A pinned session refuses `switch_model` and `set_cap`."""

    KIND: ClassVar[SessionKind] = PINNED
    why: str


@dataclass(frozen=True)
class CatalogModel:
    """What a catalog model is, as a person choosing a rung's model reads it:
    its windows, who serves it, and the name its provider knows it by. Never its `base_url` or
    key: an endpoint can carry a credential."""

    context_window: int
    max_output_tokens: int
    provider: str
    model_id: str | None = None
    """The provider's name for it (`[models.<name>].model_id`, `claude-opus-4-1`), which the
    catalog name (`frontier`) is the config's own word for: what the TUI's header shows.
    None from a server before wire 4.11."""


@dataclass(frozen=True)
class ModelWiring:
    """A session's wiring, derived from the engine config and its `model_switched` lines."""

    rungs: Mapping[str, str]
    """Rung to catalog name, as of now."""
    catalog: tuple[str, ...]
    """Every `[models.<name>]`: the switch targets."""
    switches: tuple[ModelSwitch, ...]
    """This session's, in order."""
    roles: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)
    """Role to rung, as the engine config assigns them (`EngineConfig.assignment`):
    which rung each role's calls are made on."""
    models: Mapping[str, CatalogModel] = field(
        default_factory=dict[str, CatalogModel], hash=False
    )
    """Each catalog model's windows and provider, by catalog name."""


_SWITCH: Final = TypeAdapter(ModelSwitch)
_PINNED: Final = TypeAdapter(Pinned)


def wired(rungs: Mapping[str, str], moves: Iterable[ModelSwitch]) -> dict[str, str]:
    """`rungs` with each switch applied in order: the session's wiring."""
    now = dict(rungs)
    for move in moves:
        now[move.rung] = move.to_model
    return now


def rungs_sha(rungs: Mapping[str, str]) -> str:
    """The wiring's digest: rung to catalog name, in the one canonical encoding."""
    return sha256_hex(canonical_json(dict(rungs)))


# -- the fold -------------------------------------------------------------------------------


def switches_of(state: SessionState, payload: Payload) -> tuple[tuple[int, ModelSwitch], ...]:
    """The session's switches, in order, each with the seq of its line. A rewind keeps them
    (`rewind.REWIND`): a switch is the person's command, not a turn."""
    return tuple(
        (event.seq, _SWITCH.validate_json(payload(event)))
        for event, _record in state.lines()
        if event.kind == MODEL_SWITCHED
    )


def pin_of(state: SessionState, payload: Payload) -> Pinned | None:
    """The session's pin, or None for a session no eval harness created."""
    for event, _record in state.lines():
        if event.kind == PINNED:
            return _PINNED.validate_json(payload(event))
    return None


# -- the writers ----------------------------------------------------------------------------


def record_switch(session: Session, switch: ModelSwitch) -> SessionEvent:
    """Write `switch`; fsynced, since `model_switched` is in `FSYNC_KINDS`.
    Whether the rung is declared and the model in the catalog is `switch_model`'s check: the
    log cannot know the config."""
    why = rung_name_error(switch.rung)
    if why is not None:
        raise PayloadError(why)
    if switch.from_model == switch.to_model:
        raise PayloadError(f"a switch moves rung {switch.rung} to another model")
    pinned = pin_of(session.state, lambda e: session.blob(e.payload_sha))
    if pinned is not None:
        raise PayloadError(f"the session is pinned ({pinned.why}); its models do not change")
    return session.append(switch)


def record_pinned(session: Session, why: str = PINNED_EVAL) -> SessionEvent:
    """Pin the session's models and caps: once, by `create_session(pinned=True)`."""
    if pin_of(session.state, lambda e: session.blob(e.payload_sha)) is not None:
        raise PayloadError("the session is pinned already")
    return session.append(Pinned(why))
