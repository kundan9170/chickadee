"""The scout rung's model.

`rungs.scout` names a catalog model — a mid-class one by default, because an
open-weight coder model handed tools may under-call them. The rung is config-swappable
and telemetry decides when to drop it to the coder's model.

A rung that cannot be reached never blocks the session: `locate` fails
`UNAVAILABLE` at once, and Zeus's own ladder is the fallback. A model without
function calling fails the same way, since the scout's tools cannot reach it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chickadee.adapters.base import first_attempt_params, load_template
from chickadee.adapters.frontier import TEMPLATES
from chickadee.client.request import ModelClient
from chickadee.config import EngineConfig, RungEndpoint
from chickadee.limits import window
from chickadee.schema.adapter import RetryParams

__all__ = [
    "PROMPT",
    "ROLE",
    "ScoutRung",
    "nudge",
    "prompt",
    "scout_rung",
]

ROLE = "locate"
"""The cast's role on the scout rung (`roles.py`); its caps are the register's `locate.*`."""

PROMPT = "locate.md"
"""`roles.py` names it `frontier/locate.md`: the scout's own template, all its prompt."""


@dataclass(frozen=True)
class ScoutRung:
    model: str
    """The catalog name — what the ledger charges, never the rung's."""
    endpoint: RungEndpoint
    client: ModelClient
    rung_name: str
    """The rung the `locate` role runs on, as the config resolved it (`scout_rung`): what the
    scout's reader sees as. Required, never defaulted: a default would be a second
    lookup, the cast's rung, blind to a role the config moved."""

    @property
    def native_tools(self) -> bool:
        """False: the client sends no `tools` array, so the scout has no tool channel
        until the TOOL text protocol carries its tools, and `locate` is `UNAVAILABLE`."""
        return self.endpoint.native_tools

    @property
    def window(self) -> int:
        return window(self.endpoint)

    def params(self) -> RetryParams:
        """First-attempt sampling, the model's whole output cap, and its configured
        reasoning (per model, never a retry lever): the one rule in `adapters.base`."""
        return first_attempt_params(self.endpoint)


def scout_rung(
    config: EngineConfig, client_for: Callable[[RungEndpoint], ModelClient]
) -> ScoutRung:
    """The model the `locate` role's rung names in this config."""
    rung = config.rung_name_for(ROLE)
    name = config.rungs[rung]
    endpoint = config.models[name]
    return ScoutRung(model=name, endpoint=endpoint, client=client_for(endpoint), rung_name=rung)


def prompt() -> str:
    return load_template(TEMPLATES, PROMPT)


def nudge() -> str:
    """The frontier family's format nudge, for one re-ask when the findings do not parse."""
    return load_template(TEMPLATES, "nudge.md")
