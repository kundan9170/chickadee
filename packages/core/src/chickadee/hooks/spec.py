"""One `[[hooks]]` entry, as a layer wrote it and the loader marked it.

```toml
[[hooks]]
event = "round_ended"
name = "changelog" # unique per event: what a higher layer names to change it
command = ["./.chickadee/hooks/check-changelog.sh"] # or: use = "<built-in>"
role = "worker" # optional matchers, each narrowing
model = "qwen*"
tool = "shell"
program = ["pytest"]
paths = ["src/**"]
timeout_s = 10
on_error = "allow" # or "deny": fail closed
enabled = true
[hooks.with] # a built-in's parameters, checked by the built-in
```

`layer` is where the loader found the entry's `use` or `command` (`state/resolve`), never
written in a file: a repository's script is executable config, inert until accepted.
Entries merge across layers by `(event, name)`, so a person turns a
repository's hook off with `enabled = false` under its name, and the entry keeps the layer that
defined it.
"""

from __future__ import annotations

import re
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, model_validator

from chickadee.hooks.payloads import HookRole
from chickadee.hooks.points import HookPoint

__all__ = ["HOOK_NAME", "MAX_TIMEOUT_S", "HookSpec"]

HOOK_NAME: Final = re.compile(r"^[a-z][a-z0-9_-]*$")
"""A hook's name: what a `hook` line, the handoff and a higher layer call it."""

MAX_TIMEOUT_S: Final = 120
"""A script past two minutes holds a role's loop longer than any model call should."""


def _not_a_boolean(value: object) -> object:
    # `True` is an `int`; a hook's timeout written `true` is a mistake, never one second.
    if isinstance(value, bool):
        raise ValueError("a number, not a boolean")
    return value


class HookSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    event: HookPoint
    name: Annotated[str, Field(pattern=HOOK_NAME.pattern)]
    use: str | None = None
    """A built-in's name (`hooks/builtins.py`)."""
    command: tuple[str, ...] | None = None
    """An argv, never a shell string: quoting is where injection lives."""
    role: HookRole | None = None
    model: str | None = None
    """A glob over the role's model name."""
    tool: str | None = None
    """A tool's registry name, whatever the model wrote."""
    program: tuple[str, ...] = ()
    """Program names, any of which a shell call's line runs (`shell_line.read_line`)."""
    paths: tuple[str, ...] = ()
    """Globs over the event's repository paths."""
    timeout_s: Annotated[
        int, BeforeValidator(_not_a_boolean), Field(gt=0, le=MAX_TIMEOUT_S)
    ] = 30
    on_error: Literal["allow", "deny"] = "allow"
    enabled: Annotated[bool, Field(strict=True)] = True
    with_: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue], alias="with")
    layer: str = "user"

    @model_validator(mode="after")
    def _one_kind(self) -> HookSpec:
        if self.use is None and self.command is None and not self.enabled:
            return self  # a switch-off by name, of a hook a lower layer or a default defines
        if (self.use is None) == (self.command is None):
            raise ValueError(
                f"hook {self.name!r} names {'both' if self.use else 'neither of'} `use` and "
                "`command`: a hook is a built-in or a script, one of the two"
            )
        if self.command is not None and not (self.command and self.command[0]):
            raise ValueError(f"hook {self.name!r}'s `command` is an empty argv")
        if self.with_ and self.command is not None:
            raise ValueError(
                f"hook {self.name!r} is a script, and `with` is a built-in's parameters; a "
                "script reads its own arguments from `command`"
            )
        return self

    @property
    def kind(self) -> Literal["builtin", "script"]:
        return "builtin" if self.use is not None else "script"

    @property
    def key(self) -> tuple[str, str]:
        """Its identity across layers."""
        return (self.event.value, self.name)
