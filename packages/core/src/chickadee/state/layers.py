"""The five configuration layers, in precedence order.

```
1 defaults the dataclass defaults — RunConfig and the rest
2 user $XDG_CONFIG_HOME/chickadee/config.toml
3 project <repo>/.chickadee/config.toml
4 local <repo-local>/work.toml: the person's work folder for this clone
5 environment CHICKADEE_* variables
```

**Project beats user, and that is not the same rule as approval.** For ordinary
configuration the project knows things the person does not — which linters exist, which
profiles apply, what the toolchain is — so a personal preference must not silently override a
*fact about the repository*. Approval preferences do not merge at all; they compose through
D79's `max()`, which is why "repo `always_ask` beats user `never_ask`" needs no special case.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tomllib
from collections.abc import Mapping
from typing import Any, cast

from chickadee.state.local import LOCAL_FILE
from chickadee.state.roots import Roots

CONFIG_NAME = "config.toml"
ENV_PREFIX = "CHICKADEE_"
"""§8. One prefix, so the namespace is searchable."""

RESERVED_ENV = frozenset({"CHICKADEE", "CHICKADEE_SESSION"})
"""The two markers every spawn carries, set by `process.supervisor.scrub_env`. Never
config: a `chickadee` started under the harness (`!chickadee status`, a hook) inherits
them, and read as a key, `CHICKADEE_SESSION` was an unknown `session` that refused the
load. A child's environment is not a config override."""

LAYER_ORDER = ("defaults", "user", "project", "local", "environment")
"""Lowest to highest. `merge` is associative, so this tuple is the whole specification."""


class LayerReadError(Exception):
    """A layer could not be read. Always names the file.

    **Not** `ConfigError`: that name belongs to the engine's public hierarchy, and
    a second class with the same name would not be caught by a host writing
    `except chickadee.errors.ConfigError`. The facade maps this to the public one; the twin
    that looks catchable and is not is the thing worth avoiding.
    """


@dataclasses.dataclass(frozen=True)
class Layer:
    """One layer's contribution, kept separate until `resolve` merges them.

    Separate rather than folded together as they load, because §7 records **which layer
    supplied each winning value** — and that is unrecoverable once they are one dict.
    """

    name: str
    source: pathlib.Path | None
    body: Mapping[str, Any]


def _read_toml(path: pathlib.Path, layer: str) -> Mapping[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise LayerReadError(f"{path} ({layer} layer) is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise LayerReadError(f"{path} ({layer} layer) could not be read: {exc}") from exc


def _from_env(environ: Mapping[str, str]) -> dict[str, Any]:
    """`CHICKADEE_A__B=1` -> `{"a": {"b": 1}}`.

    Double underscore nests; a single underscore is part of a key, because real keys have
    them (`hard_cost_cap_usd`) and a separator that collides with the names it separates is
    a separator that cannot be used.
    """
    out: dict[str, Any] = {}
    for raw, value in environ.items():
        if not raw.startswith(ENV_PREFIX) or raw == ENV_PREFIX or raw in RESERVED_ENV:
            continue
        parts = [p.lower() for p in raw[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        cursor = out
        for part in parts[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise LayerReadError(
                    f"{raw}: {'.'.join(parts)} is set both as a value and as a table"
                )
            # Every table on this path was made by `setdefault` above, so its keys are the
            # lowered name parts: strings, which `isinstance` alone cannot tell the checker.
            cursor = cast("dict[str, Any]", nxt)
        cursor[parts[-1]] = _coerce(value)
    return out


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            continue
    return value


def load_layers(
    roots: Roots,
    *,
    defaults: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[Layer]:
    """Read every layer that exists. A missing file is an empty layer, never an error."""
    env = os.environ if environ is None else environ
    user = roots.user / CONFIG_NAME
    project = roots.project / CONFIG_NAME
    local = roots.repo_local / LOCAL_FILE
    return [
        Layer("defaults", None, dict(defaults or {})),
        Layer(
            "user",
            user if user.is_file() else None,
            _read_toml(user, "user") if user.is_file() else {},
        ),
        Layer(
            "project",
            project if project.is_file() else None,
            _read_toml(project, "project") if project.is_file() else {},
        ),
        # The file holds `[work]`'s keys at its top (`state.local`): `dir = "…"`.
        Layer(
            "local",
            local if local.is_file() else None,
            {"work": dict(_read_toml(local, "local"))} if local.is_file() else {},
        ),
        Layer("environment", None, _from_env(env)),
    ]


def engine_layers(
    engine_config: pathlib.Path, *, environ: Mapping[str, str] | None = None
) -> list[Layer]:
    """The two layers the person sets, read with no repository: the engine config at
    `engine_config` and the environment. For a surface that runs before any repository is
    named, `chickadee setup`. A missing file is an empty layer, as in `load_layers`."""
    env = os.environ if environ is None else environ
    return [
        Layer(
            "user",
            engine_config if engine_config.is_file() else None,
            _read_toml(engine_config, "user") if engine_config.is_file() else {},
        ),
        Layer("environment", None, _from_env(env)),
    ]
