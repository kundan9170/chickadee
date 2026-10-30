"""Merging configuration layers.

Three rules, and the third is the one with an argument behind it.

- **scalars** — the higher layer wins.
- **tables** — merge recursively; rule 1 applies at the leaves.
- **arrays of tables** — merge **by identity key**, never by position and never by
  concatenation.

OpenCode concatenates arrays and de-duplicates, on the correct argument that a project adding
a plugin must not erase the global ones — which plain replacement gets wrong. But our array
entries carry identity: a hook is `(event, command)`, an MCP server has a `name`. Pure
concatenation would run a hook **twice** when both layers declare it, and would offer no way
to *override* one. Keyed merge is concatenation's benefit without that cost.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard, cast

IDENTITY_KEYS: dict[str, tuple[str, ...]] = {
    "hooks": ("event", "name"),
    "mcp.servers": ("name",),
}
"""Which fields identify an entry, per array-of-tables path.

A closed table on purpose: an array whose identity nobody declared cannot be merged
sensibly, and guessing (first field? whole body?) would be a rule nobody could predict.
`merge` refuses such an array rather than inventing one.
"""


class MergeError(Exception):
    """A layer could not be merged, with the path that could not be merged."""


class UnknownKey(MergeError):
    """A key no schema declares, naming file, key and layer.

    Its own type because its remedy is different from every other merge failure — the user
    made a typo, and the fix is one character, not a redesign. A misspelled
    `hard_cost_cap_usd` otherwise silently leaves the cap unset, which is the failure
    `RunStartP`'s docstring already promises does not happen.
    """


def _key_of(entry: Mapping[str, Any], fields: Sequence[str], where: str) -> tuple[Any, ...]:
    missing = [f for f in fields if f not in entry]
    if missing:
        raise MergeError(
            f"{where}: entry is missing its identity field(s) {missing}; "
            f"entries there are identified by {list(fields)}"
        )
    return tuple(entry[f] for f in fields)


def merge_arrays(
    lower: Sequence[Mapping[str, Any]],
    higher: Sequence[Mapping[str, Any]],
    *,
    path: str,
) -> list[dict[str, Any]]:
    """Keyed merge, preserving lower-layer order and appending genuinely new entries.

    Order is not cosmetic for `[[hooks]]`: hooks fire in declaration order, so a merge that
    reordered them would change behaviour while claiming to combine configuration.
    """
    fields = IDENTITY_KEYS.get(path)
    if fields is None:
        raise MergeError(
            f"{path}: array-of-tables has no declared identity key, so two layers "
            f"declaring it cannot be merged. Declare it in IDENTITY_KEYS or make the "
            f"field a table keyed by name"
        )
    out: list[dict[str, Any]] = [dict(e) for e in lower]
    index = {_key_of(e, fields, path): i for i, e in enumerate(out)}
    for entry in higher:
        key = _key_of(entry, fields, path)
        if key in index:
            out[index[key]] = merge(out[index[key]], dict(entry), path=path)
        else:
            index[key] = len(out)
            out.append(dict(entry))
    return out


def merge(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    path: str = "",
) -> dict[str, Any]:
    """Merge `higher` over `lower`. Associative over layers, so the order is the whole spec."""
    out: dict[str, Any] = dict(lower)
    for key, hi in higher.items():
        here = f"{path}.{key}" if path else key
        lo = out.get(key)
        if is_table(lo) and is_table(hi):
            out[key] = merge(lo, hi, path=here)
        elif _is_table_array(lo) and _is_table_array(hi):
            out[key] = merge_arrays(lo, hi, path=here)
        else:
            out[key] = hi
    return out


def is_table(value: object) -> TypeGuard[Mapping[str, Any]]:
    """A table: a TOML table, or the nested dict an environment layer builds.

    `isinstance(value, Mapping)` is the whole test. The guard exists for the type checker,
    which cannot know from `isinstance` alone that a table's keys are strings; every layer
    that produces one parses TOML or splits an environment variable name, so they are.
    """
    return isinstance(value, Mapping)


def _is_table_array(value: object) -> TypeGuard[Sequence[Mapping[str, Any]]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    items = cast("Sequence[object]", value)
    return len(items) > 0 and all(is_table(v) for v in items)


@dataclasses.dataclass(frozen=True)
class LayerProvenance:
    """Which layer supplied the winning value for one key.

    Named `LayerProvenance` rather than `Provenance`: `schema/dag.py` already has the latter,
    for how a packet came to exist. Two `Provenance` types would read as one concept in
    tracebacks and imports while answering unrelated questions.

    A `config_sha` alone says two runs differed. This says **which layer moved**, which is
    the question anyone actually has when they are looking.
    """

    key: str
    layer: str


def provenance(layers: Sequence[tuple[str, Mapping[str, Any]]]) -> list[LayerProvenance]:
    """Last layer to set each leaf key wins, in the order given (lowest first)."""
    winner: dict[str, str] = {}
    for name, body in layers:
        for key in _leaves(body):
            winner[key] = name
    return [LayerProvenance(key=k, layer=winner[k]) for k in sorted(winner)]


def _leaves(body: Mapping[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in body.items():
        here = f"{prefix}.{key}" if prefix else key
        if is_table(value):
            out.extend(_leaves(value, here))
        else:
            out.append(here)
    return out
