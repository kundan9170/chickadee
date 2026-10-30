"""JSON Schema goldens for a script hook's contract.

What a script reads on stdin at each point (`protocol.input_model`) and the JSON it may answer
with (`protocol.ScriptOutput`): the contract a script's author writes against. Additive-only
from G3, as the box's are; how a golden is rendered and written is `chickadee.goldens`'s.
"""

from __future__ import annotations

import pathlib

from pydantic import BaseModel

from chickadee.goldens import GoldenSet
from chickadee.hooks.points import HookPoint
from chickadee.hooks.protocol import ScriptOutput, input_model

__all__ = ["GOLDENS"]

_CONTRACTS: dict[str, type[BaseModel]] = {
    f"{point.value}_input": input_model(point) for point in HookPoint
}
_CONTRACTS["output"] = ScriptOutput

GOLDENS = GoldenSet(
    contracts=_CONTRACTS,
    directory=pathlib.Path(__file__).resolve().parent / "golden",
)


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    for path in GOLDENS.export_all():
        print(path)
