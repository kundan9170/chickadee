"""JSON Schema goldens for the box's contract.

`CoderRequest` and `CoderReport` freeze at G3, additive-only from then. They live here rather
than in `schema/**`, which is frozen; how a golden is rendered
and written is `chickadee.goldens`'s.
"""

from __future__ import annotations

import pathlib

from chickadee.box.report import CoderReport
from chickadee.box.request import CoderRequest
from chickadee.goldens import GoldenSet

GOLDENS = GoldenSet(
    contracts={
        "coder_request": CoderRequest,
        "coder_report": CoderReport,
    },
    directory=pathlib.Path(__file__).resolve().parent / "golden",
)


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    for path in GOLDENS.export_all():
        print(path)
