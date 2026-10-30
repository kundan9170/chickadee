"""`models.py` is what the generator writes from the wire golden today: a golden that moved
without a regeneration fails here, naming the command that fixes it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from chickadee_wire import models

TOOL = Path(__file__).resolve().parents[1] / "tools" / "generate_models.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_models", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_committed_models_are_the_generated_ones(tmp_path: Path) -> None:
    tool = _tool()
    fresh = tmp_path / "models.py"
    tool.generate(fresh)
    assert fresh.read_text(encoding="utf-8") == tool.MODELS.read_text(encoding="utf-8"), (
        "the wire golden moved: run `uv run python packages/wire/tools/generate_models.py`"
    )


def test_every_response_request_and_record_kind_has_a_type() -> None:
    golden = _tool().GOLDEN

    doc = json.loads(golden.read_text(encoding="utf-8"))
    assert {name for name in doc["responses"] if not hasattr(models, name)} == set()
    readable = {k for k, v in doc["session_records"].items() if not v.get("x-opaque")}
    assert set(models.SESSION_RECORDS) == readable
    assert {name for name in doc["requests"] if not hasattr(models, name)} == set()
    journal = json.loads(_tool().JOURNAL.read_text(encoding="utf-8"))
    assert set(models.RUN_RECORDS) == set(journal["$defs"]["EventKind"]["enum"])
