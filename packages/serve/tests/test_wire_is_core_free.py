"""The wire contract imports nothing from the engine, so a client can depend on it alone."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("module", ["chickadee_serve.frames", "chickadee_serve.routes"])
def test_importing_the_contract_loads_no_engine_module(module: str) -> None:
    probe = textwrap.dedent(
        f"""
        import sys
        import {module}
        engine = [m for m in sys.modules if m == "chickadee" or m.startswith("chickadee.")]
        loaded = sorted(engine)
        print(",".join(loaded))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"{module} pulled in engine modules: {out}"
