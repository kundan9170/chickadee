"""The committed recordings are exactly what their builders write today: the demo session
(`demo_session.build`) and its coder call's run (`recorded_run.build`)."""

from __future__ import annotations

from pathlib import Path

import recorded_run
from demo_session import FIXTURE, build


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_fixture_is_regenerated_from_the_writers(tmp_path: Path) -> None:
    rebuilt = build(tmp_path / "demo")
    assert _files(rebuilt) == _files(FIXTURE), (
        "the demo recording drifted from the writers; regenerate it with "
        "`uv run python packages/serve/tests/demo_session.py` and review the diff"
    )


def test_the_recorded_run_is_regenerated_from_the_journal_writer(tmp_path: Path) -> None:
    rebuilt = recorded_run.build(tmp_path / "run")
    assert _files(rebuilt) == _files(recorded_run.FIXTURE), (
        "the recorded run drifted from the journal writer; regenerate it with "
        "`uv run python packages/serve/tests/recorded_run.py` and review the diff"
    )
