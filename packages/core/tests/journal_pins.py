"""The run journals recorded from the packet loop, before contracts v2, at the tests' root so
every suite imports one copy.

**The recordings** are `fixtures/journals/<name>/`, each a whole run directory recorded from
the orchestrator box, plus `T-3.55`'s run journal (`fixtures/records/run/`). Contracts v2
refuses every one of them, and that refusal is what they pin now
(`journal/test_journal_projection_pins.py`): a recorded packet-loop journal is refused, never
misread. Their golden values, what each reader returned over them, retired with the readers'
packet answers; the
readers are pinned over v2 journals their tests write.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from chickadee.journal import paths
from chickadee.state.roots import Roots

FIXTURES: Final = Path(__file__).resolve().parent / "fixtures"
JOURNALS: Final = FIXTURES / "journals"

RECORDED: Final[Mapping[str, Path]] = {
    "repack": JOURNALS / "repack",
    "rejected": JOURNALS / "rejected",
    "ambiguous_exec": JOURNALS / "ambiguous_exec",
    "budget_cap": JOURNALS / "budget_cap",
    "split_or_replan": JOURNALS / "split_or_replan",
    "started": JOURNALS / "started",
    "records_run": FIXTURES / "records" / "run",
}
"""Every recording, by name. `records_run` is `T-3.55`'s."""


def scratch_roots(base: Path) -> Roots:
    """Roots under `base` for a recording to be installed into: no repository needed, since
    only the run's directory is read."""
    return Roots(
        project=base,
        repo_local=base / "repo",
        checkout_local=base / "checkout",
        user=base / "user",
    )


def install(name: str, roots: Roots, run_id: str, *, journal: bytes | None = None) -> Path:
    """Copy a recording into `roots` as run `run_id`'s directory, its journal replaced by
    `journal` when given (a cut). Returns the run directory."""
    target = paths.run_dir(roots, run_id)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(RECORDED[name], target)
    if journal is not None:
        (target / "journal.jsonl").write_bytes(journal)
    return target
