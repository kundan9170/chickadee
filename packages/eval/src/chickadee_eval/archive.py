"""Result archive layout: `eval/results/<date>-<config-sha>/`.

The config sha is in the path rather than inside a file, so two runs of the same benchmark
under different configuration cannot land in the same directory and be compared as though
they were the same experiment. A result whose configuration nobody can reproduce is not a
result, and a path is harder to lose than a field.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any

RESULTS_ROOT = pathlib.Path("eval") / "results"

CONFIG_NAME = "config.json"
PREREGISTRATION_NAME = "preregistration.json"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.jsonl"
REPORT_NAME = "report.md"

DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[0-9a-f]{12}$")


def config_sha(config: dict[str, Any]) -> str:
    """Twelve hex chars over the canonical JSON form.

    Sorted keys and fixed separators so the same configuration hashes the same regardless
    of how it was assembled — otherwise key order alone would fork the archive.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def run_dir(
    date: str, config: dict[str, Any], root: pathlib.Path | None = None
) -> pathlib.Path:
    """The directory one run archives into. `date` is ISO `YYYY-MM-DD`."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")
    base = root if root is not None else pathlib.Path.cwd()
    return base / RESULTS_ROOT / f"{date}-{config_sha(config)}"


def create(
    date: str,
    config: dict[str, Any],
    *,
    root: pathlib.Path | None = None,
    preregistration: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Open an archive and write its snapshot before any result exists.

    The preregistration is written **here**, at creation, not alongside the results:
    a file that appears at the same time as the numbers it constrains is not
    evidence of what was promised beforehand. Refuses to reopen a directory that already
    holds results, because appending a second run's rows to a first run's archive makes
    both unreadable.
    """
    d = run_dir(date, config, root)
    if (d / RESULTS_NAME).exists():
        raise FileExistsError(
            f"{d} already holds results; a new run gets a new archive, or the two are "
            "silently merged into one set nobody can separate"
        )
    d.mkdir(parents=True, exist_ok=True)
    _write(d / CONFIG_NAME, config)
    if manifest is not None:
        _write(d / MANIFEST_NAME, manifest)
    if preregistration is not None:
        if not preregistration.get("frozen_at"):
            raise ValueError(
                "preregistration is not frozen; freeze it before opening the archive, "
                "since its value is entirely in having preceded the results (EV-12)"
            )
        _write(d / PREREGISTRATION_NAME, preregistration)
    return d


def _write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_archive(path: pathlib.Path) -> bool:
    return path.is_dir() and bool(DIR_RE.match(path.name)) and (path / CONFIG_NAME).exists()


def find(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every archived run, oldest first. Used by the A/B differ (Phase 6)."""
    base = (root if root is not None else pathlib.Path.cwd()) / RESULTS_ROOT
    return sorted((p for p in base.glob("*") if is_archive(p)), key=lambda p: p.name)
