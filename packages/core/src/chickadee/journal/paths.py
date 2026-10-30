"""Storage layout.

Plain JSONL per run. No database: SQLite belongs to the code graph, not to an append-only
log that is read start-to-finish exactly once per resume.
"""

from __future__ import annotations

import pathlib

from chickadee.state.roots import Roots

RUNS_DIR = "runs"
JOURNAL_NAME = "journal.jsonl"
BLOBS_DIR = "blobs"
OPERATIONS_DIR = "operations"
CONFIG_NAME = "config.json"
LOCK_NAME = "LOCK"
GC_LOG_NAME = "gc.log"
"""One line per run per `gc` — what was pruned, how many bytes, the cutoff, who.
Repo-local beside `runs/`, and a plain file rather than a journal: a journal cannot record
its own blobs' deletion without naming what it deleted."""

# Flat blob dir; shard only past this, and not before.
BLOB_SHARD_THRESHOLD = 10_000


def runs_root(roots: Roots) -> pathlib.Path:
    """Run journals, blobs and operations, under the **repo-local** root.

    Takes `Roots`, never a path. A journal belongs to the **clone**, not to a checkout: one
    session's history must not split across worktrees, and `Roots.repo_local` is the only
    thing that means that. A path parameter here is what let an earlier security finding
    exist — it looked right and resolved to whatever the caller happened to hold,
    which under the default isolation mode is a directory that dies with the episode.
    """
    return roots.repo_local / RUNS_DIR


def run_dir(roots: Roots, run_id: str) -> pathlib.Path:
    return runs_root(roots) / run_id


def journal_path(roots: Roots, run_id: str) -> pathlib.Path:
    return run_dir(roots, run_id) / JOURNAL_NAME


def blobs_dir(roots: Roots, run_id: str) -> pathlib.Path:
    return run_dir(roots, run_id) / BLOBS_DIR


def operations_dir(roots: Roots, run_id: str) -> pathlib.Path:
    return run_dir(roots, run_id) / OPERATIONS_DIR


def lock_path(roots: Roots, run_id: str) -> pathlib.Path:
    return run_dir(roots, run_id) / LOCK_NAME


def gc_log_path(roots: Roots) -> pathlib.Path:
    """The repo-local `gc.log`. One per clone, like `runs/`."""
    return roots.repo_local / GC_LOG_NAME
