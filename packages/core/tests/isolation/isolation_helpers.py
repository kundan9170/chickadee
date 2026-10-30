"""Isolation wrap helpers. Not `conftest.py`."""

from __future__ import annotations

from pathlib import Path

from chickadee.isolation.types import Isolation


def iso_for(
    cwd: Path,
    argv: tuple[str, ...],
    *,
    repo_writable: bool = False,
    network_off: bool = True,
) -> Isolation:
    return Isolation(
        argv=argv,
        cwd=cwd,
        network_off=network_off,
        repo_writable=repo_writable,
    )


RUNS = 'while [ "$1" != "--" ] && [ $# -gt 0 ]; do shift; done; shift; exec "$@"\n'
"""A sandbox stand-in that runs the wrapped command: everything up to `--` is its own flags."""


def fake_binary(bin_dir: Path, name: str, body: str) -> Path:
    """A shell script called `name` in `bin_dir`, for a probe to find on `PATH`. The probe
    is about what a backend does when it runs, so a stand-in that runs (or fails the way a
    restricted kernel makes the real one fail) is the honest fixture where the real binary
    is absent."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path
