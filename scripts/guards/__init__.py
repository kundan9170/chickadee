"""Shared types for the D148 guards.

A guard is one module beside this one. It asserts that a concept has exactly one home by
reading source text and reporting every line that implements the concept somewhere else.
`run.py` discovers the guards, runs them, and compares each against the debt it already
tolerates, so a guard can land while its theme is still moving callers.

A guard imports **only the standard library**. That keeps the whole ratchet
inside CI's cheap `audits` job, which runs before and independently of the Python toolchain,
and means no dependency change can break a check on the code it guards. A guard that wants to
import the package it guards has the wrong design: it reads source text, never behaviour.

This module holds the two types and nothing else, because a guard's rule and the debt it
tolerates belong together in one file — splitting them would be the second home D148 exists
to remove.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Guard", "Violation"]


@dataclass(frozen=True)
class Violation:
    """One line that implements a concept outside its home."""

    path: str
    """Repo-relative, forward slashes, so a baseline reads identically on every platform."""

    line: int
    """1-indexed, to match an editor and a terminal's `file:line`."""

    text: str
    """The offending line, stripped. Printed so a reader need not open the file to judge it."""


@dataclass(frozen=True)
class Guard:
    """One concept, one home, and the debt currently tolerated against that rule."""

    name: str
    """Must equal the module's filename stem. Two names for one guard is the defect this
    wave is fixing, at one-file scale, so the runner refuses the disagreement."""

    rule: str
    """One line of prose, printed on every failure. The person who trips a guard is usually
    an agent that has never read D148: a line number teaches nothing, while "every durable
    write goes through `chickadee.durable`" routes them to the home."""

    corpus: tuple[str, ...]
    """Globs, relative to the repository root, naming what `find` reads. The runner refuses
    a corpus that matches no files — that is how a guard keeps passing after the directory
    it was written against is split out from under it."""

    find: Callable[[Sequence[Path]], Sequence[Violation]]
    """Reads the corpus and returns every line implementing the concept outside its home."""

    baseline: Mapping[str, int] = field(default_factory=dict)
    """Repo-relative path to the count tolerated there today; an absent path means zero.
    Carried from the module's `BASELINE` assignment, which `run.py --tighten` rewrites."""
