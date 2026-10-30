"""A language server as the index sees one, scripted: it judges each file of a pass it has
problems for, from the file's bytes on disk then. Shared by the graph's tests and the
box's."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chickadee.graph.diagnostics import Diagnostic, Verdict
from chickadee.graph.lsp import DefQuery


def _sha(root: Path, rel: str) -> str:
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


@dataclass
class Judging:
    """A server, as the indexer sees one: it judges each file of a pass it has a verdict for,
    from the file's bytes on disk then. `problems` maps a path to what it finds there; a path
    it has no entry for, it does not judge."""

    root: Path
    problems: dict[str, tuple[Diagnostic, ...]] = field(default_factory=dict)
    judged: list[list[str]] = field(default_factory=list)
    rechecked: list[list[str]] = field(default_factory=list)
    """What the indexer handed it to ask again after each pass (`edges.Rechecking`)."""
    started: dict[str, tuple[Diagnostic, ...]] = field(default_factory=dict)
    """What it finds in a file's start, given bytes rather than the disk's (`judge`)."""
    asked_start: list[list[str]] = field(default_factory=list)

    def recheck(self, paths: Sequence[str], asked_at: int) -> None:
        self.rechecked.append(list(paths))

    def definition(self, name: str, file: str, line: int) -> str | None:
        return None

    def prefetch(
        self,
        queries: Sequence[DefQuery],
        lookup: Callable[[str, int], str | None],
        files: Sequence[str] = (),
    ) -> tuple[Verdict, ...]:
        mine = sorted(f for f in files if f in self.problems)
        self.judged.append(mine)
        return tuple(
            Verdict(rel, _sha(self.root, rel), "fake", self.problems[rel]) for rel in mine
        )

    def judge(self, texts: Mapping[str, bytes]) -> tuple[Verdict, ...]:
        mine = sorted(rel for rel in texts if rel in self.started)
        self.asked_start.append(sorted(texts))
        return tuple(
            Verdict(rel, hashlib.sha256(texts[rel]).hexdigest(), "fake", self.started[rel])
            for rel in mine
        )
