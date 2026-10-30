"""Escapes are self-eliminating.

Every escape is counted per repository. One that recurs in **more than 30%** of `locate`
calls is promoted into the snapshot's `promoted` block: it re-runs as part of the
deterministic snapshot, under the same read-only sandbox, and thereafter costs no tokens and
needs no request. Rarity becomes a measured, decreasing quantity, and the escape rate says
whether the structured tools were enough.

The store is `snapshot_hints.json` — per command `{command, added_at, hit_rate,
last_output_sha}`, the four fields §6 names, plus the `hits` and `runs` the rate is the
quotient of, so a rate is recomputed rather than drifted. **Where** the file lives is the
caller's: `14-scout.md` §6 wrote `.chickadee/`, which `31-state.md` has since made the
tracked *project* scope, and a promoted command a repository could ship to every clone would
be a command the repository chose to run — so the path is passed in, not derived here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from chickadee.clock import stamp
from chickadee.durable import write_atomic

__all__ = ["HINTS_FILE", "PROMOTE_ABOVE", "Hint", "Hints"]

HINTS_FILE = "snapshot_hints.json"

PROMOTE_ABOVE = 0.30
"""Strictly more than 30% of `locate` calls in this repository."""


@dataclass
class Hint:
    command: str
    added_at: str
    hit_rate: float
    last_output_sha: str | None
    hits: int = 0


class Hints:
    """The per-repository escape record. A missing or unreadable file starts empty: a hint
    is an optimisation, and losing one costs an escape, never a wrong answer."""

    def __init__(self, path: Path, *, now: Callable[[], str] = stamp) -> None:
        self.path = path
        self._now = now
        self.runs = 0
        self._hints: dict[str, Hint] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        data = cast(dict[str, object], raw)
        runs = data.get("runs")
        entries = data.get("hints")
        if not isinstance(runs, int) or not isinstance(entries, list):
            return
        self.runs = runs
        for entry in cast(list[object], entries):
            if not isinstance(entry, dict):
                continue
            e = cast(dict[str, object], entry)
            command, hits = e.get("command"), e.get("hits")
            if not isinstance(command, str) or not isinstance(hits, int):
                continue
            sha = e.get("last_output_sha")
            added = e.get("added_at")
            self._hints[command] = Hint(
                command=command,
                added_at=added if isinstance(added, str) else self._now(),
                hit_rate=hits / runs if runs else 0.0,
                last_output_sha=sha if isinstance(sha, str) else None,
                hits=hits,
            )

    def record_run(self, asked: Iterable[tuple[str, str | None]]) -> None:
        """One `locate` call, and the commands it ran or recalled (`Escapes.asked()`). A
        command counts once per call however often it was asked in it."""
        self.runs += 1
        for command, sha in dict(asked).items():
            hint = self._hints.get(command)
            if hint is None:
                hint = Hint(command, self._now(), 0.0, sha)
                self._hints[command] = hint
            hint.hits += 1
            if sha is not None:
                hint.last_output_sha = sha
        for hint in self._hints.values():
            hint.hit_rate = hint.hits / self.runs

    def promoted(self) -> tuple[str, ...]:
        """Commands recurring in more than `PROMOTE_ABOVE` of calls, most frequent first."""
        chosen = [h for h in self._hints.values() if h.hit_rate > PROMOTE_ABOVE]
        chosen.sort(key=lambda h: (-h.hit_rate, h.command))
        return tuple(h.command for h in chosen)

    def get(self, command: str) -> Hint | None:
        return self._hints.get(command)

    def save(self) -> None:
        """Write the whole record or none of it (`durable.write_atomic`), so a reader never
        sees half of it and a crash never names bytes that did not reach the disk."""
        body = {"runs": self.runs, "hints": [asdict(h) for h in self._hints.values()]}
        write_atomic(self.path, (json.dumps(body, indent=2, sort_keys=True) + "\n").encode())
