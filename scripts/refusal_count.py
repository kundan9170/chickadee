#!/usr/bin/env python3
"""Count what a run's refusals cost (D209 decision 5): run on the owner's before and after runs
of the same task on the same model, so the table in `tools/foreign.py` grows only if the count
drops.

    python3 scripts/refusal_count.py <run's journal.jsonl | its run directory> [--role worker]

It reads the journal's `tool_call` lines, in order, and prints, per role:

- the refused calls, by failure (`denied_reason`'s first word: `UNKNOWN_TOOL`, `DENIED`, ...);
- the identical consecutive repeats: a call with the tool and arguments (`args_sha`) of the
  call before it;
- for each path, the calls spent before its first successful `read` (the role's calls before
  it, refused or not).

Standard library only, and it reads nothing but the journal: a run is never re-run to count it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

_FAILURE = re.compile(r"[A-Z][A-Z_]+")


@dataclass(frozen=True)
class Call:
    role: str
    tool: str
    args_sha: str
    denied: str | None
    args: Mapping[str, object] | None
    """The arguments as the journal's preview holds them; None when the preview was cut."""


@dataclass
class Count:
    calls: int = 0
    refused: Counter[str] = field(default_factory=Counter[str])
    repeats: int = 0
    first_read: dict[str, int] = field(default_factory=dict[str, int])
    """Each path, and the role's calls before its first successful `read`."""


def calls(lines: Iterable[str]) -> Iterator[Call]:
    """The journal's tool calls, in order. A line that is not JSON (a torn tail) is skipped."""
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("kind") != "tool_call":
            continue
        p = record.get("payload") or {}
        yield Call(
            role=str(p.get("role", "")),
            tool=str(p.get("tool", "")),
            args_sha=str(p.get("args_sha", "")),
            denied=p.get("denied_reason"),
            args=_args(p.get("args_preview")),
        )


def _args(preview: object) -> Mapping[str, object] | None:
    if not isinstance(preview, str):
        return None
    try:
        args = json.loads(preview)
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def failure(denied: str) -> str:
    """A refusal's kind: the first upper-case word of what the model was told."""
    found = _FAILURE.match(denied.strip())
    return found.group(0) if found else "OTHER"


def count(journal: Iterable[Call], role: str | None = None) -> dict[str, Count]:
    out: dict[str, Count] = {}
    last: dict[str, tuple[str, str]] = {}
    for call in journal:
        if role is not None and call.role != role:
            continue
        c = out.setdefault(call.role, Count())
        key = (call.tool, call.args_sha)
        if last.get(call.role) == key:
            c.repeats += 1
        last[call.role] = key
        if call.denied is not None:
            c.refused[failure(call.denied)] += 1
        elif call.tool == "read" and call.args is not None:
            path = call.args.get("path")
            if isinstance(path, str) and path not in c.first_read:
                c.first_read[path] = c.calls
        c.calls += 1
    return out


def render(counts: Mapping[str, Count]) -> str:
    rows: list[str] = []
    for role, c in counts.items():
        refused = sum(c.refused.values())
        rows.append(
            f"{role}: {c.calls} calls, {refused} refused, {c.repeats} identical repeats"
        )
        for kind, n in c.refused.most_common():
            rows.append(f"  refused {kind}: {n}")
        for path, before in c.first_read.items():
            rows.append(f"  first read of {path}: after {before} calls")
    return "\n".join(rows) if rows else "no tool calls"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count what a run's refusals cost (D209).")
    parser.add_argument(
        "journal", type=Path, help="a run's journal.jsonl, or its run directory"
    )
    parser.add_argument("--role", help="count one role's calls only (worker, athena, ...)")
    args = parser.parse_args(argv)
    path: Path = args.journal
    if path.is_dir():
        path = path / "journal.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2
    print(render(count(calls(text.splitlines()), args.role)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
