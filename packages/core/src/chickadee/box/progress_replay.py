"""What the progress watchdog would have done on runs already recorded: its thresholds
checked against real rounds for false alarms, at no model's cost.

    python -m chickadee.box.progress_replay [REPO] [--run RUN_ID...] [--set NAME=VALUE...]

Each worker round of each run is rebuilt from its journal as a resumed round is
(`TaskProgress.rebuilt`) and replayed through the ladder (`hooks.watchdog.replay`), whether or
not the watchdog was bound when it ran. For each round: its calls, how it ended, where the
watchdog would have nudged and handed it up, and the calls made after that point, which a
hand-up would have saved if the stall was real. Reading a round as a real stall or a false
alarm is the reader's: the tool says where to look.

The thresholds are checked, never fitted `--set` tries another
value, and a default moves only with a stated reason.

Nothing is written and no model is called. A round whose trees or blobs are gone (a `gc`, a
swept snapshot) is named as unreadable, never guessed at.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from chickadee.box.progress import TaskProgress
from chickadee.hooks.payloads import WorkerProgress
from chickadee.hooks.watchdog import WatchdogParams, replay
from chickadee.journal import paths
from chickadee.journal.errors import JournalError
from chickadee.journal.projection import RunProjection
from chickadee.limits import limit
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile
from chickadee.roles import TASK_WORKER
from chickadee.schema.journal import TaskDelegatedP, TaskReportedP
from chickadee.state.roots import Roots, roots

__all__ = ["RoundReplay", "main", "render", "replay_run"]


@dataclass(frozen=True)
class RoundReplay:
    """One worker round, replayed."""

    run: str
    task: str
    round: int
    calls: int
    ended: str | None
    """How the round ended, from its report line; None for one still open."""
    rungs: tuple[tuple[str, str, int], ...] = ()
    """Each rung the watchdog would have given: its name, its meter, the call it came after."""
    changes: int = 0
    longest: int = 0
    calls_left: int | None = None
    unreadable: str | None = None
    """Why the task could not be replayed, when it could not."""

    @property
    def after_hand_up(self) -> int:
        """The calls made after the watchdog would have handed the round up: what a real
        stall cost, or, for a false alarm, the work it would have cut."""
        up = [at for rung, _, at in self.rungs if rung != "nudge"]
        return self.calls - up[0] if up else 0


def replay_run(
    run: RunProjection,
    run_id: str,
    blob: Callable[[str], bytes],
    root: Path,
    params: WatchdogParams,
    *,
    profiles: Sequence[LanguageProfile] = (),
    call_cap: int | None = None,
) -> list[RoundReplay]:
    """Every worker round of `run`, replayed under `params`. `blob` reads the run's blobs;
    `root` is the repository whose objects hold its trees."""
    out: list[RoundReplay] = []
    for task, lines in run.tasks.items():
        if not any(isinstance(p.payload, TaskDelegatedP) for p in lines):
            continue
        ended = {
            p.payload.round: p.payload.stopped
            for p in lines
            if isinstance(p.payload, TaskReportedP)
        }
        rounds: list[tuple[int, WorkerProgress]] = []
        try:
            TaskProgress.rebuilt(
                run,
                task,
                blob,
                root,
                profiles=profiles,
                call_cap=call_cap,
                registry_name=TASK_WORKER.registry_name,
                on_round=lambda r, p, kept=rounds: kept.append((r, p)),
            )
        except Exception as exc:  # a diagnostic says what it could not read, and goes on
            out.append(RoundReplay(run_id, task, 0, 0, None, unreadable=_why(exc)))
            continue
        out.extend(_replayed(run_id, task, r, p, ended.get(r), params) for r, p in rounds)
    return out


def _why(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0]


def _replayed(
    run_id: str,
    task: str,
    round_: int,
    progress: WorkerProgress,
    ended: str | None,
    params: WatchdogParams,
) -> RoundReplay:
    ladder = replay(progress, params)
    # A rung is given after a response's last call: where it lands, counted in calls.
    last_call = {c.response: i for i, c in enumerate(progress.calls, start=1)}
    return RoundReplay(
        run_id,
        task,
        round_,
        len(progress.calls),
        ended,
        tuple((d.rung, d.meter, last_call[d.response]) for d in ladder.due),
        ladder.changes,
        ladder.longest,
        progress.calls_left,
    )


def render(rows: Sequence[RoundReplay], params: WatchdogParams) -> str:
    """The rounds, then what they add up to."""
    lines: list[str] = []
    run: str | None = None
    for row in rows:
        if row.run != run:
            run = row.run
            lines.append(f"run {run}")
        if row.unreadable is not None:
            lines.append(f"  {row.task}: not replayed: {row.unreadable}")
            continue
        left = "" if row.calls_left is None else f", {row.calls_left} calls left at its start"
        lines.append(
            f"  {row.task} round {row.round}: {row.calls} calls, ended {row.ended or 'open'}"
            f"{left}; {row.changes} change(s), longest stretch without one {row.longest}"
        )
        if row.rungs:
            said = "; ".join(f"{r} ({meter}) after call {at}" for r, meter, at in row.rungs)
            cut = row.after_hand_up
            lines.append(f"    {said}" + (f"; {cut} call(s) after the hand-up" if cut else ""))
    read = [r for r in rows if r.unreadable is None]
    up = [r for r in read if any(rung != "nudge" for rung, _, _ in r.rungs)]
    nudged = [r for r in read if r.rungs]
    stuck = [r for r in read if r.ended in ("looped", "limit") and r not in up]
    lines += [
        "",
        f"thresholds: {params.model_dump_json()}",
        f"{len(read)} round(s) replayed, {len(rows) - len(read)} task(s) unreadable",
        f"{len(nudged)} nudged, {len(up)} handed up, "
        f"{sum(r.after_hand_up for r in up)} call(s) made after a hand-up point",
        f"{len(stuck)} round(s) ended looped or at a limit with no hand-up: "
        "look there for what the watchdog misses",
    ]
    return "\n".join(lines)


def _params(pairs: Sequence[str]) -> WatchdogParams:
    """`NAME=VALUE` pairs over the defaults, typed by the parameter model."""
    raw: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--set takes NAME=VALUE, got {pair!r}")
        raw[name.strip()] = value.strip()
    return WatchdogParams.model_validate(raw, strict=False)


def _runs(found: Roots, named: Sequence[str]) -> list[str]:
    if named:
        return list(named)
    top = paths.runs_root(found)
    if not top.is_dir():
        return []
    return sorted(d.name for d in top.iterdir() if (d / paths.JOURNAL_NAME).is_file())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chickadee.box.progress_replay",
        description="Replay the progress watchdog over a repository's recorded runs.",
    )
    parser.add_argument("repo", nargs="?", default=".", help="the repository (default: here)")
    parser.add_argument("--run", action="append", default=[], help="a run id; repeatable")
    parser.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE", help="a threshold"
    )
    parser.add_argument(
        "--cap", type=int, default=None, help="the task's call cap (default: the register's)"
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    try:
        params = _params(args.set)
    except ValueError as exc:
        print(f"progress_replay: {exc}", file=sys.stderr)
        return 2
    found = roots(repo)
    cap = args.cap if args.cap is not None else limit("worker.max_tool_calls").value
    profiles = ProfileSet.load(repo).all()
    # The one fold a reader outside an engine uses (`chickadee replay`'s, C2).
    from chickadee.engine import replay_run as folded

    rows: list[RoundReplay] = []
    for run_id in _runs(found, args.run):
        try:
            run = folded(repo, run_id)
        except (JournalError, OSError, ValueError) as exc:
            rows.append(RoundReplay(run_id, "-", 0, 0, None, unreadable=_why(exc)))
            continue
        # Read as the other read-only readers read a run's blobs (`context/project.py`,
        # `inspect.py`): from its directory, never through the writer's `BlobStore`.
        where = paths.blobs_dir(found, run_id)

        def blob(sha: str, where: Path = where) -> bytes:
            return (where / sha).read_bytes()

        rows += replay_run(
            run, run_id, blob, repo, params, profiles=profiles, call_cap=int(cap)
        )
    if not rows:
        print(f"progress_replay: no worker rounds recorded under {paths.runs_root(found)}")
        return 1
    print(render(rows, params))
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line
    raise SystemExit(main())
