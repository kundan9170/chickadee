"""`chickadee-eval`: one command to measure the harness before and after a change.

    chickadee-eval run --label before # the smoke subset, into a new archive
    chickadee-eval run --subset e1-dev --label after
    chickadee-eval diff before after # solve rate, cost, the problems that flipped
    chickadee-eval list
    chickadee-eval research --judge <model> --label before # the researcher

**A run is the complete system on real problems** (`arms/chickadee.py`): each problem opens a
session with the person's own config, and the hidden tests decide. The archive records which
harness ran (its commit, and a digest of any uncommitted change) and which models (the
resolved engine config), so two runs that differ in either are two archives, never one.

**Archives live outside the repository being measured** (`~/.chickadee-eval`, or
`CKD_EVAL_HOME`), so `list` and `diff` work from anywhere and a run never leaves files in
the harness's own checkout.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chickadee.engine as engine_facade
from chickadee.engine import (
    ChickadeeError,
    HostConfig,
    LayerReadError,
    MergeError,
    load_host_config,
)
from chickadee_eval import archive, report, toolchain
from chickadee_eval.arms.chickadee import ARM_NAME, ChickadeeArm, open_host_engine
from chickadee_eval.differ import ArchiveError, compare_archives, load_records, summarize
from chickadee_eval.manifest import Manifest, Problem, Subset
from chickadee_eval.metrics import aggregate
from chickadee_eval.problems.polyglot import ProblemError
from chickadee_eval.research import score as research
from chickadee_eval.research.judge import ModelJudge, check_judge
from chickadee_eval.research.questions import QUESTIONS, load_questions
from chickadee_eval.research.run import REPORTS_NAME, ResearchRun
from chickadee_eval.runner import (
    RunTerminal,
    SandboxUnavailable,
    TestBodyLeak,
    load_all,
    run_arm,
    score_hidden,
    seal,
)

MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "e1.toml"
DEFAULT_SUBSET = "e1-smoke"
PROBLEMS_NAME = "problems"
"""Each problem's kept files under its id: test output, diff, journals (`ChickadeeArm.keep`)."""
RESEARCH_KIND = "research"
"""An archive's `kind` when the researcher's evaluation made it: `diff` compares two
of a kind."""
STATS_NAME = "chickadee.jsonl"
"""Each problem's `ChickadeeStats`, beside `results.jsonl`: what the system did on the way."""

# Not `CHICKADEE_*`: that prefix is the config's environment layer, which reads every such
# variable as a config key and refuses one it does not know, so the run would not start.
POLYGLOT_ENV = "CKD_POLYGLOT_ROOT"
HOME_ENV = "CKD_EVAL_HOME"
POLYGLOT_CLONE = "git clone https://github.com/Aider-AI/polyglot-benchmark"


class CommandError(Exception):
    """What the person must fix before the command can run, in the words that say how."""


def results_home(given: Path | None) -> Path:
    if given is not None:
        return given
    env = os.environ.get(HOME_ENV)
    return Path(env) if env else Path.home() / ".chickadee-eval"


def load_manifest(path: Path = MANIFEST) -> Manifest:
    return Manifest(**tomllib.loads(path.read_text(encoding="utf-8")))


def pick_subset(manifest: Manifest, name: str) -> Subset:
    for subset in manifest.subsets:
        if subset.name == name:
            return subset
    names = ", ".join(s.name for s in manifest.subsets)
    raise CommandError(f"no subset {name!r} in the manifest. Do: pass --subset one of {names}")


def polyglot_root(given: Path | None) -> Path:
    root = given or (Path(env) if (env := os.environ.get(POLYGLOT_ENV)) else None)
    if root is None or not (root / "python" / "exercises" / "practice").is_dir():
        raise CommandError(
            f"no Polyglot benchmark checkout at {root or '(unset)'}. Do: `{POLYGLOT_CLONE}`, "
            f"then pass --polyglot-root <dir> or set {POLYGLOT_ENV}"
        )
    return root


def select(problems: Sequence[Problem], only: str | None, limit: int | None) -> list[Problem]:
    """`--problems` by id or by the exercise's name, then `--limit`, in the subset's order:
    chosen before any is loaded, so a run of one needs only that one on disk."""
    chosen = list(problems)
    if only:
        wanted = {w.strip() for w in only.split(",") if w.strip()}
        chosen = [p for p in chosen if p.id in wanted or p.id.rsplit("/", 1)[-1] in wanted]
        if not chosen:
            raise CommandError(
                f"--problems {only!r} names no problem in this subset. Do: pass ids or "
                "exercise names from `chickadee-eval list --problems`"
            )
    return chosen[:limit] if limit else chosen


def harness_identity() -> dict[str, Any]:
    """Which harness is measured: the commit of the checkout `chickadee` is imported from,
    and a digest of its uncommitted change, so an edit not yet committed is a new archive."""
    src = Path(engine_facade.__file__).resolve().parent

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=src, capture_output=True, text=True, check=False
        )
        return done.stdout if done.returncode == 0 else ""

    commit = git("rev-parse", "HEAD").strip()
    change = git("diff", "HEAD")
    return {
        "commit": commit or "unknown",
        "uncommitted": hashlib.sha256(change.encode()).hexdigest()[:12] if change else "",
    }


def engine_identity() -> dict[str, Any]:
    """The engine config a problem's session resolves (models, limits, prices), as JSON."""
    return host_config().engine.model_dump(mode="json")


def host_config() -> HostConfig:
    """The engine config a run's session resolves, read the way the arm reads it, from a
    scratch repository: the person's user config is what applies."""
    with tempfile.TemporaryDirectory(prefix="e1-config-") as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True, capture_output=True)
        return load_host_config(Path(tmp))


def find_archive(home: Path, name: str) -> Path:
    """An archive by path, by directory name, or by label (the newest with it)."""
    direct = Path(name)
    if archive.is_archive(direct):
        return direct
    found = archive.find(home)
    for path in reversed(found):
        if path.name == name or _config(path).get("label") == name:
            return path
    raise CommandError(
        f"no archive named or labelled {name!r} under {home / archive.RESULTS_ROOT}. Do: run "
        "`chickadee-eval list` and pass a label or a directory it shows"
    )


def _config(path: Path) -> dict[str, Any]:
    try:
        blob = json.loads((path / archive.CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return blob if isinstance(blob, dict) else {}


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    subset = pick_subset(manifest, args.subset)
    arm = next((a for a in manifest.arms if a.name == ARM_NAME), None)
    if arm is None:
        raise CommandError(
            f"the manifest declares no {ARM_NAME!r} arm, and its cost cap is there"
        )
    if args.cost_cap is not None:
        arm = arm.model_copy(update={"hard_cost_usd": args.cost_cap})
    root = polyglot_root(args.polyglot_root)
    home = results_home(args.results_root)
    chosen = select(subset.problems, args.problems, args.limit)
    problems = load_all(root, ((p.id, p.language) for p in chosen))

    harness = harness_identity()
    label = args.label or harness["commit"][:12]
    config = {
        "label": label,
        "arm": ARM_NAME,
        "subset": subset.name,
        "problems": [p.id for p in problems],
        "max_turns": args.max_turns,
        "harness": harness,
        "engine": engine_identity(),
    }
    stores = toolchain.prepare({p.language for p in problems}, home / "toolchains")
    try:
        sealed = seal(require_sandbox=not args.allow_unsandboxed)
    except SandboxUnavailable as exc:
        raise CommandError(
            f"{exc} Do: install bubblewrap or unshare, or pass --allow-unsandboxed to compare "
            "runs on your own machine"
        ) from exc
    if not sealed.enforced:
        print(
            "warning: the tests run unsandboxed (--allow-unsandboxed); the numbers are for "
            "your own comparison, not for a report (D22)",
            file=sys.stderr,
        )
    date = datetime.date.today().isoformat()
    try:
        out = archive.create(date, config, root=home, manifest=manifest.model_dump(mode="json"))
    except FileExistsError as exc:
        raise CommandError(
            f"{exc}. Do: pass a new --label; this harness and config already ran today"
        ) from exc

    oracles = {p.id: p for p in problems}
    driver = ChickadeeArm(
        open_engine=open_host_engine,
        oracles=oracles,
        score=lambda problem, ws: score_hidden(
            problem, ws, sealed, timeout_s=args.test_timeout
        ),
        max_turns=args.max_turns,
        turn_timeout_s=args.turn_timeout,
        keep=out / PROBLEMS_NAME,
    )
    print(f"{label}: {len(problems)} problems from {subset.name} -> {out}", flush=True)
    try:
        summary = run_arm(
            problems,
            _Announced(driver),
            arm,
            sealed=sealed,
            timeout_s=args.test_timeout,
            results_path=out / archive.RESULTS_NAME,
            stores=stores,
        )
    finally:
        with (out / STATS_NAME).open("w", encoding="utf-8") as handle:
            for stats in driver.stats:
                handle.write(json.dumps(stats.as_json(), sort_keys=True) + "\n")
    metrics = aggregate(summary.records)
    (out / archive.REPORT_NAME).write_text(
        report.render(metrics, sealed=sealed, title=f"E1 {subset.name}: {label}"),
        encoding="utf-8",
    )
    print(headline(out))
    print(f"  each problem's test output, diff and journals: {out / PROBLEMS_NAME}")
    if summary.terminal is not RunTerminal.COMPLETE:
        print(f"stopped early: {summary.terminal} (cap {arm.hard_cost_usd} USD)")
    return 0


class _Announced:
    """The arm, saying each problem's outcome as it ends: a run takes minutes per problem."""

    def __init__(self, driver: ChickadeeArm) -> None:
        self._driver = driver

    @property
    def name(self) -> str:
        return self._driver.name

    def attempt(self, problem: Any, workspace: Path) -> Any:
        result = self._driver.attempt(problem, workspace)
        # The arm's own scoring of its last turn; the runner's rerun is what the archive keeps.
        mark = "pass" if result.solved_claim else "fail"
        print(f"  {mark} {problem.id}  {result.detail}  ${result.cost_usd:.4f}", flush=True)
        return result


def headline(path: Path) -> str:
    """One archive in two lines: its label and subset, then solve rate, cost and the counts."""
    config = _config(path)
    if config.get("kind") == RESEARCH_KIND:
        s = research.summarize(_research_attempts(path))
        return (
            f"{path.name}  {config.get('label', '?')}  research\n"
            f"  {s.attempts} rounds  facts {s.facts.render()}  quotes {s.quoted.render()}  "
            f"preferred {s.preferred.render()}  cost {s.cost_usd:.4f} USD"
        )
    try:
        records = load_records(path)
    except ArchiveError as exc:
        return f"{path.name}  {config.get('label', '?')}  (no results: {exc})"
    metrics = aggregate(records)
    m = next(iter(metrics.values()))
    stats = _stats(path)
    refused = sum(int(s.get("refused_calls", 0)) for s in stats)
    rounds = sum(int(s.get("rounds", 0)) for s in stats)
    return (
        f"{path.name}  {config.get('label', '?')}  {config.get('subset', '?')}\n"
        f"  solved {m.solved}/{m.denominator} ({m.solve_rate.render(percent=True)})  "
        f"cost {m.total_cost_usd:.4f} USD  rounds {rounds}  refused calls {refused}"
    )


def _stats(path: Path) -> list[dict[str, Any]]:
    target = path / STATS_NAME
    if not target.is_file():
        return []
    return [
        json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line
    ]


def cmd_research(args: argparse.Namespace) -> int:
    """The researcher's evaluation: every question, `--repeats` times, each a
    round of its own, scored by the harness's citation check, the preferred hosts, and the
    judge."""
    try:
        chosen = load_questions(args.questions_file)
    except (OSError, ValueError) as exc:
        raise CommandError(
            f"the question set {args.questions_file} could not be read: {exc}. "
            "Do: pass --questions-file <a question set's .toml>"
        ) from exc
    try:
        questions = chosen.chosen(args.questions, args.limit)
    except ValueError as exc:
        raise CommandError(f"{exc}. Do: pass --questions ids from the question set") from exc
    host = host_config()
    if not args.no_judge:
        if not args.judge:
            raise CommandError(
                "no judge named. Do: pass --judge <a model of your engine config's [models]>, "
                "not the researcher's own, or --no-judge to score only the harness's checks"
            )
        try:
            check_judge(host, args.judge)
        except ValueError as exc:
            raise CommandError(f"{exc}. Do: pass --judge another model") from exc
    judge = None if args.no_judge else args.judge
    harness = harness_identity()
    label = args.label or harness["commit"][:12]
    config = {
        "kind": RESEARCH_KIND,
        "label": label,
        "questions": [q.id for q in questions],
        "question_set": chosen.version,
        "repeats": args.repeats,
        "judge": judge,
        "harness": harness,
        "engine": host.engine.model_dump(mode="json"),
    }
    home = results_home(args.results_root)
    try:
        out = archive.create(datetime.date.today().isoformat(), config, root=home)
    except FileExistsError as exc:
        raise CommandError(
            f"{exc}. Do: pass a new --label; this harness and config already ran today"
        ) from exc
    rounds = len(questions) * args.repeats
    print(f"{label}: {len(questions)} questions x {args.repeats} = {rounds} rounds -> {out}")
    run = ResearchRun(
        open_engine=open_host_engine,
        judge=None if judge is None else lambda engine: ModelJudge(engine, judge),
        repeats=args.repeats,
        say=lambda line: print(line, flush=True),
    )
    with tempfile.TemporaryDirectory(prefix="research-") as workspace:
        try:
            run.run(questions, Path(workspace), out)
        finally:
            summary = research.summarize(run.attempts)
            (out / archive.REPORT_NAME).write_text(
                summary.render(f"Researcher: {label}"), encoding="utf-8"
            )
    print(headline(out))
    print(f"  each round's report: {out / REPORTS_NAME}")
    return 0


def _research_attempts(path: Path) -> list[research.ResearchRound]:
    target = path / archive.RESULTS_NAME
    if not target.is_file():
        return []
    return [
        research.ResearchRound.from_json(json.loads(line))
        for line in target.read_text(encoding="utf-8").splitlines()
        if line
    ]


def cmd_diff(args: argparse.Namespace) -> int:
    home = results_home(args.results_root)
    before, after = find_archive(home, args.before), find_archive(home, args.after)
    kinds = {_config(before).get("kind"), _config(after).get("kind")}
    if RESEARCH_KIND in kinds:
        if kinds != {RESEARCH_KIND}:
            raise CommandError(
                "one archive is a research run and the other is not. Do: diff two of a kind"
            )
        lhs = research.summarize(_research_attempts(before))
        rhs = research.summarize(_research_attempts(after))
        print(research.compare(lhs, rhs))
        return 0
    diff = compare_archives(before, after, arm=ARM_NAME)
    lhs = aggregate(load_records(before)).get(ARM_NAME)
    rhs = aggregate(load_records(after)).get(ARM_NAME)
    if lhs is not None and rhs is not None:
        print(summarize(lhs, rhs))
    print(diff.render())
    print(_stats_delta(before, after))
    return 0


def _stats_delta(before: Path, after: Path) -> str:
    """The system's counts, before and after: what a harness change moves besides the rate."""
    keys = ("turns", "tasks", "rounds", "refused_calls", "model_calls")
    lhs, rhs = _stats(before), _stats(after)
    rows = [
        f"| {k} | {sum(int(s.get(k, 0)) for s in lhs)} | {sum(int(s.get(k, 0)) for s in rhs)} |"
        for k in keys
    ]
    return "| count | before | after |\n|---|---|---|\n" + "\n".join(rows)


def cmd_list(args: argparse.Namespace) -> int:
    if args.problems:
        manifest = load_manifest(args.manifest)
        for subset in manifest.subsets:
            print(f"{subset.name} ({subset.kind}, {subset.count})")
            for p in subset.problems:
                print(f"  {p.id}")
        return 0
    home = results_home(args.results_root)
    found = archive.find(home)
    if not found:
        print(f"no archives under {home / archive.RESULTS_ROOT}")
    for path in found:
        print(headline(path))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="chickadee-eval",
        description="Measure the harness before and after a change (D222).",
    )
    top.add_argument("--results-root", type=Path, help=f"archives' home (default ${HOME_ENV})")
    top.add_argument("--manifest", type=Path, default=MANIFEST)
    sub = top.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a subset into a new archive")
    run.add_argument("--subset", default=DEFAULT_SUBSET)
    run.add_argument("--label", help="a name to diff by (default: the harness commit)")
    run.add_argument("--problems", help="comma-separated ids or exercise names")
    run.add_argument("--limit", type=int)
    run.add_argument("--polyglot-root", type=Path, help=f"default ${POLYGLOT_ENV}")
    run.add_argument("--max-turns", type=int, default=3)
    run.add_argument(
        "--cost-cap", type=float, help="USD; the run stops past it (default: the arm's cap)"
    )
    run.add_argument("--turn-timeout", type=float, default=3600.0, help="seconds per turn")
    run.add_argument("--test-timeout", type=int, default=120, help="seconds per test run")
    run.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help="run the tests without a network sandbox (macOS has none)",
    )
    run.set_defaults(func=cmd_run)

    diff = sub.add_parser("diff", help="compare two archives")
    diff.add_argument("before", help="a label, an archive's directory name, or its path")
    diff.add_argument("after")
    diff.set_defaults(func=cmd_diff)

    research = sub.add_parser("research", help="the researcher's questions (D232)")
    research.add_argument("--label", help="a name to diff by (default: the harness commit)")
    research.add_argument("--judge", help="a model of your engine config's [models]")
    research.add_argument(
        "--no-judge", action="store_true", help="score only the harness's own checks"
    )
    research.add_argument("--questions", help="comma-separated question ids")
    research.add_argument("--limit", type=int)
    research.add_argument("--repeats", type=int, default=3, help="rounds a question (3)")
    research.add_argument("--questions-file", type=Path, default=QUESTIONS)
    research.set_defaults(func=cmd_research)

    listing = sub.add_parser("list", help="the archives, or with --problems the subsets")
    listing.add_argument("--problems", action="store_true")
    listing.set_defaults(func=cmd_list)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (CommandError, TestBodyLeak, ArchiveError, ProblemError) as exc:
        print(f"chickadee-eval: {exc}", file=sys.stderr)
        return 1
    except (ChickadeeError, LayerReadError, MergeError, toolchain.ProvisionError) as exc:
        print(f"chickadee-eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
