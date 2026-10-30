"""The E1 loop: seal → fresh worktree → dispatch arm → run tests → record.

Five steps, and four of them exist to stop the measurement from flattering itself.

**Seal before problem 1, or abort.** `require_sandbox=true` means a run
that cannot enforce isolation stops rather than quietly continuing at `sandbox=none`.
Runtime contamination was measured at 63% of one frontier model's SWE-bench Pro
resolutions and sealing cost it 14.1pp, so an unsealed number is not a number. Aborting
*before* the first problem matters: aborting halfway leaves a partial set whose absences
are correlated with whatever broke.

**Fresh worktree per problem.** No state carries between problems inside an arm, so
a solution cannot be assembled across two attempts and a leftover `__pycache__` cannot make
the next problem pass.

**Arms are injected, never imported.** The runner takes an `ArmDriver` callable, so the E1
control and the Chickadee arm plug in without this module knowing either
exists — and the loop is testable with a fake arm rather than a model.

**Caps are checked against spend the arm reports.** Every manifest arm carries a
finite `hard_cost_usd`; crossing it ends the run with a recorded terminal rather than an
overspend nobody notices until the invoice.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from chickadee.schema import SandboxLevel
from chickadee_eval import toolchain
from chickadee_eval.manifest import Arm, Language
from chickadee_eval.problems import polyglot
from chickadee_eval.problems.polyglot import LoadedProblem, Outcome, PromptView

# Variables that would let a sealed subprocess reach the network or spend money. Removed
# rather than blanked: a tool that reads `HTTPS_PROXY=""` may treat it as "no proxy" or as
# a malformed one, and the two behave differently.
NETWORK_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "no_proxy",
)
CREDENTIAL_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

# Kept from the parent so a sealed interpreter still starts. Everything else is dropped.
ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SystemRoot", "NODE_PATH")

PROBE_TIMEOUT_S = 5


class SandboxUnavailable(RuntimeError):
    """`require_sandbox=true` and no backend can enforce it. Raised before problem 1."""


class CapExceeded(RuntimeError):
    """An arm's declared `hard_cost_usd` was crossed. Terminal, and recorded."""


class RunTerminal(StrEnum):
    COMPLETE = "complete"
    CAP = "cap"
    ABORTED = "aborted"


ARM_ISOLATION_IN_PROCESS = (
    "none — the arm is an injected in-process callable (§8) and is not wrapped. It needs "
    "its own model endpoint, so a network namespace around it would break the thing being "
    "measured. What is sealed is the execution of benchmark code and the oracle's absence "
    "from the arm's workspace."
)


@dataclass(frozen=True)
class Seal:
    """What isolation was actually achieved, stated rather than assumed.

    Three fields where the first version had one, because they are three different claims
    and collapsing them let a row read `sandbox=native` when nothing had been wrapped:

    - `available` — what this machine *could* enforce, from executable discovery
    - `applied` — what actually wraps the benchmark's own test process
    - `arm_isolation` — prose, because the honest answer for an in-process arm is not a
      `SandboxLevel` value

    `network_verified` stays separate from all three: a backend can be present, and applied,
    and still not prove a socket fails. A run that could not probe says so, exactly as a
    skipped gate records its reason — the alternative is a report that reads
    identically whether or not sealing worked.
    """

    available: SandboxLevel
    applied: SandboxLevel
    arm_isolation: str
    network_verified: bool
    verification: str
    env: Mapping[str, str]

    @property
    def enforced(self) -> bool:
        """Whether benchmark code actually ran wrapped. Reads `applied`, never `available`."""
        return self.applied is not SandboxLevel.NONE


def _scrub(parent: Mapping[str, str]) -> dict[str, str]:
    keep = {k: v for k, v in parent.items() if k in ENV_KEEP}
    for k in list(keep):
        if k in NETWORK_ENV or any(m in k.upper() for m in CREDENTIAL_ENV_MARKERS):
            del keep[k]
    return keep


def available_backend() -> SandboxLevel:
    """What this machine could enforce. **Availability, not application** — `wrap()` is what
    turns this into isolation, and `Seal.applied` is what a record may claim."""
    if shutil.which("bwrap"):
        return SandboxLevel.CONTAINER
    if shutil.which("unshare"):
        return SandboxLevel.NATIVE
    return SandboxLevel.NONE


def wrap(argv: Sequence[str], level: SandboxLevel) -> list[str]:
    """Prefix `argv` so it runs inside the named sandbox.

    `unshare -rn` maps the caller to root in a new user namespace and gives the child a
    fresh network namespace with only a down loopback — no route, so `connect()` fails at
    the kernel rather than at a proxy that could be reconfigured. `-r` is what makes it work
    without privileges.
    """
    if level is SandboxLevel.CONTAINER:
        return ["bwrap", "--unshare-net", "--dev-bind", "/", "/", *argv]
    if level is SandboxLevel.NATIVE:
        return ["unshare", "-rn", *argv]
    return list(argv)


def _wrap_works(level: SandboxLevel, env: Mapping[str, str]) -> bool:
    """Whether the backend actually runs a child here.

    Present on PATH is not the same as usable: `unshare` needs unprivileged user namespaces,
    which some kernels and container runtimes disable. Discovering that per problem would
    turn one policy failure into N scoring failures, so it is settled once at seal time.
    """
    py = shutil.which("python3")
    if py is None:
        return False
    try:
        proc = subprocess.run(
            wrap([py, "-c", "pass"], level),
            env=dict(env),
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _probe_network(env: Mapping[str, str], level: SandboxLevel) -> tuple[bool, str]:
    """Try to prove a sealed child cannot reach the network.

    Probes **through the same wrapper the benchmark's tests get**, so the answer describes
    the isolation actually applied rather than a bare subprocess that happens to share the
    parent's env. Probing unwrapped was the first version's mistake: it measured the env
    scrub and reported it as though it measured the sandbox.

    A positive result is worth having and a negative one must not be guessed at, so this
    reports three states and never two: proven unreachable, proven reachable, or not
    established.
    """
    py = shutil.which("python3")
    if py is None:
        return False, "no python3 to probe with"
    code = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('example.com', 443), timeout=3).close()\n"
        "except Exception:\n"
        "    sys.exit(0)\n"
        "sys.exit(7)\n"
    )
    try:
        proc = subprocess.run(
            wrap([py, "-c", code], level),
            env=dict(env),
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"probe did not complete: {type(exc).__name__}"
    if proc.returncode == 0:
        return True, "connect() to example.com:443 failed inside the sealed environment"
    if proc.returncode == 7:
        return False, "connect() SUCCEEDED — the environment is not sealed"
    return False, f"probe exited {proc.returncode}, inconclusive"


def seal(*, require_sandbox: bool = True, probe: bool = True) -> Seal:
    """Establish isolation, or refuse to start.

    `require_sandbox` defaults to true because that is what an eval config sets, and
    because defaulting it the other way would make the safest-looking call the least safe
    one.
    """
    env = _scrub(os.environ)
    available = available_backend()
    # Availability is not application, and the gap between them is where the first version
    # reported `sandbox=native` over an unwrapped process. A backend that cannot actually
    # start a child here is treated as absent.
    applied = available if _wrap_works(available, env) else SandboxLevel.NONE
    if require_sandbox and applied is SandboxLevel.NONE:
        raise SandboxUnavailable(
            f"no sandbox could be applied (available={available}) and require_sandbox=true: "
            "the run aborts before problem 1 rather than degrading to sandbox=none "
            "(D22, EX-2). An unsealed number is not a number."
        )
    verified, why = (
        _probe_network(env, applied) if probe else (False, "probe disabled by caller")
    )
    return Seal(
        available=available,
        applied=applied,
        arm_isolation=ARM_ISOLATION_IN_PROCESS,
        network_verified=verified,
        verification=why,
        env=env,
    )


@dataclass(frozen=True)
class Attempt:
    """What an arm reports back. Cost is required — an unpriced arm cannot be compared."""

    solved_claim: bool
    """The arm's own belief. The tests decide; this is recorded to compare the two."""
    cost_usd: float
    attempts: int = 1
    prompts: tuple[str, ...] = ()
    """Every prompt rendered, kept so EV-3 can assert D7 over the real run, not a fixture."""
    detail: str = ""


class ArmDriver(Protocol):
    """One approach under test. `workspace` already holds the fresh problem copy."""

    @property
    def name(self) -> str: ...

    def attempt(self, problem: PromptView, workspace: Path) -> Attempt: ...


@dataclass
class ProblemRecord:
    """One row of `results.jsonl`."""

    problem_id: str
    language: str
    arm: str
    outcome: str
    solved: bool
    solved_claim: bool
    exit_code: int
    attempts: int
    cost_usd: float
    wall_ms: int
    sandbox: str
    network_verified: bool
    detail: str = ""
    oracle_planted: bool = False
    """The arm wrote a file where the hidden tests go. Overwritten either way; recorded
    because attempting it is itself a finding."""
    deps_digest: str = ""
    """Which pinned dependency tree ran this problem, or empty when the language needs
    none. A JavaScript result scored under a different jest is a different result, and a row
    that cannot say which one ran is a number without a provenance."""
    leaked_line_numbers: list[int] = field(default_factory=list[int])
    """Which lines of the hidden file reached a prompt — **numbers, never the text**. The
    first version serialized the matched lines, so a leak wrote the oracle it had just
    detected into `results.jsonl`, and an archive that anyone may share carried the answer
    key. A number identifies the line for diagnosis without reproducing it."""

    @property
    def leaked(self) -> bool:
        return bool(self.leaked_line_numbers)

    @property
    def counts_in_denominator(self) -> bool:
        """`infra_skip` leaves the denominator: the problem was never attempted, and
        counting it as unsolved would understate the solve rate for an infrastructure
        reason."""
        return self.outcome != Outcome.INFRA_SKIP


@contextmanager
def fresh_worktree(
    problem: LoadedProblem, stores: Mapping[Language, toolchain.Store] | None = None
) -> Iterator[Path]:
    """A private copy of the problem **without its hidden tests**.

    The first version copied the whole problem root, including the oracle, and handed that
    directory to the arm. Scanning `Attempt.prompts` afterwards did not close the hole it
    opened: an arm that reads `lib_test.py` off disk, hard-codes exactly what it demands and
    reports no prompt at all scores `solved` with nothing flagged. The prompt scan protects
    the loader-to-prompt path; this protects the only other one.

    So the tests are absent while the arm works and are written in for scoring by
    `restore_oracle()`. `PromptView` deliberately carries no `root`, so the arm is not handed
    a path back to the original either.

    **And the reference solution is absent too.** Deleting the test file was not enough: the
    real corpus ships `.meta/example.py` and `.meta/proof.ci.js` — the complete worked
    solution — plus `.meta/tests.toml` enumerating every case. Copying the problem root put
    all of it in the arm's workspace. An arm that reads `.meta/example.py`, writes it to the
    stub and reports no prompt at all scores `solved` with nothing flagged, because
    `leaks()` scans prompts for hidden *test body* lines and a solution is neither a test nor
    a body. That is a larger hole than the one this function was written to close, and it was
    invisible for the same reason as #60: the fixtures these tests use have no `.meta/`.

    Private directories are therefore dropped wholesale, using the same `is_private`
    predicate the loader uses to keep the answer key out of stub selection. Dot-*files* at
    the problem root are kept — `.npmrc`, `.eslintrc` and `.babelrc` are part of how the
    tests run.

    `stores` supplies pre-provisioned dependencies. They are linked in after the
    copy, so a problem shipping its own `node_modules` keeps it.
    """
    # A Polyglot id is a path (`python/exercises/practice/grep`), and a prefix with a `/` names
    # a directory that does not exist: every real problem failed here before its arm started.
    tmp = Path(tempfile.mkdtemp(prefix=f"e1-{problem.id.replace('/', '-')}-"))

    def ignore(directory: str, names: list[str]) -> set[str]:
        rel = Path(directory).relative_to(problem.root)
        base = rel.parts[0] if rel.parts else ""
        if base.startswith("."):
            return set(names)
        return {n for n in names if n.startswith(".") and (Path(directory) / n).is_dir()}

    try:
        shutil.copytree(problem.root, tmp, dirs_exist_ok=True, ignore=ignore)
        (tmp / problem.test_path).unlink(missing_ok=True)
        store = (stores or {}).get(problem.language)
        if store is not None:
            toolchain.link_into(tmp, store)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def restore_oracle(workspace: Path, problem: LoadedProblem) -> bool:
    """Write the hidden tests in for scoring. Returns True if the arm had planted a file
    where the oracle goes.

    An arm cannot fake the oracle — this always overwrites — but a file appearing at that
    path is an arm trying to, and that is worth recording rather than silently clobbering.
    """
    dst = workspace / problem.test_path
    planted = dst.exists()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(problem.test_source(), encoding="utf-8")
    return planted


def score_hidden(
    problem: LoadedProblem, workspace: Path, sealed: Seal, *, timeout_s: int = 120
) -> tuple[bool, str]:
    """The hidden tests run against `workspace` mid-attempt, for an arm's retry: `(passed,
    output)`. The oracle is written in, run under the seal, and **removed** before this
    returns: an arm whose model can read files would otherwise read the
    tests on its next turn, and no prompt scan sees a read. A file the arm had put where the
    oracle goes is put back afterwards, as `restore_oracle` would find it at the end."""
    dst = workspace / problem.test_path
    planted = dst.read_bytes() if dst.is_file() else None
    restore_oracle(workspace, problem)
    try:
        result = polyglot.run_tests(
            workspace,
            problem,
            env=sealed.env,
            timeout_s=timeout_s,
            wrapper=wrap((), sealed.applied),
        )
    finally:
        if planted is None:
            dst.unlink(missing_ok=True)
        else:
            dst.write_bytes(planted)
    output = "\n".join(part for part in (result.stdout, result.stderr, result.detail) if part)
    return result.outcome is Outcome.PASS, output


def run_problem(
    problem: LoadedProblem,
    driver: ArmDriver,
    sealed: Seal,
    *,
    timeout_s: int = 120,
    stores: Mapping[Language, toolchain.Store] | None = None,
) -> ProblemRecord:
    """Seal is already held; do the middle three steps for one problem."""
    started = time.monotonic()
    if not polyglot.toolchain_available(problem.language):
        return ProblemRecord(
            problem_id=problem.id,
            language=problem.language,
            arm=driver.name,
            outcome=Outcome.INFRA_SKIP,
            solved=False,
            solved_claim=False,
            exit_code=-1,
            attempts=0,
            cost_usd=0.0,
            wall_ms=0,
            sandbox=sealed.applied,
            network_verified=sealed.network_verified,
            detail=f"{polyglot.TOOLCHAINS[problem.language].binary} not on PATH",
        )

    with fresh_worktree(problem, stores) as ws:
        # The arm works without the oracle present. It is written in only now, for
        # scoring, so reading it off disk is not a route the arm ever had.
        attempt = driver.attempt(problem.prompt_view(), ws)
        planted = restore_oracle(ws, problem)
        # EV-3 over the real run rather than a fixture.
        leaked = sorted({n for p in attempt.prompts for n, _ in polyglot.leaks(p, problem)})
        result = polyglot.run_tests(
            ws,
            problem,
            env=sealed.env,
            timeout_s=timeout_s,
            wrapper=wrap((), sealed.applied),
        )

    return ProblemRecord(
        problem_id=problem.id,
        language=problem.language,
        arm=driver.name,
        outcome=result.outcome,
        solved=result.outcome is Outcome.PASS,
        solved_claim=attempt.solved_claim,
        exit_code=result.exit_code,
        attempts=attempt.attempts,
        cost_usd=attempt.cost_usd,
        wall_ms=int((time.monotonic() - started) * 1000),
        sandbox=sealed.applied,
        network_verified=sealed.network_verified,
        detail=result.detail or attempt.detail,
        oracle_planted=planted,
        deps_digest=store.digest if (store := (stores or {}).get(problem.language)) else "",
        leaked_line_numbers=leaked,
    )


@dataclass
class RunSummary:
    terminal: RunTerminal
    records: list[ProblemRecord]
    spent_usd: float
    seal: Seal
    leaked: bool

    @property
    def solved(self) -> int:
        return sum(1 for r in self.records if r.solved)

    @property
    def denominator(self) -> int:
        return sum(1 for r in self.records if r.counts_in_denominator)

    @property
    def infra_skipped(self) -> int:
        return sum(1 for r in self.records if not r.counts_in_denominator)


class TestBodyLeak(RuntimeError):
    """A leak fails the run loudly and its results are discarded."""


def run_arm(
    problems: Sequence[LoadedProblem],
    driver: ArmDriver,
    arm: Arm,
    *,
    sealed: Seal | None = None,
    require_sandbox: bool = True,
    timeout_s: int = 120,
    results_path: Path | None = None,
    stores: Mapping[Language, toolchain.Store] | None = None,
) -> RunSummary:
    """The whole loop for one arm.

    Raises `TestBodyLeak` rather than returning it: §7 says a leak discards the results, and
    a caller that received a summary could archive it before noticing the flag.
    """
    sealed = sealed or seal(require_sandbox=require_sandbox)
    records: list[ProblemRecord] = []
    spent = 0.0
    terminal = RunTerminal.COMPLETE

    handle = results_path.open("a", encoding="utf-8") if results_path else None
    try:
        for problem in problems:
            record = run_problem(problem, driver, sealed, timeout_s=timeout_s, stores=stores)
            # Checked before anything is appended. The first version wrote and flushed the
            # row first, so the "results discarded" left the invalid row on disk — and,
            # because the row carried the matched text, the oracle with it.
            if record.leaked:
                raise TestBodyLeak(
                    f"{record.problem_id}: hidden test line(s) "
                    f"{record.leaked_line_numbers} of {problem.test_path} reached a "
                    "rendered prompt (D7, §5.1). Results discarded; nothing was written."
                )
            records.append(record)
            if handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                handle.flush()
            spent += record.cost_usd
            if spent > arm.hard_cost_usd:
                # Recorded, not raised: §5.9 makes a cap terminal a scored outcome under the
                # preregistered missing rule, not an exception the caller has to interpret.
                terminal = RunTerminal.CAP
                break
    finally:
        if handle:
            handle.close()

    return RunSummary(
        terminal=terminal, records=records, spent_usd=spent, seal=sealed, leaked=False
    )


def load_all(root: Path, entries: Iterable[tuple[str, Language]]) -> list[LoadedProblem]:
    """Load a manifest subset's problems, in the order given (problem order is fixed)."""
    return [polyglot.load(root, pid, lang) for pid, lang in entries]
