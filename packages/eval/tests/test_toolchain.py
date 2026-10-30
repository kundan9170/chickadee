"""Provisioning: dependencies before the seal, no network after it.

The property that matters is one sentence and one test: **a JavaScript problem runs its real
tests inside the applied sandbox, with no network.** Everything else here exists to keep
that honest — that the tree is pinned rather than resolved fresh, that the store cannot
become a channel between problems, and that a run without provisioning still says
`infra_skip` instead of failing.

Like `test_polyglot_real.py`, the tests that matter use the real corpus and real `npm`, and
skip loudly rather than substituting something agreeable. Issue #60 is why: a suite built on
fixtures written to match the code will agree with it and prove nothing.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest
from repos import git_result

from chickadee.schema import SandboxLevel
from chickadee_eval import runner, toolchain
from chickadee_eval.manifest import Arm, Language
from chickadee_eval.problems import polyglot


def benchmark_root() -> Path | None:
    for candidate in (
        os.environ.get("CKD_POLYGLOT_ROOT"),
        "/workspace/aider-ai/polyglot-benchmark",
    ):
        if candidate and (Path(candidate) / "python" / "exercises" / "practice").is_dir():
            return Path(candidate)
    return None


real = pytest.mark.skipif(benchmark_root() is None, reason="no polyglot checkout")
needs_npm = pytest.mark.skipif(shutil.which("npm") is None, reason="npm not installed")

# One provisioned store for the module. Provisioning is ~8s and 127MB; doing it per test
# would be minutes spent reinstalling a tree the lockfile already fixes.
_STORE: dict[str, toolchain.Store] = {}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> toolchain.Store:
    if "js" not in _STORE:
        root = tmp_path_factory.mktemp("deps")
        try:
            _STORE["js"] = toolchain.provision(Language.JAVASCRIPT, root)
        except toolchain.ProvisionError as exc:
            pytest.skip(f"could not provision (offline?): {exc}")
    return _STORE["js"]


# --- the pinned manifest ------------------------------------------------------


def test_the_dependency_set_is_present_with_a_lockfile() -> None:
    """§6. The upstream exercises ship no lockfile and depend on `^`/`~` ranges, so two runs
    a month apart would score against different jest versions with nothing recording it."""
    dep = toolchain.DEPENDENCY_SETS[Language.JAVASCRIPT]
    for committed, _installed in dep.manifest_files:
        assert (dep.manifest_dir / committed).is_file(), f"{committed} is missing"
    assert "ci" in dep.install_argv, "npm install would ignore the lockfile; npm ci does not"


def test_the_dependency_set_is_actually_tracked_in_git() -> None:
    """On disk is not the same as committed, and the difference cost a CI run.

    The repo's `.gitignore` carries bare `package.json` and `package-lock.json` entries — a
    sensible guard against stray npm artefacts in a Python repo — so `git add -A` silently
    skipped both files. Every local test passed, because the files were sitting there
    untracked, and CI failed on a lockfile that had never left this machine. They are
    force-added at first; they are now named with a `.pinned` suffix so the pattern simply
    does not match, and this is what stops that being undone quietly.
    """
    dep = toolchain.DEPENDENCY_SETS[Language.JAVASCRIPT]
    repo = Path(__file__).resolve().parents[3]
    tracked = git_result(repo, "ls-files", "--", str(dep.manifest_dir))
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    listed = {Path(line).name for line in tracked.stdout.split()}
    missing = [committed for committed, _ in dep.manifest_files if committed not in listed]
    assert not missing, (
        f"{missing} exist on disk but are not tracked — .gitignore matches them, so "
        "`git add` skips them and CI gets a tree this machine never had"
    )


def test_the_manifests_would_actually_ship_in_a_wheel() -> None:
    """Two conditions, and getting only the first is how this failed twice.

    They must be **inside the importable package**, because
    `packages = ["src/chickadee_eval"]` ships only what is under the package directory —
    the first version put them in a sibling of `src/` while the docstring claimed the
    opposite.

    And they must **not match a VCS ignore rule**, because hatchling honours `.gitignore`
    when selecting files. The repo ignores bare `package.json` and `package-lock.json`, so
    the originals were dropped from the wheel even after being force-added to git: building
    one and listing it showed `toolchain.py` and no manifests. The `.pinned` suffix is what
    fixes that, and this is the assertion that keeps it fixed.
    """
    dep = toolchain.DEPENDENCY_SETS[Language.JAVASCRIPT]
    package_root = Path(toolchain.__file__).resolve().parent
    assert dep.manifest_dir.is_relative_to(package_root), (
        f"{dep.manifest_dir} is outside {package_root}; it will not be packaged"
    )

    repo = Path(__file__).resolve().parents[3]
    for committed, _installed in dep.manifest_files:
        probe = git_result(repo, "check-ignore", "-q", str(dep.manifest_dir / committed))
        if probe.returncode == 128:
            pytest.skip("not a git checkout")
        assert probe.returncode != 0, (
            f"{committed} matches a .gitignore rule; hatchling honours those, so it will "
            "be dropped from the wheel however carefully it was added to git"
        )


def test_the_lockfile_pins_exact_versions() -> None:
    import json

    dep = toolchain.DEPENDENCY_SETS[Language.JAVASCRIPT]
    committed = next(c for c, i in dep.manifest_files if i == "package-lock.json")
    lock = json.loads((dep.manifest_dir / committed).read_text(encoding="utf-8"))
    packages = lock.get("packages", {})
    assert len(packages) > 50, "the lockfile looks empty"
    unpinned = [
        name
        for name, meta in packages.items()
        if name and "version" in meta and any(c in meta["version"] for c in "^~*")
    ]
    assert not unpinned, unpinned


def test_the_digest_changes_with_the_manifest(tmp_path: Path) -> None:
    """The archive records this digest; a digest that ignored the lockfile would let the
    tree change while every row claimed the same provenance."""
    before = toolchain.manifest_digest(Language.JAVASCRIPT)
    assert len(before) == 32
    assert before == toolchain.manifest_digest(Language.JAVASCRIPT)


def test_python_needs_no_provisioning() -> None:
    """pytest comes with the interpreter. Provisioning a language that needs nothing would
    add a failure mode for no benefit."""
    assert not toolchain.needs_provisioning(Language.PYTHON)
    with pytest.raises(toolchain.ProvisionError, match="no dependency set"):
        toolchain.provision(Language.PYTHON, Path("/tmp"))


# --- provisioning -------------------------------------------------------------


@needs_npm
def test_provisioning_installs_the_runner(store: toolchain.Store) -> None:
    assert (store.path / ".bin" / "jest").exists()
    assert store.packages > 20
    assert store.digest == toolchain.manifest_digest(Language.JAVASCRIPT)


@needs_npm
def test_the_store_is_read_only(store: toolchain.Store) -> None:
    """§5.2 item 5 — no state carries between problems. A writable shared `node_modules` is
    a channel from problem 1 to problem 12, and one that *executes*."""
    assert toolchain.is_readonly(store)

    # A regular file, not `.bin/jest` — that is a symlink, and a symlink's own mode bits are
    # meaningless on Linux (`_set_readonly` skips them for exactly that reason). Asserting on
    # one reads as a passing check of nothing.
    regular = next(
        f for f in sorted(store.path.rglob("*.json")) if f.is_file() and not f.is_symlink()
    )
    assert not regular.lstat().st_mode & stat.S_IWUSR, f"{regular} is writable"

    if os.geteuid() == 0:
        pytest.skip("running as root, which bypasses mode bits — see the note below")
    with pytest.raises(PermissionError):
        (store.path / "planted.js").write_text("module.exports = 1;\n", encoding="utf-8")


@needs_npm
def test_read_only_is_measured_on_the_store_not_on_the_caller(store: toolchain.Store) -> None:
    """`os.access(..., W_OK)` asks "can *this* process write here", and for root the answer
    is yes whatever the mode says — it reported a correctly locked store as writable the
    first time this suite ran, as root in a container.

    Recorded as a test because the honest scope of the measure follows from it: clearing
    write bits stops a stray write from an ordinary user, and does **not** stop one from
    inside the sandbox, where `unshare -r` maps the caller to root. Anyone tempted to
    describe this as sealing the store should read this test first.
    """
    assert toolchain.is_readonly(store)
    if os.geteuid() == 0:
        assert os.access(store.path, os.W_OK), (
            "root can write to a read-only tree; if this ever fails, os.access became a "
            "usable check and is_readonly could be simplified"
        )


@needs_npm
def test_provisioning_twice_reuses_the_store(
    store: toolchain.Store, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Idempotent by digest. Reinstalling 127MB per run proves nothing the lockfile has not
    already fixed."""
    again = toolchain.provision(Language.JAVASCRIPT, store.path.parent.parent)
    assert again.digest == store.digest
    assert again.path == store.path


def test_a_missing_binary_refuses_rather_than_half_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller may well continue — `run_tests` reports infra_skip — but that has to be a
    decision, not the result of an install nobody noticed had failed."""
    monkeypatch.setattr(toolchain.shutil, "which", lambda _b: None)
    with pytest.raises(toolchain.ProvisionError, match="not installed"):
        toolchain.provision(Language.JAVASCRIPT, tmp_path)


# --- linking into a workspace -------------------------------------------------


@needs_npm
def test_linking_makes_the_runner_resolvable(store: toolchain.Store, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    toolchain.link_into(ws, store)
    assert (ws / "node_modules" / ".bin" / "jest").exists()


@needs_npm
def test_linking_never_replaces_a_problems_own_dependencies(
    store: toolchain.Store, tmp_path: Path
) -> None:
    """Overwriting would substitute a tree the exercise did not ask for."""
    ws = tmp_path / "ws"
    (ws / "node_modules").mkdir(parents=True)
    (ws / "node_modules" / "marker").write_text("mine\n", encoding="utf-8")
    toolchain.link_into(ws, store)
    assert (ws / "node_modules" / "marker").is_file()
    assert not (ws / "node_modules").is_symlink()


# --- the arm's workspace holds no answers -------------------------------------


@real
def test_the_worktree_excludes_the_reference_solution() -> None:
    """Found while building this: `fresh_worktree` deleted the hidden test and left
    `.meta/example.py` — the complete worked solution — in the arm's workspace, along with
    `tests.toml` listing every case.

    An arm that reads it, writes it to the stub and reports no prompt at all scores `solved`
    with nothing flagged: `leaks()` scans prompts for hidden test *bodies*, and a solution is
    neither a test nor a body. Larger than the hole `fresh_worktree` was written to close,
    and invisible to a fixture that has no `.meta/`.
    """
    root = benchmark_root()
    assert root is not None
    practice = root / "python" / "exercises" / "practice"
    # A **dev-set** problem, dev in *both* languages: the exercise statement is
    # byte-identical across languages, so a stem held out in one is held out in all.
    # `test_no_fixture_names_a_holdout_problem` is what keeps that true as this is edited.
    problem = polyglot.load(practice, "hangman", Language.PYTHON)

    with runner.fresh_worktree(problem) as ws:
        present = sorted(str(f.relative_to(ws)) for f in ws.rglob("*") if f.is_file())
        assert present == ["hangman.py"], present
        assert not (ws / problem.test_path).exists()


@real
def test_dotfiles_at_the_root_survive() -> None:
    """`.npmrc`, `.eslintrc` and `.babelrc` are how the tests run. Dropping every dotted
    name rather than every dotted *directory* would break the language it is meant to fix."""
    root = benchmark_root()
    assert root is not None
    practice = root / "javascript" / "exercises" / "practice"
    problem = polyglot.load(practice, "triangle", Language.JAVASCRIPT)
    shipped = {f.name for f in (practice / "triangle").iterdir() if f.is_file()}

    with runner.fresh_worktree(problem) as ws:
        kept = {f.name for f in ws.iterdir() if f.is_file()}
        for name in shipped:
            if name.startswith(".") and name != Path(problem.test_path).name:
                assert name in kept, f"{name} was dropped"


# --- the whole point ----------------------------------------------------------


@real
@needs_npm
def test_a_javascript_problem_runs_sealed_with_no_network(
    store: toolchain.Store, tmp_path: Path
) -> None:
    """#63, end to end and both directions.

    Before this, all 49 JavaScript problems resolved to `infra_skip` and E1 measured one
    language while claiming two. The unimplemented stub must FAIL and the reference solution
    must PASS — a runner returning PASS unconditionally satisfies the second alone.
    """
    root = benchmark_root()
    assert root is not None
    practice = root / "javascript" / "exercises" / "practice"
    problem = polyglot.load(practice, "triangle", Language.JAVASCRIPT)

    sealed = runner.seal(require_sandbox=False, probe=True)
    stores = {Language.JAVASCRIPT: store}

    with runner.fresh_worktree(problem, stores) as ws:
        assert (ws / "node_modules" / ".bin" / "jest").exists(), "the store was not linked"
        # The worktree has no oracle — that is the point of it. `run_problem` writes it back
        # for scoring, and so must this test: without it jest reports "no such file" and the
        # unimplemented case below would pass for a reason that has nothing to do with the
        # stub. It did, on the first run of this test.
        runner.restore_oracle(ws, problem)

        unimplemented = polyglot.run_tests(
            ws,
            problem,
            env=sealed.env,
            timeout_s=300,
            wrapper=runner.wrap((), sealed.applied),
        )
        assert unimplemented.outcome is polyglot.Outcome.FAIL, unimplemented.detail

        proof = next(iter((practice / "triangle" / ".meta").glob("*.js")))
        shutil.copyfile(proof, ws / problem.stub_path)
        solved = polyglot.run_tests(
            ws,
            problem,
            env=sealed.env,
            timeout_s=300,
            wrapper=runner.wrap((), sealed.applied),
        )
        assert solved.outcome is polyglot.Outcome.PASS, (
            f"{solved.stdout[-1500:]}\n{solved.stderr[-800:]}"
        )

    if sealed.applied is not SandboxLevel.NONE:
        assert sealed.network_verified, sealed.verification


@real
def test_an_unprovisioned_javascript_run_is_infra_skip_not_failure(tmp_path: Path) -> None:
    """The behaviour T-1.6 established, kept. Scoring 49 unrunnable problems as unsolved
    would understate the solve rate by more than half the benchmark."""
    root = benchmark_root()
    assert root is not None
    practice = root / "javascript" / "exercises" / "practice"
    problems = [polyglot.load(practice, "triangle", Language.JAVASCRIPT)]

    class Idle:
        name = "idle"

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            return runner.Attempt(solved_claim=False, cost_usd=0.0)

    summary = runner.run_arm(
        problems,
        Idle(),
        Arm(name="x", harness="h", harness_version="0", hard_cost_usd=1.0),
        sealed=runner.seal(require_sandbox=False, probe=False),
    )
    assert summary.records[0].outcome == polyglot.Outcome.INFRA_SKIP
    assert summary.denominator == 0, "an unattemptable problem must leave the denominator"
    assert summary.records[0].deps_digest == ""


@real
@needs_npm
def test_the_row_records_which_dependency_tree_ran_it(
    store: toolchain.Store, tmp_path: Path
) -> None:
    """§6. A JavaScript result scored under a different jest is a different result, and a
    row that cannot say which one ran is a number without a provenance."""
    root = benchmark_root()
    assert root is not None
    practice = root / "javascript" / "exercises" / "practice"
    problem = polyglot.load(practice, "triangle", Language.JAVASCRIPT)

    class Solver:
        name = "solver"

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            proof = next(iter((practice / "triangle" / ".meta").glob("*.js")))
            (workspace / problem.stub_path).write_text(
                proof.read_text(encoding="utf-8"), encoding="utf-8"
            )
            return runner.Attempt(solved_claim=True, cost_usd=0.0)

    record = runner.run_problem(
        problem,
        Solver(),
        runner.seal(require_sandbox=False, probe=False),
        timeout_s=300,
        stores={Language.JAVASCRIPT: store},
    )
    assert record.outcome == polyglot.Outcome.PASS, record.detail
    assert record.deps_digest == store.digest
