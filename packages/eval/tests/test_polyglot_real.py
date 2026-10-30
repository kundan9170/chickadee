"""The loader, against the benchmark it exists to load.

`test_sealing.py`, `test_caps.py` and `test_protocol.py` all exercise `polyglot.load` — and
all of them build a fixture written to match the loader, so they agreed with it and proved
nothing about the world. The loader passed every one of them while loading **0 of 83** real
Polyglot problems, and one of its failure modes resolved the stub to `.meta/example.py`, the
reference solution, in all 34 Python problems.

So this file has one rule: **no fixtures**. Every test here reads a real checkout, and the
ones that cannot are skipped loudly rather than replaced with something agreeable.

Point `CKD_POLYGLOT_ROOT` at a checkout of `Aider-AI/polyglot-benchmark`. CI clones it;
locally, `/workspace/aider-ai/polyglot-benchmark` is picked up automatically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from chickadee_eval.manifest import Language
from chickadee_eval.problems import polyglot

EXPECTED = {Language.PYTHON: 34, Language.JAVASCRIPT: 49}
SUBDIR = {Language.PYTHON: "python", Language.JAVASCRIPT: "javascript"}


def benchmark_root() -> Path | None:
    for candidate in (
        os.environ.get("CKD_POLYGLOT_ROOT"),
        "/workspace/aider-ai/polyglot-benchmark",
    ):
        if candidate and (Path(candidate) / "python" / "exercises" / "practice").is_dir():
            return Path(candidate)
    return None


real = pytest.mark.skipif(benchmark_root() is None, reason="no polyglot checkout")


def practice(language: Language) -> Path:
    root = benchmark_root()
    assert root is not None
    return root / SUBDIR[language] / "exercises" / "practice"


def problem_ids(language: Language) -> list[str]:
    return sorted(d.name for d in practice(language).iterdir() if d.is_dir())


def test_the_checkout_is_present_or_genuinely_absent() -> None:
    """A skip is honest. A skip nobody notices is the whole of #60.

    If a checkout exists it must be usable, so a half-present or restructured one fails here
    instead of quietly turning every test below into a pass-by-absence.
    """
    root = benchmark_root()
    if root is None:
        pytest.skip("no polyglot checkout; real-loader coverage is genuinely absent")
    for language, expected in EXPECTED.items():
        assert len(problem_ids(language)) == expected, (
            f"{language}: the benchmark changed size; the committed manifests describe a "
            "pool that no longer exists"
        )


# --- defect 1: instructions are nested -----------------------------------------


@real
@pytest.mark.parametrize("language", list(Language))
def test_every_problem_loads(language: Language) -> None:
    """The headline. Was 0/83; the failure was `(d / name).is_file()` at the root only,
    while the benchmark keeps instructions at `.docs/instructions.md`."""
    failures: list[str] = []
    for pid in problem_ids(language):
        try:
            polyglot.load(practice(language), pid, language)
        except polyglot.ProblemError as exc:
            failures.append(f"{pid}: {exc}")
    assert not failures, failures


@real
def test_the_appendix_is_included_when_there_is_one() -> None:
    """26 problems ship `instructions.append.md` with requirements the main file omits.
    Dropping it hands the model an incomplete problem and lowers the solve rate for a
    reason that is not the model's."""
    checked = 0
    for language in Language:
        for pid in problem_ids(language):
            appendix = practice(language) / pid / ".docs" / "instructions.append.md"
            if not appendix.is_file():
                continue
            loaded = polyglot.load(practice(language), pid, language)
            extra = appendix.read_text(encoding="utf-8").strip()
            assert extra and extra in loaded.instructions, f"{pid}: appendix dropped"
            checked += 1
    assert checked >= 20, f"only {checked} problems with an appendix were exercised"


# --- defect 2: the stub was the reference solution -----------------------------


@real
@pytest.mark.parametrize("language", list(Language))
def test_no_stub_is_ever_a_reference_solution(language: Language) -> None:
    """The defect that would have invalidated every number the rig produced.

    `sorted()` put `.meta/example.py` first, because `.` sorts before letters. The arm would
    have been handed the answer key as its stub — and `leaks()` cannot catch it, because it
    scans for hidden *test body* lines and a reference solution is neither.
    """
    offenders: list[str] = []
    for pid in problem_ids(language):
        loaded = polyglot.load(practice(language), pid, language)
        parts = Path(loaded.stub_path).parts
        if any(part.startswith(".") for part in parts[:-1]):
            offenders.append(f"{pid}: {loaded.stub_path}")
    assert not offenders, offenders


@real
@pytest.mark.parametrize("language", list(Language))
def test_the_stub_is_the_file_the_test_imports(language: Language) -> None:
    """Stronger than "not under a dotdir": the stub must be the module the oracle scores.

    Derived from the test's own filename, which is the only rule that resolves all 83 — 50
    problems have more than one candidate with the right suffix (`babel.config.js` in every
    JavaScript exercise, `test_utils.py` in `paasio`).
    """
    tc = polyglot.TOOLCHAINS[language]
    for pid in problem_ids(language):
        loaded = polyglot.load(practice(language), pid, language)
        test_name = Path(loaded.test_path).name
        suffix = next(s for s in tc.test_suffixes if test_name.endswith(s))
        expected = f"{test_name[: -len(suffix)]}{tc.stub_suffix}"
        assert Path(loaded.stub_path).name == expected, (
            f"{pid}: test {test_name!r} scores {expected!r} but the stub is "
            f"{loaded.stub_path!r}"
        )


@real
@pytest.mark.parametrize("language", list(Language))
def test_the_stub_is_never_the_shipped_solution(language: Language) -> None:
    """Exact, not heuristic.

    The first version of this test compared *sizes* and failed on seven real problems:
    `binary`'s skeleton is 360 bytes of comment banner and `throw` bodies against a 321-byte
    reference solution, so "the stub must be smaller" is simply false. A size rule that
    fires on correct input is worse than no rule — it teaches people to delete the test.
    What actually matters is identity, and that can be checked exactly.
    """
    offenders: list[str] = []
    for pid in problem_ids(language):
        d = practice(language) / pid
        loaded = polyglot.load(practice(language), pid, language)
        for example in (d / ".meta").glob(f"*{polyglot.TOOLCHAINS[language].stub_suffix}"):
            if loaded.stub_source == example.read_text(encoding="utf-8"):
                offenders.append(f"{pid}: stub is byte-identical to {example.name}")
    assert not offenders, offenders


# --- defect 3: the JavaScript test suffix --------------------------------------


@real
def test_javascript_tests_are_found_by_their_real_suffix() -> None:
    """The benchmark uses `.spec.js`; the toolchain declared `.test.js`, so all 49 problems
    raised "expected exactly one... found 0"."""
    assert ".spec.js" in polyglot.TOOLCHAINS[Language.JAVASCRIPT].test_suffixes
    for pid in problem_ids(Language.JAVASCRIPT):
        loaded = polyglot.load(practice(Language.JAVASCRIPT), pid, Language.JAVASCRIPT)
        assert loaded.test_path.endswith(".spec.js"), f"{pid}: {loaded.test_path}"


@real
@pytest.mark.parametrize("language", list(Language))
def test_test_names_are_extracted(language: Language) -> None:
    """`test_names` is the only part of the oracle §5.1 lets into a prompt. Empty for every
    problem would not fail anything else, and would quietly remove the DONE WHEN signal."""
    empty = [
        pid
        for pid in problem_ids(language)
        if not polyglot.load(practice(language), pid, language).test_names
    ]
    assert not empty, f"{language}: no test names extracted for {empty[:5]}"


# --- defect 4: node --test cannot run these ------------------------------------


@real
def test_an_unprovisioned_javascript_workspace_is_infra_skip(tmp_path: Path) -> None:
    """Not a failure. The problem could never have been attempted, so §7 keeps it out of
    the denominator — scoring 49 unrunnable problems as unsolved would understate the solve
    rate by more than half the benchmark."""
    pid = problem_ids(Language.JAVASCRIPT)[0]
    loaded = polyglot.load(practice(Language.JAVASCRIPT), pid, Language.JAVASCRIPT)
    ws = tmp_path / "ws"
    shutil.copytree(practice(Language.JAVASCRIPT) / pid, ws)

    run = polyglot.run_tests(ws, loaded, env={"PATH": os.environ["PATH"]}, timeout_s=60)
    assert run.outcome is polyglot.Outcome.INFRA_SKIP
    assert "not provisioned" in run.detail


@real
def test_a_python_problem_runs_for_real(tmp_path: Path) -> None:
    """End to end on the real toolchain: unimplemented stub fails, reference solution passes.

    Both directions, because a runner that reported PASS unconditionally would satisfy the
    second alone — and that is the shape of an oracle that has stopped oracling.
    """
    # A **dev-set** problem: a suite pinned to held-out problems turns every future
    # instrument fix into tuning on the holdout, one commit at a time. Dev in *both*
    # languages, because the exercise statement is byte-identical across them.
    pid = "bottle-song"
    src = practice(Language.PYTHON) / pid
    if not src.is_dir():
        pytest.skip(f"{pid} is not in this checkout")
    loaded = polyglot.load(practice(Language.PYTHON), pid, Language.PYTHON)
    ws = tmp_path / "ws"
    shutil.copytree(src, ws)
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/root")}

    unimplemented = polyglot.run_tests(ws, loaded, env=env, timeout_s=180)
    assert unimplemented.outcome is polyglot.Outcome.FAIL, unimplemented.detail

    shutil.copyfile(ws / ".meta" / "example.py", ws / loaded.stub_path)
    solved = polyglot.run_tests(ws, loaded, env=env, timeout_s=180)
    assert solved.outcome is polyglot.Outcome.PASS, (
        f"the shipped reference solution does not pass its own tests:\n"
        f"{solved.stdout[-1500:]}\n{solved.stderr[-500:]}"
    )


@real
def test_a_javascript_problem_runs_for_real(tmp_path: Path) -> None:
    """The same both-directions check through jest.

    `node --test` — the original argv — cannot run these at all: `expect` is a jest global
    and the imports are extensionless, so it dies at import before any assertion. Gated on
    npm, since the dependencies must be installed before the workspace is sealed.
    """
    if shutil.which("npm") is None:
        pytest.skip("npm not installed; JavaScript execution is genuinely unverified here")
    pid = "triangle"  # dev in both languages
    src = practice(Language.JAVASCRIPT) / pid
    if not src.is_dir():
        pytest.skip(f"{pid} is not in this checkout")
    loaded = polyglot.load(practice(Language.JAVASCRIPT), pid, Language.JAVASCRIPT)
    ws = tmp_path / "ws"
    shutil.copytree(src, ws)
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/root")}

    install = subprocess.run(
        ["npm", "install", "--silent", "--no-audit", "--no-fund"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if install.returncode != 0:
        pytest.skip(f"npm install failed (offline?): {install.stderr[-300:]}")

    unimplemented = polyglot.run_tests(ws, loaded, env=env, timeout_s=300)
    assert unimplemented.outcome is polyglot.Outcome.FAIL, unimplemented.detail

    shutil.copyfile(ws / ".meta" / "proof.ci.js", ws / loaded.stub_path)
    solved = polyglot.run_tests(ws, loaded, env=env, timeout_s=300)
    assert solved.outcome is polyglot.Outcome.PASS, (
        f"the shipped reference solution does not pass its own tests:\n"
        f"{solved.stdout[-1500:]}\n{solved.stderr[-500:]}"
    )


# --- the oracle never reaches the prompt view ----------------------------------


@real
@pytest.mark.parametrize("language", list(Language))
def test_no_prompt_view_carries_oracle_text(language: Language) -> None:
    """EV-3 over the real corpus rather than a fixture.

    The stub and instructions are shown to the model; the test body never is. With the old
    stub selection this would have failed on every Python problem, since the stub *was* the
    solution — which is the check the fixture-based suite could not perform.
    """
    offenders: list[str] = []
    for pid in problem_ids(language):
        loaded = polyglot.load(practice(language), pid, language)
        view = loaded.prompt_view()
        rendered = f"{view.instructions}\n{view.stub_source}"
        if polyglot.leaks(rendered, loaded):
            offenders.append(pid)
    assert not offenders, f"{language}: oracle text reachable from the prompt view: {offenders}"


@real
def test_no_problem_that_quotes_its_own_tests_is_read_as_a_leak() -> None:
    """The false-positive case — swept rather than named.

    Some problem statements specify the exact strings their tests assert on, so the hidden
    file and the instructions share whole lines. `run_arm` raises `TestBodyLeak` on any leak
    and discards the run, so one problem statement doing its job would have thrown away an
    entire arm's execution, after paying for every problem before it. That is the defect
    this test was written for.

    The first version named the single problem that exposed it. That problem is in the
    confirmation set, and a suite pinned to held-out problems turns every future instrument
    fix into tuning on the holdout, one commit at a time. A sweep is strictly stronger: it
    still covers that problem, it covers the other 82 as well, and it names nothing, so
    there is no fixture to tune `leaks()` against.

    Two guards keep it from passing vacuously — some problem must have the property at all,
    and at least one must have it strongly. Without those, a corpus that stopped quoting its
    tests would turn this into a green test that exercises nothing.
    """
    quoting: dict[str, int] = {}
    offenders: list[str] = []
    for language in Language:
        for pid in problem_ids(language):
            loaded = polyglot.load(practice(language), pid, language)
            shared = [
                line.strip()
                for line in loaded.test_source().splitlines()
                if len(line.strip()) >= polyglot.MIN_MARKER
                and line.strip() in loaded.instructions
            ]
            if not shared:
                continue
            quoting[f"{SUBDIR[language]}/{pid}"] = len(shared)
            if polyglot.leaks(loaded.instructions, loaded):
                offenders.append(f"{SUBDIR[language]}/{pid}")

    assert quoting, (
        "no problem in this checkout quotes its own test lines, so this test proves "
        "nothing — the corpus changed and the false-positive case needs re-finding"
    )
    assert max(quoting.values()) >= 20, (
        f"the strongest overlap left is {max(quoting.values())} lines, down from 80; the "
        "case this test was written for has weakened and it no longer carries its weight"
    )
    assert not offenders, (
        f"a problem statement doing its job is read as a test-body leak, which discards the "
        f"whole run: {offenders}"
    )


@real
def test_a_genuine_leak_is_still_caught() -> None:
    """The exclusion must not have turned the check off.

    A line from the hidden file that the problem does *not* already show must still be
    reported — otherwise the fix for `food-chain` would be a hole rather than a refinement.
    """
    root = practice(Language.PYTHON)
    loaded = polyglot.load(root, problem_ids(Language.PYTHON)[0], Language.PYTHON)
    hidden = [
        line.strip()
        for line in loaded.test_source().splitlines()
        if len(line.strip()) >= polyglot.MIN_MARKER
        and line.strip() not in loaded.instructions
        and line.strip() not in loaded.stub_source
        and not polyglot.TOOLCHAINS[Language.PYTHON].test_name_re.search(line)
    ]
    assert hidden, "no hidden-only lines to test with; pick a different problem"
    assert polyglot.leaks(f"here it is:\n{hidden[0]}\n", loaded)
