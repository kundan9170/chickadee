"""Polyglot problem loader and local test invocation.

An E1 problem is four things on disk: instructions, a stub carrying signatures with empty
bodies, hidden tests, and a language. Three of those may be shown to a model. The fourth
never may, and this module's shape is mostly about that fourth one.

**D7, enforced structurally rather than by discipline.** The rule is that test
*names* may appear in a packet's DONE WHEN while test *bodies* never enter any prompt. A
flag saying "don't show this" is obeyed by code that remembers to check it, so instead the
readable fields are reachable through `prompt_view()` and the bodies only through
`test_source()` — a method named so that its appearance in a render path is visible in
review and in a grep. `leaks()` then asserts the property over rendered text, because §5.1
requires the protocol to be *asserted, not trusted*: comparability with published Polyglot
numbers depends on it, and a leak makes every number the rig produces meaningless.

**Why this carries its own toolchain table instead of reading a LanguageProfile.** Plan §3
and EN-1 restrict this package to `chickadee.engine` and `chickadee.schema`; `profiles` is
harness-internal. That restriction is also the right answer here for a second reason — §5
says the eval must not be helped by the thing it measures, and running the benchmark's
tests through the harness's own command templates would let a profile change silently move
a benchmark number.

**A missing toolchain is not a model failure.** `python` absent means the problem
could never have been attempted, so it resolves to `infra_skip` and leaves the denominator
rather than counting as an unsolved problem, which would understate the solve rate for a
reason unrelated to the harness. The same applies to a JavaScript workspace whose `jest` is
not provisioned — under D22 the sandbox has no network, so dependencies are installed before
the seal or the language does not run, and 49 unrunnable problems scored as unsolved would
understate the rate by more than half the benchmark.

**Every rule here is answerable to the real corpus.** The first version of this
module passed a full test suite and loaded **0 of 83** real Polyglot problems, because every
test built a fixture written to match the loader. Four defects hid behind that agreement:
instructions live at `.docs/instructions.md` and were looked for only at the root; the
JavaScript suffix is `.spec.js`, not `.test.js`; `node --test` cannot run a jest suite at
all; and — worst — `sorted()` put `.meta/example.py` first, so the *stub* handed to the arm
was the reference solution, in all 34 Python problems. `leaks()` could not catch that: it
scans for hidden test *bodies*, and an answer key is neither a test nor a body.

`test_polyglot_real.py` is the answer to that, and its rule is that it uses no fixtures.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from chickadee_eval.manifest import Language

INSTRUCTION_NAMES = ("instructions.md", "instructions.txt", "README.md")
"""Filenames, searched at the problem root **and one level down**. The real benchmark keeps
them at `.docs/instructions.md`; the original non-recursive lookup found none of them."""

APPENDIX_NAMES = ("instructions.append.md",)
"""Appended to the instructions when present — 26 of the 83 problems have one, and it
carries requirements the main file leaves out. Dropping it hands the model an incomplete
problem statement, which lowers the solve rate for a reason that is not the model's."""

PRIVATE_DIRS_ARE_HIDDEN = True
"""Dot-directories are never searched for a stub or a test. `.meta/` holds the reference
solution (`example.py`, `proof.ci.js`), and `sorted()` puts it *first* because `.` sorts
before letters — so the original code handed the arm the answer key as its stub, in all 34
Python problems. `leaks()` cannot catch that: it scans for hidden *test body* lines, and a
reference solution is neither a test nor a body. See issue #60."""

CONFIG_STEMS = frozenset({"babel.config", "jest.config", "eslint.config", "vite.config"})
"""Build configuration that shares the stub's suffix. Every JavaScript problem ships
`babel.config.js`, which is why stub selection is derived from the test filename rather than
from "the only file with the right extension"."""


# Per language: stub suffix, how a test file is recognised, the argv template that runs it,
# and the binary whose absence means `infra_skip`. Templates are argv lists, never shell
# strings — a problem id or path is benchmark-supplied text, and one containing a shell
# metacharacter must stay one argument rather than becoming a second command.
@dataclass(frozen=True)
class Toolchain:
    binary: str
    argv: tuple[str, ...]
    """`{test}` is substituted per token against the sealed workspace copy; `{runner}`
    against `runner_relpath` resolved inside the workspace."""
    stub_suffix: str
    test_suffixes: tuple[str, ...]
    """Plural because the same language is spelled differently by different suites: the
    Polyglot JavaScript exercises use `.spec.js`, while `.test.js` is the more common
    convention elsewhere. Exactly one file may match across all of them — two would make the
    scoring set ambiguous, and a benchmark whose oracle is ambiguous is not an oracle."""
    test_name_re: re.Pattern[str]
    runner_relpath: str | None = None
    """A runner that lives in the workspace rather than on `PATH`. `None` means the language
    runs from `binary` alone."""


TOOLCHAINS: Mapping[Language, Toolchain] = {
    Language.PYTHON: Toolchain(
        binary="python3",
        argv=("python3", "-m", "pytest", "{test}", "-q", "--no-header"),
        stub_suffix=".py",
        test_suffixes=("_test.py",),
        test_name_re=re.compile(r"^\s*def\s+(test_\w+)", re.M),
    ),
    Language.JAVASCRIPT: Toolchain(
        binary="node",
        # `node --test` was the original choice and cannot run these at all: the exercises
        # are jest suites — `expect` is a jest global, and the imports are extensionless, so
        # they need babel's resolution. Verified against the real benchmark, where it fails
        # at import before a single assertion runs.
        argv=("{runner}", "--ci", "--runTestsByPath", "{test}"),
        stub_suffix=".js",
        test_suffixes=(".spec.js", ".test.js"),
        test_name_re=re.compile(r"""["'`]([^"'`\n]+)["'`]\s*,\s*(?:async\s*)?\(""", re.M),
        runner_relpath="node_modules/.bin/jest",
    ),
}


class Outcome(StrEnum):
    """What running a problem's tests concluded."""

    PASS = "pass"
    FAIL = "fail"
    INFRA_SKIP = "infra_skip"
    """Toolchain absent. Excluded from denominators and listed in the report."""
    TIMEOUT = "timeout"


class ProblemError(ValueError):
    """A problem that cannot be loaded. Raised at load, never mid-run."""


@dataclass(frozen=True)
class PromptView:
    """Exactly what a model may be shown about a problem.

    A separate type rather than a subset convention: a render path that accepts this cannot
    reach a test body by accident, because the body is not a field on it.
    """

    id: str
    language: Language
    instructions: str
    stub_path: str
    stub_source: str
    test_names: tuple[str, ...]
    """Names only. §5.1 permits these in DONE WHEN; the bodies they belong to never."""


@dataclass(frozen=True)
class LoadedProblem:
    """One problem, materialized from disk."""

    id: str
    language: Language
    root: Path
    instructions: str
    stub_path: str
    stub_source: str
    test_path: str
    test_names: tuple[str, ...]
    _test_source: str

    def prompt_view(self) -> PromptView:
        return PromptView(
            id=self.id,
            language=self.language,
            instructions=self.instructions,
            stub_path=self.stub_path,
            stub_source=self.stub_source,
            test_names=self.test_names,
        )

    def test_source(self) -> str:
        """The hidden test body. **Never render this into a prompt.**

        Deliberately a method with an unmistakable name rather than a field, so that any
        path from problem to prompt that touches it is visible in review and greppable in
        CI. `leaks()` is the assertion that backs the convention up.
        """
        return self._test_source


def is_private(path: Path, base: Path) -> bool:
    """Whether `path` lies under a dot-directory of `base`.

    The single most important predicate in this module. `.meta/` holds the reference
    solution, and anything that walks into it while looking for a stub is looking for the
    answer key. Applied to the *directory* parts only, so `.eslintrc` as a file is fine.
    """
    return any(part.startswith(".") for part in path.relative_to(base).parts[:-1])


def _visible(base: Path, pattern: str) -> list[Path]:
    return sorted(p for p in base.rglob(pattern) if p.is_file() and not is_private(p, base))


def find_instructions(d: Path) -> str:
    """The problem statement, plus its appendix when there is one.

    Searched at the root and one level down, because the benchmark keeps them under
    `.docs/` — the one place a dot-directory has to be readable, and the reason
    `is_private` is applied to stub and test discovery rather than to everything.
    """
    parts: list[str] = []
    for names, required in ((INSTRUCTION_NAMES, True), (APPENDIX_NAMES, False)):
        found = None
        for name in names:
            direct = d / name
            if direct.is_file():
                found = direct
                break
            nested = sorted(p for p in d.glob(f"*/{name}") if p.is_file())
            if nested:
                found = nested[0]
                break
        if found is not None:
            parts.append(found.read_text(encoding="utf-8").strip())
        elif required:
            raise ProblemError(f"none of {names} present at {d} or one level below")
    return "\n\n".join(parts) + "\n"


def find_test(d: Path, tc: Toolchain, problem_id: str) -> Path:
    """The one file that scores the attempt."""
    tests = [p for sfx in tc.test_suffixes for p in _visible(d, f"*{sfx}")]
    if len(tests) != 1:
        # Zero means nothing can score the attempt; more than one means the scoring set is
        # ambiguous, and a benchmark whose oracle is ambiguous is not an oracle.
        raise ProblemError(
            f"{problem_id}: expected exactly one test file matching {tc.test_suffixes}, "
            f"found {len(tests)}: {[str(t.relative_to(d)) for t in tests]}"
        )
    return tests[0]


def find_stub(d: Path, test: Path, tc: Toolchain, problem_id: str) -> Path:
    """The file the arm is asked to implement, **derived from the test's own filename**.

    `affine_cipher_test.py` names `affine_cipher.py`; `affine-cipher.spec.js` names
    `affine-cipher.js`. That rule resolves all 83 real problems, and it is the only rule
    that could: 50 of them have more than one candidate with the right suffix — every
    JavaScript exercise ships `babel.config.js`, and `paasio` ships a `test_utils.py`
    helper. "The first file with the right extension" picked `.meta/example.py`.

    A fallback exists for layouts that do not follow the convention, and it is deliberately
    strict: exactly one visible non-config candidate, or an error. Guessing here is how the
    answer key became the stub.
    """
    for sfx in tc.test_suffixes:
        if test.name.endswith(sfx):
            derived = test.parent / f"{test.name[: -len(sfx)]}{tc.stub_suffix}"
            if derived.is_file():
                return derived
            break

    candidates = [
        p
        for p in _visible(d, f"*{tc.stub_suffix}")
        if p != test
        and not any(p.name.endswith(s) for s in tc.test_suffixes)
        and p.name.removesuffix(tc.stub_suffix) not in CONFIG_STEMS
    ]
    if len(candidates) != 1:
        raise ProblemError(
            f"{problem_id}: cannot identify the stub. The test {test.name!r} does not name "
            f"one, and {len(candidates)} visible candidates remain: "
            f"{[str(c.relative_to(d)) for c in candidates]}"
        )
    return candidates[0]


def load(root: Path, problem_id: str, language: Language) -> LoadedProblem:
    """Load one problem directory. Raises `ProblemError` rather than degrading."""
    d = root / problem_id
    if not d.is_dir():
        raise ProblemError(f"{problem_id}: no directory at {d}")
    tc = TOOLCHAINS[language]

    try:
        instructions = find_instructions(d)
    except ProblemError as exc:
        raise ProblemError(f"{problem_id}: {exc}") from exc

    test = find_test(d, tc, problem_id)
    stub = find_stub(d, test, tc, problem_id)

    # Belt and braces on the one mistake that silently invalidates every number the rig
    # produces. `find_stub` already searches visible files only; this is the assertion that
    # a future edit to that search cannot quietly undo.
    if is_private(stub, d):
        raise ProblemError(
            f"{problem_id}: resolved the stub to {stub.relative_to(d)}, which is under a "
            "private directory. That is where reference solutions live, and handing one to "
            "an arm as its stub would make every solve rate meaningless (#60)."
        )

    test_source = test.read_text(encoding="utf-8")
    return LoadedProblem(
        id=problem_id,
        language=language,
        root=d,
        instructions=instructions,
        stub_path=str(stub.relative_to(d)),
        stub_source=stub.read_text(encoding="utf-8"),
        test_path=str(test.relative_to(d)),
        test_names=tuple(dict.fromkeys(tc.test_name_re.findall(test_source))),
        _test_source=test_source,
    )


# Substrings shorter than this are too common to attribute to a test body — `assert` or a
# closing brace would match half of any prompt and turn the leak check into noise.
#
# The cost is a real blind spot, stated rather than hidden: a hidden file whose every line
# is either a test name or under the floor cannot be caught by this check at all. Real
# Polyglot assertions are comfortably longer, but a problem with one-line bodies is
# protected only by the oracle's absence from the arm's workspace, not by this scan. The
# two defenses are independent for that reason.
MIN_MARKER = 24


def oracle_lines(problem: LoadedProblem) -> tuple[tuple[int, str], ...]:
    """The hidden file's lines that must never reach a prompt, numbered, stripped: the one
    reading of D7 that `leaks` checks a prompt against and `redact_oracle` cleans a test run's
    output by, so what the redaction lets through is what the check accepts. Each is at least
    `MIN_MARKER` long, is not a test's name line, and is not already in the visible problem
    (`leaks` says why for each)."""
    tc = TOOLCHAINS[problem.language]
    visible = f"{problem.instructions}\n{problem.stub_source}"
    return tuple(
        (n, line)
        for n, raw in enumerate(problem.test_source().splitlines(), start=1)
        if len(line := raw.strip()) >= MIN_MARKER
        and not tc.test_name_re.search(raw)
        and line not in visible
    )


def leaks(rendered: str, problem: LoadedProblem) -> list[tuple[int, str]]:
    """Test-body lines from `problem` that appear in `rendered` **and could only have come
    from the hidden file**.

    Two exclusions, and both are the difference between a check that runs and a check that
    gets switched off:

    *Test names.* The signature line a name appears on is permitted in DONE WHEN, so
    matching on it would fail every legitimate prompt.

    *Lines the problem already shows.* A line present in the instructions or the stub was
    always going to be in the prompt, legitimately, and its appearance carries no
    information from the hidden file. Found on the real benchmark: `food-chain`'s tests
    assert on song lyrics that the instructions **specify verbatim**, so 80 lines matched
    and `run_arm` would have raised `TestBodyLeak` and discarded the entire run — an arm
    killed by a problem statement doing its job. The old fixture-based tests could not see
    this, because a fixture's instructions and tests share no prose.

    This does not weaken the property. The question is whether the *hidden file* reached a
    prompt, and a line the visible problem statement already contains is not evidence of
    that either way.

    Returns `(line_number, text)` so a caller can identify a leak without the caller having
    to keep the oracle. Only the numbers reach a results file — see `runner.ProblemRecord`.
    """
    return [(n, line) for n, line in oracle_lines(problem) if line in rendered]


def toolchain_available(language: Language) -> bool:
    return shutil.which(TOOLCHAINS[language].binary) is not None


@dataclass(frozen=True)
class TestRun:
    outcome: Outcome
    exit_code: int
    stdout: str
    stderr: str
    wall_ms: int
    detail: str = ""


def run_tests(
    workspace: Path,
    problem: LoadedProblem,
    *,
    env: Mapping[str, str],
    timeout_s: int = 120,
    wrapper: Sequence[str] = (),
) -> TestRun:
    """Run a problem's hidden tests against whatever is in `workspace`.

    `env` is the sealed environment and is passed rather than inherited, so benchmark code
    cannot pick up the parent's credentials or proxy settings.

    `wrapper` prefixes the argv so the test process runs **inside** the sandbox rather than
    beside a report that one exists — `runner.wrap()` builds it, and passing `level` lets the
    `runner.wrap()` builds it. Benchmark code is untrusted by construction: it is whatever
    the arm just wrote, plus a test file that executes it.

    The prefix is passed in rather than derived here so this module never imports `runner`,
    which imports it — the dependency runs one way.
    """
    import time

    tc = TOOLCHAINS[problem.language]
    if shutil.which(tc.binary) is None:
        return TestRun(Outcome.INFRA_SKIP, -1, "", "", 0, f"{tc.binary} not on PATH")

    runner = ""
    if tc.runner_relpath is not None:
        resolved = workspace / tc.runner_relpath
        if not resolved.exists():
            # the own rule, applied honestly: the problem could never have been attempted,
            # so it leaves the denominator rather than scoring as unsolved. Under D22 the
            # sandbox has no network, so the dependencies cannot be fetched here — they are
            # provisioned into the problem before the seal, or the language does not run.
            # Reporting these as failures would understate the solve rate by 49 problems.
            return TestRun(
                Outcome.INFRA_SKIP,
                -1,
                "",
                "",
                0,
                f"{tc.runner_relpath} not provisioned in the workspace; "
                f"{problem.language} dependencies must be installed before sealing (D22)",
            )
        runner = str(resolved)

    argv = [
        *wrapper,
        *(
            tok.replace("{test}", problem.test_path).replace("{runner}", runner)
            for tok in tc.argv
        ),
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=workspace,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        # PATH resolved a moment ago and does not now, or the interpreter is broken.
        # Infrastructure either way, so it leaves the denominator rather than scoring.
        return TestRun(Outcome.INFRA_SKIP, -1, "", "", 0, f"{tc.binary} vanished from PATH")
    except subprocess.TimeoutExpired:
        wall = int((time.monotonic() - started) * 1000)
        return TestRun(Outcome.TIMEOUT, -1, "", "", wall, f"exceeded {timeout_s}s")

    wall = int((time.monotonic() - started) * 1000)
    outcome = Outcome.PASS if proc.returncode == 0 else Outcome.FAIL
    return TestRun(outcome, proc.returncode, proc.stdout, proc.stderr, wall)
