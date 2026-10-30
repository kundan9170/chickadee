"""Perturbation is seeded, reproducible, and leaves the problem solvable.

The solvability property is the one that carries weight, and it is the one a fixture cannot
establish. A perturbation that broke a problem would lower the solve rate, be read as
evidence *against* memorisation, and be exactly backwards — so this file proves solvability
the only way that counts: it perturbs a real Polyglot problem **together with its reference
solution** and runs the perturbed tests against it. If a rename missed a file, the reference
solution stops passing.

Those tests skip when the benchmark checkout is absent, which is the honest trade — but the
skip is loud, and `test_the_real_benchmark_tests_are_not_silently_skipped` asserts the
checkout is either present and usable or genuinely missing, never present and quietly
unreadable. Issue #60 is what that guard is for: `polyglot.py` passed a full test suite
built entirely on a fixture I had written to match it, and loaded 0 of 83 real problems.

Point `CKD_POLYGLOT_ROOT` at a checkout of `Aider-AI/polyglot-benchmark` to run them.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from chickadee_eval.manifest import Language, Manifest, SubsetKind
from chickadee_eval.problems import perturb


def survives(name: str, text: str) -> bool:
    """Whether `name` is still present as a standalone identifier component.

    The same boundary `rename_in` uses. A plain substring check would flag `decoder`, which
    is a different word the perturbation is right to leave alone."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text) is not None


SEED = 20260820

STUB = '''\
def encode(plaintext, a, b):
    """Encipher plaintext."""
    raise NotImplementedError


class CipherError(Exception):
    pass
'''

TEST = """\
from affine_cipher import encode, CipherError


def test_encode_shifts_each_letter():
    assert encode("test", 5, 7) == "ybty"
"""

DOCS = """\
# Affine Cipher

You are given a plaintext string. Write a function `encode` that enciphers it.

For example, `encode("test", 5, 7)` returns `"ybty"`.
"""


def make_problem(root: Path, pid: str = "sample-exercise") -> Path:
    d = root / pid
    (d / ".docs").mkdir(parents=True)
    (d / ".meta").mkdir(parents=True)
    (d / ".docs" / "instructions.md").write_text(DOCS, encoding="utf-8")
    (d / "affine_cipher.py").write_text(STUB, encoding="utf-8")
    (d / "affine_cipher_test.py").write_text(TEST, encoding="utf-8")
    (d / ".meta" / "example.py").write_text("def encode(p, a, b):\n    return p\n", "utf-8")
    return d


# --- determinism --------------------------------------------------------------


def test_the_same_seed_produces_the_same_plan(tmp_path: Path) -> None:
    make_problem(tmp_path)
    a = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    b = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    assert a == b


def test_a_different_seed_produces_a_different_plan(tmp_path: Path) -> None:
    make_problem(tmp_path)
    a = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    b = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED + 1)
    assert a.renames != b.renames


def test_names_are_stable_across_processes() -> None:
    """Pinned literals, not a self-consistency check.

    `random.Random`'s stream is a CPython implementation detail; a perturbed set that
    regenerated differently on another interpreter would fail the one property D23 needs.
    Hard-coding the expected output is what makes that a test rather than a hope — if the
    derivation changes, this fails and someone has to decide whether the committed
    perturbed set is still the set that was run.
    """
    assert perturb.coined_name(SEED, "sample-exercise", "encode") == "trailan"
    assert perturb.coined_name(SEED, "sample-exercise", "CipherError") == "lorak"
    assert perturb.coined_name(SEED, "other-exercise", "encode") == "rekloun"


def test_two_identifiers_in_one_problem_get_different_names(tmp_path: Path) -> None:
    make_problem(tmp_path)
    p = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    news = [r.new for r in p.renames]
    assert len(set(news)) == len(news)


# --- the handle is actually removed -------------------------------------------


def test_the_original_name_does_not_survive_inside_the_new_one(tmp_path: Path) -> None:
    """A suffix would not do. `affine_cipher_x7` still contains `affine_cipher`, and the
    retrieval that enables is the thing D23 controls for."""
    make_problem(tmp_path)
    p = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    for rename in p.renames:
        assert rename.old.lower() not in rename.new.lower()


def test_the_module_name_is_perturbed_too(tmp_path: Path) -> None:
    """The filename is as strong a retrieval key as the function name."""
    make_problem(tmp_path)
    p = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    assert p.module_rename is not None
    assert p.module_rename.old == "affine_cipher"


def test_the_perturbed_tree_contains_none_of_the_original_identifiers(tmp_path: Path) -> None:
    make_problem(tmp_path)
    out = tmp_path / "out"
    p = perturb.generate(tmp_path, out, "sample-exercise", Language.PYTHON, SEED)

    dest = out / p.perturbed_id
    olds = {r.old for r in p.renames} | {p.module_rename.old if p.module_rename else ""}
    for path in dest.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md"}:
            text = path.read_text(encoding="utf-8")
            for old in olds:
                if old:
                    assert not survives(old, text), f"{old!r} survived in {path.name}"


def test_the_case_shape_of_a_name_is_preserved(tmp_path: Path) -> None:
    """A PascalCase class replaced by a snake_case name is a second signal in the diff, and
    one a model could learn to read as 'this is the perturbed set'."""
    make_problem(tmp_path)
    p = perturb.plan(tmp_path / "sample-exercise", "sample-exercise", Language.PYTHON, SEED)
    by_old = {r.old: r.new for r in p.renames}
    assert by_old["CipherError"][0].isupper()
    assert by_old["encode"][0].islower()


# --- renaming is a single pass ------------------------------------------------


def test_renames_do_not_chain() -> None:
    """Sequential passes let an earlier replacement be rewritten by a later one, so
    `a -> b` then `b -> c` silently turns every original `a` into `c`."""
    renames = (perturb.Rename("a", "b"), perturb.Rename("b", "c"))
    assert perturb.rename_in("a b", renames) == "b c"


def test_renames_respect_word_boundaries() -> None:
    renames = (perturb.Rename("encode", "zaikel"),)
    assert perturb.rename_in("encoder encode encoded", renames) == "encoder zaikel encoded"


# --- prose is rephrased, never re-specified -----------------------------------


def test_prose_is_rephrased() -> None:
    text, edits = perturb.tweak_prose("You are given a list. For example, this.")
    assert edits == 2
    assert "You receive" in text and "As an illustration" in text


def test_prose_substitution_never_reaches_inside_code() -> None:
    """Identifiers and worked examples live in code spans. Rewriting an example's text
    would alter what the problem asks for — the one thing this generator promises not to."""
    src = "Text. `For example` stays.\n\n```\nYou are given x\n```\n"
    text, _ = perturb.tweak_prose(src)
    assert "`For example`" in text
    assert "```\nYou are given x\n```" in text


def test_an_unperturbable_problem_raises(tmp_path: Path) -> None:
    """An unperturbed problem in the perturbed set reports a memorisation delta of zero for
    a problem that was never perturbed, and reads as evidence that nothing is memorised."""
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(perturb.PerturbError, match=r"no \.py stub"):
        perturb.generate(tmp_path, tmp_path / "out", "empty", Language.PYTHON, SEED)


def test_the_reference_solution_is_never_used_as_the_stub(tmp_path: Path) -> None:
    """Issue #60: the loader picks `.meta/example.py` because `.` sorts before letters.
    Planning renames against the answer key would rename the wrong declarations."""
    d = make_problem(tmp_path)
    (d / ".meta" / "example.py").write_text(
        "def totally_different_name(x):\n    return x\n", encoding="utf-8"
    )
    p = perturb.plan(d, "sample-exercise", Language.PYTHON, SEED)
    assert {r.old for r in p.renames} == {"encode", "CipherError"}


# --- against the real benchmark -----------------------------------------------


def benchmark_root() -> Path | None:
    for candidate in (
        os.environ.get("CKD_POLYGLOT_ROOT"),
        "/workspace/aider-ai/polyglot-benchmark",
    ):
        if candidate and (Path(candidate) / "python" / "exercises" / "practice").is_dir():
            return Path(candidate)
    return None


real = pytest.mark.skipif(benchmark_root() is None, reason="no polyglot checkout")


def test_the_real_benchmark_tests_are_not_silently_skipped() -> None:
    """A skip is honest; a skip nobody notices is how #60 happened.

    If a checkout is present it must be usable. The failure this catches is a half-present
    or restructured checkout that makes every real test skip while the suite still reports
    green — which is indistinguishable, from the summary line, from having tested anything.
    """
    root = benchmark_root()
    if root is None:
        pytest.skip("no polyglot checkout; real-benchmark coverage is genuinely absent")
    practice = root / "python" / "exercises" / "practice"
    assert len([d for d in practice.iterdir() if d.is_dir()]) > 20


@real
def test_every_real_python_problem_can_be_planned() -> None:
    """Planned, not sampled. #60 was three defects that a single happy-path example would
    have missed and a full sweep found immediately."""
    root = benchmark_root()
    assert root is not None
    practice = root / "python" / "exercises" / "practice"
    failures: list[str] = []
    for d in sorted(x for x in practice.iterdir() if x.is_dir()):
        try:
            p = perturb.plan(d, d.name, Language.PYTHON, SEED)
            if p.is_empty:
                failures.append(f"{d.name}: nothing perturbed")
        except perturb.PerturbError as exc:
            failures.append(f"{d.name}: {exc}")
    assert not failures, failures


@real
def test_every_real_javascript_problem_can_be_planned() -> None:
    root = benchmark_root()
    assert root is not None
    practice = root / "javascript" / "exercises" / "practice"
    failures: list[str] = []
    for d in sorted(x for x in practice.iterdir() if x.is_dir()):
        try:
            p = perturb.plan(d, d.name, Language.JAVASCRIPT, SEED)
            if p.is_empty:
                failures.append(f"{d.name}: nothing perturbed")
        except perturb.PerturbError as exc:
            failures.append(f"{d.name}: {exc}")
    assert not failures, failures


@real
def test_a_real_problem_stays_solvable_after_perturbation(tmp_path: Path) -> None:
    """The property D23 actually needs, proved the only way that counts.

    The reference solution is perturbed alongside the problem and then run against the
    perturbed tests. A rename that missed a file shows up here as a failing solution — and
    a perturbation that broke the problem would otherwise have been read as evidence
    against memorisation, which is exactly backwards.
    """
    root = benchmark_root()
    assert root is not None
    practice = root / "python" / "exercises" / "practice"

    checked = 0
    # Real ids from the benchmark, **all from the dev subset in every language**.
    # Named explicitly rather than sampled so a failure names a problem someone can go and
    # look at — and the `checked >= 3` floor below is what stops a renamed or removed
    # exercise from turning this into a test that silently exercises nothing.
    # `test_no_fixture_names_a_holdout_problem` keeps this list off the holdout as it is
    # edited, and is where the "in every language" part is justified.
    for name in ("bottle-song", "dot-dsl", "forth", "hangman", "react"):
        src = practice / name
        if not src.is_dir() or not (src / ".meta" / "example.py").is_file():
            continue
        p = perturb.generate(practice, tmp_path, name, Language.PYTHON, SEED)
        dest = tmp_path / p.perturbed_id

        # The candidate file is replaced by the perturbed reference solution: this asks
        # whether a *correct* implementation still passes, not whether the stub does.
        stub = next(
            f
            for f in dest.glob("*.py")
            if not f.name.endswith("_test.py") and not f.name.startswith(".")
        )
        example = next(iter((dest / ".meta").glob("*.py")))
        shutil.copyfile(example, stub)

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", str(dest)],
            capture_output=True,
            text=True,
            cwd=dest,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"{name}: the reference solution stopped passing after perturbation — a rename "
            f"missed a file.\n{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}"
        )
        checked += 1

    assert checked >= 3, f"only {checked} problems were actually exercised"


@real
def test_a_real_perturbed_problem_keeps_none_of_its_original_names(tmp_path: Path) -> None:
    root = benchmark_root()
    assert root is not None
    practice = root / "python" / "exercises" / "practice"
    src = next(d for d in sorted(practice.iterdir()) if d.is_dir())

    p = perturb.generate(practice, tmp_path, src.name, Language.PYTHON, SEED)
    dest = tmp_path / p.perturbed_id
    olds = {r.old for r in p.renames}
    if p.module_rename:
        olds.add(p.module_rename.old)

    for path in dest.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for old in olds:
            assert not survives(old, text), f"{old!r} survived in {path.relative_to(dest)}"


# --- the committed manifest ---------------------------------------------------
# These live here rather than in a `test_manifest.py` because `manifest.py` belongs to
# T-0.9 and the manifests themselves belong to T-1.2 — this file is the one T-1.2 owns. The
# schema is already tested next to its model; what is tested here is whether the committed
# data describes the real benchmark, which is the whole of D33's "recorded, not assumed".

MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "e1.toml"


def load_manifest() -> Manifest:
    return Manifest(**tomllib.loads(MANIFEST.read_text(encoding="utf-8")))


def test_the_committed_manifest_parses() -> None:
    """A manifest that does not validate is a plan, not a protocol."""
    m = load_manifest()
    assert m.experiment == "e1"
    assert {s.kind for s in m.subsets} == set(SubsetKind)


def test_the_recorded_counts_are_the_ones_in_the_file() -> None:
    """Pinned, so a set that quietly changed size between runs fails here rather than
    showing up at G3 as a number that moved for an unexplained reason."""
    m = load_manifest()
    sizes = {s.name: s.count for s in m.subsets}
    assert sizes == {"e1-dev": 30, "e1-smoke": 8, "e1-confirmation": 53, "e1-perturbed": 20}

    dev = m.subset(SubsetKind.DEV)
    conf = m.subset(SubsetKind.CONFIRMATION)
    assert dev is not None and conf is not None
    assert dev.by_language() == {Language.PYTHON: 12, Language.JAVASCRIPT: 18}
    assert conf.by_language() == {Language.PYTHON: 22, Language.JAVASCRIPT: 31}


def test_dev_and_confirmation_partition_the_whole_pool() -> None:
    """Confirmation is *the remaining* problems, so the two must cover the pool
    exactly — a gap would be problems nobody ever runs, silently shrinking the benchmark."""
    m = load_manifest()
    dev = {p.id for p in m.subset(SubsetKind.DEV).problems}  # type: ignore[union-attr]
    conf = {p.id for p in m.subset(SubsetKind.CONFIRMATION).problems}  # type: ignore[union-attr]
    assert not dev & conf
    assert len(dev | conf) == 83


def test_the_smoke_subset_is_drawn_from_dev() -> None:
    """A smoke run is a quick look, tuned on freely, so it may hold only dev problems;
    and `subset(DEV)` must still mean the whole dev set, which it does while dev comes first."""
    m = load_manifest()
    dev = m.subset(SubsetKind.DEV)
    smoke = next(s for s in m.subsets if s.name == "e1-smoke")
    assert dev is not None and dev.name == "e1-dev"
    assert {p.id for p in smoke.problems} <= {p.id for p in dev.problems}
    assert smoke.by_language() == {Language.PYTHON: 4, Language.JAVASCRIPT: 4}
    held_out = {
        p.id.rsplit("/", 1)[-1]
        for p in m.subset(SubsetKind.CONFIRMATION).problems  # type: ignore[union-attr]
    }
    assert not {p.id.rsplit("/", 1)[-1] for p in smoke.problems} & held_out, (
        "a smoke stem is held out in the other language, and the statement is the same (#70)"
    )


def test_ids_are_paths_because_bare_exercise_names_collide() -> None:
    """Bare names are not unique across languages, and `Subset` rejects duplicate ids.

    Both halves are asserted. That every id is a path is the rule; that some *stems* repeat
    across languages is the reason for it — without the second assertion the first reads as
    a style preference, and someone would shorten the ids and rediscover this the hard way.
    """
    m = load_manifest()
    all_ids = [p.id for subset in m.subsets for p in subset.problems]
    assert all("/" in i for i in all_ids), "an id lost its language prefix"

    stems = [i.rsplit("/", 1)[-1] for i in all_ids]
    collisions = sorted({s for s in stems if stems.count(s) > 1})
    assert collisions, "no stem collisions left; paths may no longer be necessary — recheck"


def test_every_perturbed_problem_derives_from_a_confirmation_problem() -> None:
    """§5.3 compares the perturbed delta against the delta on the original set, and at a
    gate the original set is confirmation. A perturbation of a dev problem would put the
    two deltas on different populations and compare them anyway."""
    m = load_manifest()
    conf = {p.id for p in m.subset(SubsetKind.CONFIRMATION).problems}  # type: ignore[union-attr]
    pert = m.subset(SubsetKind.PERTURBED)
    assert pert is not None and pert.seed is not None
    for p in pert.problems:
        assert p.id.endswith(perturb.PERTURBED_SUFFIX), p.id
        assert p.id.removesuffix(perturb.PERTURBED_SUFFIX) in conf, p.id


@real
def test_every_id_in_the_manifest_is_a_real_directory() -> None:
    """The property D33 is actually about.

    "Real counts recorded, not assumed" is not satisfied by a plausible list. Every id here
    was read off a checkout, and this is what keeps that true as the file is edited.
    """
    root = benchmark_root()
    assert root is not None
    m = load_manifest()
    missing: list[str] = []
    for subset in m.subsets:
        for p in subset.problems:
            src = p.id.removesuffix(perturb.PERTURBED_SUFFIX)
            if not (root / src).is_dir():
                missing.append(f"{subset.name}: {p.id}")
    assert not missing, missing


@real
def test_the_pool_has_not_changed_size_under_the_manifest() -> None:
    """If the benchmark gains or loses exercises, the committed split no longer partitions
    it — and the failure should be here, not in a gate run six weeks later."""
    root = benchmark_root()
    assert root is not None
    counts = {
        lang: len([d for d in (root / lang / "exercises" / "practice").iterdir() if d.is_dir()])
        for lang in ("python", "javascript")
    }
    assert counts == {"python": 34, "javascript": 49}


@real
def test_every_perturbed_entry_can_actually_be_generated(tmp_path: Path) -> None:
    """A manifest naming a perturbation the generator cannot produce is a run that dies
    partway through, after paying for everything before it."""
    root = benchmark_root()
    assert root is not None
    m = load_manifest()
    pert = m.subset(SubsetKind.PERTURBED)
    assert pert is not None and pert.seed is not None

    for problem in pert.problems:
        src = problem.id.removesuffix(perturb.PERTURBED_SUFFIX)
        plan = perturb.plan(root / src, src, problem.language, pert.seed)
        assert not plan.is_empty, src
        assert perturb.perturbed_id(src) == problem.id


# --- the suite keeps its hands off the holdout --------------------------
# EV-11 is "tune on dev, gate on confirmation". Nothing here is tuning in the ordinary
# sense — no model has run and no result has been fitted — but a *test suite* pinned to
# held-out problems tunes the instrument on the holdout one commit at a time: every future
# loader or runner fix gets iterated against those problems until they pass. G1 recorded
# five such fixtures as an exposure to fix; this is the check that fixes it and keeps it
# fixed, because a rule nobody enforces is a rule that lasts until the next hurry.

TESTS_DIR = Path(__file__).resolve().parent


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Docstrings are prose, not fixtures.

    A rationale that says *why* `leaks()` once misfired may name the problem it misfired on
    — that mention loads nothing and is iterated against nothing. What the rule is about is
    a value the test actually uses.
    """
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, holders) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            out.add(id(first.value))
    return out


def _named_stems(path: Path) -> dict[str, int]:
    """Every problem stem a source file names as a value, mapped to its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        if not isinstance(node.value, str) or not node.value:
            continue
        tail = node.value.rsplit("/", 1)[-1]
        stem = tail.split(".", 1)[0]
        # A problem id has no whitespace. Requiring the whole component to match is what
        # keeps prose like "connect() failed inside the sealed environment" out of this.
        if stem and not any(c.isspace() for c in stem):
            found.setdefault(stem, node.lineno)
    return found


def test_no_fixture_names_a_holdout_problem() -> None:
    """Issue #70, asserted rather than reviewed.

    Comparison is by **stem, across both languages**, which is stricter than the manifest's
    per-language split and deliberately so: the exercise statement is shared, so reading
    `python/bowling` is reading the statement of `javascript/bowling` too.
    `test_the_statement_is_the_same_problem_in_both_languages` is the evidence for that,
    checked against the corpus rather than assumed here.

    Failure is easy to fix — `test_the_holdout_free_stems_are_not_empty` prints the stems
    that are safe in each language.
    """
    conf = {
        p.id.rsplit("/", 1)[-1].removesuffix(perturb.PERTURBED_SUFFIX)
        for p in load_manifest().subset(SubsetKind.CONFIRMATION).problems  # type: ignore[union-attr]
    }
    assert conf, "no confirmation set to check against"

    offenders: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        for stem, line in _named_stems(path).items():
            if stem in conf:
                offenders.append(f"{path.name}:{line} names {stem!r}")
    assert not offenders, (
        "these fixtures are held-out problems, so every future instrument fix would be "
        "iterated against the holdout until they pass (#70, EV-11):\n  "
        + "\n  ".join(offenders)
    )


def test_the_holdout_free_stems_are_not_empty() -> None:
    """The rule must leave something to pick from, per language.

    A guard that forbids every available fixture is not a guard, it is a wall — and the
    next person to hit it would delete it rather than route around it. This also prints the
    list, so a `test_no_fixture_names_a_holdout_problem` failure comes with its own remedy.
    """
    m = load_manifest()
    conf = {
        p.id.rsplit("/", 1)[-1].removesuffix(perturb.PERTURBED_SUFFIX)
        for p in m.subset(SubsetKind.CONFIRMATION).problems  # type: ignore[union-attr]
    }
    dev = m.subset(SubsetKind.DEV)
    assert dev is not None
    for language in Language:
        safe = sorted(
            {p.id.rsplit("/", 1)[-1] for p in dev.problems if p.language is language} - conf
        )
        assert len(safe) >= 5, f"{language}: only {safe} are usable as fixtures"


@real
def test_the_statement_is_the_same_problem_in_both_languages() -> None:
    """Why `test_no_fixture_names_a_holdout_problem` compares stems and not ids.

    26 exercises appear in both languages, and for every one of them `.docs/instructions.md`
    is **byte-identical**. The tests differ and the stub differs; the problem a person reads
    does not. So a stem held out in one language is held out in both, and a per-language
    rule would let `python/bowling` be read freely while `javascript/bowling` is the thing
    being held back.

    The language-specific appendix is excluded on purpose: it genuinely differs for 15 of
    the 26, and claiming identity of the whole rendered instructions would be false.
    """
    root = benchmark_root()
    assert root is not None

    def statement(language: Language, pid: str) -> str | None:
        base = root / language.value / "exercises" / "practice"
        path = base / pid / ".docs" / "instructions.md"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def ids(language: Language) -> set[str]:
        base = root / language.value / "exercises" / "practice"
        return {d.name for d in base.iterdir() if d.is_dir()}

    shared = sorted(ids(Language.PYTHON) & ids(Language.JAVASCRIPT))
    assert len(shared) >= 20, f"only {len(shared)} shared stems; the argument has weakened"
    differing = [
        pid
        for pid in shared
        if statement(Language.PYTHON, pid) != statement(Language.JAVASCRIPT, pid)
    ]
    assert not differing, (
        f"these no longer share a statement across languages: {differing} — the stem-wide "
        "holdout rule is stricter than it needs to be for them"
    )


# --- JavaScript, end to end ---------------------------------------------------
# Greptile P2 on #61: twelve of the twenty perturbed problems are JavaScript, and the only
# execution test was hard-coded to Python. `apply()`'s filename rename and import rewrite
# were covered by `plan()` alone — which checks a shape, not a fact, and is precisely how
# issue #60 happened. Two tiers now: a static wiring check over every JS problem in the
# manifest, and one real jest run.

IMPORT_RE = re.compile(r"import\s*\{([^}]*)\}\s*from\s*['\"]\./([^'\"]+)['\"]")


def _js_test_file(dest: Path) -> Path:
    return next(f for f in dest.rglob("*.spec.js") if ".meta" not in f.parts)


@real
def test_every_perturbed_javascript_problem_is_wired_up(tmp_path: Path) -> None:
    """The import must resolve, and the names it asks for must be exported.

    This is the failure mode that matters for `apply()`: the module file is renamed in one
    place and the import rewritten in another, and if the two disagree the problem is
    unrunnable. Static, so it can cover all twelve rather than one.
    """
    root = benchmark_root()
    assert root is not None
    m = load_manifest()
    pert = m.subset(SubsetKind.PERTURBED)
    assert pert is not None and pert.seed is not None

    js = [p for p in pert.problems if p.language is Language.JAVASCRIPT]
    assert len(js) == 12, "the manifest's JavaScript perturbations changed; recheck this"

    for problem in js:
        src = problem.id.removesuffix(perturb.PERTURBED_SUFFIX)
        p = perturb.generate(root, tmp_path, src, Language.JAVASCRIPT, pert.seed)
        dest = tmp_path / p.perturbed_id

        spec = _js_test_file(dest)
        match = IMPORT_RE.search(spec.read_text(encoding="utf-8"))
        assert match, f"{src}: no relative import found in {spec.name}"

        names, module = match.group(1), match.group(2)
        assert (dest / f"{module}.js").is_file(), (
            f"{src}: the test imports './{module}' but no such file exists — the filename "
            f"rename and the import rewrite disagree"
        )
        stub_source = (dest / f"{module}.js").read_text(encoding="utf-8")
        for name in (n.strip() for n in names.split(",") if n.strip()):
            assert re.search(
                rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", stub_source
            ), f"{src}: the test imports {name!r} but the stub does not export it"

        olds = {r.old for r in p.renames} | (
            {p.module_rename.old} if p.module_rename else set()
        )
        for path in dest.rglob("*.js"):
            text = path.read_text(encoding="utf-8")
            for old in olds:
                assert not survives(old, text), f"{src}: {old!r} survived in {path.name}"


@real
def test_a_real_javascript_problem_stays_solvable_after_perturbation(tmp_path: Path) -> None:
    """The Python solvability test's counterpart, run through jest.

    Gated on npm being reachable — the exercises pull babel and jest — and skipped rather
    than failed when it is not, because an offline machine is not a broken perturbation.
    """
    root = benchmark_root()
    assert root is not None
    if shutil.which("npm") is None:
        pytest.skip("npm not installed; JavaScript solvability is genuinely unverified here")

    name = "javascript/exercises/practice/triangle"  # dev in both languages
    p = perturb.generate(root, tmp_path, name, Language.JAVASCRIPT, SEED)
    dest = tmp_path / p.perturbed_id

    spec = _js_test_file(dest)
    stub = dest / spec.name.replace(".spec.js", ".js")
    proof = next(iter((dest / ".meta").glob("*.js")))
    shutil.copyfile(proof, stub)

    install = subprocess.run(
        ["npm", "install", "--silent", "--no-audit", "--no-fund"],
        cwd=dest,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if install.returncode != 0:
        pytest.skip(f"npm install failed (offline?): {install.stderr[-300:]}")

    proc = subprocess.run(
        ["npx", "jest"], cwd=dest, capture_output=True, text=True, timeout=600
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"{name}: the reference solution stopped passing after perturbation — a rename "
        f"missed a file.\n{output[-2000:]}"
    )

    # Exercism marks all but the first case `xtest`, so "0 passed, 14 skipped" is a green
    # run that proves nothing. The count is what makes this test mean something.
    passed = re.search(r"Tests:.*?(\d+) passed", output)
    assert passed and int(passed.group(1)) >= 1, f"no test actually executed:\n{output[-800:]}"
