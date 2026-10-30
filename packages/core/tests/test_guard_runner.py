"""The D148 guard runner.

Every case runs over **fixture** guards in `tmp_path`, never the repository's real ones. A
test that asserts against live guards starts failing the day a theme lands a real violation,
which would make the ratchet's own test the first thing anyone disables.

No test imported `scripts/` before this one, so the runner is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_GUARDS = REPO_ROOT / "scripts" / "guards"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_guard_runner", REAL_GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, which is None for a module that is mid-exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# A guard module, parameterised. `find` reports one violation per line holding the needle,
# which is what every real guard does: read source text, never behaviour.
GUARD_SRC = """
from pathlib import Path

from guards import Guard, Violation

{baseline}


def find(files):
    out = []
    for path in files:
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if "{needle}" in line:
                out.append(Violation(path=path.name, line=i, text=line.strip()))
    return tuple(out)


GUARD = Guard(
    name="{name}",
    rule="{rule}",
    corpus=("{corpus}",),
    find=find,
    baseline=BASELINE,
)
"""


PKG = "fixguards"
"""The fixture package's directory name, deliberately not `guards`. The guard modules below
import `Guard` and `Violation` from the real package, so these cases exercise the real types
rather than a copy — two classes of the same name would fail `isinstance` for a reason that
has nothing to do with the runner."""


def _set_floor(root: Path, spec: str) -> None:
    """Declare the fixture repository's interpreter floor, where the real one declares it."""
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "fixture"\nrequires-python = "{spec}"\n'
    )


def _world(root: Path) -> Path:
    """A repository root holding a pyproject, a guards package and a `src/` to read.

    The pyproject is there because the real root has one: the runner reads the interpreter
    floor from it before anything else, and a root without one is refused. Its floor
    is the project's own, which the suite's interpreter meets."""
    root.mkdir(parents=True, exist_ok=True)
    _set_floor(root, ">=3.12")
    pkg = root / PKG
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    src = root / "src"
    src.mkdir()
    (src / "a.py").write_text("import flock\nx = 1\n")
    (src / "b.py").write_text("y = 2\n")
    return root


@pytest.fixture
def world(tmp_path: Path) -> Path:
    return _world(tmp_path)


def write_guard(
    world: Path, name: str, *, baseline: str, needle: str = "flock", corpus: str = "src/*.py"
) -> None:
    (world / PKG / f"{name}.py").write_text(
        GUARD_SRC.format(
            baseline=baseline,
            needle=needle,
            name=name,
            rule=f"{name} has one home",
            corpus=corpus,
        )
    )


def run(world: Path, *, tighten: bool = False) -> int:
    return runner.run(world / PKG, world, tighten=tighten)


def test_no_guards_is_clean_and_says_so(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner lands before any guard exists, so an empty package passes — but it
    must print the count. "0 guards" and "all guards pass" reading the same is the trap."""
    assert run(world) == 0
    out = capsys.readouterr().out
    assert "0 guards" in out
    assert "GUARDS PASSED" in out


def test_violations_equal_to_baseline_pass(world: Path) -> None:
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1}')
    assert run(world) == 0


def test_a_violation_above_baseline_fails_and_names_file_line_and_rule(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (world / "src" / "b.py").write_text("y = 2\nimport flock\n")
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1}')
    assert run(world) == 1
    out = capsys.readouterr().out
    assert "GUARDS FAILED" in out
    assert "b.py" in out
    assert "b.py:2" in out
    assert "one_flock has one home" in out


def test_a_failing_file_prints_five_rows_and_counts_the_rest(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rows say where to look; past five they are counted, not printed. The count
    and the verdict are unchanged."""
    (world / "src" / "b.py").write_text("".join(f"import flock  # {i}\n" for i in range(1, 9)))
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1}')
    assert run(world) == 1
    out = capsys.readouterr().out
    assert "b.py has 8 violation(s), baseline allows 0" in out
    assert "b.py:5:" in out and "b.py:6:" not in out
    assert "... and 3 more" in out


def test_slack_fails_and_points_at_tighten(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The owner's call (2026-10-03): a baseline looser than the code advertises protection it
    is not giving, so it fails — with the one command that fixes it."""
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 4}')
    assert run(world) == 1
    out = capsys.readouterr().out
    assert "down to 1 from a baseline of 4" in out
    assert "--tighten" in out


def test_a_baseline_naming_a_deleted_file_is_slack(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_guard(
        world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1, "gone.py": 2}'
    )
    assert run(world) == 1
    assert "gone.py is down to 0 from a baseline of 2" in capsys.readouterr().out


def test_a_module_without_GUARD_fails_rather_than_being_skipped(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (world / PKG / "broken.py").write_text("x = 1\n")
    assert run(world) == 1
    assert "no module-level `GUARD`" in capsys.readouterr().out


def test_a_name_disagreeing_with_its_filename_fails(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = GUARD_SRC.format(
        baseline="BASELINE: dict[str, int] = {}",
        needle="flock",
        name="something_else",
        rule="r",
        corpus="src/*.py",
    )
    (world / PKG / "one_flock.py").write_text(src)
    assert run(world) == 1
    assert "expected 'one_flock'" in capsys.readouterr().out


def test_a_corpus_matching_nothing_fails(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A guard written against a directory that is later split out from under it would
    otherwise report clean forever."""
    write_guard(
        world, "one_flock", baseline="BASELINE: dict[str, int] = {}", corpus="moved/*.py"
    )
    assert run(world) == 1
    assert "corpus matches no files" in capsys.readouterr().out


def test_an_interpreter_below_the_floor_is_refused_before_any_guard_is_imported(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-3.96. The floor is moved where the runner reads it, the fixture's pyproject, rather
    than by patching `sys.version_info`. The guard module raises when imported, so reaching
    discovery at all would surface as that error instead of the refusal."""
    _set_floor(world, ">=99.0")
    (world / PKG / "loud.py").write_text('raise RuntimeError("a guard was imported")\n')
    assert run(world) == 1
    out = capsys.readouterr().out
    assert "below the project's floor of 99.0" in out
    assert "requires-python" in out


def test_tighten_refuses_below_the_floor_and_rewrites_nothing(world: Path) -> None:
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 4}')
    before = (world / PKG / "one_flock.py").read_text()
    _set_floor(world, ">=99.0")
    assert run(world, tighten=True) == 1
    assert (world / PKG / "one_flock.py").read_text() == before


def test_a_root_without_a_pyproject_is_refused(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the floor the runner cannot know the source parses, so it refuses rather than
    guess — the same rule as a corpus that matches no files."""
    (world / "pyproject.toml").unlink()
    assert run(world) == 1
    assert "cannot read the interpreter floor" in capsys.readouterr().out


def test_a_requires_python_with_no_lower_bound_is_refused(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_floor(world, "<4")
    assert run(world) == 1
    assert "names no '>=X.Y' floor" in capsys.readouterr().out


def test_the_floor_is_read_from_a_compound_specifier(world: Path) -> None:
    _set_floor(world, ">=3.12,<4")
    assert runner.floor(world) == (3, 12)


def test_tighten_lowers_the_baseline_and_the_next_run_is_clean(world: Path) -> None:
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 4}')
    assert run(world) == 1
    assert run(world, tighten=True) == 0
    assert '"a.py": 1,' in (world / PKG / "one_flock.py").read_text()
    assert run(world) == 0


def test_tighten_drops_a_path_that_is_now_clean(world: Path) -> None:
    write_guard(
        world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1, "b.py": 3}'
    )
    assert run(world, tighten=True) == 0
    text = (world / PKG / "one_flock.py").read_text()
    assert "b.py" not in text
    assert run(world) == 0


def test_tighten_is_idempotent(world: Path) -> None:
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 9}')
    assert run(world, tighten=True) == 0
    first = (world / PKG / "one_flock.py").read_text()
    assert run(world, tighten=True) == 0
    assert (world / PKG / "one_flock.py").read_text() == first


def test_an_empty_baseline_renders_as_an_empty_dict(world: Path) -> None:
    """`--tighten` on a guard with no violations must leave parseable source, not `{\n}`."""
    write_guard(
        world, "no_needle", baseline='BASELINE: dict[str, int] = {"a.py": 2}', needle="zzz"
    )
    assert run(world, tighten=True) == 0
    assert "BASELINE: dict[str, int] = {}" in (world / PKG / "no_needle.py").read_text()
    assert run(world) == 0


def test_each_discovery_reads_the_directory_it_was_given(world: Path, tmp_path: Path) -> None:
    """Two fixture packages share a name in different directories; the second must not be
    served from the first's import cache. The runner discovers once, but a function that
    only works once is a trap for everything that calls it twice."""
    write_guard(world, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1}')
    assert run(world) == 0

    other = _world(tmp_path / "other")
    (other / "src" / "a.py").write_text("import flock\nimport flock\n")
    write_guard(other, "one_flock", baseline='BASELINE: dict[str, int] = {"a.py": 1}')
    assert run(other) == 1


def test_the_repositorys_own_guards_run(capsys: pytest.CaptureFixture[str]) -> None:
    """The real package, at whatever state the wave has reached. Asserts the runner starts
    and reports — never a violation count, which is each theme's to move."""
    assert runner.run() == 0 or "GUARDS FAILED" in capsys.readouterr().out


def test_every_guard_in_the_real_package_is_discovered() -> None:
    """The real directory is the package `Guard` lives in, which no fixture directory is.
    Discovery purged it and re-imported it, so each real guard built a second `Guard` class
    and failed `isinstance`: the first real guard (`storage`) found it."""
    stems = sorted(
        p.stem
        for p in REAL_GUARDS.glob("*.py")
        if p.name not in runner.NOT_GUARDS and not p.name.startswith("_")
    )
    assert [g.name for g in runner.discover(REAL_GUARDS)] == stems


def test_render_baseline_is_sorted(world: Path) -> None:
    rendered = runner.render_baseline({"z.py": 1, "a.py": 2})
    assert rendered.index('"a.py"') < rendered.index('"z.py"')


@pytest.fixture(autouse=True)
def _clean_sys_path() -> object:
    """`discover` puts a guards package's parent on `sys.path`; tmp directories must not
    accumulate there across the session."""
    before = list(sys.path)
    yield
    sys.path[:] = before
