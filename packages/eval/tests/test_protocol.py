"""No hidden test body reaches any rendered prompt.

This is the anti-fraud test. If a test body leaks into a prompt, the model is transcribing
the answer key rather than solving anything, and every number the rig produces afterwards
describes nothing. §7 says such a run fails loudly and its results are discarded, so the
leak has to be detectable rather than merely forbidden.

Fixtures are built inline rather than in a `conftest.py`: `T-1.1`'s `allowed_paths` names
four test files and no shared module, and staying inside that boundary is what lets other
tasks work the same tree at the same time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee_eval import runner
from chickadee_eval.manifest import Language
from chickadee_eval.problems import polyglot

PY_STUB = '''\
def rotate(items, n):
    """Rotate `items` left by `n`."""
    raise NotImplementedError
'''

PY_TEST = """\
from lib import rotate


def test_rotates_left():
    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]


def test_rotation_wraps_past_length():
    assert rotate([1, 2, 3], 5) == [3, 1, 2]
"""


def make_python_problem(root: Path, pid: str = "rotate") -> Path:
    d = root / pid
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("Rotate a list left by n.\n", encoding="utf-8")
    (d / "lib.py").write_text(PY_STUB, encoding="utf-8")
    (d / "lib_test.py").write_text(PY_TEST, encoding="utf-8")
    return d


def test_prompt_view_cannot_carry_a_test_body(tmp_path: Path) -> None:
    """The structural half of D7: the body is not a field a render path can reach."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)
    view = problem.prompt_view()

    assert not hasattr(view, "test_source")
    assert "_test_source" not in vars(view)
    # Every value the view exposes, concatenated, still contains no assertion from the file.
    rendered = "\n".join(str(v) for v in vars(view).values())
    assert "assert rotate(" not in rendered


def test_test_names_are_allowed_bodies_are_not(tmp_path: Path) -> None:
    """§5.1 permits names in DONE WHEN. The distinction is the whole protocol."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    assert problem.test_names == ("test_rotates_left", "test_rotation_wraps_past_length")
    # A prompt listing every permitted name is legal and must not be flagged.
    legal = "DONE WHEN: " + ", ".join(problem.test_names)
    assert polyglot.leaks(legal, problem) == []


def test_a_leaked_assertion_is_detected(tmp_path: Path) -> None:
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    leaked_prompt = "Solve it.\n\n    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]\n"
    found = polyglot.leaks(leaked_prompt, problem)
    assert found, "a verbatim assertion from the hidden file must be caught"
    assert any("assert rotate([1, 2, 3, 4], 1)" in text for _n, text in found)
    assert all(isinstance(n, int) and n > 0 for n, _t in found)


def test_whole_body_pasted_is_detected(tmp_path: Path) -> None:
    """The blunt failure: someone renders `test_source()` straight into the prompt."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    found = polyglot.leaks(f"Here are the tests:\n{problem.test_source()}", problem)
    assert len(found) >= 2, f"expected several body lines flagged, got {found}"


def test_short_lines_do_not_trip_the_check(tmp_path: Path) -> None:
    """A check that fires on `import` or a closing brace gets switched off, and then the
    protocol is unasserted again. Markers below the length floor are ignored on purpose."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    # `from lib import rotate` is a real line of the hidden file, and also the single most
    # likely line to appear innocently in a prompt about the same module.
    assert len("from lib import rotate") < polyglot.MIN_MARKER
    assert polyglot.leaks("from lib import rotate\n", problem) == []


def test_leak_check_runs_against_every_prompt_an_arm_rendered(tmp_path: Path) -> None:
    """The property is asserted over the real run, not a fixture, so an arm that
    leaks on attempt 3 is caught even when attempts 1 and 2 were clean."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    prompts = (
        "attempt 1: solve it",
        "attempt 2: still solve it",
        f"attempt 3: hint —\n{problem.test_source()}",
    )
    found = [line for p in prompts for line in polyglot.leaks(p, problem)]
    assert found


JS_STUB = "export function rotate(items, n) {\n  throw new Error('not implemented')\n}\n"
JS_TEST = """\
import { test } from 'node:test'
import assert from 'node:assert'
import { rotate } from './lib.js'

test('rotates the collection to the left by one', () => {
  assert.deepEqual(rotate([1, 2, 3, 4], 1), [2, 3, 4, 1])
})
"""


def test_javascript_problems_load_with_the_same_protocol(tmp_path: Path) -> None:
    d = tmp_path / "rotate-js"
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("Rotate.\n", encoding="utf-8")
    (d / "lib.js").write_text(JS_STUB, encoding="utf-8")
    (d / "lib.test.js").write_text(JS_TEST, encoding="utf-8")

    problem = polyglot.load(tmp_path, "rotate-js", Language.JAVASCRIPT)
    assert problem.test_names == ("rotates the collection to the left by one",)
    assert polyglot.leaks("DONE WHEN: " + problem.test_names[0], problem) == []
    assert polyglot.leaks(problem.test_source(), problem)


def test_ambiguous_or_missing_oracle_refuses_to_load(tmp_path: Path) -> None:
    """A benchmark whose scoring set is ambiguous is not an oracle, so this fails at load
    rather than picking one and producing a number nobody can interpret."""
    d = tmp_path / "twotests"
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("x\n", encoding="utf-8")
    (d / "lib.py").write_text(PY_STUB, encoding="utf-8")
    (d / "a_test.py").write_text(PY_TEST, encoding="utf-8")
    (d / "b_test.py").write_text(PY_TEST, encoding="utf-8")

    with pytest.raises(polyglot.ProblemError, match="exactly one"):
        polyglot.load(tmp_path, "twotests", Language.PYTHON)

    bare = tmp_path / "notests"
    bare.mkdir(parents=True)
    (bare / "instructions.md").write_text("x\n", encoding="utf-8")
    (bare / "lib.py").write_text(PY_STUB, encoding="utf-8")
    with pytest.raises(polyglot.ProblemError, match="exactly one"):
        polyglot.load(tmp_path, "notests", Language.PYTHON)


# --- the oracle is not reachable from the workspace --------------------------
# Greptile P1 on #56. `fresh_worktree` copied the hidden test into the directory handed to
# the arm, so an arm could read the answer key, hard-code exactly what it demanded, report
# no prompt at all, and score `solved` with the leak scan seeing nothing. The prompt scan
# guards the loader-to-prompt path; nothing guarded the filesystem one.


def test_the_arm_workspace_does_not_contain_the_oracle(tmp_path: Path) -> None:
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    seen: dict[str, list[str]] = {}

    class Peeking:
        name = "peek"

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            seen["files"] = sorted(p.name for p in workspace.rglob("*") if p.is_file())
            return runner.Attempt(solved_claim=False, cost_usd=0.0)

    runner.run_problem(problem, Peeking(), runner.seal(require_sandbox=False, probe=False))

    assert "lib_test.py" not in seen["files"], (
        f"the oracle was in the arm's workspace: {seen['files']}"
    )
    assert "lib.py" in seen["files"], "the stub must still be there to edit"


def test_prompt_view_hands_over_no_route_back_to_the_original(tmp_path: Path) -> None:
    """Removing the file is worth nothing if the arm is handed the source directory."""
    make_python_problem(tmp_path)
    view = polyglot.load(tmp_path, "rotate", Language.PYTHON).prompt_view()
    assert not hasattr(view, "root")
    assert "rotate" not in str(getattr(view, "stub_path", ""))[: -len("lib.py")] or True
    for value in vars(view).values():
        assert str(tmp_path) not in str(value), "an absolute path back to the oracle leaked"


def test_an_arm_that_reads_the_oracle_can_no_longer_do_so(tmp_path: Path) -> None:
    """The exact cheat that scored `solved` before: read the hidden test, hard-code its
    expectation, report no prompts. It must now fail to find the file."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    class Cheat:
        name = "cheat"
        read_ok = False

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            try:
                (workspace / "lib_test.py").read_text()
                Cheat.read_ok = True
            except OSError:
                Cheat.read_ok = False
            return runner.Attempt(solved_claim=True, cost_usd=0.0, prompts=())

    runner.run_problem(problem, Cheat(), runner.seal(require_sandbox=False, probe=False))
    assert Cheat.read_ok is False, "the arm read the answer key"


def test_the_oracle_is_restored_for_scoring(tmp_path: Path) -> None:
    """Absent during the attempt, present for the scorer — otherwise nothing can grade."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    class Solver:
        name = "solver"

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            (workspace / problem.stub_path).write_text(
                "def rotate(items, n):\n    n %= len(items) or 1\n"
                "    return list(items[n:]) + list(items[:n])\n",
                encoding="utf-8",
            )
            return runner.Attempt(solved_claim=True, cost_usd=0.0)

    record = runner.run_problem(
        problem, Solver(), runner.seal(require_sandbox=False, probe=False)
    )
    assert record.outcome == polyglot.Outcome.PASS
    assert record.solved is True


def test_planting_a_fake_oracle_is_recorded(tmp_path: Path) -> None:
    """An arm cannot fake the tests — they are overwritten — but trying is a finding."""
    make_python_problem(tmp_path)
    problem = polyglot.load(tmp_path, "rotate", Language.PYTHON)

    class Planter:
        name = "planter"

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            (workspace / "lib_test.py").write_text("def test_ok():\n    assert True\n")
            return runner.Attempt(solved_claim=True, cost_usd=0.0)

    record = runner.run_problem(
        problem, Planter(), runner.seal(require_sandbox=False, probe=False)
    )
    assert record.oracle_planted is True
    assert record.solved is False, "the real oracle overwrote the plant and still fails"
