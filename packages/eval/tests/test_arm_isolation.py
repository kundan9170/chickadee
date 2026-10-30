"""Arm isolation: two arms may differ in exactly one variable, and roles cannot blur.

`19-eval` §5.7 — *"P0 and P1 share a config file with a single `planner.enabled` flip, and
the runner asserts the two resolved configs differ in no other field before either arm
starts."* Before, because the cost of finding out afterwards is the whole run.

And the boundary: the E1 external diagnostic is an anti-strawman check, *not* an E2
headline. That distinction survives only if it is computed rather than asserted by whoever
writes the report — which is months later, when only the numbers remain.

Also here: D4's anti-strawman duty on the naive control, and the §2/§5.1 collision over
retry feedback. The control is the arm G1 scrutinises, so a weakening of it is a finding
against the harness's numbers rather than a saving.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from chickadee.client import ChatRequest
from chickadee.schema import Completion, RetryParams, StopReason
from chickadee.schema import Usage as ModelUsage
from chickadee_eval.arms import api_baseline, e1_naive
from chickadee_eval.arms.api_baseline import ApiBaselineArm, HarnessSpec, Role, Usage
from chickadee_eval.arms.e1_naive import ArmPrices, NaiveConfig
from chickadee_eval.manifest import Language
from chickadee_eval.problems import polyglot
from chickadee_eval.protocol import validate_arms_differ_in_one_variable

STUB = "def f(x):\n    raise NotImplementedError\n"
TEST = (
    "from lib import f\n\n\n"
    "def test_f_doubles_its_argument():\n"
    '    assert f(2) == 4, "doubling two has to give four"\n'
)


def make_problem(root: Path, pid: str = "p") -> polyglot.LoadedProblem:
    d = root / pid
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("Double it.\n", encoding="utf-8")
    (d / "lib.py").write_text(STUB, encoding="utf-8")
    (d / "lib_test.py").write_text(TEST, encoding="utf-8")
    return polyglot.load(root, pid, Language.PYTHON)


# --- §5.7: one variable, checked before either run ---------------------------


def test_two_arms_differing_in_one_field_are_accepted() -> None:
    a = dict(e1_naive.arm_config(NaiveConfig(model_id="m", max_attempts=3)))
    b = dict(a, max_attempts=2)
    assert validate_arms_differ_in_one_variable(a, b) == ["max_attempts"]


def test_two_arms_differing_in_several_fields_are_visible() -> None:
    """The check reports every difference, not just the first. A pair differing in three
    fields that reported one would be worse than no check at all."""
    a = dict(e1_naive.arm_config(NaiveConfig(model_id="m", max_attempts=3)))
    b = dict(a, max_attempts=2, model_id="other", max_output_tokens=1)
    assert validate_arms_differ_in_one_variable(a, b) == [
        "max_attempts",
        "max_output_tokens",
        "model_id",
    ]


def test_arm_config_is_flat() -> None:
    """§5.7 compares field by field. A nested value compares as one opaque blob, so two arms
    differing in three nested fields would read as differing in one — the check passing for
    the reason it exists to catch."""
    config = e1_naive.arm_config(NaiveConfig(model_id="m"))
    for key, value in config.items():
        assert not isinstance(value, dict), f"{key} is nested"


def test_identical_arms_report_no_difference() -> None:
    a = dict(e1_naive.arm_config(NaiveConfig(model_id="m")))
    assert validate_arms_differ_in_one_variable(a, dict(a)) == []


# --- §2: the two roles of the api_baseline driver ----------------------------


def spec() -> HarnessSpec:
    return HarnessSpec(
        name="aider", version="0.60.0", argv=("aider", "--yes"), model_id="frontier-x"
    )


def test_an_e1_diagnostic_arm_is_never_headline_eligible() -> None:
    """The E1 external diagnostic is an anti-strawman check only."""
    arm = ApiBaselineArm(spec=spec(), role=Role.E1_DIAGNOSTIC)
    arm.last_usage = Usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
    assert arm.headline_eligible is False


def test_an_e2_baseline_with_auditable_accounting_is_eligible() -> None:
    arm = ApiBaselineArm(spec=spec(), role=Role.E2_BASELINE)
    arm.last_usage = Usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
    assert arm.headline_eligible is True


def test_an_e2_baseline_without_accounting_is_not_eligible() -> None:
    arm = ApiBaselineArm(spec=spec(), role=Role.E2_BASELINE)
    arm.last_usage = Usage()
    assert arm.headline_eligible is False


def test_tokens_without_a_price_are_not_headline_eligible() -> None:
    """Greptile P1 on #58. Tokens and dollars are separate facts, and the headline claim is
    about cost — an arm whose cost column is a defaulted zero cannot carry one."""
    arm = ApiBaselineArm(spec=spec(), role=Role.E2_BASELINE)
    arm.last_usage = Usage(prompt_tokens=100, completion_tokens=50)
    assert arm.last_usage.auditable is True, "tokens are present"
    assert arm.last_usage.priced is False, "money is not"
    assert arm.headline_eligible is False


def test_headline_eligibility_cannot_be_set_by_a_caller() -> None:
    """Computed, never assigned — a flag could be flipped by the report writer."""
    arm = ApiBaselineArm(spec=spec(), role=Role.E1_DIAGNOSTIC)
    with pytest.raises(AttributeError):
        arm.headline_eligible = True  # type: ignore[misc]


def test_a_headline_built_on_diagnostic_arms_is_refused() -> None:
    diagnostic = ApiBaselineArm(spec=spec(), role=Role.E1_DIAGNOSTIC)
    diagnostic.last_usage = Usage(prompt_tokens=1, completion_tokens=1)
    with pytest.raises(api_baseline.BaselineBlocked, match="anti-strawman"):
        api_baseline.assert_not_headline([diagnostic])


def test_the_role_is_required() -> None:
    """Inferring it would let a diagnostic run become a headline by omission."""
    with pytest.raises(TypeError):
        ApiBaselineArm(spec=spec())  # type: ignore[call-arg]


# --- §7: refusal rather than a downgraded baseline ---------------------------


def test_usage_reports_unknown_as_none_not_zero() -> None:
    """Zero and unknown are different facts and only one of them can be summed. A harness
    reporting nothing must not look like the cheapest arm."""
    usage = api_baseline.normalize_usage("no accounting here")
    assert usage.prompt_tokens is None
    assert usage.cost_usd is None
    assert not usage.auditable


def test_usage_is_read_from_json_when_the_harness_emits_it() -> None:
    out = '{"usage": {"prompt_tokens": 120, "completion_tokens": 30, "cost_usd": 0.004}}'
    usage = api_baseline.normalize_usage(out)
    assert (usage.prompt_tokens, usage.completion_tokens) == (120, 30)
    assert usage.cost_usd == 0.004
    assert usage.auditable


def test_usage_is_read_from_prose_when_that_is_all_there_is() -> None:
    usage = api_baseline.normalize_usage("Tokens: 1,200 sent, 340 received. Cost: $0.0123")
    assert usage.completion_tokens == 340
    assert usage.cost_usd == 0.0123


def test_a_version_mismatch_blocks_rather_than_being_absorbed() -> None:
    """§6 pins every external harness version. A pin nobody checks is a note."""
    s = HarnessSpec(
        name="aider",
        version="0.60.0",
        argv=("python3", "--version"),
        model_id="m",
        version_argv=("python3", "--version"),
    )
    with pytest.raises(api_baseline.BaselineBlocked, match="does not describe"):
        api_baseline.verify_pin(s)


def test_a_missing_harness_blocks() -> None:
    s = HarnessSpec(name="nope", version="1", argv=("definitely-not-installed",), model_id="m")
    with pytest.raises(api_baseline.BaselineBlocked, match="not installed"):
        api_baseline.verify_pin(s)


def test_an_unpriced_timeout_blocks_e2_rather_than_billing_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Greptile P1 on #58. A timeout is not a free run — the harness billed for everything
    it did before the clock ran out, and $0 in the archive would understate the baseline's
    cost, which is the one number the comparison is about."""
    problem = make_problem(tmp_path)

    def boom(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(cmd="aider", timeout=1, output=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", boom)
    arm = ApiBaselineArm(spec=spec(), role=Role.E2_BASELINE, timeout_s=1)
    with pytest.raises(api_baseline.AccountingUnavailable, match="unaccounted"):
        arm.attempt(problem.prompt_view(), tmp_path / "p")


def test_a_timeout_keeps_whatever_accounting_the_harness_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial accounting beats an invented zero — and it is what makes the E2 refusal
    message able to say how much was already spent."""
    problem = make_problem(tmp_path)

    def boom(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(
            cmd="aider", timeout=1, output=b"Tokens: 900 sent, 100 received.", stderr=None
        )

    monkeypatch.setattr(subprocess, "run", boom)
    arm = ApiBaselineArm(spec=spec(), role=Role.E1_DIAGNOSTIC, timeout_s=1)
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert arm.last_usage.prompt_tokens == 900
    assert "unpriced" in result.detail and "exceeded" in result.detail


def test_a_diagnostic_arm_labels_its_unpriced_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = make_problem(tmp_path)

    class Proc:
        returncode = 0
        stdout = "prompt_tokens: 100 completion_tokens: 20"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: Proc())
    arm = ApiBaselineArm(spec=spec(), role=Role.E1_DIAGNOSTIC)
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert result.cost_usd == 0.0
    assert "unpriced" in result.detail
    assert "in=100" in result.detail and "out=20" in result.detail


def test_tokens_without_a_price_block_an_e2_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zero does not stay put: it reaches the archive, `spent_usd` and the
    `hard_cost_usd` cap, so a baseline could spend its whole budget with the cap never
    seeing a dollar."""
    problem = make_problem(tmp_path)

    class Proc:
        returncode = 0
        stdout = "prompt_tokens: 100 completion_tokens: 20"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: Proc())
    arm = ApiBaselineArm(spec=spec(), role=Role.E2_BASELINE)
    with pytest.raises(api_baseline.AccountingUnavailable, match="no cost"):
        arm.attempt(problem.prompt_view(), tmp_path / "p")


# --- one driver, many problems -----------------------------------------------
# Greptile P1 on #58. `runner.run_arm` builds one driver and hands it every problem in
# turn, so anything an arm keeps on `self` outlives the problem it belongs to.


def test_the_arm_redacts_against_the_problem_it_was_given(tmp_path: Path) -> None:
    """The serious half of the finding. With a single `LoadedProblem` fixed at
    construction, the second problem's failures were redacted against the *first*
    problem's oracle — so its real oracle lines survived into the retry prompt.
    `run_problem` scans against the right problem and would have raised, discarding the
    run, but the arm was unusable past one problem either way."""
    first = make_problem(tmp_path, "p1")
    second_dir = tmp_path / "p2"
    second_dir.mkdir()
    (second_dir / "instructions.md").write_text("Triple it.\n", encoding="utf-8")
    (second_dir / "lib.py").write_text(STUB, encoding="utf-8")
    (second_dir / "lib_test.py").write_text(
        "from lib import f\n\n\n"
        "def test_f_triples_its_argument():\n"
        '    assert f(3) == 9, "tripling three has to give nine"\n',
        encoding="utf-8",
    )
    second = polyglot.load(tmp_path, "p2", Language.PYTHON)

    second_failure = (
        "=================================== FAILURES ===================================\n"
        "    def test_f_triples_its_argument():\n"
        '>       assert f(3) == 9, "tripling three has to give nine"\n'
        "E       AssertionError: tripling three has to give nine\n"
    )
    scores = iter([(False, second_failure), (True, "")])
    arm = e1_naive.NaiveArm(
        client=_StubClient(["x\n", "y\n"]),
        config=NaiveConfig(model_id="m", max_attempts=2),
        oracles={first.id: first, second.id: second},
        score=lambda _ws: next(scores),
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    arm.attempt(second.prompt_view(), ws)

    retry = arm.prompts[1]
    assert polyglot.leaks(retry, second) == [], "the second problem's oracle reached a prompt"
    assert "tripling three has to give nine" not in retry.split("AssertionError")[0]


def test_an_unregistered_problem_is_refused_not_guessed(tmp_path: Path) -> None:
    """An arm that cannot identify the oracle cannot promise anything about D7, and
    guessing is how the wrong oracle reached a prompt in the first place."""
    known = make_problem(tmp_path, "known")
    stranger = make_problem(tmp_path, "stranger")
    arm = e1_naive.NaiveArm(
        client=_StubClient(["x\n"]),
        config=NaiveConfig(model_id="m", max_attempts=1, temperature_ladder=(0.0,)),
        oracles={known.id: known},
        score=lambda _ws: (True, ""),
    )
    with pytest.raises(KeyError, match="no oracle registered"):
        arm.attempt(stranger.prompt_view(), tmp_path / "ws")


def test_per_problem_state_does_not_accumulate_across_problems(tmp_path: Path) -> None:
    """Cumulative prompts put unrelated text in each record, re-scanned every earlier
    problem's prompts against every later oracle, and grew the archive quadratically. A
    cumulative token count in a per-problem row is simply not the number the row claims."""
    a = make_problem(tmp_path, "a")
    b = make_problem(tmp_path, "b")
    arm = e1_naive.NaiveArm(
        client=_StubClient(["x\n"] * 4, tokens=(100, 50)),
        config=NaiveConfig(model_id="m", max_attempts=1, temperature_ladder=(0.0,)),
        oracles={a.id: a, b.id: b},
        score=lambda _ws: (True, ""),
    )
    wsa, wsb = tmp_path / "wsa", tmp_path / "wsb"
    wsa.mkdir()
    wsb.mkdir()
    first = arm.attempt(a.prompt_view(), wsa)
    second = arm.attempt(b.prompt_view(), wsb)

    assert len(first.prompts) == 1
    assert len(second.prompts) == 1, "the second record carried the first problem's prompt"
    assert arm.tokens_in == 100, "token counts are per-problem, not cumulative"
    assert "Triple" not in first.prompts[0]


def test_the_baseline_driver_also_resets_its_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = make_problem(tmp_path, "a")
    b = make_problem(tmp_path, "b")

    class Proc:
        returncode = 0
        stdout = '{"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.1}}'
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: Proc())
    arm = ApiBaselineArm(spec=spec(), role=Role.E1_DIAGNOSTIC)
    wsa, wsb = tmp_path / "wsa", tmp_path / "wsb"
    wsa.mkdir()
    wsb.mkdir()
    arm.attempt(a.prompt_view(), wsa)
    second = arm.attempt(b.prompt_view(), wsb)
    assert len(second.prompts) == 1


# --- D4/D5: the control is not crippled --------------------------------------


def test_the_retry_ladder_may_not_descend() -> None:
    """Same or higher on retry, never lower. A descending ladder makes retries quieter
    rather than more varied, and the control looks worse for a reason unrelated to it."""
    with pytest.raises(ValueError, match="must not descend"):
        NaiveConfig(model_id="m", max_attempts=3, temperature_ladder=(0.4, 0.2, 0.0))


def test_every_attempt_has_a_declared_temperature() -> None:
    with pytest.raises(ValueError, match="every attempt needs"):
        NaiveConfig(model_id="m", max_attempts=3, temperature_ladder=(0.0,))


def test_the_control_gets_the_whole_stub_not_an_excerpt(tmp_path: Path) -> None:
    """Packing is the harness's lever. Lending it to the control would blur the pair."""
    problem = make_problem(tmp_path)
    view = problem.prompt_view()
    arm = e1_naive.NaiveArm(
        client=_StubClient(["def f(x):\n    return x * 2\n"]),
        config=NaiveConfig(model_id="m", max_attempts=1, temperature_ladder=(0.0,)),
        oracles={problem.id: problem},
        score=lambda _ws: (True, ""),
    )
    arm.attempt(view, tmp_path / "p")
    assert problem.stub_source in arm.prompts[0]
    assert problem.instructions in arm.prompts[0]


# --- §2 vs §5.1: failure feedback is verbatim minus what D7 forbids ----------


RAW_PYTEST = """\
=================================== FAILURES ===================================
_________________________ test_f_doubles_its_argument __________________________

    def test_f_doubles_its_argument():
>       assert f(2) == 4, "doubling two has to give four"
E       AssertionError: doubling two has to give four
E       assert 0 == 4
E        +  where 0 = f(2)

lib_test.py:5: AssertionError
"""


def test_redaction_removes_the_oracle_line(tmp_path: Path) -> None:
    problem = make_problem(tmp_path)
    text, removed = e1_naive.redact_oracle(RAW_PYTEST, problem)
    assert removed >= 1
    assert "assert f(2) == 4" not in text


def test_redacted_output_passes_the_leak_scan(tmp_path: Path) -> None:
    """The property that makes this safe to feed back: after redaction the check that would
    discard the run finds nothing."""
    problem = make_problem(tmp_path)
    text, _ = e1_naive.redact_oracle(RAW_PYTEST, problem)
    assert polyglot.leaks(text, problem) == []


def test_raw_output_would_have_failed_the_leak_scan(tmp_path: Path) -> None:
    """The other half: without redaction this really does leak, so the test above is
    testing something."""
    problem = make_problem(tmp_path)
    assert polyglot.leaks(RAW_PYTEST, problem)


def test_redaction_keeps_everything_the_oracle_did_not_contribute(tmp_path: Path) -> None:
    """Verbatim is read as *not summarised*. The diagnosis a developer would act on —
    exception type, computed values, location — survives intact."""
    problem = make_problem(tmp_path)
    text, _ = e1_naive.redact_oracle(RAW_PYTEST, problem)
    assert "AssertionError: doubling two has to give four" in text
    assert "where 0 = f(2)" in text
    assert "lib_test.py:5" in text
    assert "assert 0 == 4" in text


def test_redaction_preserves_line_positions(tmp_path: Path) -> None:
    """A traceback whose line numbers shifted is less useful than one with a visible hole."""
    problem = make_problem(tmp_path)
    text, _ = e1_naive.redact_oracle(RAW_PYTEST, problem)
    assert len(text.splitlines()) == len(RAW_PYTEST.splitlines())
    assert e1_naive.REDACTED in text


JS_SPEC = """\
import { solve } from './puzzle';

describe('Puzzle', () => {
  test('solves the three letter puzzle', () => {
    const puzzle = 'I + BB == ILL';
    const expected = { I: 1, B: 9, L: 0 };
    expect(solve(puzzle)).toEqual(expected);
  });
});
"""

RAW_JEST = """\
 FAIL  ./puzzle.spec.js
  Puzzle
    \u2715 solves the three letter puzzle (3 ms)

  \u25cf Puzzle \u203a solves the three letter puzzle

    expect(received).toEqual(expected) // deep equality

    Expected: {"B": 9, "I": 1, "L": 0}
    Received: null

       5 |     const puzzle = 'I + BB == ILL';
       6 |     const expected = { I: 1, B: 9, L: 0 };
    >  7 |     expect(solve(puzzle)).toEqual(expected);
         |                           ^
       8 |   });

      at Object.toEqual (puzzle.spec.js:7:27)
"""


def make_js_problem(root: Path) -> polyglot.LoadedProblem:
    d = root / "js"
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("Solve the puzzle.\n", encoding="utf-8")
    (d / "puzzle.js").write_text("export const solve = () => null;\n", encoding="utf-8")
    (d / "puzzle.spec.js").write_text(JS_SPEC, encoding="utf-8")
    return polyglot.load(root, "js", Language.JAVASCRIPT)


def test_jest_code_frames_are_redacted_and_pass_the_leak_scan(tmp_path: Path) -> None:
    """jest echoes the test's source inside a code frame (`> 7 | expect(...)`), which no
    whole-line match sees: the frame's lines went into a JavaScript retry, the leak scan
    refused it, and the run was discarded (2026-10-24). A line holding an oracle line is
    redacted, whatever wraps it; the diagnosis around it stays."""
    problem = make_js_problem(tmp_path)
    assert polyglot.leaks(RAW_JEST, problem), "the raw output really does leak"
    text, removed = e1_naive.redact_oracle(RAW_JEST, problem)
    assert polyglot.leaks(text, problem) == []
    assert removed == 3
    assert 'Expected: {"B": 9, "I": 1, "L": 0}' in text and "Received: null" in text
    assert "puzzle.spec.js:7:27" in text
    assert len(text.splitlines()) == len(RAW_JEST.splitlines())


def test_the_arm_feeds_redacted_failure_into_its_retry(tmp_path: Path) -> None:
    """End to end: attempt 1 fails, attempt 2's prompt carries the diagnosis and not the
    oracle."""
    problem = make_problem(tmp_path)
    scores = iter([(False, RAW_PYTEST), (True, "")])
    arm = e1_naive.NaiveArm(
        client=_StubClient(["def f(x):\n    return 0\n", "def f(x):\n    return x * 2\n"]),
        config=NaiveConfig(model_id="m", max_attempts=2, temperature_ladder=(0.0, 0.2)),
        oracles={problem.id: problem},
        score=lambda _ws: next(scores),
    )
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")

    assert result.attempts == 2
    retry = arm.prompts[1]
    assert "AssertionError: doubling two has to give four" in retry
    assert "assert f(2) == 4" not in retry
    assert polyglot.leaks(retry, problem) == []


def test_every_prompt_the_arm_rendered_is_reported(tmp_path: Path) -> None:
    """EV-3 scans what the arm reports, so an arm that under-reported would evade it."""
    problem = make_problem(tmp_path)
    scores = iter([(False, RAW_PYTEST), (False, RAW_PYTEST), (False, RAW_PYTEST)])
    arm = e1_naive.NaiveArm(
        client=_StubClient(["x\n"] * 3),
        config=NaiveConfig(model_id="m", max_attempts=3),
        oracles={problem.id: problem},
        score=lambda _ws: next(scores),
    )
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert len(result.prompts) == 3


# --- cost: derived from tokens, or absent and labelled -----------------------
# `Completion.usage` carries tokens, not money. The first version of this arm read a
# `cost_usd` attribute off it, which does not exist — so `getattr(..., 0.0)` made every run
# report exactly zero, and the cost comparison would have crowned the naive control the
# cheapest arm on the strength of a typo.


PRICES = ArmPrices(usd_per_input_token=1e-6, usd_per_output_token=3e-6)


def test_cost_is_computed_from_reported_tokens(tmp_path: Path) -> None:
    problem = make_problem(tmp_path)
    arm = e1_naive.NaiveArm(
        client=_StubClient(["def f(x):\n    return x * 2\n"], tokens=(1000, 500)),
        config=NaiveConfig(model_id="m", max_attempts=1, temperature_ladder=(0.0,)),
        oracles={problem.id: problem},
        score=lambda _ws: (True, ""),
        prices=PRICES,
    )
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert result.cost_usd == pytest.approx(1000 * 1e-6 + 500 * 3e-6)


def test_cost_accumulates_across_retries(tmp_path: Path) -> None:
    """Each attempt is a paid call. A cost that reported only the last one would make a
    three-try arm look as cheap as a one-try arm, which is the comparison §5.9 exists for."""
    problem = make_problem(tmp_path)
    scores = iter([(False, RAW_PYTEST), (False, RAW_PYTEST), (True, "")])
    arm = e1_naive.NaiveArm(
        client=_StubClient(["x\n"] * 3, tokens=(1000, 500)),
        config=NaiveConfig(model_id="m", max_attempts=3),
        oracles={problem.id: problem},
        score=lambda _ws: next(scores),
        prices=PRICES,
    )
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert result.attempts == 3
    assert result.cost_usd == pytest.approx(3 * (1000 * 1e-6 + 500 * 3e-6))


def test_an_unpriced_arm_says_so_rather_than_reporting_a_bare_zero(tmp_path: Path) -> None:
    """Zero and unpriced are different facts. Only one of them means "cheap"."""
    problem = make_problem(tmp_path)
    arm = e1_naive.NaiveArm(
        client=_StubClient(["def f(x):\n    return x * 2\n"], tokens=(1000, 500)),
        config=NaiveConfig(model_id="m", max_attempts=1, temperature_ladder=(0.0,)),
        oracles={problem.id: problem},
        score=lambda _ws: (True, ""),
    )
    result = arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert result.cost_usd == 0.0
    assert "unpriced" in result.detail
    # The tokens are still there, so the run can be priced afterwards from the archive.
    assert "in=1000" in result.detail and "out=500" in result.detail


def test_unpriced_cached_input_is_charged_at_the_full_rate() -> None:
    """Pessimistic when the cache discount is unknown — the direction that cannot flatter
    the arm being measured."""
    completion = Completion(
        text="",
        usage=ModelUsage(prompt_tokens=1000, completion_tokens=0, cached_prompt_tokens=900),
        latency_ms=1,
        stop_reason=StopReason.END_TURN,
    )
    undiscounted = ArmPrices(usd_per_input_token=1e-6, usd_per_output_token=0.0)
    assert e1_naive.price(completion, undiscounted) == pytest.approx(1000 * 1e-6)

    discounted = ArmPrices(
        usd_per_input_token=1e-6,
        usd_per_output_token=0.0,
        usd_per_cached_input_token=1e-7,
    )
    assert e1_naive.price(completion, discounted) == pytest.approx(100 * 1e-6 + 900 * 1e-7)


def test_the_ladder_temperature_reaches_the_client(tmp_path: Path) -> None:
    """D5 is only a property of the run if the declared ladder is what gets sent."""
    problem = make_problem(tmp_path)
    scores = iter([(False, RAW_PYTEST), (False, RAW_PYTEST), (True, "")])
    client = _StubClient(["x\n"] * 3)
    arm = e1_naive.NaiveArm(
        client=client,
        config=NaiveConfig(model_id="m", max_attempts=3, temperature_ladder=(0.0, 0.2, 0.4)),
        oracles={problem.id: problem},
        score=lambda _ws: next(scores),
    )
    arm.attempt(problem.prompt_view(), tmp_path / "p")
    assert [p.temperature for p in client.params] == [0.0, 0.2, 0.4]


# --- whole-file extraction ---------------------------------------------------


def test_a_fenced_reply_yields_the_file() -> None:
    assert e1_naive.extract_file("```python\ndef f():\n    return 1\n```") == (
        "def f():\n    return 1\n"
    )


def test_the_naive_arm_sends_its_prompt_as_one_system_and_one_user_message(
    tmp_path: Path,
) -> None:
    """Pinned before `complete()` retires. The request is the arm's
    instructions as the system message and the rest of the rendered prompt as the user
    message, at the ladder's first temperature and the configured output cap, with no
    affinity key and no tools: what `complete()` sent, and what `send` must send after."""
    problem = make_problem(tmp_path)
    client = _StubClient(["def f(x):\n    return x * 2\n"])
    arm = e1_naive.NaiveArm(
        client=client,
        config=NaiveConfig(
            model_id="m", max_attempts=1, temperature_ladder=(0.0,), max_output_tokens=512
        ),
        oracles={problem.id: problem},
        score=lambda _ws: (True, ""),
    )
    arm.attempt(problem.prompt_view(), tmp_path / "p")

    (request,) = client.requests
    rendered = arm.prompts[0]
    assert rendered.startswith(e1_naive.INSTRUCTIONS)
    system, user = request.messages
    assert (system.role, system.content) == ("system", e1_naive.INSTRUCTIONS)
    assert (user.role, user.content) == ("user", rendered[len(e1_naive.INSTRUCTIONS) :])
    assert request.params == RetryParams(temperature=0.0, max_tokens=512)
    assert request.session_key is None and request.tools == ()


def test_an_unfenced_reply_is_taken_as_the_file() -> None:
    """Permissive on purpose: rejecting a good file for a formatting reason would make the
    control fail for a harness reason, which is the strawman G1 looks for."""
    assert e1_naive.extract_file("def f():\n    return 1\n") == "def f():\n    return 1\n"


class _StubClient:
    """A `ModelClient` that replays canned completions. Keeps the loop testable without a
    model, which is also what stops these tests from costing money.

    It returns real `Completion` objects rather than a duck-typed stand-in: the arm reads
    `usage.prompt_tokens` to price itself, and a stub with a looser shape than the protocol
    would let a field disappear from `Completion` without a single test noticing.
    """

    def __init__(self, replies: list[str], *, tokens: tuple[int, int] = (100, 50)) -> None:
        self._replies = iter(replies)
        self._tokens = tokens
        self.params: list[RetryParams] = []
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        self.params.append(request.params)
        return Completion(
            text=next(self._replies),
            usage=ModelUsage(prompt_tokens=self._tokens[0], completion_tokens=self._tokens[1]),
            latency_ms=1,
            stop_reason=StopReason.END_TURN,
        )

    def close(self) -> None:
        """The protocol's; a replay holds nothing to release."""
