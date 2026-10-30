"""Repeats, stopping rule and CI fixed before results; paired keys complete.

Every assertion here is about *ordering*. The schema's value is not that it holds these
fields but that it held them before anyone saw a number, so the tests exercise the
sequence, not the shape.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from chickadee_eval import (
    Arm,
    CIMethod,
    CommonPolicy,
    Language,
    Manifest,
    MissingPolicy,
    Preregistration,
    Problem,
    Subset,
    SubsetKind,
    archive,
    validate_arms_differ_in_one_variable,
)

SHA = "a" * 64


def _common() -> CommonPolicy:
    return CommonPolicy(
        task_statement_sha=SHA,
        repo_snapshot_sha=SHA,
        visible_tests_sha=SHA,
        acceptance_oracle_sha=SHA,
        tools_available=("read", "write", "shell"),
        tool_permissions=("no-network",),
        test_commands=("pytest -q",),
        per_command_timeout_s=120,
        per_task_timeout_s=900,
        human_intervention="none",
        handoff_policy="fresh-worktree",
    )


def _prereg(**over: object) -> Preregistration:
    base: dict[str, object] = {
        "experiment": "E2",
        "frontier_model": "frontier-x",
        "frontier_revision": "2026-08-01",
        "slm_model": "qwen-coder",
        "slm_revision": "2026-07-01",
        "arm_versions": {"baseline": "1.2.3", "chickadee": "0.1.0"},
        "task_order_seed": 7,
        "paired_seeds": {"t1": 11, "t2": 12},
        "repeats": 3,
        "stopping_rule": "all tasks x 3 repeats, no early stop",
        "attempt_policy": "one run per task per repeat",
        "accuracy_margin_pp": 5.0,
        "cost_ratio_max": 0.5,
        "ci_method": CIMethod.BOOTSTRAP_PAIRED,
        "ci_level": 0.95,
        "cost_formula": "prompt+completion at list price, cache split",
        "slm_shadow_price": "hosted-market-equivalent",
        "cache_treatment": "unknown cache counted as a miss",
        "missing_policy": MissingPolicy.UNSOLVED,
        "hard_cost_usd_per_arm": {"baseline": 20.0, "chickadee": 20.0},
        "common": _common(),
    }
    return Preregistration.model_validate({**base, **over})


def test_freezing_is_a_one_way_door() -> None:
    """Re-freezing would let the protocol be edited between a peek at the results and the
    record of what was promised — the one thing preregistration exists to prevent."""
    p = _prereg()
    assert not p.is_frozen
    frozen = p.freeze("2026-08-18T09:00:00")
    assert frozen.is_frozen
    with pytest.raises(ValueError, match="already frozen"):
        frozen.freeze("2026-08-18T10:00:00")


def test_the_original_is_untouched_by_freezing() -> None:
    """Frozen models: `freeze` returns a new value rather than mutating, so a caller
    holding the draft cannot be surprised by it changing underneath."""
    p = _prereg()
    p.freeze("2026-08-18T09:00:00")
    assert not p.is_frozen


def test_every_arm_needs_a_finite_cap() -> None:
    with pytest.raises(ValidationError, match="no hard cost cap"):
        _prereg(hard_cost_usd_per_arm={"baseline": 20.0})
    with pytest.raises(ValidationError, match="not a cap"):
        _prereg(hard_cost_usd_per_arm={"baseline": 20.0, "chickadee": 0.0})


@pytest.mark.parametrize(
    "field, bad",
    [("repeats", 0), ("ci_level", 0.5), ("ci_level", 1.0), ("accuracy_margin_pp", 0.0)],
)
def test_inference_parameters_must_be_usable(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        _prereg(**{field: bad})


def test_pairing_must_be_complete_before_a_run() -> None:
    """Pairing is the entire basis of the inference: unpaired, a difference between arms
    is confounded with a difference between tasks."""
    p = _prereg()
    assert p.pairing_is_complete(("t1", "t2")) == []
    assert p.pairing_is_complete(("t1", "t2", "t3")) == ["t3"]


def test_stopping_rule_and_ci_method_are_required() -> None:
    """Chosen after the fact, each is a knob that can be turned until the answer is the
    one you wanted, and no reader can tell that happened."""
    for missing in ("stopping_rule", "ci_method", "repeats", "missing_policy"):
        payload = _prereg().model_dump()
        del payload[missing]
        with pytest.raises(ValidationError):
            Preregistration.model_validate(payload)


def test_ablation_arms_must_differ_in_exactly_one_variable() -> None:
    """Two differences make the readout uninterpretable, and finding that out
    afterwards costs the whole run."""
    p0 = {"decompose": False, "model": "m", "budget": 5.0}
    assert validate_arms_differ_in_one_variable(p0, {**p0, "decompose": True}) == ["decompose"]
    two = validate_arms_differ_in_one_variable(p0, {**p0, "decompose": True, "budget": 9.0})
    assert two == ["budget", "decompose"]


# -- manifests ---------------------------------------------------------------


def _subset(kind: SubsetKind, ids: tuple[str, ...], seed: int | None = None) -> Subset:
    return Subset(
        name=kind.value,
        kind=kind,
        problems=tuple(
            Problem(id=i, language=Language.PYTHON if n % 2 else Language.JAVASCRIPT)
            for n, i in enumerate(ids)
        ),
        seed=seed,
    )


def test_counts_are_recorded_not_assumed() -> None:
    """A manifest saying "~30 problems" is a plan; one saying which 30 is evidence."""
    s = _subset(SubsetKind.DEV, ("p1", "p2", "p3"))
    assert s.count == 3
    assert sum(s.by_language().values()) == 3


def test_the_confirmation_set_cannot_share_problems_with_dev() -> None:
    """Otherwise the gate measures what the tuning fitted."""
    with pytest.raises(ValidationError, match="both the dev and confirmation"):
        Manifest(
            experiment="E1",
            subsets=(
                _subset(SubsetKind.DEV, ("p1", "p2")),
                _subset(SubsetKind.CONFIRMATION, ("p2", "p3")),
            ),
            arms=(Arm(name="a", harness="h", harness_version="1", hard_cost_usd=5.0),),
        )


def test_a_perturbed_subset_must_be_reproducible() -> None:
    """A perturbation nobody can regenerate is an anecdote."""
    with pytest.raises(ValidationError, match="needs a seed"):
        _subset(SubsetKind.PERTURBED, ("p1",))
    assert _subset(SubsetKind.PERTURBED, ("p1",), seed=42).seed == 42


def test_only_a_perturbed_subset_carries_a_seed() -> None:
    with pytest.raises(ValidationError, match="only a perturbed subset"):
        _subset(SubsetKind.DEV, ("p1",), seed=1)


def test_duplicate_problem_ids_are_rejected() -> None:
    """A duplicate silently doubles a problem's weight in the denominator."""
    with pytest.raises(ValidationError, match="duplicate problem ids"):
        _subset(SubsetKind.DEV, ("p1", "p1"))


def test_an_arm_without_a_finite_cap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Arm(name="a", harness="h", harness_version="1", hard_cost_usd=0.0)


# -- archive -----------------------------------------------------------------


def test_config_sha_is_order_independent() -> None:
    """Key order alone must not fork the archive."""
    assert archive.config_sha({"a": 1, "b": 2}) == archive.config_sha({"b": 2, "a": 1})
    assert archive.config_sha({"a": 1}) != archive.config_sha({"a": 2})


def test_run_dir_carries_the_config_sha(tmp_path: pathlib.Path) -> None:
    d = archive.run_dir("2026-08-18", {"model": "x"}, root=tmp_path)
    assert d.parent.name == "results"
    assert d.name.startswith("2026-08-18-")
    assert archive.run_dir("2026-08-18", {"model": "y"}, root=tmp_path) != d


def test_archive_refuses_an_unfrozen_preregistration(tmp_path: pathlib.Path) -> None:
    """A preregistration written at the same time as the results it constrains is not
    evidence of what was promised beforehand."""
    with pytest.raises(ValueError, match="not frozen"):
        archive.create(
            "2026-08-18", {"m": 1}, root=tmp_path, preregistration=_prereg().model_dump()
        )


def test_archive_writes_the_snapshot_before_results(tmp_path: pathlib.Path) -> None:
    p = _prereg().freeze("2026-08-18T09:00:00")
    d = archive.create(
        "2026-08-18",
        {"m": 1},
        root=tmp_path,
        preregistration=json.loads(p.model_dump_json()),
    )
    assert (d / archive.CONFIG_NAME).exists()
    stored = json.loads((d / archive.PREREGISTRATION_NAME).read_text())
    assert stored["frozen_at"] == "2026-08-18T09:00:00"
    assert not (d / archive.RESULTS_NAME).exists()


def test_archive_refuses_to_reopen_a_run_that_has_results(tmp_path: pathlib.Path) -> None:
    """Appending a second run's rows to a first run's archive makes both unreadable."""
    d = archive.create("2026-08-18", {"m": 1}, root=tmp_path)
    (d / archive.RESULTS_NAME).write_text("{}\n")
    with pytest.raises(FileExistsError, match="already holds results"):
        archive.create("2026-08-18", {"m": 1}, root=tmp_path)


def test_find_lists_only_real_archives(tmp_path: pathlib.Path) -> None:
    archive.create("2026-08-18", {"m": 1}, root=tmp_path)
    (tmp_path / archive.RESULTS_ROOT / "not-an-archive").mkdir()
    assert [p.name for p in archive.find(tmp_path)] == [
        archive.run_dir("2026-08-18", {"m": 1}, root=tmp_path).name
    ]


def test_a_bad_date_is_refused() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        archive.run_dir("18-08-2026", {"m": 1})
