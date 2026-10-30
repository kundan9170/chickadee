"""A role whose template, never-evicted floors, one maximum tool result
and output reserve exceed its rung's window is refused at load, naming role, rung and both
numbers; a worker rung whose `max_output_tokens` is below `worker.output_floor` is refused."""

from __future__ import annotations

from typing import Any

import pytest

from chickadee.config import EngineConfig, RungEndpoint
from chickadee.errors import ConfigError
from chickadee.limits import LIMITS, check_fit, resolve, tokens_for_chars
from chickadee.roles import DEFAULT_RUNGS, ROLES, assign, role


def _rung(**over: Any) -> RungEndpoint:
    fields: dict[str, Any] = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 131_072,
        "max_output_tokens": 16_384,
    }
    return RungEndpoint(**{**fields, **over})


def _resolved(coder: RungEndpoint, overrides: dict[str, float | int] | None = None) -> Any:
    rungs = {name: _rung() for name in DEFAULT_RUNGS}
    rungs["low-param"] = coder
    return resolve(LIMITS, ROLES, rungs, overrides or {}, assignment=assign({}))


def test_every_role_fits_a_128k_rung() -> None:
    rung = _rung()
    resolved = _resolved(rung)
    for r in ROLES:
        check_fit(r, rung, resolved, template_tokens=2_000, rung_name=assign({})[r.name])


def test_a_worker_that_cannot_fit_is_refused_naming_role_rung_and_both_numbers() -> None:
    # 28K window: 0.15 of it (4,200) + 8,000 result + 16,000 output + 1,000 template.
    coder = _rung(context_window=28_000, max_output_tokens=16_000)
    with pytest.raises(ConfigError) as exc:
        check_fit(
            role("worker"),
            coder,
            _resolved(coder),
            template_tokens=1_000,
            rung_name=assign({})["worker"],
        )
    msg = str(exc.value)
    assert "'worker'" in msg and "'low-param'" in msg
    assert "29200" in msg, "the floors"
    assert "28000" in msg, "the window"


def test_the_window_is_input_limit_when_set() -> None:
    coder = _rung(context_window=131_072, input_limit=28_000)
    with pytest.raises(ConfigError, match=r"\(28000\)"):
        check_fit(
            role("worker"),
            coder,
            _resolved(coder),
            template_tokens=1_000,
            rung_name=assign({})["worker"],
        )


def test_lowering_the_fixed_rows_lets_a_small_rung_run() -> None:
    """A smaller coder rung is refused by name, and runs once those two rows are
    lowered for it."""
    coder = _rung(context_window=32_768, max_output_tokens=8_000)
    resolved = _resolved(
        coder, {"limits.worker.result_max": 4_000, "limits.worker.output_floor": 8_000}
    )
    check_fit(
        role("worker"), coder, resolved, template_tokens=1_000, rung_name=assign({})["worker"]
    )


def test_a_worker_rung_below_the_output_floor_is_refused() -> None:
    coder = _rung(max_output_tokens=8_192)
    with pytest.raises(ConfigError, match=r"worker\.output_floor \(16000\)"):
        check_fit(
            role("worker"),
            coder,
            _resolved(coder),
            template_tokens=0,
            rung_name=assign({})["worker"],
        )


def test_the_output_floor_binds_only_the_worker() -> None:
    small = _rung(max_output_tokens=4_096)
    check_fit(
        role("locate"),
        small,
        _resolved(_rung()),
        template_tokens=0,
        rung_name=assign({})["locate"],
    )


def test_zeus_counts_its_larger_inline_cap_as_one_result() -> None:
    # 8,192 bytes / 3.5 = 2,341 tokens; 1,000 + 2,341 + 16,000 = 19,341 > 19,000.
    zeus = _rung(context_window=19_000, max_output_tokens=16_000)
    with pytest.raises(ConfigError, match="19341"):
        check_fit(
            role("zeus"),
            zeus,
            _resolved(_rung()),
            template_tokens=1_000,
            rung_name=assign({})["zeus"],
        )


def test_athena_counts_her_largest_result_as_one_result() -> None:
    """She can take a command's output through `request_exec`, bounded
    by the shell's byte cap: 8,192 bytes / 3.5 = 2,341 tokens. Since D214 a read of hers may be
    raised to `athena.result_max` (8,000 tokens), the larger, so that is her one-result floor
    (it was the shell's 2,341, and 3,341 with the output). She holds no row whatever is
    evicted: 8,000 + 1,000 output = 9,000 > 3,000."""
    planner = _rung(context_window=3_000, max_output_tokens=1_000)
    with pytest.raises(ConfigError, match="9000"):
        check_fit(
            role("athena"),
            planner,
            _resolved(_rung()),
            template_tokens=0,
            rung_name=assign({})["athena"],
        )


def _raw(coder: dict[str, Any]) -> dict[str, Any]:
    """A catalog with one model per rung, the coder's changed by `coder`."""
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 131_072,
        "max_output_tokens": 16_384,
    }
    models = {f"{name}-model": dict(model) for name in DEFAULT_RUNGS}
    models["low-param-model"] = {**model, **coder}
    return {
        "models": models,
        "rungs": {name: f"{name}-model" for name in DEFAULT_RUNGS},
        "spend": {"hard_cost_usd": 5.0},
    }


def test_the_config_refuses_at_load_a_role_that_cannot_fit() -> None:
    with pytest.raises(ConfigError, match="cannot fit one request on rung 'low-param'"):
        EngineConfig.with_defaults(
            _raw({"context_window": 28_000, "max_output_tokens": 16_000})
        )


def test_the_config_refuses_at_load_a_worker_below_its_output_floor() -> None:
    with pytest.raises(ConfigError, match="output_floor"):
        EngineConfig.with_defaults(_raw({"max_output_tokens": 4_096}))


def test_the_config_loads_a_small_rung_once_its_rows_are_lowered() -> None:
    raw = _raw({"context_window": 32_768, "max_output_tokens": 8_000})
    raw["limits"] = {"worker": {"result_max": 4_000, "output_floor": 8_000}}
    EngineConfig.with_defaults(raw)


def test_a_role_template_counts_its_own_file_only() -> None:
    """Moved: the family preamble and protocol retired with the packet loop, so a
    template is measured by its own file, what its request sends; one not written yet is 0."""
    from chickadee.config import (  # pyright: ignore[reportPrivateUsage]
        _TEMPLATES,
        _template_tokens,
    )
    from chickadee.work import DEFAULT_DIR, with_work_dir

    assert _template_tokens("frontier", "does-not-exist.md", DEFAULT_DIR) == 0
    raw = (_TEMPLATES / "frontier" / "templates" / "zeus.md").read_text(encoding="utf-8")
    # As its request sends it: `work.dir` named.
    text = with_work_dir(raw, DEFAULT_DIR)
    assert _template_tokens("frontier", "zeus.md", DEFAULT_DIR) == tokens_for_chars(len(text))


def test_the_worker_is_measured_by_the_tools_its_request_sends() -> None:
    """The worker's tools are measured as its request sends them: each
    enabled built-in's native schema and description, the one assembly `box/worker` sends, so
    the load measures what a run sends. The packet worker's frames and protocols retired with
    it: the task worker calls tools natively."""
    from dataclasses import replace

    from chickadee.adapters.base import BOX_DESCRIPTIONS, tool_descriptions
    from chickadee.config import _worker_tools_tokens  # pyright: ignore[reportPrivateUsage]
    from chickadee.context.request import est_tools
    from chickadee.roles import TASK_WORKER
    from chickadee.tools.builtin import BUILTINS
    from chickadee.tools.meta import shown
    from chickadee.tools.validate import resident

    specs = tuple(
        TASK_WORKER.resolved(t.spec)
        for n in TASK_WORKER.enabled
        if (t := BUILTINS.get(n)) is not None
    )
    measured = _worker_tools_tokens(TASK_WORKER)
    # Moved: the worker's own descriptions, and its specs as its policy resolves them.
    # And with `why` first in each, as the request sends them.
    assert measured == est_tools(
        shown(resident(specs, tool_descriptions(specs, *BOX_DESCRIPTIONS)))
    )
    # Fewer tools offered is fewer tokens sent.
    assert 0 < _worker_tools_tokens(replace(TASK_WORKER, enabled=("read",))) < measured


def test_a_partial_resolved_table_is_a_harness_bug_not_a_config_error() -> None:
    from chickadee.errors import HarnessBug

    rung = _rung()
    partial = resolve(
        LIMITS[:3], ROLES, {name: rung for name in DEFAULT_RUNGS}, {}, assignment=assign({})
    )
    with pytest.raises(HarnessBug):
        check_fit(
            role("worker"), rung, partial, template_tokens=0, rung_name=assign({})["worker"]
        )
