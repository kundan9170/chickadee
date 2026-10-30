"""D5 ladder: temperature never decreases. D195: every attempt asks for the model's output cap.
AD-4 no stop sequences. The rule is `adapters.base.retry_params`, which the family adapters'
`retry_policy` wrapped until they retired with the packet loop; its tests read it
directly."""

from __future__ import annotations

from typing import cast

import pytest

from chickadee.adapters.base import first_attempt_params, retry_params
from chickadee.client import ModelClient
from chickadee.config import RungEndpoint
from chickadee.schema.adapter import RetryParams
from chickadee.scout.rung import ScoutRung

CAP = 4096


def test_temperature_never_decreases() -> None:
    temps = [retry_params(n, CAP).temperature for n in range(1, 6)]
    assert temps[0] == 0.2
    assert temps[1:] == [0.4] * 4
    assert temps == sorted(temps)


def test_no_stop_sequences() -> None:
    params = retry_params(1, CAP)
    assert params.stop == ()


def test_retry_params_have_no_constrained_decoding_fields() -> None:
    """JSON mode is a rung endpoint flag, not a retry lever, and never for code."""
    assert "json_mode" not in RetryParams.model_fields
    assert "response_format" not in RetryParams.model_fields
    params = retry_params(1, CAP)
    assert params.reasoning is None


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_every_attempt_asks_for_the_models_output_cap(attempt: int) -> None:
    """#565: AD-3 sized the request from a target estimate nothing ever bound, so every
    worker request asked for the 1,024-token floor and a file plus a class was cut off."""
    params = retry_params(attempt, 16_384)
    assert params.max_tokens == 16_384


# -- the first call's sampling, one rule ----------------------------------


def _endpoint(**over: object) -> RungEndpoint:
    fields: dict[str, object] = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 65536,
        "max_output_tokens": 16384,
    }
    return RungEndpoint.model_validate({**fields, **over})


def _the_scouts_old_rule(ep: RungEndpoint) -> RetryParams:
    """`ScoutRung.params()` as it read before the rule moved, and `engine._params` copied it
    line for line: the reference the shared function must keep."""
    base = retry_params(1, ep.max_output_tokens)
    update: dict[str, object] = {"reasoning": ep.reasoning}
    if ep.temperature is not None:
        update["temperature"] = ep.temperature
    return base.model_copy(update=update)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"temperature": 0.7},
        {"reasoning": {"level": "high"}},
        {"temperature": 0.0, "max_output_tokens": 4096},
    ],
    ids=["defaults", "temperature", "reasoning", "zero-temperature"],
)
def test_the_shared_first_call_rule_is_the_scouts(over: dict[str, object]) -> None:
    ep = _endpoint(**over)
    shared = first_attempt_params(ep)

    assert shared == _the_scouts_old_rule(ep)
    assert ScoutRung("m", ep, cast("ModelClient", None), "scout").params() == shared
    assert shared.max_tokens == ep.max_output_tokens, "the model's whole output cap"
    assert shared.stop == ()
