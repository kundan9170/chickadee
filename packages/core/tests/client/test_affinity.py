"""Cache affinity injection."""

from __future__ import annotations

import json
import pathlib

import httpx2
import pytest
from client_helpers import FakeEndpoint, ok_body, scripted
from pydantic import ValidationError

from chickadee.client import AffinityRow, AffinityTable, OpenAIClient, to_request
from chickadee.schema import Prompt, RetryParams
from chickadee.schema.adapter import CacheMode, CacheObs, Usage

PROMPT = Prompt(prefix="P", volatile="V")
PARAMS = RetryParams(temperature=0.2, max_tokens=512)


def _send(provider: str, session_key: str | None) -> dict:
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    client = OpenAIClient(FakeEndpoint(provider=provider), transport=transport)
    client.send(to_request(PROMPT, PARAMS, session_key=session_key))
    return json.loads(seen[0].content)


def test_hosted_row_injects_the_key() -> None:
    body = _send("fireworks", "run-7:t3")
    assert body["prompt_cache_key"] == "run-7:t3"


def test_local_rows_inject_nothing() -> None:
    """A single local instance has one prefix cache; there is nothing to pin it to, and a
    stray body field is a 400 on some servers and silently ignored on others."""
    for provider in ("local", "llama_cpp", "vllm"):
        body = _send(provider, "run-7:t3")
        assert "prompt_cache_key" not in body, provider


def test_unknown_provider_injects_nothing() -> None:
    assert "prompt_cache_key" not in _send("some-new-vendor", "k")


def test_no_session_key_injects_nothing() -> None:
    assert "prompt_cache_key" not in _send("fireworks", None)


def test_candidates_share_one_session_key() -> None:
    """Best-of-n passes one key for all candidates (01 §5.2 row 5) so parallel samples hit
    one replica's cache instead of each paying for a cold prefix."""
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    client = OpenAIClient(FakeEndpoint(provider="fireworks"), transport=transport)
    for _ in range(3):
        client.send(to_request(PROMPT, PARAMS, session_key="run-7:t3"))
    keys = {json.loads(r.content)["prompt_cache_key"] for r in seen}
    assert keys == {"run-7:t3"}


def test_header_style_row() -> None:
    table = AffinityTable((AffinityRow(provider="hdr", header="X-Cache-Key"),))
    transport, seen = scripted(httpx2.Response(200, json=ok_body()))
    client = OpenAIClient(FakeEndpoint(provider="hdr"), affinity=table, transport=transport)
    client.send(to_request(PROMPT, PARAMS, session_key="abc"))
    assert seen[0].headers["x-cache-key"] == "abc"


def test_rows_load_from_toml(tmp_path: pathlib.Path) -> None:
    """Adding a provider is a TOML entry, not code — otherwise `core` grows a branch
    per vendor, which is how provider assumptions leak in and stay."""
    f = tmp_path / "affinity.toml"
    f.write_text('[[affinity]]\nprovider = "newvendor"\nbody_field = "cache_id"\n')
    table = AffinityTable.from_toml(f)
    assert table.row("newvendor").body_field == "cache_id"


def test_malformed_row_rejects(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "affinity.toml"
    f.write_text('[[affinity]]\nprovider = "x"\nunknown_key = 1\n')
    with pytest.raises(ValidationError):
        AffinityTable.from_toml(f)


# --- cache mode -------------------------------------------------------


def test_explicit_providers_carry_no_routing_key() -> None:
    """An `explicit` provider caches what the assembler marked, wherever it lands.

    There is no replica to pin, so a row that also injected a key would be claiming a
    mechanism the provider does not have.
    """
    table = AffinityTable()
    for provider in ("anthropic", "bedrock", "openrouter"):
        row = table.row(provider)
        assert row.cache_mode is CacheMode.EXPLICIT
        assert row.body_field is None and row.header is None


def test_unknown_provider_declares_none_rather_than_assuming() -> None:
    """The D18 rule ("inject nothing rather than guess") extended to caching.

    Assuming a cache that is absent makes every request look like a permanent miss, which
    CX-1 would read as a cache-cold transition and compact on forever.
    """
    assert AffinityTable().row("acme-llm").cache_mode is CacheMode.NONE


def test_warm_is_unknowable_not_false_when_nothing_is_reported() -> None:
    """Invariant 20 in one property: a negative is a fact only when coverage is total."""
    silent = Usage(prompt_tokens=100, completion_tokens=10)
    assert CacheObs.from_usage(silent, CacheMode.AUTOMATIC).warm is None

    reported_miss = Usage(prompt_tokens=100, completion_tokens=10, cached_prompt_tokens=0)
    assert CacheObs.from_usage(reported_miss, CacheMode.AUTOMATIC).warm is False

    hit = Usage(prompt_tokens=20, completion_tokens=10, cached_prompt_tokens=80)
    assert CacheObs.from_usage(hit, CacheMode.AUTOMATIC).warm is True


def test_mode_none_makes_every_observation_unknowable() -> None:
    """Even a provider that reports numbers tells us nothing once we declared no cache."""
    hit = Usage(prompt_tokens=20, completion_tokens=10, cached_prompt_tokens=80)
    assert CacheObs.from_usage(hit, CacheMode.NONE).warm is None
