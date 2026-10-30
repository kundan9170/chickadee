"""`/rung`: the session's wiring and the catalog, a switch from the
rung's next call, and each refusal said with its reason."""

from __future__ import annotations

from typing import Any

import hosting
import pytest
from engines import MODEL, raw_config
from repl_driver import Typed, repl

OTHER = "frontier-b"


@pytest.fixture(autouse=True)
def catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hosted config with `OTHER` in its catalog and priced, so a switch to it loads."""

    def raw() -> dict[str, Any]:
        tables = raw_config()
        tables["models"][OTHER] = {**MODEL, "model_id": OTHER}
        return tables

    def priced(tables: dict[str, Any]) -> dict[str, Any]:
        price = {"price_in": 1.0, "price_out": 2.0}
        tables["prices"] = {tables["rungs"]["high-param"]: price, OTHER: price}
        return tables

    monkeypatch.setattr(hosting, "raw_config", raw)
    monkeypatch.setattr(hosting, "with_prices", priced)


def test_rung_lists_the_wiring_and_the_catalog_and_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed = ["/rung", "/rung high-param", f"/rung high-param {OTHER}", "/rung", "/exit"]
    with repl(monkeypatch, [], [Typed(t) for t in typed]) as (source, _code):
        out = source.out
    assert "  high-param   high-param-model" in out
    assert any(line.startswith("  catalog: ") and OTHER in line for line in out)
    assert any(line.startswith("  others: ") and "high-param-model" not in line for line in out)
    assert f"high-param: high-param-model → {OTHER}, from its next call" in out
    assert f"  high-param   {OTHER}  (switched this session)" in out


def test_each_refusal_prints_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    typed = [
        "/rung high-param no-such-model",
        "/rung no-such-rung",
        "/rung high-param high-param-model",
        "/rung a b c",
        "/rung",
        "/exit",
    ]
    with repl(monkeypatch, [], [Typed(t) for t in typed]) as (source, _code):
        out = "\n".join(source.out)
    assert "/rung: " in out and "not in the catalog" in out
    assert "no rung no-such-rung; the rungs are " in out
    assert "already runs on" in out
    assert "/rung [rung] [model]" in out
    assert "switched this session" not in out, "no refusal moved anything"
