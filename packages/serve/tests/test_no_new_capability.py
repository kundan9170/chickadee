"""SV-1 as code: every route reaches exactly one `PacketEngine` member, and nothing else.

The route table is the whole surface. Each row names the facade member it calls; this holds
every name to `PacketEngine`, except the members the facade does not have yet, each named with
the task that adds it. That list only shrinks: when a member lands, its entry must go, or the
second test fails.
"""

from __future__ import annotations

import inspect

from chickadee.engine import PacketEngine
from chickadee_serve import app
from chickadee_serve.routes import ROUTES

AWAITING_FACADE: dict[str, str] = {}
"""Facade members a route names that `PacketEngine` lacks today, and who adds each. Empty
since T-5.7 declared `models` and `switch_model` (unbuilt), so every route's
answer has a type; a new entry needs the task that adds the member."""


def test_every_route_reaches_one_facade_member_or_none() -> None:
    for row in ROUTES:
        if row.facade is None:
            assert (row.method, row.path) == ("GET", "/"), "only GET / reaches nothing (D117)"
            continue
        on_engine = hasattr(PacketEngine, row.facade) or row.facade == "interaction"
        assert on_engine or row.facade in AWAITING_FACADE, (
            f"{row.method} {row.path} reaches {row.facade!r}, which PacketEngine does not have "
            f"and no task is named to add"
        )


def test_awaited_members_are_still_missing() -> None:
    """The ratchet: a member that has landed comes off the list."""
    landed = [name for name in AWAITING_FACADE if hasattr(PacketEngine, name)]
    assert not landed, f"now on PacketEngine; remove from AWAITING_FACADE: {landed}"


def test_the_answer_route_reaches_the_engines_own_port() -> None:
    """`interaction` is `PacketEngine`'s injected port, set in its constructor."""
    assert "interaction" in inspect.signature(PacketEngine.__init__).parameters


def test_every_row_has_a_handler_and_no_handler_lacks_a_row() -> None:
    rows = {(r.method, r.path) for r in ROUTES}
    assert rows == set(app.HANDLERS)


def test_rows_are_unique() -> None:
    keys = [(r.method, r.path) for r in ROUTES]
    assert len(keys) == len(set(keys))


def test_the_app_calls_the_facade_only_through_the_rows_member() -> None:
    """No handler names a facade method itself: each calls `call.facade(...)`, which reaches the
    member its row names, or `_method` with the row's name. A literal attribute call on the
    facade elsewhere in the module would be a capability the table does not show."""
    source = inspect.getsource(app)
    assert "served.facade." not in source
    assert "facade.facade" not in source
