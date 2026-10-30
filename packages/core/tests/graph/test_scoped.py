"""The coder cannot query outside packet scope, and a denial is typed."""

from __future__ import annotations

from chickadee.graph.scoped import ScopedGraph


def coder(graph, scope=("app.py",), **kw):
    return ScopedGraph(graph=graph, role="coder", scope=scope, **kw)


def test_the_coder_may_use_its_granted_methods(indexed) -> None:
    g = coder(indexed)
    assert g.resolve("Widget").ok
    assert g.sig_of("app.Widget.render").ok
    assert g.globals_in("app.py").ok


def test_an_out_of_scope_file_is_denied_not_silently_empty(indexed) -> None:
    """A denial is a runtime-correction signal. `()` would read as "nothing there",
    which is how a coder concludes a symbol does not exist when it was merely out of scope.
    """
    answer = coder(indexed).globals_in("helpers.py")
    assert not answer.ok
    assert answer.denied is not None
    assert "outside packet scope" in answer.denied.reason
    assert answer.denied.scope == ("app.py",)


def test_an_out_of_scope_symbol_is_denied_rather_than_reported_absent(indexed) -> None:
    answer = coder(indexed).resolve("helper")
    assert not answer.ok
    assert answer.denied is not None
    assert "outside packet scope" in answer.denied.reason


def test_a_genuinely_absent_symbol_is_an_empty_result_not_a_denial(indexed) -> None:
    """The distinction the previous test depends on: absent and forbidden differ."""
    answer = coder(indexed).resolve("no_such_symbol_anywhere")
    assert answer.ok
    assert answer.items == ()


def test_the_coder_cannot_reach_ungranted_methods(indexed) -> None:
    answer = coder(indexed).symbols_in("app.py")
    assert not answer.ok
    assert answer.denied is not None
    assert "not available to the coder rung" in answer.denied.reason


def test_the_coder_is_capped_at_depth_one(indexed) -> None:
    answer = coder(indexed).callers("helpers.helper", depth=2)
    assert not answer.ok
    assert answer.denied is not None
    assert "depth" in answer.denied.reason


def test_the_call_budget_is_enforced(indexed) -> None:
    g = coder(indexed, max_calls=2)
    assert g.resolve("Widget").ok
    assert g.resolve("Widget").ok
    third = g.resolve("Widget")
    assert not third.ok
    assert third.denied is not None
    assert "budget exhausted" in third.denied.reason


def test_results_are_capped_and_say_so(indexed) -> None:
    g = ScopedGraph(graph=indexed, role="planner", max_results=1)
    answer = g.symbols_in("app.py")
    assert answer.ok
    assert len(answer.items) == 1
    assert answer.truncated, "a capped result must announce the cap"


def test_repository_scoped_roles_are_not_packet_bounded(indexed) -> None:
    """Every non-coder rung that asks the graph. `rescue` and `reader` named roles D96
    retired, and D131 took them out of the type."""
    for role in ("planner", "scout"):
        g = ScopedGraph(graph=indexed, role=role, scope=("app.py",))
        assert g.globals_in("helpers.py").ok, f"{role} has repository scope (§5.1)"


def test_an_empty_scope_denies_rather_than_granting_everything(indexed) -> None:
    """Fail closed. An empty scope meaning "no restriction" turns a forgotten argument into
    a silently removed boundary."""
    g = ScopedGraph(graph=indexed, role="coder", scope=())
    # `resolve` and `sig_of` are on the coder's allowlist, so these reach `_in_scope` and
    # fail only if the empty scope denies. `symbols_in` would be rejected by the method
    # allowlist first, and would keep passing even if empty scopes became permissive.
    assert not g.resolve("Widget").ok
    assert not g.sig_of("app.Widget.render").ok
    assert not g.globals_in("app.py").ok


def test_sig_of_is_scope_checked(indexed) -> None:
    """It read straight through to the store, so the boundary held everywhere except the
    method most likely to be asked."""
    answer = coder(indexed, scope=("app.py",)).sig_of("helper")
    assert not answer.ok
    assert answer.denied is not None
    assert "outside packet scope" in answer.denied.reason


def test_where_defined_is_scope_checked(indexed) -> None:
    answer = coder(indexed, scope=("app.py",)).where_defined("helper")
    assert not answer.ok


def test_depth_above_one_is_rejected_for_every_role(indexed) -> None:
    """The signature promised a traversal depth the implementation never provided."""
    answer = ScopedGraph(graph=indexed, role="planner").callers("helpers.helper", depth=3)
    assert not answer.ok
    assert answer.denied is not None
    assert "blast_radius" in answer.denied.reason
