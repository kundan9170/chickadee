"""`35-limits.md` §3 is a rendering of `LIMITS`: the table in the doc is
`render_limits(LIMITS)` byte for byte; every row applies to a role in the cast or to `all`;
every row names a measurement row."""

from __future__ import annotations

import pytest

from chickadee.errors import ConfigError
from chickadee.limits import (
    HARNESS_OWNERS,
    LIMITS,
    Limit,
    harness_limits,
    limit,
    render_limits,
)
from chickadee.roles import ROLES


def test_names_are_unique_and_prefixed_by_what_they_apply_to() -> None:
    names = [lim.name for lim in LIMITS]
    assert len(names) == len(set(names))
    for lim in LIMITS:
        assert lim.name.split(".", 1)[0] == lim.applies_to, lim.name


def test_every_applies_to_is_a_role_or_all_or_harness_wide() -> None:
    """A row owned by no role is named as such (`HARNESS_OWNERS`): never an
    owner that is a role's typo."""
    cast = {r.name for r in ROLES} | {"all"} | HARNESS_OWNERS
    assert {lim.applies_to for lim in LIMITS} <= cast
    assert not HARNESS_OWNERS & {r.name for r in ROLES}


def test_a_harness_wide_row_is_one_number_and_in_no_roles_table() -> None:
    """`lsp.servers_max` and `lsp.idle_s` are the harness's, set once; so are
    the skills index's and `find_skill`'s rows, how many agents run at once, and how often
    one is followed up; D227 decision 8's bounds on what hooks may steer, and how long
    a worker's stop waits on the re-check; and what one web call may take and return;
    and what the person may attach to one turn. The server cap is 8
    since B2, a language's linter counting as
    one."""
    assert harness_limits(LIMITS, {}) == {
        "agents.max_parallel": 3,
        "agents.max_follow_ups": 3,
        "lsp.servers_max": 8,
        "lsp.idle_s": 600,
        "skills.index_max_rows": 24,
        "skills.index_desc_chars": 280,
        "skills.find_results": 5,
        "hooks.max_continues": 1,
        "hooks.max_retries": 1,
        "hooks.max_steering": 3,
        "hooks.recheck_wait_s": 30,
        "web.fetch_max_chars": 40000,
        "web.fetch_connect_s": 8,
        "web.fetch_timeout_s": 15,
        "web.search_results": 5,
        "web.search_snippet_chars": 300,
        "web.fetch_page_s": 20,
        "web.url_max_chars": 2000,
        "turn.attachments_max": 8,
        "turn.attachment_bytes_max": 24 * 1024 * 1024,
        "turn.attachment_file_max_bytes": 25 * 1024 * 1024,
        "turn.pdf_file_max_bytes": 20 * 1024 * 1024,
        "turn.pdf_pages_max": 100,
        "turn.pdf_text_min_chars": 32,
        "turn.text_file_max_bytes": 1024 * 1024,
        "turn.text_inline_bytes": 32 * 1024,
    }
    assert harness_limits(LIMITS, {"limits.lsp.servers_max": 2})["lsp.servers_max"] == 2
    with pytest.raises(ConfigError, match="positive integer"):
        harness_limits(LIMITS, {"limits.lsp.idle_s": 0})


def test_fractions_are_fractions_and_counts_are_positive_ints() -> None:
    for lim in LIMITS:
        if lim.unit == "window_fraction":
            assert isinstance(lim.value, float) and 0 < lim.value <= 1, lim.name
        else:
            assert isinstance(lim.value, int) and lim.value > 0, lim.name


def test_zeus_inline_caps_are_registered_not_changed() -> None:
    """The numbers `zeus/tools/spec.py` carried in older versions, unchanged."""
    assert limit("zeus.inline_bytes").value == 4096
    assert limit("zeus.shell_inline_bytes").value == 8192
    assert limit("zeus.result_lines").value == 200


def test_an_unknown_name_is_a_config_error_naming_the_register() -> None:
    with pytest.raises(ConfigError, match="limits register"):
        limit("worker.max_tool_call")


def test_a_default_with_more_places_renders_them() -> None:
    """Rounding 0.125 to 0.12 would publish a number the code does not apply, and the
    drift test could not tell."""
    lim = Limit("worker.x", 0.125, "window_fraction", "packet", "worker", "why", "W2")
    assert "| 0.125 |" in render_limits((lim,))
    assert "| 0.30 |" in render_limits((limit("worker.max_cumulative_result_tokens"),))
