"""The validation gate drops what does not
exist and counts it; the render for Zeus carries every field; no targets is `UNRESOLVED`,
naming what was searched; `UNAVAILABLE` is a typed failure, not a report; and nothing is
trimmed here."""

from __future__ import annotations

import json

from scout_helpers import CALLER, TARGET

from chickadee.schema.adapter import ParseError
from chickadee.schema.findings import Findings, Pointer, Unresolved
from chickadee.scout import findings as findings_mod
from chickadee.scout.findings import MAX_FACTS, Checker, Draft, Report, gate, parse_draft


def _gate(checker: Checker, **fields: object) -> tuple[Findings, int, int]:
    return gate(Draft.model_validate(fields), checker, truncated=False)


def test_the_gate_drops_hallucinated_pointers_and_counts_them(checker: Checker) -> None:
    seeded = [
        TARGET,
        {"path": "pkg/net/nowhere.py", "line_start": 1, "line_end": 2},  # no such file
        {"path": "pkg/net/client.py", "line_start": 1, "line_end": 400},  # past the end
        {"path": "pkg/net/client.py", "line_start": 1, "line_end": 2, "symbol": "HttpClient"},
        {"path": "pkg/net/client.py", "line_start": 6, "line_end": 11, "symbol": "Nope"},
        {"path": "../outside.py", "line_start": 1, "line_end": 1},  # escapes the root
        {"path": "/etc/hostname", "line_start": 1, "line_end": 1},
        {"path": "pkg/net/client.py", "line_start": 9, "line_end": 3},  # malformed
        "pkg/net/client.py:6",  # not a pointer at all
    ]
    findings, dropped, _ = _gate(checker, targets=seeded)
    assert [p.path for p in findings.targets] == ["pkg/net/client.py"]
    assert findings.targets[0].symbol == "HttpClient"
    assert dropped == len(seeded) - 1


def test_a_symbol_is_checked_where_the_graph_has_the_file_and_in_the_text_where_not(
    checker: Checker,
) -> None:
    qualified = {**TARGET, "symbol": "HttpClient.request", "line_start": 9, "line_end": 11}
    in_yaml = {"path": "config/app.yaml", "line_start": 1, "line_end": 2, "symbol": "timeout"}
    not_in_yaml = {"path": "config/app.yaml", "line_start": 1, "line_end": 2, "symbol": "retry"}
    findings, dropped, _ = _gate(checker, targets=[qualified, in_yaml, not_in_yaml])
    assert [(p.path, p.symbol) for p in findings.targets] == [
        ("pkg/net/client.py", "HttpClient.request"),
        ("config/app.yaml", "timeout"),
    ]
    assert dropped == 1


def test_callers_name_the_callee_and_the_edge_class_is_the_graph_s(checker: Checker) -> None:
    wrong_line = {**CALLER, "line_start": 1, "line_end": 1}
    claimed = {**CALLER, "edge_class": "resolved"}
    findings, dropped, _ = _gate(checker, callers=[claimed, wrong_line])
    (caller,) = findings.callers
    assert caller.path == "pkg/api/fetcher.py" and dropped == 1
    assert caller.edge_class in {"resolved", "fallback", "unknown"}
    # The model's word for it is not taken on a target either.
    (target,) = _gate(checker, targets=[{**TARGET, "edge_class": "resolved"}])[0].targets
    assert target.edge_class is None


def test_facts_travel_only_when_the_workspace_confirms_the_path(checker: Checker) -> None:
    facts = [
        "httpx declared in pyproject.toml",
        "src/auth/ does not exist",
        "pkg/net/client.py does not exist",
        "the retry logic is probably in the middleware",
        "tests live in tests/test_client.py",
        42,
    ]
    findings, _, dropped_facts = _gate(checker, facts=facts)
    assert findings.facts == (
        "httpx declared in pyproject.toml",
        "src/auth/ does not exist",
        "tests live in tests/test_client.py",
    )
    assert dropped_facts == 3


def test_a_fact_worded_as_an_absence_must_name_only_absent_paths(checker: Checker) -> None:
    """The safe direction, documented: "missing" reads as a claim that the path is absent, so
    a true fact about something missing *inside* a present file is dropped, never taken."""
    facts = ["pkg/net/client.py is missing a retry parameter", "src/auth/ is missing"]
    findings, _, dropped_facts = _gate(checker, facts=facts)
    assert findings.facts == ("src/auth/ is missing",) and dropped_facts == 1


def test_facts_past_the_cap_are_counted_as_dropped(checker: Checker) -> None:
    facts = ["httpx declared in pyproject.toml"] * (MAX_FACTS + 5)
    findings, _, dropped_facts = _gate(checker, facts=facts)
    assert len(findings.facts) == MAX_FACTS and dropped_facts == 5


def test_unresolved_keeps_its_question_and_only_real_candidates(checker: Checker) -> None:
    unresolved = [
        {
            "question": "which config",
            "candidates": ["config/app.yaml", "nope.yaml", "HttpClient"],
        },
        {"candidates": ["config/app.yaml"]},
    ]
    findings, _, _ = _gate(checker, unresolved=unresolved)
    assert findings.unresolved == (
        Unresolved(question="which config", candidates=("config/app.yaml", "HttpClient")),
    )


def test_over_inclusion_is_not_trimmed_here(checker: Checker) -> None:
    """Missing core evidence hurts more than redundant context; the packer trims."""
    spans = [
        {"path": "pkg/net/client.py", "line_start": s, "line_end": e}
        for s in range(1, 12)
        for e in range(s, 12)
    ]
    findings, dropped, _ = _gate(checker, targets=spans, evidence=spans)
    assert len(findings.targets) == len(spans) == 66 and dropped == 0
    assert len(findings.evidence) == 66


def test_a_malformed_pointer_is_one_drop_not_a_lost_report() -> None:
    text = "```json\n" + json.dumps({"targets": [TARGET, {"path": 3}], "facts": []}) + "\n```"
    draft = parse_draft(text)
    assert isinstance(draft, Draft) and len(draft.targets) == 2
    assert isinstance(parse_draft("the client is in pkg/net"), ParseError)


def test_a_single_line_pointer_may_say_line(checker: Checker) -> None:
    findings, dropped, _ = _gate(checker, callers=[{"path": "pkg/api/fetcher.py", "line": 7}])
    assert dropped == 0 and findings.callers[0].line_start == findings.callers[0].line_end == 7


# -- the render for Zeus -----------------------------------------------------------------


def test_the_render_carries_every_field_and_the_dropped_count() -> None:
    findings = Findings(
        targets=(
            Pointer(path="pkg/net/client.py", line_start=6, line_end=11, symbol="HttpClient"),
        ),
        callers=(
            Pointer(
                path="pkg/api/fetcher.py",
                line_start=7,
                line_end=7,
                symbol="HttpClient.request",
                edge_class="resolved",
            ),
        ),
        facts=("httpx declared in pyproject.toml",),
        unresolved=(Unresolved(question="which config", candidates=("config/app.yaml",)),),
        evidence=(Pointer(path="pkg/net/client.py", line_start=3, line_end=3),),
        truncated=True,
    )
    text = Report(
        question="the http client",
        findings=findings,
        outcome="found",
        model="local-mid",
        calls=4,
        escapes=1,
        tokens=2100,
        dropped_pointers=2,
        dropped_facts=1,
    ).render()
    assert text.splitlines()[0].startswith(
        "## FINDINGS    scout: local-mid · 4 calls · 1 escape"
    )
    assert "  - pkg/net/client.py:6-11   HttpClient" in text
    assert "  - pkg/api/fetcher.py:7   → HttpClient.request   (resolved)" in text
    assert "  - httpx declared in pyproject.toml" in text
    assert "'which config' → 1 candidate(s): config/app.yaml" in text
    assert "  - pkg/net/client.py:3" in text
    assert "gate: dropped 2 pointer(s) and 1 fact(s)" in text
    assert text.splitlines()[-1] == "truncated: true"
    assert "UNRESOLVED" not in text


def test_no_targets_is_unresolved_naming_what_was_searched() -> None:
    report = Report(
        question="where is the retry limit enforced",
        findings=Findings(),
        outcome="unresolved",
        model="local-mid",
        searched=("symbol 'retry'", "grep 'max_retries'"),
    )
    text = report.render()
    assert (
        "UNRESOLVED{question: 'where is the retry limit enforced', "
        "searched: [symbol 'retry', grep 'max_retries']}"
    ) in text
    assert text.splitlines()[-1] == "truncated: false"


def test_unavailable_renders_the_typed_failure_not_a_report() -> None:
    report = Report(
        question="where is the retry limit enforced",
        findings=Findings(),
        outcome="unavailable",
        model="local-mid",
        unavailable="no_tool_calling",
        detail="native_tools = false",
    )
    text = report.render()
    assert text.startswith("UNAVAILABLE{rung: 'local-mid', reason: no_tool_calling")
    assert "FINDINGS" not in text and "UNRESOLVED" not in text and "truncated" not in text


def test_nothing_reads_names_out_of_the_question() -> None:
    """The name shortcut and the third signal are gone, and with them the guessing."""
    for gone in ("named_in", "only_names", "from_names"):
        assert not hasattr(findings_mod, gone), gone
