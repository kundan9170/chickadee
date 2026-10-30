"""D213 decisions 4 and 5, Amendment 1 decisions 3 and 4: one agent, one round, its report.

An agent starts from its prompt, its task and the handles passed, never Zeus's conversation; it
reads through its own tools; its report is one shape whose citations are checked by code; a
limit makes it `partial`, never a made-up report; and every message is on the session log.
"""

from __future__ import annotations

from pathlib import Path

from agents_helpers import (
    AgentClient,
    LineReader,
    agents_for,
    reader_profile,
    report,
    turn_calls,
    turn_text,
    write_repo,
)

from chickadee.context.citations import seen_in_reads
from chickadee.context.fit import render_lines
from chickadee.context.kept_lines import AgentRecord
from chickadee.context.results import InvalidArgument
from chickadee.context.session import Session
from chickadee.context.spans import Span
from chickadee.schema.journal import AgentMessageP, AgentReportP, AgentStartP


def _session(tmp_path: Path) -> Session:
    return Session.create(tmp_path / "sessions", "S-agents")


def _payloads(session: Session) -> list[object]:
    return [
        r.line.payload
        for _, r in session.state.lines()
        if isinstance(r, AgentRecord) and r.line is not None
    ]


def test_an_agent_reads_and_reports_with_its_citation_checked(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    client = AgentClient(
        turn_calls(("read", {"path": "src/auth.py"})),
        turn_text(report("- `login` checks the user first: `src/auth.py:1-2`.")),
    )
    try:
        out = agents_for(session, repo, client).spawn(
            None, "reader", "Where is the user checked?", (), call_id="z1"
        )
        assert isinstance(out, str)
        assert "[agent A1 · round 0 · reader · complete · 1 tool calls" in out
        assert "1 checked · 1 against the lines read" in out
        assert "FLAGGED" not in out
        kinds = [type(p) for p in _payloads(session)]
        assert kinds[0] is AgentStartP and kinds[-1] is AgentReportP
        assert AgentMessageP in kinds
    finally:
        session.close()


def test_the_agent_sees_only_its_task_and_the_handles_passed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    session.record_user("a secret plan only Zeus has seen")
    client = AgentClient(turn_text(report()))
    try:
        agents_for(session, repo, client, handles={"h1": "the handle's text"}).spawn(
            None, "reader", "Look at this.", ("h1",), call_id="z1"
        )
        sent = "\n".join(str(m.content) for m in client.requests[0].messages)
        assert "Look at this." in sent and "the handle's text" in sent
        assert "a secret plan" not in sent
    finally:
        session.close()


def test_a_citation_of_no_file_is_flagged_and_one_never_read_checked_by_location(
    tmp_path: Path,
) -> None:
    """The report is kept whole; a citation is labelled, never a reason to drop a finding. One
    of a file that is not there is FLAGGED; one of lines the agent never read passes on its
    location only, since nothing recorded what it saw there."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    client = AgentClient(
        turn_text(
            report("- It returns a token: `src/auth.py:3`.\n- It caches: `src/cache.py:9`.")
        )
    )
    try:
        out = agents_for(session, repo, client).spawn(
            None, "reader", "What does login return?", (), call_id="z1"
        )
        assert isinstance(out, str)
        assert "FLAGGED: src/cache.py:9 NO_FILE" in out
        assert "1 by location only" in out
        (done,) = [p for p in _payloads(session) if isinstance(p, AgentReportP)]
        assert (done.findings, done.flagged) == (2, 1)
    finally:
        session.close()


def test_a_missing_section_is_asked_for_once_then_the_round_failed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    client = AgentClient(turn_text("## Summary\nOnly this."), turn_text("Still only prose."))
    try:
        out = agents_for(session, repo, client).spawn(None, "reader", "Look.", (), call_id="z1")
        assert isinstance(out, str)
        assert " failed " in out and "never had every section" in out
        assert "## Findings" in str(client.requests[1].messages[-1].content)
        assert "## Summary" not in out.split("\n", 2)[-1]
    finally:
        session.close()


def test_at_its_call_cap_the_next_turn_is_its_report_marked_partial(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    client = AgentClient(
        turn_calls(("read", {"path": "src/auth.py"})),
        turn_calls(("read", {"path": "src/auth.py", "lines": "1-1"})),
        turn_text(report(not_done="the callers")),
    )
    try:
        agents = agents_for(
            session, repo, client, profiles={"reader": reader_profile(max_tool_calls=1)}
        )
        out = agents.spawn(None, "reader", "Look.", (), call_id="z1")
        assert isinstance(out, str)
        assert " partial " in out and "the callers" in out
    finally:
        session.close()


def test_unknown_names_are_refused_with_what_there_is(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        agents = agents_for(session, repo, AgentClient())
        unknown = agents.spawn(None, "writer", "Fix it.", (), call_id="z1")
        assert (
            isinstance(unknown, InvalidArgument)
            and "Do: send one of: reader" in unknown.render()
        )
        nothing = agents.spawn(None, None, "Fix it.", (), call_id="z2")
        assert isinstance(nothing, InvalidArgument)
        handle = agents.spawn(None, "reader", "Look.", ("h9",), call_id="z3")
        assert isinstance(handle, InvalidArgument) and "h9" in handle.render()
        assert _payloads(session) == []
    finally:
        session.close()


def test_agents_are_numbered_from_the_log(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        for n in (1, 2):
            client = AgentClient(turn_text(report()))
            out = agents_for(session, repo, client).spawn(
                None, "reader", "Look.", (), call_id=f"z{n}"
            )
            assert isinstance(out, str) and out.startswith(f"[agent A{n} ")
    finally:
        session.close()
    again = Session.open(tmp_path / "sessions", "S-agents")
    try:
        client = AgentClient(turn_text(report()))
        out = agents_for(again, write_repo(tmp_path / "repo"), client).spawn(
            None, "reader", "Look.", (), call_id="z3"
        )
        assert isinstance(out, str) and out.startswith("[agent A3 ")
    finally:
        again.close()


def test_its_reads_go_through_its_own_reader(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    own = LineReader(repo)
    client = AgentClient(turn_calls(("read", {"path": "src/auth.py"})), turn_text(report()))
    try:
        agents_for(session, repo, client, reader=own).spawn(
            None, "reader", "Look.", (), call_id="z1"
        )
        assert own.calls == ["read src/auth.py"]
    finally:
        session.close()


def test_a_read_spelt_another_way_is_checked_against_what_it_showed(tmp_path: Path) -> None:
    """The second review of #640: reads were kept by the path as the model sent it, so a read
    of `./src/auth.py` never met its citation `src/auth.py` and was checked by location only.
    They are kept as a citation is checked (`citations.repository_path`)."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    client = AgentClient(
        turn_calls(("read", {"path": "./src/auth.py"})),
        turn_text(report("- `login` checks the user first: `src/auth.py:1-2`.")),
    )
    try:
        out = agents_for(session, repo, client).spawn(
            None, "reader", "Where is the user checked?", (), call_id="z1"
        )
        assert isinstance(out, str) and "1 against the lines read" in out
    finally:
        session.close()


def test_a_line_shown_cut_is_never_compared() -> None:
    """The second review of #640: the agents' check parsed what a read showed by a copy of the
    citation gate's, without its rule that a line shown cut is never compared, so a citation
    of a long line, unchanged, was CHANGED. It is the gate's own now (`seen_in_reads`)."""
    lines = ["a = 1", "b = " + "x" * 400, "c = 3"]
    shown = render_lines(lines, [Span(1, 3)], cut={2: 20})
    seen = seen_in_reads({"a.py": [shown]})
    assert seen("a.py", Span(1, 1)) == ("a = 1",)
    assert seen("a.py", Span(2, 2)) is None, "checked on its location only"
    assert seen("a.py", Span(1, 3)) is None
