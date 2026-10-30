"""Seeded bad citations rejected; moved lines detected.

The gate checks evidence LOCATIONS, never the truth of a claim. It is mechanical, so it can be
exact about the part it checks, and it says which part that was: a citation checked against
the lines the citer read passed on its content, one checked only against the file passed on
its location. A citation to a file that is not there, a line past its end, a path outside the
repository or a secret is rejected, and the answer carrying it is not served as checked.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from context_helpers import Caller, png, reader_for, write_files

import chickadee.context.citations as citations_module
from chickadee.context.citations import (
    Checked,
    LineSource,
    Verdict,
    WebSource,
    claims,
    extract,
    gate,
    seen_in,
    web_extract,
    workspace_lines,
)
from chickadee.context.session import Session
from chickadee.context.spans import Span
from chickadee.secrets.denylist import Denial

SOURCE = "".join(
    f"{line}\n"
    for line in (
        "import os",
        "",
        "def limit():",
        "    cap = 10",
        "    return cap",
        "",
        "def name():",
        '    return "x"',
        "",
        "LIMIT = limit()",
    )
)


class Asked:
    """A `LineSource` that remembers every path it was asked for."""

    def __init__(self, inner: LineSource) -> None:
        self.inner = inner
        self.paths: list[str] = []

    def __call__(self, path: str) -> Sequence[str] | Denial | None:
        self.paths.append(path)
        return self.inner(path)


@pytest.fixture
def repo_ws(tmp_path: Path, ws: Path) -> Path:
    write_files(
        ws,
        {
            "a.py": SOURCE,
            "Makefile": "all:\n\tpytest\n",
            "docs/guide.md": "# Guide\n\nRun it.\n",
            "pkg/sub/mod.py": "x = 1\n",
            ".env": "TOKEN=hunter2\n",
            "keys/server.pem": "-----BEGIN-----\n",
            "assets/logo.png": png(4, 4),
        },
    )
    write_files(tmp_path, {"outside.py": "x = 1\n"})
    os.symlink(ws / ".env", ws / "notes.txt")
    os.symlink(tmp_path / "outside.py", ws / "escape.py")
    return ws


def verdicts(answer: str, ws: Path) -> list[tuple[str, Verdict]]:
    return [
        (c.citation.text, c.verdict) for c in gate(answer, read=workspace_lines(ws)).checked
    ]


# -- extraction -------------------------------------------------------------------------------


def test_every_citation_shape_is_extracted_and_a_url_is_not() -> None:
    text = (
        "See a.py:3, b/c.py:4-6, d.py:7\N{EN DASH}9 and .github/ci.yml#L2-L5. "
        "Also e.py:12:5 and https://example.com/src/a.py:40."
    )
    found = [(c.path, c.start, c.end) for c in extract(text)]
    assert found == [
        ("a.py", 3, 3),
        ("b/c.py", 4, 6),
        ("d.py", 7, 9),
        (".github/ci.yml", 2, 5),
        ("e.py", 12, 12),
    ]


def test_a_shape_that_names_no_file_is_not_a_citation(repo_ws: Path) -> None:
    """`host:port`, a time and a ratio have the shape; only a file makes one a citation. A
    bare name with a source suffix is one even when missing, so an invented `utils.py:12`
    cannot slip through as prose."""
    answer = "It listens on localhost:8080 from 12:30, at a 1.5:2 ratio, per example.com:443."
    assert gate(answer, read=workspace_lines(repo_ws)).checked == ()
    assert verdicts("Built by Makefile:2 and utils.py:12.", repo_ws) == [
        ("Makefile:2", Verdict.OK),
        ("utils.py:12", Verdict.NO_FILE),
    ]


# -- seeded bad citations ---------------------------------------------------------------------


BAD = [
    pytest.param("utils.py:12", Verdict.NO_FILE, id="invented-file"),
    pytest.param("src/cache/store.py:3", Verdict.NO_FILE, id="invented-path"),
    pytest.param("assets/logo.png:1", Verdict.NO_FILE, id="not-text"),
    pytest.param("pkg/sub:1", Verdict.NO_FILE, id="a-directory"),
    # SECRET since T-3.87: the line source returns the denial of where
    # the link lands, as the Reader refuses it, where it was NO_FILE.
    pytest.param("notes.txt:1", Verdict.SECRET, id="a-link-to-a-secret"),
    pytest.param("escape.py:1", Verdict.NO_FILE, id="a-link-out-of-the-repo"),
    pytest.param("a.py:11", Verdict.OUT_OF_RANGE, id="past-the-end"),
    pytest.param("a.py:9-12", Verdict.OUT_OF_RANGE, id="running-off-the-end"),
    pytest.param("a.py:0", Verdict.OUT_OF_RANGE, id="line-zero"),
    pytest.param("a.py:5-3", Verdict.OUT_OF_RANGE, id="backwards"),
    pytest.param("/etc/passwd:1", Verdict.OUTSIDE, id="absolute"),
    pytest.param("../other/a.py:1", Verdict.OUTSIDE, id="escaping"),
    pytest.param("~/.ssh/config:1", Verdict.OUTSIDE, id="home"),
    pytest.param(".env:1", Verdict.SECRET, id="dotenv"),
    pytest.param("keys/server.pem:1", Verdict.SECRET, id="pem"),
]


@pytest.mark.parametrize(("cited", "verdict"), BAD)
def test_a_seeded_bad_citation_is_rejected_and_fails_the_answer(
    repo_ws: Path, cited: str, verdict: Verdict
) -> None:
    answer = f"The cap is set at a.py:4. It is read back at {cited}."
    result = gate(answer, read=workspace_lines(repo_ws))
    assert [(c.citation.text, c.verdict) for c in result.checked] == [
        ("a.py:4", Verdict.OK),
        (cited, verdict),
    ]
    assert not result.passed
    assert [c.citation.text for c in result.rejected] == [cited]
    payload = result.payload()
    assert (payload.passed, payload.checked) == (False, 2)


def test_a_secret_is_refused_unread(repo_ws: Path) -> None:
    asked = Asked(workspace_lines(repo_ws))
    result = gate("The token is at .env:1 and keys/server.pem:1.", read=asked)
    assert [c.verdict for c in result.checked] == [Verdict.SECRET, Verdict.SECRET]
    assert asked.paths == []


def test_an_answer_that_cites_nothing_is_not_served_as_checked(repo_ws: Path) -> None:
    result = gate("The cap is ten. It is read once.", read=workspace_lines(repo_ws))
    assert result.checked == ()
    assert not result.passed
    assert (result.claims, result.cited_claims, result.coverage) == (2, 0, 0.0)


def test_an_answer_with_uncited_claims_is_not_served_as_checked(repo_ws: Path) -> None:
    """Every served claim cites. One valid citation does not vouch for its neighbour."""
    result = gate(
        "The cap is set at a.py:4. It is never read twice.", read=workspace_lines(repo_ws)
    )
    assert [c.passed for c in result.checked] == [True]
    assert (result.claims, result.cited_claims, result.coverage) == (2, 1, 0.5)
    assert not result.passed
    assert result.payload().passed is False


def test_the_limits_line_says_what_was_checked_and_on_what_basis(
    session: Session, repo_ws: Path
) -> None:
    call = Caller(session, reader_for(session, repo_ws))
    session.record_user("where is the cap?")
    call("read", path="a.py", lines="3-5")
    answer = "The cap is set at a.py:4.\nIt is used at a.py:10.\nIt is named in utils.py:1."
    result = gate(answer, read=workspace_lines(repo_ws), seen=seen_in(session.state))
    assert result.render().splitlines() == [
        "[citations: 3 checked · 1 against the lines read, 1 by location only · "
        "1 rejected · 3 of 3 claims cited]",
        "  utils.py:1 NO_FILE: no readable text file utils.py",
    ]
    assert result.payload().coverage == 1.0


def test_claims_are_counted_from_prose_not_code_or_headings() -> None:
    answer = (
        "# Findings\n\n- The cap is ten.\n- It is read once. Then cached.\n\n```\nx = 1\n```"
    )
    assert claims(answer) == ("The cap is ten.", "It is read once.", "Then cached.")


# -- moved lines --------------------------------------------------------------------------


def test_lines_that_moved_are_found_where_they_are_now(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": SOURCE})
    call = Caller(session, reader_for(session, ws))
    session.record_user("where is the cap?")
    call("read", path="a.py", lines="1-10")
    answer = "The cap is set at a.py:4-5."
    read, seen = workspace_lines(ws), seen_in(session.state)
    (still,) = gate(answer, read=read, seen=seen).checked
    assert (still.verdict, still.basis) == (Verdict.OK, "content")

    write_files(ws, {"a.py": "# one\n# two\n# three\n" + SOURCE})
    result = gate(answer, read=read, seen=seen)
    (moved,) = result.checked
    assert (moved.verdict, moved.basis, moved.now) == (Verdict.MOVED, "content", Span(7, 8))
    assert moved.render() == "a.py:4-5 MOVED (now a.py:7-8): the lines cited moved"
    assert not result.passed


def test_lines_that_are_gone_are_changed_and_whitespace_is_not(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"a.py": SOURCE})
    call = Caller(session, reader_for(session, ws))
    session.record_user("where is the cap?")
    call("read", path="a.py", lines="1-10")
    seen = seen_in(session.state)

    def checked() -> Checked:
        (one,) = gate("The cap is set at a.py:4.", read=workspace_lines(ws), seen=seen).checked
        return one

    write_files(ws, {"a.py": SOURCE.replace("    cap = 10\n", "    cap = 10   \n")})
    assert (checked().verdict, checked().basis) == (Verdict.OK, "content")
    write_files(ws, {"a.py": SOURCE.replace("cap = 10", "cap = 20")})
    assert (checked().verdict, checked().why) == (Verdict.CHANGED, "the lines cited are gone")


def test_a_moved_block_is_found_nearest_where_it_was_cited() -> None:
    lines = ["x", "dup", "y", "z", "dup", "w", "dup"]
    seen = {Span(4, 4): ("dup",)}
    (moved,) = gate(
        "It is at a.py:4.",
        read=lambda path: lines if path == "a.py" else None,
        seen=lambda path, span: seen.get(span),
    ).checked
    assert (moved.verdict, moved.now) == (Verdict.MOVED, Span(5, 5))


def test_lines_the_citer_never_read_pass_on_location_only(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": SOURCE})
    call = Caller(session, reader_for(session, ws))
    session.record_user("where is the cap?")
    call("read", path="a.py", lines="1-2")
    (one,) = gate(
        "The cap is set at a.py:4.", read=workspace_lines(ws), seen=seen_in(session.state)
    ).checked
    assert (one.verdict, one.basis) == (Verdict.OK, "location")


def test_a_line_the_citer_saw_cut_is_checked_on_its_location(
    session: Session, ws: Path
) -> None:
    """The view showed the head of a long line and its size. Comparing that head with the
    whole line would call an unchanged line CHANGED; its location is all that is known."""
    long_line = "DATA = [" + ", ".join(str(i) for i in range(3000)) + "]"
    write_files(ws, {"big.py": f"import os\n{long_line}\n"})
    call = Caller(session, reader_for(session, ws))
    session.record_user("what is DATA?")
    view = call("read", path="big.py", lines="2-2")
    assert "…[+" in view.text
    (one,) = gate(
        "DATA is defined at big.py:2.", read=workspace_lines(ws), seen=seen_in(session.state)
    ).checked
    assert (one.verdict, one.basis) == (Verdict.OK, "location")


# -- pure code, no model -----------------------------------------------------------------------


def test_the_gate_is_pure_and_imports_no_model_side_module(repo_ws: Path) -> None:
    source = Path(citations_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    model_side = ("chickadee.client", "chickadee.adapters", "chickadee.zeus", "openai", "httpx")
    assert not [m for m in imported if m.startswith(model_side)]
    answer = "The cap is set at a.py:4. It is used in utils.py:3."
    first = gate(answer, read=workspace_lines(repo_ws))
    assert gate(answer, read=workspace_lines(repo_ws)) == first


# -- web citations ------------------------------------------------------

PAGE_URL = "https://docs.python.org/3/library/re.html"
PAGE = (
    "# re — Regular expression operations\n\n"
    "Compile a regular expression **pattern** into a [regular expression object](https://"
    "docs.python.org/3/library/re.html#re-objects), which can be used for matching.\n\n"
    "`re.compile(pattern, flags=0)`"
)


def fetched(pages: dict[str, str]) -> WebSource:
    return pages.get


def test_a_web_citation_is_a_url_then_its_quote_in_straight_or_curly_quotes() -> None:
    text = (
        f'It compiles first ({PAGE_URL} "Compile a regular expression"). '
        f"See also <{PAGE_URL}#re-objects> “which can be used for matching”."
    )
    found = web_extract(text, claim=3)
    assert [(c.url, c.quote, c.claim) for c in found] == [
        (PAGE_URL, "Compile a regular expression", 3),
        (f"{PAGE_URL}#re-objects", "which can be used for matching", 3),
    ]


def test_a_quote_on_the_page_is_quoted_whitespace_links_and_emphasis_folded() -> None:
    answer = (
        f'- `compile` takes a pattern: {PAGE_URL}#top "Compile a regular expression pattern '
        f'into a  regular expression object"\n'
        f'- Its signature: {PAGE_URL} "re.compile(pattern, flags=0)"'
    )
    result = gate(answer, read=lambda _: None, fetched=fetched({PAGE_URL: PAGE}))
    assert [c.verdict for c in result.checked] == [Verdict.QUOTED, Verdict.QUOTED]
    assert result.passed and result.web
    assert result.render() == (
        "[citations: 2 checked · 0 against the lines read, 0 by location only, 2 quoted from "
        "the page fetched · 0 rejected · 2 of 2 claims cited]"
    )


def test_a_page_never_fetched_and_words_not_on_it_are_flagged_never_dropped() -> None:
    answer = (
        f'- Not read: https://example.org/blog "re is slow"\n'
        f'- Misquoted: {PAGE_URL} "re.compile is deprecated"\n'
        f'- Wrong case: {PAGE_URL} "compile a regular expression"'
    )
    result = gate(answer, read=lambda _: None, fetched=fetched({PAGE_URL: PAGE}))
    assert [c.verdict for c in result.checked] == [
        Verdict.NOT_FETCHED,
        Verdict.NOT_ON_PAGE,
        Verdict.NOT_ON_PAGE,
    ]
    assert not result.passed and len(result.rejected) == 3
    assert result.rejected[0].render() == (
        'https://example.org/blog "re is slow" NOT_FETCHED: this round never fetched it'
    )


def test_without_the_round_s_pages_every_web_citation_is_not_fetched() -> None:
    result = gate(f'- {PAGE_URL} "Compile a regular expression"', read=lambda _: None)
    assert [c.verdict for c in result.checked] == [Verdict.NOT_FETCHED]


def test_a_quote_that_looks_like_a_file_citation_is_not_one(repo_ws: Path) -> None:
    page = "Edit app.py:12 to change the limit."
    answer = f'- The guide says so: {PAGE_URL} "Edit app.py:12 to change the limit."'
    result = gate(answer, read=workspace_lines(repo_ws), fetched=fetched({PAGE_URL: page}))
    assert [c.verdict for c in result.checked] == [Verdict.QUOTED]


def test_a_file_only_answer_reads_as_it_did_before_web_citations(repo_ws: Path) -> None:
    result = gate("- `limit()` is defined at a.py:3", read=workspace_lines(repo_ws))
    assert not result.web and "quoted" not in result.render()
