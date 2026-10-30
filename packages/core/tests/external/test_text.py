"""A fetched body as the text a model reads (`external/fetch/text.py`)."""

from __future__ import annotations

import pytest

from chickadee.external.fetch.text import html_to_text, is_text, page_text

URL = "https://docs.example.org/guide/start.html"


def test_html_keeps_what_a_page_says_and_drops_what_runs_or_navigates() -> None:
    html = """<html><head><title>Ignored</title><style>p{}</style>
    <script>steal()</script></head>
    <body><nav><a href="/">Home</a></nav>
    <h1>Getting started</h1>
    <p>Install it with <code>pip install x</code> &amp; then
    <a href="../api.html">read the API</a>.</p>
    <h2>Steps</h2>
    <ul><li>one</li><li>two<ol><li>first</li><li>second</li></ol></li></ul>
    <pre>  def f():
        return 1
</pre>
    <form><input name="q"><button>Go</button></form>
    <table><tr><th>key</th><th>value</th></tr><tr><td>a</td><td>1</td></tr></table>
    </body></html>"""
    assert html_to_text(html, URL) == (
        "# Getting started\n\n"
        "Install it with `pip install x` & then [read the API](https://docs.example.org/api.html).\n\n"
        "## Steps\n\n"
        "- one\n- two\n  1. first\n  2. second\n\n"
        "```\n  def f():\n        return 1\n```\n\n"
        "key | value\na | 1"
    )


def test_a_page_with_no_heading_starts_with_its_title() -> None:
    assert html_to_text("<title>The  page</title><p>text</p>", URL) == "# The page\n\ntext"


def test_a_link_to_nowhere_is_its_text_alone() -> None:
    html = '<p><a href="#top">top</a> <a href="javascript:x()">run</a> <a>bare</a></p>'
    assert html_to_text(html, URL) == "top run bare"


def test_markup_that_is_not_well_formed_still_comes_back_as_its_text() -> None:
    assert html_to_text("<p>one<p>two <b>bold<div>three", URL) == "one\n\ntwo bold\n\nthree"


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("text/html; charset=utf-8", True),
        ("application/xhtml+xml", True),
        ("text/plain", True),
        ("text/markdown", True),
        ("application/json", True),
        ("application/ld+json", True),
        ("", True),
        ("application/pdf", False),
        ("image/png", False),
        ("application/octet-stream", False),
    ],
)
def test_which_types_are_text(kind: str, text: bool) -> None:
    assert is_text(kind) is text


def test_plain_text_and_json_are_kept_as_they_are() -> None:
    assert page_text(b"<not html>\n", "text/plain", URL) == "<not html>\n"
    assert page_text(b'{"a": 1}', "application/json", URL) == '{"a": 1}'
    assert page_text(b"%PDF-1.7", "application/pdf", URL) is None


def test_a_body_is_decoded_by_its_declared_charset() -> None:
    body = "café".encode("latin-1")
    assert page_text(body, "text/plain; charset=ISO-8859-1", URL) == "café"
    assert page_text(body, "text/plain; charset=no-such-codec", URL) == "caf�"
