"""A fetched body as the text a model reads.

HTML becomes Markdown-shaped text: headings as `#`, links as `[text](absolute url)`, list items
as `- `, `<pre>` kept verbatim in a fence and `<code>` in backticks. Scripts, styles, forms and
navigation are dropped, since none of them is what a page says. `text/plain`, Markdown and JSON
are kept as they are. Anything else is not text, and `page_text` says so with None.

Built on the standard library's `html.parser`, which never fails on bad markup: a page that is
not well formed still comes back as its text.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Final
from urllib.parse import urljoin

__all__ = ["html_to_text", "is_text", "media_type", "page_text"]

_KEPT: Final = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "application/json",
        "text/csv",
        # A search API that streams its one answer as a server-sent event (the keyless Exa
        # trial's MCP endpoint): text, read as it is.
        "text/event-stream",
    }
)
_HTML: Final = frozenset({"text/html", "application/xhtml+xml"})
_DROPPED: Final = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "nav",
        "form",
        "button",
        "select",
        "head",
    }
)
_VOID: Final = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "wbr",
    }
)
_BLOCKS: Final = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "aside",
        "blockquote",
        "figure",
        "figcaption",
        "table",
        "thead",
        "tbody",
        "tr",
        "dl",
        "dt",
        "dd",
        "ul",
        "ol",
        "details",
        "summary",
    }
)
_HEADINGS: Final = {f"h{n}": n for n in range(1, 7)}
_SPACE = re.compile(r"\s+")


def media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def is_text(content_type: str) -> bool:
    """Whether a body of this `Content-Type` becomes text. An absent type is taken as text, as
    a browser sniffs one."""
    kind = media_type(content_type)
    return not kind or kind in _HTML or kind in _KEPT or kind.endswith(("+json", "+xml"))


def page_text(body: bytes, content_type: str, url: str) -> str | None:
    """`body` as text, decoded by its declared charset (UTF-8 when it names none), or None when
    its type is not text. `url` resolves the page's relative links."""
    if not is_text(content_type):
        return None
    text = body.decode(_charset(content_type), errors="replace")
    kind = media_type(content_type)
    if kind in _HTML or (not kind and text.lstrip()[:1] == "<"):
        return html_to_text(text, url)
    return text


def html_to_text(html: str, url: str) -> str:
    parser = _Text(url)
    parser.feed(html)
    parser.close()
    return parser.result()


def _charset(content_type: str) -> str:
    match = re.search(r"charset=\"?([\w.:-]+)", content_type, re.IGNORECASE)
    name = match.group(1) if match else "utf-8"
    try:
        "".encode(name)
    except LookupError:
        return "utf-8"
    return name


class _Text(HTMLParser):
    """Blocks of text, each flushed whole when the next begins."""

    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._url = url
        self._title: list[str] = []
        self._in_title = False
        self._blocks: list[tuple[str, str]] = []
        """Each block, and `list` or `row` for a list item or a table row: a run of either
        stands on lines of its own rather than in paragraphs."""
        self._line: list[str] = []
        self._prefix = ""
        self._dropped = 0
        self._pre = 0
        self._pre_text: list[str] = []
        self._links: list[tuple[str | None, int]] = []
        self._lists: list[list[int]] = []
        """Each open list's next number, or -1 for a bulleted one."""
        self._cell = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _DROPPED:
            self._dropped += tag not in _VOID
        elif not (self._dropped or self._pre or self._open_block(tag)):
            self._open_inline(tag, dict(attrs))

    def _open_block(self, tag: str) -> bool:
        """A tag that starts a block of its own, opened; False for any other."""
        if tag in _HEADINGS:
            self._flush()
            self._prefix = "#" * _HEADINGS[tag] + " "
        elif tag == "li":
            self._flush()
            self._prefix = "  " * max(0, len(self._lists) - 1) + self._mark()
        elif tag in ("ul", "ol"):
            self._flush()
            self._lists.append([1 if tag == "ol" else -1])
        elif tag == "pre":
            self._flush()
            self._pre = 1
            self._pre_text = []
        elif tag in _BLOCKS or tag in ("br", "hr"):
            self._flush()
            if tag == "blockquote":
                self._prefix = "> "
        else:
            return False
        return True

    def _open_inline(self, tag: str, attrs: dict[str, str | None]) -> None:
        if tag in ("td", "th"):
            if self._cell:
                self._line.append(" | ")
            self._cell += 1
        elif tag == "a":
            self._links.append((attrs.get("href"), len(self._line)))
        elif tag == "code":
            self._line.append("`")
        elif tag == "img" and attrs.get("alt"):
            self._line.append(attrs["alt"] or "")

    def _mark(self) -> str:
        """A list item's mark: a bullet, or the open numbered list's next number."""
        counter = self._lists[-1] if self._lists else [-1]
        if counter[0] < 0:
            return "- "
        counter[0] += 1
        return f"{counter[0] - 1}. "

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in _DROPPED:
            self._dropped = max(0, self._dropped - 1)
            return
        if self._dropped:
            return
        if self._pre:
            if tag == "pre":
                body = "".join(self._pre_text).strip("\n")
                self._blocks.append((f"```\n{body}\n```", ""))
                self._pre = 0
            return
        if tag == "a" and self._links:
            href, start = self._links.pop()
            text = _SPACE.sub(" ", "".join(self._line[start:])).strip()
            target = self._target(href)
            if text and target:
                self._line[start:] = [f"[{text}]({target})"]
        elif tag == "code":
            self._line.append("`")
        elif tag in ("ul", "ol"):
            self._flush()
            if self._lists:
                self._lists.pop()
        elif tag == "tr":
            self._flush()
            self._cell = 0
        elif tag in _HEADINGS or tag == "li" or tag in _BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif self._dropped:
            return
        elif self._pre:
            self._pre_text.append(data)
        else:
            self._line.append(data)

    def result(self) -> str:
        self._flush()
        title = _SPACE.sub(" ", "".join(self._title)).strip()
        blocks = self._blocks
        if title and not (blocks and blocks[0][0].startswith("# ")):
            blocks = [(f"# {title}", ""), *blocks]
        out: list[str] = []
        for i, (text, run) in enumerate(blocks):
            if i:
                out.append("\n" if run and run == blocks[i - 1][1] else "\n\n")
            out.append(text)
        return "".join(out)

    def _flush(self) -> None:
        text = _SPACE.sub(" ", "".join(self._line)).strip()
        if text and text != "``":
            mark = self._prefix.lstrip()[:1]
            run = "row" if self._cell else "list" if mark and mark in "-0123456789" else ""
            self._blocks.append((self._prefix + text, run))
        self._line = []
        self._prefix = ""
        self._links = []

    def _target(self, href: str | None) -> str | None:
        if not href or href.startswith(("#", "javascript:", "mailto:", "data:")):
            return None
        return urljoin(self._url, href.strip())
