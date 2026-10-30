"""`web_fetch`'s guards, each in its order: the
first that fails decides, and each refuses before anything is sent."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from chickadee.context.results import UrlRefused
from chickadee.external.urls import (
    DOCS_DOMAINS,
    Admitted,
    Resolver,
    SeenUrls,
    check_url,
    find_urls,
    normalise,
)

PUBLIC = "93.184.216.34"


def resolver(*addresses: str) -> Resolver:
    def resolve(host: str, port: int) -> Sequence[str]:
        del host, port
        return addresses

    return resolve


def unresolvable(host: str, port: int) -> Sequence[str]:
    raise OSError(f"no address for {host}:{port}")


def seen(*urls: str) -> SeenUrls:
    out = SeenUrls()
    for url in urls:
        out.admit_url(url)
    return out


def check(
    url: str, *, given: SeenUrls | None = None, addresses: Sequence[str] = (PUBLIC,)
) -> Admitted | UrlRefused:
    return check_url(url, seen=given or seen(url), max_chars=2000, resolve=resolver(*addresses))


@pytest.mark.parametrize(
    ("url", "normal"),
    [
        ("HTTPS://Docs.Python.ORG/3/", "https://docs.python.org/3/"),
        ("https://a.com:443/x#frag", "https://a.com/x"),
        ("http://a.com:80", "http://a.com/"),
        ("http://a.com:8080/x?q=1", "http://a.com:8080/x?q=1"),
        ("http://[::1]:8000/", "http://[::1]:8000/"),
        ("ftp://a.com/x", None),
        ("not a url", None),
    ],
)
def test_normalise_gives_one_form_to_compare(url: str, normal: str | None) -> None:
    assert normalise(url) == normal


def test_find_urls_drops_what_ends_a_sentence_and_an_unbalanced_bracket() -> None:
    text = (
        "See https://a.com/x. Or (https://b.com/y), and "
        "[docs](https://en.wikipedia.org/wiki/Foo_(bar)). Also <https://c.com/z>"
    )
    assert find_urls(text) == (
        "https://a.com/x",
        "https://b.com/y",
        "https://en.wikipedia.org/wiki/Foo_(bar)",
        "https://c.com/z",
    )


def test_a_url_that_appeared_is_admitted_with_the_addresses_it_resolved_to() -> None:
    got = check("https://Example.com/page?id=7#top")
    assert got == Admitted(
        "https://example.com/page?id=7", "example.com", 443, (PUBLIC,), False
    )


def test_a_url_nobody_gave_is_refused_before_anything_else() -> None:
    """Rule 1 is first: a URL the model wrote itself is how what it read is sent out."""
    got = check_url(
        "https://evil.example/?q=secret", seen=SeenUrls(), max_chars=2000, resolve=unresolvable
    )
    assert got == UrlRefused("https://evil.example/?q=secret", "not_seen")
    assert isinstance(got, UrlRefused) and "search for the page" in got.render()


def test_a_guess_on_a_documentation_domain_is_admitted_without_a_query_only() -> None:
    empty = SeenUrls()
    got = check_url(
        "https://docs.python.org/3/library/re.html",
        seen=empty,
        max_chars=2000,
        resolve=resolver(PUBLIC),
    )
    assert isinstance(got, Admitted) and got.guessed
    asked = "https://docs.python.org/3/search.html?q=secret"
    refused = check_url(asked, seen=empty, max_chars=2000, resolve=resolver(PUBLIC))
    assert refused == UrlRefused(asked, "not_seen")


def test_a_host_that_only_looks_like_a_documentation_domain_is_not_one() -> None:
    for url in ("https://docs.python.org.evil.com/", "https://evil.com/docs.python.org/"):
        got = check_url(url, seen=SeenUrls(), max_chars=2000, resolve=resolver(PUBLIC))
        assert got == UrlRefused(url, "not_seen")


def test_the_person_s_own_documentation_domains_are_guessable_too() -> None:
    got = check_url(
        "https://docs.example.org/guide",
        seen=SeenUrls(),
        docs=DOCS_DOMAINS | {"docs.example.org"},
        max_chars=2000,
        resolve=resolver(PUBLIC),
    )
    assert isinstance(got, Admitted) and got.guessed


def test_the_shipped_list_holds_no_host_whose_pages_its_users_write() -> None:
    written_by_users = {
        "github.com",
        "gist.github.com",
        "raw.githubusercontent.com",
        "readthedocs.io",
        "docs.rs",
        "pastebin.com",
        "medium.com",
    }
    assert not DOCS_DOMAINS & written_by_users
    assert not [h for h in DOCS_DOMAINS if h.endswith((".github.io", ".readthedocs.io"))]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a.com/x", "https:///x", "a.com"])
def test_a_url_that_is_not_http_is_one_nothing_showed(url: str) -> None:
    """`SeenUrls` holds only http(s) URLs, so guard 1 refuses every other kind."""
    given = SeenUrls()
    given.admit(f"see {url} and https://a.com/")
    assert check(url, given=given) == UrlRefused(url, "not_seen")


def test_a_url_past_the_length_cap_is_refused() -> None:
    url = "https://a.com/" + "x" * 2000
    assert check(url) == UrlRefused(url, "too_long")


@pytest.mark.parametrize(
    "url",
    [
        "https://user:hunter2@a.com/",
        "https://a.com/x?api_key=sk_live_0123456789abcdef",
    ],
)
def test_a_url_carrying_a_credential_is_refused(url: str) -> None:
    assert check(url) == UrlRefused(url, "credential")


@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        ("10.0.0.7",),
        ("169.254.169.254",),
        ("::1",),
        ("fd00::1",),
        ("::ffff:127.0.0.1",),
        # NAT64's well-known prefix and the old compatible form carry an IPv4 address, which
        # Python counts as global whatever it is.
        ("64:ff9b::a9fe:a9fe",),
        ("64:ff9b::7f00:1",),
        ("::7f00:1",),
        ("224.0.0.1",),
        (PUBLIC, "192.168.1.1"),
    ],
)
def test_a_host_with_any_private_address_is_refused(addresses: tuple[str, ...]) -> None:
    assert check("https://a.com/", addresses=addresses) == UrlRefused(
        "https://a.com/", "private_address"
    )


def test_a_public_address_behind_nat64_is_admitted() -> None:
    got = check("https://a.com/", addresses=("64:ff9b::808:808",))
    assert isinstance(got, Admitted) and got.addresses == ("64:ff9b::808:808",)


def test_a_host_that_does_not_resolve_is_refused() -> None:
    url = "https://nowhere.invalid/"
    got = check_url(url, seen=seen(url), max_chars=2000, resolve=unresolvable)
    assert got == UrlRefused(url, "unresolvable")
