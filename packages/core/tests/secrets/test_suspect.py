"""Advisory only: it reports, and never cuts."""

from __future__ import annotations

import copy

from chickadee.secrets.suspect import scan, scan_text

BUNDLE = {
    "app.py": 'API_KEY = "ghp_1234567890abcdefghij"\n',
    "sauce.py": 'secret_sauce = "tomatoes, basil and a long slow simmer"\n',
    "tests/fixtures.py": 'example_token = "aZ3kL9pQ2xR7vT1nW4yB8c"\n',
    "prose.py": 'MESSAGE = "hello world, this is ordinary prose for a user"\n',
}


def test_a_seeded_key_is_reported_with_path_and_line() -> None:
    findings = scan(BUNDLE)
    hit = next(f for f in findings if f.path == "app.py")
    assert hit.line == 1
    assert hit.kind == "github_pat"


def test_a_secret_sauce_identifier_is_not_reported() -> None:
    """The canonical false positive BL-47 names, and the reason substring matching alone
    is not enough."""
    assert not [f for f in scan(BUNDLE) if f.path == "sauce.py"]


def test_a_fixture_constant_is_not_reported() -> None:
    """`\\bexample\\b` never fires inside `example_token` — `_` is a word character — so the
    benign check works on subwords."""
    assert not [f for f in scan(BUNDLE) if f.path.startswith("tests/")]


def test_prose_is_not_reported() -> None:
    assert not [f for f in scan(BUNDLE) if f.path == "prose.py"]


def test_the_bundle_is_byte_identical_afterwards() -> None:
    """SE-11 never cuts. BL-47 rejected denial because the packer cannot tell a hardcoded
    secret from a fixture, so blanking content would be a correctness failure *caused by* a
    security feature."""
    before = copy.deepcopy(BUNDLE)
    scan(BUNDLE)
    assert before == BUNDLE


def test_scanning_returns_findings_rather_than_content() -> None:
    findings = scan_text("app.py", 'API_KEY = "ghp_1234567890abcdefghij"\n')
    assert findings
    assert not any("ghp_1234567890abcdefghij" in f.detail for f in findings), (
        "a finding that quotes the secret has copied it somewhere new"
    )


def test_a_high_entropy_assignment_is_reported() -> None:
    findings = scan_text("c.py", 'auth = "aZ3kL9pQ2xR7vT1nW4yB8cD"\n')
    assert findings and findings[0].kind in ("high_entropy", "assigned_secret")


def test_a_low_entropy_assignment_is_not() -> None:
    assert not scan_text("c.py", 'greeting = "aaaaaaaaaaaaaaaaaaaaaaaa"\n')


def test_a_multiline_private_key_is_reported() -> None:
    """Scanning line by line meant the `private_key` detector — which needs both PEM
    delimiters — could never fire, so a real key was reported as nothing at all."""
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAxyz\n-----END RSA PRIVATE KEY-----\n"
    )
    findings = scan_text("deploy/id_rsa", pem)
    assert findings, "a multiline private key must be reported"
    assert findings[0].kind == "private_key"
    assert findings[0].line == 1


def test_a_known_format_is_not_double_reported() -> None:
    """A line already flagged by a known detector is not re-flagged by the entropy pass."""
    findings = scan_text("a.py", 'API_KEY = "ghp_1234567890abcdefghij"\n')
    assert len(findings) == 1
