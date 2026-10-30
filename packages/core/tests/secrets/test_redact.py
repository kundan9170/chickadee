"""Layer 3 — detectors, identity preservation, and no false redaction."""

from __future__ import annotations

import pytest

from chickadee.secrets.redact import fingerprint, redact, redact_text

SAMPLES = {
    "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
    "github_pat": "ghp_1234567890abcdefghij",
    "openai_key": "sk-abcdefghijklmnopqrstuvwx",
    "anthropic_key": "sk-ant-abcdefghijklmnopqrst",
    "slack_token": "xoxb-1234567890-abcdefg",
    "google_key": "AIza" + "a" * 35,
    "jwt": "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4",
}


@pytest.mark.parametrize(("kind", "value"), sorted(SAMPLES.items()))
def test_each_detector_finds_its_format(kind: str, value: str) -> None:
    out, found = redact_text(f"the value is {value} here")
    assert value not in out, f"{kind} survived redaction"
    assert any(f.kind == kind for f in found), f"{kind} not reported: {found}"


def test_a_private_key_block_is_redacted_whole() -> None:
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
    out, found = redact_text(block)
    assert "MIIEowIBAAKCAQEA" not in out
    assert found[0].kind == "private_key"


def test_url_credentials_go_but_the_url_survives() -> None:
    out, _ = redact_text("db = postgres://admin:hunter2@db.internal:5432/app")
    assert "hunter2" not in out
    assert "db.internal:5432/app" in out, "the host is debugging information, not a secret"


def test_an_assigned_secret_loses_only_its_value() -> None:
    out, _ = redact_text('password = "correct-horse-battery"')
    assert "correct-horse-battery" not in out
    assert "password" in out


def test_identity_is_preserved_across_occurrences() -> None:
    """SE-4: the same secret reads the same everywhere, so a human can say 'that is the
    token from turn 3' while holding nothing they could use."""
    token = "ghp_1234567890abcdefghij"
    out, found = redact_text(f"first {token}\nsecond {token}\n")
    placeholder = f"[REDACTED:github_pat:{fingerprint(token)}]"
    assert out.count(placeholder) == 2
    assert found[0].count == 2


def test_different_secrets_get_different_placeholders() -> None:
    out, _ = redact_text("a ghp_1234567890abcdefghij b ghp_zzzzzzzzzzabcdefghij")
    assert len({line for line in out.split() if line.startswith("[REDACTED")}) == 2


@pytest.mark.parametrize(
    "code",
    [
        "def compute(x):\n    return x * 2\n",
        "MAX_RETRIES = 3\n",
        "url = 'https://example.com/docs'\n",
        "secret_sauce = 'tomatoes and basil'\n",
        "# the token is stored in the vault\n",
    ],
)
def test_ordinary_code_is_untouched(code: str) -> None:
    """A redactor that mangles source is a redactor people disable."""
    out, found = redact_text(code)
    assert out == code
    assert found == ()


def test_bytes_in_bytes_out() -> None:
    """The blob path is bytes, so a caller cannot forget to re-encode."""
    out, _ = redact(b"token: ghp_1234567890abcdefghij")
    assert isinstance(out, bytes)
    assert b"ghp_1234567890abcdefghij" not in out


def test_invalid_utf8_bytes_survive_untouched() -> None:
    """This sits on the way into every blob. `replace` was lossy in both directions —
    `b"\xff"` came back as `b"\xef\xbf\xbd"` with no secret present at all, so a redactor
    that found nothing still corrupted what it was handed."""
    raw = b"\xff\xfe some binary \x00 payload"
    out, found = redact(raw)
    assert out == raw
    assert found == ()


def test_a_secret_beside_invalid_bytes_is_still_redacted() -> None:
    raw = b"\xff token=ghp_1234567890abcdefghij \xfe"
    out, found = redact(raw)
    assert b"ghp_1234567890abcdefghij" not in out
    assert out.startswith(b"\xff") and out.endswith(b"\xfe")
    assert found
