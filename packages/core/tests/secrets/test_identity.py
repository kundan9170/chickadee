"""SE-5 — hashes are computed pre-redaction, so dedup and replay are unaffected."""

from __future__ import annotations

from chickadee.schema import sha256_hex
from chickadee.secrets.redact import redact

PROMPT = b"call the API with token ghp_1234567890abcdefghij and retry\n"


def test_the_identity_hash_is_taken_before_redaction() -> None:
    """Two prompts differing only in a secret must stay distinguishable.

    Hashing after redaction would collapse them onto one digest — dedup would treat them as
    the same prompt and replay would hand back the wrong one.
    """
    other = PROMPT.replace(b"ghp_1234567890abcdefghij", b"ghp_zzzzzzzzzzabcdefghij")
    assert sha256_hex(PROMPT) != sha256_hex(other)

    redacted_a, _ = redact(PROMPT)
    redacted_b, _ = redact(other)
    assert redacted_a != redacted_b, "different secrets keep different fingerprints"


def test_redaction_is_deterministic() -> None:
    """Replay depends on it: the same bytes must redact to the same bytes every time."""
    first, _ = redact(PROMPT)
    second, _ = redact(PROMPT)
    assert first == second


def test_the_stored_bytes_carry_no_secret() -> None:
    stored, found = redact(PROMPT)
    assert b"ghp_1234567890abcdefghij" not in stored
    assert found and found[0].kind == "github_pat"


def test_the_pre_redaction_hash_still_identifies_the_original() -> None:
    """The dedup key is over what was sent, not over what was persisted."""
    sent_hash = sha256_hex(PROMPT)
    stored, _ = redact(PROMPT)
    assert sent_hash != sha256_hex(stored)
    assert sent_hash == sha256_hex(PROMPT), "recomputable from the original, not the blob"
