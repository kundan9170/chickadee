"""Every route but `GET /` needs a token this server issued; the token file is owner-only and
goes when the server does."""

from __future__ import annotations

import stat
from pathlib import Path

from demo_session import SESSION_ID
from serve_helpers import replay

from chickadee_serve.auth import TOKEN_FILE_MODE, TokenStore, bearer
from chickadee_serve.frames import TokenClass, read_token


def test_hello_needs_no_token_and_everything_else_does(tmp_path: Path) -> None:
    with replay(tmp_path, start=False) as h:
        assert h.client.get("/").status_code == 200
        for headers in (
            {},
            {"authorization": "Bearer not-issued"},
            {"authorization": "Basic x"},
        ):
            r = h.client.get(f"/session/{SESSION_ID}", headers=headers)
            assert r.status_code == 401 and r.json()["code"] == "UNAUTHORIZED"
        assert h.client.get(f"/session/{SESSION_ID}", headers=h.headers()).status_code == 200


def test_the_token_file_is_owner_only_and_removable(tmp_path: Path) -> None:
    store = TokenStore()
    tokens = {cls: store.issue(cls) for cls in TokenClass}
    path = tmp_path / "serve.token"
    store.write(path, tokens)
    assert stat.S_IMODE(path.stat().st_mode) == TOKEN_FILE_MODE
    assert read_token(path, TokenClass.INTERACTIVE) == tokens[TokenClass.INTERACTIVE]
    TokenStore.remove(path)
    assert not path.exists()


def test_tokens_verify_by_class() -> None:
    store = TokenStore()
    a = store.issue(TokenClass.AUTOMATION)
    i = store.issue(TokenClass.INTERACTIVE)
    assert store.verify(a) is TokenClass.AUTOMATION
    assert store.verify(i) is TokenClass.INTERACTIVE
    assert store.verify(a[:-1]) is None
    assert store.verify(None) is None


def test_bearer_reads_only_the_bearer_scheme() -> None:
    assert bearer("Bearer abc") == "abc"
    assert bearer("bearer abc") == "abc"
    assert bearer("Basic abc") is None
    assert bearer("Bearer ") is None
    assert bearer(None) is None


def test_a_leftover_token_file_readable_by_others_is_replaced_owner_only(
    tmp_path: Path,
) -> None:
    """A replaced file keeps its own mode: fresh credentials never take a loose one."""
    path = tmp_path / "serve.token"
    path.write_text("{}")
    path.chmod(0o644)
    store = TokenStore()
    store.write(path, {cls: store.issue(cls) for cls in TokenClass})
    assert stat.S_IMODE(path.stat().st_mode) == TOKEN_FILE_MODE
