"""Tokens, and who an answer is recorded as coming from.

**The authority is the credential's, fixed when it was issued.** The server cannot know
whether a person was at a keyboard; it knows a token authenticated. An `interactive` token is
issued to a frontend that shows a question to a person and sends their choice, and its
answers are recorded `human`. An `automation` token's answers are recorded as the `default`
the script agreed with. Nothing in a request is consulted, so a client cannot promote itself.

**Every route but `GET /` needs a token, on a Unix socket too.** The filesystem decides who can
connect; the token class decides what their answers mean, which a socket's permissions
cannot say.

`serve.token` hands the tokens to the clients this process expects: JSON naming one token per
class, beside the socket, readable by its owner only, and removed when the server stops.
"""

from __future__ import annotations

import hmac
import json
import secrets
from pathlib import Path
from typing import Final, Literal

from chickadee.durable import unlink_durably, write_atomic
from chickadee.schema.interaction import AnswerSource
from chickadee_serve.frames import TOKEN_FILE, TokenClass, TokenFile

__all__ = [
    "TOKEN_BYTES",
    "TOKEN_FILE",
    "TOKEN_FILE_MODE",
    "TokenClass",
    "TokenStore",
    "approval_source",
    "bearer",
    "source_for",
]

TOKEN_BYTES: Final = 32
"""`secrets.token_urlsafe`'s entropy. Not a tunable anyone should lower."""

TOKEN_FILE_MODE: Final = 0o600


def source_for(token_class: TokenClass) -> AnswerSource:
    """The table: what an answer sent with this class of token is recorded as."""
    return AnswerSource.HUMAN if token_class is TokenClass.INTERACTIVE else AnswerSource.DEFAULT


def approval_source(token_class: TokenClass) -> Literal["human", "default"]:
    """The same rule for an action approval (`ApprovalDecision.source`)."""
    return "human" if token_class is TokenClass.INTERACTIVE else "default"


class TokenStore:
    """The tokens this server issued, by class. Held in memory; a restart issues new ones."""

    def __init__(self) -> None:
        self._issued: dict[str, TokenClass] = {}

    def issue(self, token_class: TokenClass) -> str:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self._issued[token] = token_class
        return token

    def verify(self, token: str | None) -> TokenClass | None:
        """The class `token` was issued with, or None. Compared in constant time against every
        issued token, so a near miss takes as long as a far one."""
        if not token:
            return None
        found: TokenClass | None = None
        for issued, token_class in self._issued.items():
            if hmac.compare_digest(issued.encode(), token.encode()):
                found = token_class
        return found

    def write(self, path: Path, tokens: dict[TokenClass, str]) -> None:
        """`serve.token`, owner-readable only, through the durable writer. Always a new file:
        a replaced file keeps its own mode, so a leftover readable by others would carry
        fresh credentials. Its writer holds the server lock (`server.running`), so nothing
        else is writing it."""
        doc = TokenFile(
            interactive=tokens[TokenClass.INTERACTIVE], automation=tokens[TokenClass.AUTOMATION]
        )
        unlink_durably(path)
        raw = json.dumps(doc.model_dump(mode="json"), sort_keys=True).encode()
        write_atomic(path, raw, mode=TOKEN_FILE_MODE)

    @staticmethod
    def remove(path: Path) -> None:
        unlink_durably(path)


def bearer(header: str | None) -> str | None:
    """The token of an `Authorization: Bearer <token>` header, or None."""
    if header is None:
        return None
    scheme, _, value = header.partition(" ")
    return value.strip() or None if scheme.lower() == "bearer" else None
