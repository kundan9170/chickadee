"""One prompt, one answer, outside any session: a host's own use of a catalog
model, such as the researcher's evaluation's judge, which grades a report the harness wrote.

Nothing is journaled and nothing is charged to a session's ledger, since no session asked: the
host is told what the answer cost and reports it as its own spend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chickadee.client.request import ChatRequest, ModelClient
from chickadee.schema.adapter import Usage

__all__ = ["ModelAnswer", "ask_once"]


@dataclass(frozen=True)
class ModelAnswer:
    text: str
    tokens: int
    """Prompt and completion, as the provider counted them."""
    cost_usd: float


def ask_once(
    client: ModelClient, request: ChatRequest, price: Callable[[Usage], float]
) -> ModelAnswer:
    """`request` sent on `client`, which is closed after: the answer, its tokens, and what
    `price` says it cost."""
    try:
        done = client.send(request)
    finally:
        client.close()
    usage = done.usage
    return ModelAnswer(done.text, usage.prompt_tokens + usage.completion_tokens, price(usage))
