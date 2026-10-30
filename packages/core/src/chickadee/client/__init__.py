"""Model transports: `ChatRequest`, and two clients over the vendors' own SDKs.

`core` assumes no runtime and no provider. `OpenAIClient` covers every OpenAI-compatible
surface; `AnthropicClient` is the one case where that layer drops caching. Everything
else vendor-specific is a config row in the affinity table, not a branch in the request path.
"""

from __future__ import annotations

from chickadee.client.affinity import DEFAULT_ROWS, AffinityRow, AffinityTable
from chickadee.client.anthropic_client import AnthropicClient
from chickadee.client.errors import (
    ClientError,
    ContextOverflow,
    CredentialError,
    RequestCancelled,
    ToolChoiceRefused,
    TransportExhausted,
)
from chickadee.client.openai_client import (
    RETRY_POLICY,
    STOP_REASON_MAP,
    EndpointSpec,
    OpenAIClient,
    RetryPolicy,
    map_stop_reason,
    probe_usage,
)
from chickadee.client.request import (
    ChatRequest,
    ImagePart,
    Message,
    ModelClient,
    TextPart,
    ToolSchema,
    to_request,
)


def client_for(endpoint: EndpointSpec) -> AnthropicClient | OpenAIClient:
    """The transport a rung's provider speaks: Anthropic's own client for
    `anthropic`, the OpenAI-compatible one for every other provider. The one place this is
    chosen, so the box's rungs reuse it rather than copying it."""
    if endpoint.provider == "anthropic":
        return AnthropicClient(endpoint)
    return OpenAIClient(endpoint)


__all__ = [
    "DEFAULT_ROWS",
    "RETRY_POLICY",
    "STOP_REASON_MAP",
    "AffinityRow",
    "AffinityTable",
    "AnthropicClient",
    "ChatRequest",
    "ClientError",
    "ContextOverflow",
    "CredentialError",
    "EndpointSpec",
    "ImagePart",
    "Message",
    "ModelClient",
    "OpenAIClient",
    "RequestCancelled",
    "RetryPolicy",
    "TextPart",
    "ToolChoiceRefused",
    "ToolSchema",
    "TransportExhausted",
    "client_for",
    "map_stop_reason",
    "probe_usage",
    "to_request",
]
