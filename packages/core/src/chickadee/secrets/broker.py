"""Credential injection for one authorized effect.

Values are injected into a single Executor or MCP call **after** authorization, and never
enter a model-visible structure, a packet, a prompt, or a journal payload. What travels
through those is the variable *name*.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from chickadee.secrets.env import looks_like_credential


class UndeclaredCredential(Exception):
    """A call asked for a variable its approval never mentioned."""


class AlphaGateClosed(Exception):
    """A mode that is not authorized until the external-alpha security gate passes."""


@dataclass(frozen=True)
class Exposure:
    """What the approval gate renders: names, never values."""

    names: tuple[str, ...]

    @property
    def summary(self) -> str:
        return f"exposes: {', '.join(self.names)}" if self.names else "exposes: nothing"

    @property
    def credential_bearing(self) -> bool:
        """D40 second-confirm class. Advisory naming, but the *class* is what drives the
        second confirmation, so it is computed here rather than eyeballed at the gate."""
        return any(looks_like_credential(n) for n in self.names)


class Broker:
    """Injects declared credential values into one authorized effect.

    **Trust is a constructor argument, not a runtime hope.** The trusted
    single-author prototype may use explicitly configured *first-party* credentials. Scoped
    third-party credentials and untrusted repositories are external-alpha gates, and they
    **fail closed** here rather than being merely discouraged in prose — §3.5 says the
    prototype's allowance "does not authorize third-party MCP, external alpha users, or
    untrusted repos", and a rule that only lives in a document is one the first integration
    walks past.
    """

    def __init__(
        self,
        source: Mapping[str, str],
        repo_trust: str = "first_party",
        alpha_gate_passed: bool = False,
    ) -> None:
        self._source = source
        self.repo_trust = repo_trust
        self.alpha_gate_passed = alpha_gate_passed

    def _check_gate(self, server_trust: str) -> None:
        if self.alpha_gate_passed:
            return
        if self.repo_trust != "first_party":
            raise AlphaGateClosed(
                f"repository trust is {self.repo_trust!r}; untrusted repositories are an "
                "external-alpha gate (SE-10) and fail closed until it passes"
            )
        if server_trust != "first_party":
            raise AlphaGateClosed(
                f"server trust is {server_trust!r}; scoped third-party credentials are an "
                "external-alpha gate (SE-10) and fail closed until it passes"
            )

    def exposure(self, declared: tuple[str, ...]) -> Exposure:
        return Exposure(tuple(declared))

    def inject(
        self,
        declared: tuple[str, ...],
        approved: tuple[str, ...],
        server_trust: str = "first_party",
    ) -> dict[str, str]:
        """Values for exactly the approved names.

        A name declared but not approved raises rather than being dropped. Silently omitting
        it would hand the command a missing variable and let it fail somewhere confusing,
        while the authorization mismatch — the thing that actually went wrong — went
        unreported.
        """
        self._check_gate(server_trust)
        missing = [n for n in declared if n not in approved]
        if missing:
            raise UndeclaredCredential(
                f"{missing} were requested but not approved; approval covers {list(approved)}"
            )
        return {n: self._source[n] for n in declared if n in self._source}
