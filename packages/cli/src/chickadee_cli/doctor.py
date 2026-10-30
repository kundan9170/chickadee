"""Renders the toolchain probe for `init` and the run header.

Presentation only. The probe itself is engine knowledge and lives in core; a surface may
import the facade and nothing else, so what arrives here is already decided and this
file only decides how it reads.
"""

from __future__ import annotations

from collections.abc import Mapping


def render(
    toolchains: Mapping[str, Mapping[str, str | None]], degraded: tuple[str, ...]
) -> str:
    """One block per profile, then the degradations in plain words.

    Degradations are printed even when the list is long. Summarising them as a count is
    how "3 gates will skip" becomes something nobody reads, and the whole point of probing
    at init is that the user learns it here rather than in a handoff.
    """
    lines: list[str] = []
    for profile in sorted(toolchains):
        lines.append(f"{profile}:")
        for binary in sorted(toolchains[profile]):
            version = toolchains[profile][binary]
            lines.append(f"  {binary:<12} {version if version else 'not found'}")
    if degraded:
        lines.append("")
        lines.append("Degraded — these will be weaker than a full run:")
        lines.extend(f"  - {d}" for d in degraded)
    return "\n".join(lines)


def banner(degraded: tuple[str, ...]) -> str | None:
    """The one-line run-header form, or None when nothing is degraded."""
    if not degraded:
        return None
    return f"degraded: {len(degraded)} check(s) will skip — see `chickadee init` output"
