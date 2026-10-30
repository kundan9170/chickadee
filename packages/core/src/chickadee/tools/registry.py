"""One registry, two sources: the built-ins, listed in one file
(`builtin/__init__.py`), and a session's MCP tools, added at run time (`with_dynamic`).

No discovery from outside the repository: a skill or a
hook cannot register a tool, and nothing is loaded from a project or plugin directory.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from chickadee.errors import HarnessBug
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Port, Tool

__all__ = ["Registry"]


class Registry:
    """Tools by name. Names are unique; a second tool under one name is a harness bug, never
    a silent replacement."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            name = tool.spec.name
            if name in self._tools:
                raise HarnessBug(f"two tools are registered as {name!r}")
            self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(t.spec for t in self._tools.values())

    def needs(self, names: Iterable[str]) -> frozenset[Port]:
        """The ports the named tools use, together: what a set of tools reaches, so a caller
        asks what a set may touch rather than which names it holds.
        A name it does not register reaches nothing."""
        found = (self._tools.get(n) for n in names)
        return frozenset(p for t in found if t is not None for p in t.needs)

    def with_dynamic(self, tools: Sequence[Tool]) -> Registry:
        """This registry and a session's MCP tools. Which of them a role
        is offered is MCP's policy (`read_only_tools`), not the enabled list, because
        their names are not known at load."""
        return Registry((*self._tools.values(), *tools))

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
