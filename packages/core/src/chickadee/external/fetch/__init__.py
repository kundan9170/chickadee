"""The engines that fetch a page `web_fetch` admitted.

An engine is given an `Admitted` URL, every guard passed, and makes one GET. It never follows
a redirect: it says where one points (`Moved`), and the `web` port decides, since only it holds
the guards. `http` is the engine here; `lightpanda`, for a page that needs its scripts run,
is not built yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from chickadee.context.results import FetchFailed
from chickadee.external.urls import Admitted

__all__ = ["CONTACT", "Engine", "Fetched", "Moved"]

CONTACT: Final = "+https://github.com/kundan9170/chickadee"
"""Who is asking, in every engine's `User-Agent`: Wikimedia's API rules ask every client for a
contact, and without one Wikipedia refuses the request outright."""


@dataclass(frozen=True)
class Fetched:
    """A page, as text."""

    url: str
    text: str
    truncated: bool
    """The body was cut at the engine's byte cap."""


@dataclass(frozen=True)
class Moved:
    """A redirect, not followed: where it points, made absolute."""

    url: str
    location: str
    status: int
    """The redirect's status; 0 when the browser followed it and landed elsewhere."""


class Engine(Protocol):
    @property
    def provider(self) -> str:
        """Its name on the `external_call` line: `web.fetch.http`."""
        ...

    def get(self, target: Admitted) -> Fetched | Moved | FetchFailed: ...
