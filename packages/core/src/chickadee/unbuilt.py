"""A facade method declared before the work behind it exists, and the task that builds it.

`18-engine.md` §2 declares the whole facade, and some of it waits on machinery that is not
built: the session handoff on the final suite, `finalize` on the Git review. Such a method is
marked `@not_built("T-x.y")`. Calling it raises `NotImplementedError` naming that task, as the
bare `raise` it replaces did, and a host can ask `owner_of(method)` without calling it.

That second half is what a served engine needs: `GET /` reports the
capabilities that answer, and a route over an unbuilt method is left out until its task lands,
with no list kept anywhere else to drift.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Final

__all__ = ["not_built", "owner_of"]

_MARK: Final = "__chickadee_not_built__"


def not_built[F: Callable[..., Any]](owner: str) -> Callable[[F], F]:
    """Mark a method unbuilt, owned by task `owner`. The body is never run."""

    def mark(fn: F) -> F:
        @functools.wraps(fn)
        def refuse(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise NotImplementedError(
                f"{fn.__name__} is not built yet; it arrives with {owner}"
            )

        setattr(refuse, _MARK, owner)
        return refuse  # pyright: ignore[reportReturnType]

    return mark


def owner_of(method: object) -> str | None:
    """The task that builds `method`, or None when it is built (or is not a marked method)."""
    owner = getattr(method, _MARK, None)
    return owner if isinstance(owner, str) else None
