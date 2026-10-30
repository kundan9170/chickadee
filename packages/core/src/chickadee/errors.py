"""Public exception hierarchy.

Internal exceptions never escape the facade. A host embedding the engine sees these
classes and nothing else, because the alternative — leaking `JournalCorrupt`,
`ResumeRefused`, `TransportExhausted` and every future component's private failure — makes
the private modules part of the API by accident, and EN-1 says they are not.

**Terminals are results, not exceptions.** A run that ends `PARTIAL`, `BUDGET_CAP` or
`ABORTED` returns a `RunResult`; nothing here is raised for it. Exceptions mean the engine
could not run the thing, not that the thing did not succeed. Conflating the two is how a
budget cap becomes indistinguishable from a crash.
"""

from __future__ import annotations

from typing import Literal


class ChickadeeError(Exception):
    """Base for every public error. Hosts may catch this to mean "the engine refused"."""


class ConfigError(ChickadeeError, ValueError):
    """Bad config, unknown rung, malformed profile, missing credentials.

    Never retryable programmatically: retrying an unparseable config produces the same
    result, and a host that loops on it burns a run doing nothing.

    **`ValueError` is in the bases on purpose, and it is load-bearing.** Most of what raises
    this raises it from inside a pydantic `model_validator`, and pydantic collects only
    `ValueError` and `AssertionError` into a `ValidationError` — anything else escapes the
    model raw, past the loaders that normalise it. `config.py` had its own
    `ConfigError(ValueError)` for exactly that reason; it is this class now, because two
    uncatchable classes under one name is the worse of the two problems.
    """


class EngineBusy(ChickadeeError):
    """A second concurrent run on one engine. Queue, or use another instance."""


class RepoLeaseBusy(ChickadeeError):
    """Another process holds the repository WRITE lease.

    Distinct from `EngineBusy`: that one is our own engine, this one is a different
    process, and only the second tells the user to go look at another terminal.
    """


class RepoNotIndexed(ChickadeeError):
    """A query arrived before `init_repo()`. Call it first."""


PlannerBlock = Literal["planner_failed", "context_exhausted", "unbuildable"]
"""Why Athena's request gave nothing to act on: the box's `BLOCKED` reason for it."""


class PlannerError(ChickadeeError):
    """Athena's request ended without an answer: her re-asks ran out or her
    endpoint failed (`planner_failed`), or the request did not fit after her one reset
    (`context_exhausted`).

    Raised only when there is nothing to return. A plan that parsed and was then rejected by
    packet-lint is a run outcome, not this; a repair that answers with nothing is R3, not this.
    The box reports either as `BLOCKED{blocked}`, before a run or during one. A preview
    raises it with `unbuildable` too, when her answer is that the design cannot be built as
    approved: a preview has no plan to show then, and the run would end `BLOCKED{unbuildable}`.
    """

    def __init__(
        self,
        detail: str,
        *,
        blocked: PlannerBlock = "planner_failed",
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.blocked: PlannerBlock = blocked


class InfraError(ChickadeeError):
    """A required tool is missing, the graph is unavailable, or a sandbox is required and
    absent.

    Separate from `PlannerError` and from any model failure on purpose: this is never the
    model's fault, and reporting it as one would corrupt the failure attribution E1 exists
    to measure.
    """


class SessionError(ChickadeeError):
    """A persistent session is missing, corrupt, or unwritable.

    Carries the path and last durable seq where it can, because "session broken" without
    them leaves the user nothing to inspect.
    """


class RunNotStarted(ChickadeeError):
    """A run's journal was asked for before it has one: the run has not started yet, or was
    stopped before it started (`BLOCKED{tier_above_approved}`, `unbuildable`). A normal state,
    never "lost" and never a session's error."""


class HandoffError(ChickadeeError):
    """Apply/squash/keep/discard cannot reconcile source state.

    The source branch and the worktree are left untouched when this is raised — a handoff
    that half-applied and then failed is worse than one that refused.
    """


class AttachmentRefused(ChickadeeError, ValueError):
    """A turn's attachment is not one the session takes: not an image, a PDF
    or a text file, or over a cap. The turn is refused whole and nothing is recorded; the
    message names the attachment, its place in the turn and why."""


class HarnessBug(ChickadeeError):
    """An invariant was violated, a sequence was impossible, or an internal structure was
    malformed.

    **This is our bug, and it is never swallowed.** Everything else here describes a
    situation the engine handles; this one says the engine's own reasoning was wrong, so
    catching it broadly would hide exactly the failures worth finding. The run aborts and
    the host is expected to report it.
    """
