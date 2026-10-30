"""What Zeus hands the box: `CoderRequest` v2.

The work is one of two kinds, and exactly one, each already approved at its stop:

- **a design** (`ApprovedDesign`): Zeus's markdown, the Request and Constraints sections the
  harness inserted, as the person approved it at stop 1. Athena plans it, asks stop 2 for her
  plan, and delegates its tasks;
- **a one-task plan** (`OneTask`): Zeus's title and brief, the brief as the person approved
  it at stop 2. The desk delegates it at once and Athena reviews it.

Each carries the sha the stop bound (invariant 15: what runs is what was shown) and the sha of
the template it was written to, so the run records which templates were in force. Nothing here
parses the markdown. The router's tier, the claims it judged and the exact path's
hunks retired with the packet loop: a write the person must know of is asked at its command,
and approval of the work is the stops'.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chickadee.schema.ids import sha256_hex
from chickadee.schema.isolation import WorkspaceMode

__all__ = [
    "ApprovedDesign",
    "AttachedFile",
    "CoderRequest",
    "Continuation",
    "OneTask",
    "WorkPin",
]

WIRE = ConfigDict(extra="forbid", frozen=True)


def _bound(text: str, sha: str, what: str) -> None:
    # A host builds the request and the engine asks no stop of its own: a text changed after
    # its stop would run while the journal named the approved one as what ran.
    if sha256_hex(text) != sha:
        raise ValueError(
            f"the {what}'s sha is not the sha of its text: it is not what was approved"
        )


class WorkPin(BaseModel):
    """The work folder a design was approved in: its name, `work.dir`
    then, and the `design/` tree stop 1 pinned, which every later read of it is answered from
    (`workspace.pinned_text`)."""

    model_config = WIRE

    name: str = Field(min_length=1)
    dir: str = Field(min_length=1)
    tree: str = Field(min_length=1)

    @property
    def folder(self) -> str:
        """The work's folder, from the repository root."""
        return f"{self.dir}/{self.name}"


class ApprovedDesign(BaseModel):
    """A design as approved at stop 1."""

    model_config = WIRE

    text: str = Field(min_length=1)
    """The markdown approved, exactly: Zeus's design with the Request and Constraints sections
    the harness inserted."""
    sha: str
    """`sha256_hex(text)`, which the stop bound; refused when it is not (invariant 15)."""
    template_sha: str
    """The design template in force when Zeus wrote it (`handoffs.resolve`)."""
    work: WorkPin | None = None
    """The folder the design is, with its pinned `design/`: `text` is its README as
    pinned. None for a design handed over as text alone, by a host."""

    @model_validator(mode="after")
    def _binds_its_text(self) -> ApprovedDesign:
        _bound(self.text, self.sha, "design")
        return self


class OneTask(BaseModel):
    """Zeus's one-task plan as approved at stop 2."""

    model_config = WIRE

    title: str = Field(min_length=1)
    brief: str = Field(min_length=1)
    """The brief approved, exactly."""
    sha: str
    """`sha256_hex(brief)`, which the stop bound; refused when it is not (invariant 15)."""
    template_sha: str
    """The brief template in force when Zeus wrote it."""

    @model_validator(mode="after")
    def _binds_its_brief(self) -> OneTask:
        _bound(self.brief, self.sha, "brief")
        return self


class Continuation(BaseModel):
    """A run that continues a work an ended or superseded run left: what it inherits, read back
    on resume from the run's config
    snapshot, so the seed never depends on a record that may have changed since."""

    model_config = WIRE

    mode: Literal["CONTINUE", "AMEND"]
    """`AMEND`: the approved design or brief changed since the work's last run."""
    prior_runs: tuple[str, ...] = Field(min_length=1)
    """The work's earlier runs, oldest first: their task books are this run's history."""
    plan_run: str | None = None
    """The run whose approved plan this one inherits; None when no plan was approved."""
    plan_version: int | None = Field(default=None, ge=1)
    """That plan's version in `plan_run`, whose pin (`plan-v<n>`) is the plan inherited."""
    plan_sha: str | None = None
    """The inherited plan's text sha, as its `plan_written` line named it."""
    plan_bound_to: str | None = None
    """The design tree that plan was approved under: a plan revision is asked again
    when the design's tree is no longer it."""
    amended_from: str | None = None
    """`AMEND`: the design tree (or a one-task brief's sha) approved before this one."""


class AttachedFile(BaseModel):
    """A file the person sent with the turn the work answers: named as the
    session numbered it, its bytes the session blob `sha` names. The Request section lists it,
    so it was approved with the work."""

    model_config = WIRE
    id: str = Field(pattern=r"^A-[1-9][0-9]*$")
    name: str = Field(min_length=1, max_length=200)
    sha: str = Field(min_length=1)


class CoderRequest(BaseModel):
    """`design` or `task`, exactly one, and where the work runs."""

    model_config = WIRE

    design: ApprovedDesign | None = None
    task: OneTask | None = None
    workspace: WorkspaceMode = "worktree"
    continues: Continuation | None = None
    """Set when this run continues a work rather than starting it: the plan it
    inherits and the runs before it. None for a work's first run."""
    attachments: tuple[AttachedFile, ...] = ()
    """The files approved with the work: each role's head lists them with a handle on
    its own reader."""

    @model_validator(mode="after")
    def _one_kind_of_work(self) -> CoderRequest:
        if (self.design is None) == (self.task is None):
            raise ValueError("exactly one of design or task")
        return self

    @property
    def work_sha(self) -> str:
        """The approved work's sha: the design's, or the brief's. What `run_start` names."""
        if self.design is not None:
            return self.design.sha
        if self.task is None:  # pragma: no cover - the validator refuses it
            raise ValueError("a request with no work")
        return self.task.sha

    @property
    def goal(self) -> str:
        """One line naming the work: the one-task plan's title, or the design's first line."""
        if self.task is not None:
            return self.task.title
        text = self.design.text if self.design is not None else ""
        first = next((line for line in text.splitlines() if line.strip()), "")
        return first.lstrip("# ").strip() or "the approved design"
