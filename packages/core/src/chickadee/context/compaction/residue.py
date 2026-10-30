"""Layer 3: what prose leaves that the record does not already hold. A model call, bounded and
detached; on a timeout the cut still commits, with empty
residue, and what it returns is marked a claim.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Final, Literal, cast

from chickadee.client.request import ChatRequest, Message, ModelClient
from chickadee.context.compaction.receipts import flat
from chickadee.context.request import est_text
from chickadee.schema.adapter import RetryParams, Usage

__all__ = [
    "L3_MAX_ITEMS",
    "L3_TIMEOUT_MS",
    "NO_RESIDUE",
    "Job",
    "ModelResidue",
    "ProseTurn",
    "ResidueModel",
    "ResidueOutcome",
    "ResidueRequest",
    "ResidueSource",
    "residue",
    "residue_text",
]


L3_MAX_ITEMS: Final = 12
"""A bound, not a target: an unbounded list is a summary wearing a list's clothes."""


L3_TIMEOUT_MS: Final = 10_000
"""A cut never waits on a model longer than the request it is making room for."""


Job = Literal["fallback", "residue"]


@dataclass(frozen=True)
class ProseTurn:
    seq: int
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class ResidueRequest:
    """What layer 3 asks, fixed in shape: turns X-Y and the record already made from them, and
    ONLY what the turns hold that the record does not — never "summarise this span". The
    wording lives with the model's adapter templates, never inline here."""

    job: Job
    first_seq: int
    last_seq: int
    turns: tuple[ProseTurn, ...]
    exclusion: tuple[str, ...]
    """The record already made — layer 2's block and any checkpoint rows. Read as an
    EXCLUSION SET, never as material: summarising a projection is compacting a compaction."""
    max_items: int


ResidueModel = Callable[[ResidueRequest], Sequence[str]]
"""The cheap rung (`role("residue")`), behind the caller's client. One line per item."""


@dataclass(frozen=True)
class ResidueSource:
    """Layer 3's inputs besides the region."""

    model: ResidueModel | None = None
    last_checkpoint_seq: int = 0
    """The narrowing point: everything at or below it already faced this question from a
    checkpoint, and asking again is the re-processing loop guard 4 forbids."""
    checkpoint_rows: tuple[str, ...] = ()
    """The typed rows those checkpoints recorded — part of the exclusion set."""
    off: str | None = None
    """Why there is no model, when the composition knows (an unpriced residue rung): what a
    cut records as its residue's failure, in place of the generic reason."""


NO_RESIDUE: Final = ResidueSource()


@dataclass
class ModelResidue:
    """`ResidueModel` over a client: the cheap rung (`role("residue")`), one request per cut,
    rendered from the role's template (`frontier/residue.md`, never inline) and read back one
    item per line.

    **Spend is held, not charged.** The call runs on `residue`'s detached thread while the
    cut holds the session's lock, so charging the session's ledger from here would wait on
    that lock while the cut waits on this answer. Each answer's usage is kept until the
    session's driver, on its own thread, `drain`s it and charges it: a call that
    outlives its timeout is still charged, at the next drain.

    **Concurrency.** `_spent` is appended from the residue thread and drained from the
    driver's; `_lock` guards it."""

    client: ModelClient
    params: RetryParams
    template: str
    _spent: list[Usage] = field(default_factory=list[Usage])
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, request: ResidueRequest) -> Sequence[str]:
        prompt = self.template.format(
            max_items=request.max_items,
            exclusion="\n".join(request.exclusion) or "(nothing)",
            first_seq=request.first_seq,
            last_seq=request.last_seq,
            turns="\n\n".join(f"[{t.seq} {t.role}] {t.text}" for t in request.turns),
        )
        sent = ChatRequest(messages=(Message(role="user", content=prompt),), params=self.params)
        completion = self.client.send(sent)
        with self._lock:
            self._spent.append(completion.usage)
        return [_unbulleted(line) for line in completion.text.splitlines()]

    def drain(self) -> tuple[Usage, ...]:
        """Every answer's usage not yet charged, once."""
        with self._lock:
            spent, self._spent = tuple(self._spent), []
        return spent


def _unbulleted(line: str) -> str:
    """The template asks for bare lines; a model that bullets them anyway is not refused."""
    stripped = line.strip()
    for mark in ("-", "*", "•"):
        if stripped == mark:
            return ""
        if stripped.startswith(f"{mark} "):
            return stripped[len(mark) + 1 :]
    return stripped


@dataclass(frozen=True)
class ResidueOutcome:
    job: Job
    items: tuple[str, ...]
    fired: bool
    """Whether the model was called."""
    failure: str | None
    """A timeout, an error, or a missing model. Recorded; never fails the cut."""
    first_seq: int = 0
    last_seq: int = 0
    input_tokens: int = 0
    """The request's estimated weight: the `H` of the payback arithmetic."""


def residue(
    job: Job, prose: Sequence[ProseTurn], exclusion: Sequence[str], source: ResidueSource
) -> ResidueOutcome:
    """§5.10. Runs after layer 2 has rendered, on the prose past the last checkpoint only.

    A failure never fails the cut: the cut commits with an empty residue and the failure is
    recorded, since a compaction that cannot complete because a model call failed is the
    market's failure reproduced from the other direction. An empty answer asserts nothing —
    it is never "nothing else was discussed", because layer 3's coverage is not total."""
    span = tuple(t for t in prose if t.seq > source.last_checkpoint_seq)
    if not span:
        return ResidueOutcome(job, (), fired=False, failure=None)
    if source.model is None:
        why = source.off or "no residue model is configured"
        return ResidueOutcome(job, (), fired=False, failure=why)
    request = ResidueRequest(
        job, span[0].seq, span[-1].seq, span, tuple(exclusion), L3_MAX_ITEMS
    )
    weight = sum(est_text(t.text) for t in span) + sum(est_text(x) for x in exclusion)
    outcome = ResidueOutcome(job, (), True, None, span[0].seq, span[-1].seq, weight)
    answer = _call_detached(source.model, request)
    try:
        items = _items(answer.result(L3_TIMEOUT_MS / 1000))
    except TimeoutError:
        return replace(outcome, failure=f"timed out after {L3_TIMEOUT_MS} ms")
    except Exception as exc:  # any model failure, by rule, becomes an empty residue
        return replace(outcome, failure=f"{type(exc).__name__}: {exc}")
    return replace(outcome, items=items)


def _call_detached(
    model: Callable[[ResidueRequest], object], request: ResidueRequest
) -> Future[object]:
    """The residue call on a daemon thread. A call already running cannot be cancelled, so
    one that outlives its timeout is abandoned — and a pool's worker would be joined at
    interpreter exit, holding the process open on a model that never answers."""
    answer: Future[object] = Future()

    def run() -> None:
        if not answer.set_running_or_notify_cancel():
            return
        try:
            answer.set_result(model(request))
        except BaseException as exc:
            answer.set_exception(exc)

    threading.Thread(target=run, name="residue", daemon=True).start()
    return answer


def _items(raw: object) -> tuple[str, ...]:
    """One line per item, blank items dropped, at most `L3_MAX_ITEMS`."""
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError("a residue is a list of lines")
    items: list[str] = []
    for item in cast("Sequence[object]", raw):
        if not isinstance(item, str):
            raise TypeError("a residue item is a line of text")
        if item.strip():
            items.append(flat(item))
    return tuple(items[:L3_MAX_ITEMS])


def residue_text(outcome: ResidueOutcome) -> str:
    span = f"{outcome.first_seq}-{outcome.last_seq}"
    rows = [f"[model recollection · not a record · turns {span}]"]
    rows.extend(f"  - {item}" for item in outcome.items)
    return "\n".join(rows)
