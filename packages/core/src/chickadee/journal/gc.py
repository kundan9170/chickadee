"""Blob gc — the record is kept, the bulk is pruned on request.

Every prompt, completion, output and pre-image is blobbed and, before this, nothing removed
one. Three rules make the removal safe enough to be a command:

- **A blob's class is derived from the journal**, never stored beside it. The field that
  names a sha names its class (`prompt_sha` → `prompt`, `stdout_sha` → `output`, …). A sha
  any *record* field names is record, whatever else names it; a field this table does not
  know is record too, because the safe direction for an unknown is "keep".
- **The record is never pruned.** Journal files, `config.json`, the handoff (the blob
  `run_end.handoff_path` names), operation manifests, and every record-class blob. `gc` opens
  the journal read-only.
- **Nothing runs on its own.** `plan` and `apply` are called when a person typed `gc` or
  enabled it as a `run_end` hook. There is no timer here and none anywhere else.

`classify` is pure; `plan` decides and unlinks nothing; `apply` unlinks exactly the plan and
writes one line per run to the repo-local `gc.log`, which is how a later miss on a pruned sha
can say *pruned, by that gc, at that time* rather than *absent* (`BlobMissing.why`).
"""

from __future__ import annotations

import datetime as dt
import getpass
import json
import pathlib
import socket
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel

from chickadee import clock
from chickadee.durable import AppendLog, BlobStore, complete_lines, lock_is_held, make_dir
from chickadee.journal import paths
from chickadee.journal.errors import BlobMissing, JournalError
from chickadee.journal.projection import project
from chickadee.journal.reader import read_events
from chickadee.schema import JournalEvent
from chickadee.schema.journal import BoxMessageP, RunEndP


class BlobClass(StrEnum):
    PROMPT = "prompt"
    COMPLETION = "completion"
    OUTPUT = "output"
    IMAGE = "image"
    PREIMAGE = "preimage"
    RECORD = "record"
    UNREFERENCED = "unreferenced"


BULK = frozenset(
    {
        BlobClass.PROMPT,
        BlobClass.COMPLETION,
        BlobClass.OUTPUT,
        BlobClass.IMAGE,
        BlobClass.PREIMAGE,
        BlobClass.UNREFERENCED,
    }
)
"""Every class a plan may name. `record` is not here and cannot be asked for."""

FIELD_CLASSES: Mapping[str, BlobClass] = {
    # Model I/O; a box role's conversation is `box_message`'s bare `sha` (`_message_class`)
    "completion_sha": BlobClass.COMPLETION,
    # Checkpoint and shell output handles; a frontier rescue's output is model output
    "stdout_sha": BlobClass.OUTPUT,
    "stderr_sha": BlobClass.OUTPUT,
    "output_sha": BlobClass.OUTPUT,
    "result_sha": BlobClass.OUTPUT,
    # The viewed image's original
    "image_sha": BlobClass.IMAGE,
    # The rollback path until the handoff
    "preimages_sha": BlobClass.PREIMAGE,
    "pre_sha": BlobClass.PREIMAGE,
    "post_sha": BlobClass.PREIMAGE,
    "post_state_sha": BlobClass.PREIMAGE,
}
"""The bulk fields by name (the table). `completion_sha` and `image_sha` are named ahead
of the contract fields that will carry them, so the day they land they are already bulk.
Every other `*_sha` field — `digest_sha`, `args_sha`, `work_sha`, the config sha — is
**record**, by not being here, and so is every bare `sha` but a box message's: `plan_written`'s
is the plan the person approved at stop 2."""

SHA_LEN = 64


@dataclass(frozen=True)
class RunRow:
    """One run's share of a plan: the shas that go, their classes, and the bytes freed."""

    run_id: str
    shas: tuple[str, ...]
    classes: frozenset[BlobClass]
    bytes: int


@dataclass(frozen=True)
class GcPlan:
    runs_root: pathlib.Path
    cutoff: dt.datetime
    classes: frozenset[BlobClass]
    rows: tuple[RunRow, ...]
    skipped: tuple[tuple[str, str], ...] = ()
    """`(run_id, why)` for every run the plan left alone, named so `--dry-run` shows them."""

    @property
    def shas(self) -> int:
        return sum(len(r.shas) for r in self.rows)

    @property
    def bytes(self) -> int:
        return sum(r.bytes for r in self.rows)


@dataclass(frozen=True)
class GcResult:
    plan: GcPlan
    unlinked: int
    bytes: int
    log: pathlib.Path
    at: str


@dataclass(frozen=True)
class GcLogLine:
    """One line of `gc.log` — what a miss on a pruned sha names."""

    run_id: str
    at: str
    classes: tuple[str, ...]
    count: int
    bytes: int
    cutoff: str
    user: str
    shas: tuple[str, ...] = ()


# -- 1. classify ---------------------------------------------------------------------------


def classify(run_dir: pathlib.Path) -> Mapping[str, BlobClass]:
    """Fold the run's journal once and name every sha's class from the field that names it.

    Pure: reads the journal and the operation manifests, lists `blobs/`, unlinks nothing.
    The mapping covers every sha a field names **and** every blob on disk; a blob no field
    names is `unreferenced`, and a sha that is named but not on disk is still classified so
    a caller can ask about it.
    """
    classes: dict[str, BlobClass] = {}

    def name(sha: str, cls: BlobClass) -> None:
        # Record wins: a sha the record needs is kept whatever else references it.
        if classes.get(sha) is BlobClass.RECORD:
            return
        classes[sha] = cls

    journal = run_dir / paths.JOURNAL_NAME
    if journal.is_file():
        for event in read_events(journal):
            message = _message_class(event.payload)
            for key, sha in blob_shas(event.payload):
                if key == "sha" and message is not None:
                    name(sha, message)
                else:
                    name(sha, FIELD_CLASSES.get(key, BlobClass.RECORD))
            # The handoff is named by a path, not a `*_sha` field, so the walk above misses
            # it; read as unreferenced, the default classes would prune the record the journal
            # keeps.
            handoff = _handoff_sha(event)
            if handoff is not None:
                name(handoff, BlobClass.RECORD)

    # Operation manifests: `{"files": {path: pre_sha}}`, the applier's shape. Their shas are
    # the rollback path, so preimage — and the manifest file itself is record.
    operations = run_dir / paths.OPERATIONS_DIR
    if operations.is_dir():
        for manifest in sorted(operations.rglob("*.json")):
            for sha in _manifest_shas(manifest):
                name(sha, BlobClass.PREIMAGE)

    blobs = run_dir / paths.BLOBS_DIR
    if blobs.is_dir():
        for entry in blobs.iterdir():
            if entry.is_file() and _looks_like_sha(entry.name) and entry.name not in classes:
                classes[entry.name] = BlobClass.UNREFERENCED
    return classes


def _message_class(payload: object) -> BlobClass | None:
    """A box message's class: the model's own turn is a completion, every other side of the
    conversation is prompt. None for any other line, whose bare `sha` is record."""
    if not isinstance(payload, BoxMessageP):
        return None
    return BlobClass.COMPLETION if payload.side == "assistant" else BlobClass.PROMPT


def blob_shas(payload: BaseModel) -> tuple[tuple[str, str], ...]:
    """Every `(field, sha)` pair a payload names a blob by, nested lists and tables included:
    the one reading of which fields are blobs, for gc and for the inspector."""
    return tuple(_sha_fields(payload.model_dump(by_alias=True)))


def _sha_fields(value: Any, key: str = "") -> Iterable[tuple[str, str]]:
    """Every `(field, sha)` pair in a payload, nested lists and tables included."""
    if isinstance(value, Mapping):
        for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
            yield from _sha_fields(v, str(k))  # pyright: ignore[reportUnknownArgumentType]
    elif isinstance(value, list | tuple):
        for v in value:  # pyright: ignore[reportUnknownVariableType]
            yield from _sha_fields(v, key)
    elif isinstance(value, str) and _names_a_sha(key) and _looks_like_sha(value):
        yield key, value


def _names_a_sha(key: str) -> bool:
    """`stdout_sha`, or v2's bare `sha` (`box_message`, `plan_written`): a field that names a
    blob. A bare one missed was read as unreferenced, and the default classes pruned it."""
    return key == "sha" or key.endswith("_sha")


def _manifest_shas(manifest: pathlib.Path) -> Iterable[str]:
    try:
        payload: object = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ()
    if not isinstance(payload, dict):
        return ()
    files: object = cast("dict[str, object]", payload).get("files")
    if not isinstance(files, dict):
        return ()
    return tuple(
        sha
        for sha in cast("dict[str, object]", files).values()
        if isinstance(sha, str) and _looks_like_sha(sha)
    )


def _handoff_sha(event: JournalEvent) -> str | None:
    """The sha of the handoff a `run_end` names (`handoff_path` is `blobs/<sha>`, relative to
    the run), or `None` for any other event or any other shape of path."""
    if not isinstance(event.payload, RunEndP) or event.payload.handoff_path is None:
        return None
    parent, _, sha = event.payload.handoff_path.partition("/")
    return sha if parent == paths.BLOBS_DIR and _looks_like_sha(sha) else None


def _looks_like_sha(text: str) -> bool:
    return len(text) == SHA_LEN and all(c in "0123456789abcdef" for c in text)


# -- 2. plan -------------------------------------------------------------------------------


def plan(
    runs_root: pathlib.Path,
    *,
    older_than: dt.datetime | dt.timedelta,
    classes: Iterable[BlobClass] = BULK,
    run: str | None = None,
) -> GcPlan:
    """Decide what `apply` would unlink. Unlinks nothing.

    Per run: skipped and named when its `LOCK` is held (a live writer) or its journal
    has no terminal (a run still in flight, or one that died before `run_end` and may be
    resumed); `preimage` blobs are excluded until the run is handed off, because until then
    they are the rollback path. Handed off is the journal's word: its `run_end`
    names a handoff blob, and the run still holds it. A run is *old* when its last journaled
    event is before the cutoff.

    **A run's blobs are its own.** `put_blob` writes to `runs/<run_id>/blobs/<sha>`, so two
    runs holding the same bytes hold two files (dedupe is *within* a run), and
    unlinking one run's copy cannot reach another's. The rule — a sha goes only when every
    run naming it qualifies — describes a **shared** store, and applied to this one it
    protected nothing while costing most of the feature: prompt blobs are byte-identical
    across runs by construction, so a single recent run pinned that content in every old run.
    It also ratcheted, because a pruned run still *names* the sha it no longer *has*, which
    made every other copy of those bytes permanently unprunable. If the store is ever shared
    (a repo-level `blobs/`), this rule has to come back with it — it is load-bearing there.

    `older_than` is the cutoff instant, or a `timedelta` back from now. `record` can not be
    asked for and is refused rather than ignored.
    """
    wanted = frozenset(classes)
    if not wanted <= BULK:
        raise JournalError(
            f"gc can only prune the bulk classes {sorted(BULK)}; asked for "
            f"{sorted(wanted - BULK)} — the record is never pruned (J-6)"
        )
    cutoff = _cutoff(older_than)
    runs = _run_dirs(runs_root, run)

    rows: list[RunRow] = []
    skipped: list[tuple[str, str]] = []
    for run_dir in runs:
        why, events = _why_skipped(run_dir, cutoff)
        if why is not None:
            skipped.append((run_dir.name, why))
            continue
        blobs = BlobStore(run_dir / paths.BLOBS_DIR)
        handoff = _handoff_sha(events[-1])
        handed_off = handoff is not None and blobs.has(handoff)
        mine = {
            sha: cls
            for sha, cls in classify(run_dir).items()
            if cls in wanted
            and (cls is not BlobClass.PREIMAGE or handed_off)
            and blobs.has(sha)
        }
        going = sorted(mine)
        if not going:
            continue
        size = sum(blobs.path(sha).stat().st_size for sha in going)
        rows.append(
            RunRow(
                run_id=run_dir.name,
                shas=tuple(going),
                classes=frozenset(mine[sha] for sha in going),
                bytes=size,
            )
        )
    return GcPlan(
        runs_root=runs_root,
        cutoff=cutoff,
        classes=wanted,
        rows=tuple(rows),
        skipped=tuple(skipped),
    )


def _cutoff(older_than: dt.datetime | dt.timedelta) -> dt.datetime:
    if isinstance(older_than, dt.timedelta):
        return dt.datetime.now(dt.UTC) - older_than
    if older_than.tzinfo is None:
        raise JournalError("gc cutoff must be timezone-aware, so it compares to the journal")
    return older_than


def _run_dirs(runs_root: pathlib.Path, run: str | None) -> list[pathlib.Path]:
    if not runs_root.is_dir():
        return []
    if run is not None:
        target = runs_root / run
        return [target] if target.is_dir() else []
    return sorted(p for p in runs_root.iterdir() if p.is_dir())


def _why_skipped(
    run_dir: pathlib.Path, cutoff: dt.datetime
) -> tuple[str | None, list[JournalEvent]]:
    """`None` and the run's events when it qualifies; otherwise the reason `--dry-run`
    prints. A qualifying run's events end in `run_end` (its terminal)."""
    if lock_is_held(run_dir / paths.LOCK_NAME):
        return "LOCK is held by a live process (J-4)", []
    journal = run_dir / paths.JOURNAL_NAME
    if not journal.is_file():
        return "no journal", []
    events = list(read_events(journal))
    if not events:
        return "no terminal — the run is in flight or resumable", events
    projection = project(events)
    # A journal the fold finds inconsistent is refused, as `replay` refused it.
    projection.require_consistent()
    if projection.end is None:
        return "no terminal — the run is in flight or resumable", events
    last = clock.parse(events[-1].ts)
    if last is None:
        return f"last event carries an unreadable timestamp {events[-1].ts!r}", events
    if last >= cutoff:
        return f"younger than the cutoff (last event {events[-1].ts})", events
    return None, events


# -- 3. apply ------------------------------------------------------------------------------


def apply(plan: GcPlan) -> GcResult:
    """Unlink exactly the plan's shas and append one `gc.log` line per run.

    The journal is never opened for writing. A sha already gone is not an error — a second
    `apply` of the same plan is a no-op with a log line saying zero — and the line is
    written after the run's unlinks so a crash between them under-reports rather than
    claims a prune that did not happen.
    """
    log = gc_log_for(plan.runs_root)
    at = clock.stamp()
    user = _user()
    unlinked = 0
    freed = 0
    make_dir(log.parent)
    # Opening repairs a tail torn by a gc that died mid-line, which would otherwise glue
    # itself to this run's first line and hide it from `pruned_record`.
    appending = AppendLog.open(log, create=True)
    try:
        for row in plan.rows:
            blobs = BlobStore(plan.runs_root / row.run_id / paths.BLOBS_DIR)
            gone: list[str] = []
            size = 0
            for sha in row.shas:
                target = blobs.path(sha)
                try:
                    size += target.stat().st_size
                    target.unlink()
                except FileNotFoundError:
                    continue
                gone.append(sha)
            line = GcLogLine(
                run_id=row.run_id,
                at=at,
                classes=tuple(sorted(c.value for c in row.classes)),
                count=len(gone),
                bytes=size,
                cutoff=plan.cutoff.isoformat(timespec="seconds"),
                user=user,
                shas=tuple(gone),
            )
            record = json.dumps(line.__dict__, sort_keys=True) + "\n"
            appending.append(record.encode("utf-8"), sync=False)
            unlinked += len(gone)
            freed += size
    finally:
        appending.close()  # the one fsync, as before; quiet if an append already failed
    return GcResult(plan=plan, unlinked=unlinked, bytes=freed, log=log, at=at)


def gc_log_for(runs_root: pathlib.Path) -> pathlib.Path:
    """`gc.log` beside `runs/`, repo-local — the same root, one up."""
    return runs_root.parent / paths.GC_LOG_NAME


def _user() -> str:
    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"
    except (OSError, KeyError):  # pragma: no cover - no passwd entry, no hostname
        return "?"


# -- 4. the miss ---------------------------------------------------------------------------


def pruned_record(runs_root: pathlib.Path, run_id: str, sha: str) -> GcLogLine | None:
    """The `gc.log` line that pruned `sha` from `run_id`, or `None`. Read on the miss."""
    log = gc_log_for(runs_root)
    if not log.is_file():
        return None
    found: GcLogLine | None = None
    # Split bytes first: a tail torn inside a character is skipped like any torn tail, never
    # a decode error on the miss it is meant to explain.
    for raw in complete_lines(log.read_bytes()):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except ValueError:  # JSONDecodeError, and UnicodeDecodeError from bad UTF-8
            continue
        if data.get("run_id") != run_id or sha not in data.get("shas", ()):
            continue
        found = GcLogLine(
            run_id=str(data["run_id"]),
            at=str(data.get("at", "")),
            classes=tuple(str(c) for c in data.get("classes", ())),
            count=int(data.get("count", 0)),
            bytes=int(data.get("bytes", 0)),
            cutoff=str(data.get("cutoff", "")),
            user=str(data.get("user", "?")),
            shas=tuple(str(s) for s in data.get("shas", ())),
        )
    return found


def blob_missing(runs_root: pathlib.Path, run_id: str, sha: str) -> BlobMissing:
    """The typed miss: `pruned`, naming the log line, or `absent`. The lookup happens here,
    on the miss, so nothing is stored beside a blob that may not exist."""
    line = pruned_record(runs_root, run_id, sha)
    if line is None:
        return BlobMissing(sha)
    return BlobMissing(sha, why="pruned", run_id=line.run_id, gc_at=line.at, user=line.user)


__all__ = [
    "BULK",
    "FIELD_CLASSES",
    "BlobClass",
    "GcLogLine",
    "GcPlan",
    "GcResult",
    "RunRow",
    "apply",
    "blob_missing",
    "classify",
    "gc_log_for",
    "plan",
    "pruned_record",
]
