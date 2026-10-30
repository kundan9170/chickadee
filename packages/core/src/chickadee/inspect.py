"""What each model was sent and said, read from what the harness records (`chickadee inspect`).

Read-only, and through the harness's own readers: a run's journal (`journal.reader.read_after`)
and its blobs, and a session's log (`context.store.SessionLog`). Nothing here writes, so it is
safe beside a running session.

- **A run:** every message Athena and the worker were given or wrote (`box_message`, full text),
  each call's charge (`model_call`) and the request its transport got (`sent_sha`), each
  tool call's facts (`tool_call`), and every other line with the blobs it names inlined.
- **A session:** Zeus's turns, his calls and each tool result as he read it, and his system
  prompt rebuilt from the `prefix` line's part blobs by `zeus.prefix.system_text`. A
  session from in older versions has no part blobs until its next turn; `prompt` is then None.

Values are plain JSON (dicts and lists) because their one reader is the inspector's page; no
harness code reads them back.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Final, cast

from chickadee.context.store import SessionLog, list_sessions, session_dir, sessions_root
from chickadee.journal.gc import blob_shas
from chickadee.journal.paths import blobs_dir, journal_path, runs_root
from chickadee.journal.reader import read_after
from chickadee.schema import JournalEvent
from chickadee.schema.journal import BoxMessageP, ModelCallP, RunStartP
from chickadee.state.roots import Roots, xdg_config_home
from chickadee.zeus.prefix import PARTS, TEMPLATES, PartName, system_text

__all__ = ["BLOB_LIMIT", "Inspection", "state_roots"]

BLOB_LIMIT: Final = 2_000_000
"""A blob larger than this is named, not inlined: a page should not stall on one file."""

_SHA: Final = re.compile(r"[0-9a-f]{64}")
_NAME: Final = re.compile(r"[A-Za-z0-9._-]+")
"""A run or session id as a path segment: nothing that climbs out of its directory."""

Json = dict[str, Any]


def state_roots(state: Path) -> Roots:
    """A state directory given directly (an eval archive, a test's tmp dir): the layout under
    it is the repo-local root's, `runs/` and `sessions/`."""
    local = state.resolve()
    return Roots(project=local, repo_local=local, checkout_local=local, user=xdg_config_home())


class Inspection:
    """The runs and sessions under one repo-local root, as the inspector's page reads them."""

    def __init__(self, roots: Roots) -> None:
        self._roots = roots

    @property
    def state(self) -> Path:
        return self._roots.repo_local

    # -- the listing ----------------------------------------------------------------------

    def index(self) -> Json:
        """Every session and run, newest first. A damaged one is listed with its error, not
        hidden."""
        sessions: list[Json] = []
        for sid in list_sessions(sessions_root(self._roots)):
            try:
                sessions.append(self._session_summary(sid))
            except Exception as exc:
                sessions.append(
                    {"id": sid, "mtime": 0, "title": None, "runs": [], "error": str(exc)}
                )
        runs: list[Json] = []
        root = runs_root(self._roots)
        if root.is_dir():
            for d in root.iterdir():
                if (d / "journal.jsonl").is_file():
                    try:
                        runs.append(self._run_summary(d.name))
                    except Exception as exc:
                        runs.append({"id": d.name, "mtime": 0, "goal": None, "error": str(exc)})
        sessions.sort(key=lambda s: s["mtime"], reverse=True)
        runs.sort(key=lambda s: s["mtime"], reverse=True)
        return {"state": str(self.state), "sessions": sessions, "runs": runs}

    def _run_summary(self, run_id: str) -> Json:
        path = journal_path(self._roots, run_id)
        stat = path.stat()
        first = read_after(path, 0, limit=1).events
        goal = (
            first[0].payload.goal if first and isinstance(first[0].payload, RunStartP) else None
        )
        # The last complete line says whether the run ended; reading only the file's end keeps
        # a listing cheap however long a run grows.
        outcome = None
        with path.open("rb") as fh:
            fh.seek(max(0, stat.st_size - 4096))
            for line in reversed(fh.read().splitlines()):
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
                if last.get("kind") == "run_end":
                    outcome = last.get("payload", {}).get("outcome")
                break
        return {"id": run_id, "goal": goal, "mtime": stat.st_mtime, "outcome": outcome}

    def _session_summary(self, sid: str) -> Json:
        root = sessions_root(self._roots)
        log = SessionLog.read(root, sid)
        runs: list[str] = []
        title: str | None = None
        for event in log.events():
            if event.run_id and event.run_id not in runs:
                runs.append(event.run_id)
            if title is None and event.kind == "turn":
                payload = json.loads(log.payload(event))
                if payload.get("role") == "user":
                    title = str(payload.get("text", ""))[:120]
        mtime = (session_dir(root, sid) / "session.jsonl").stat().st_mtime
        return {"id": sid, "mtime": mtime, "title": title, "runs": runs}

    # -- a run ----------------------------------------------------------------------------

    def run(self, run_id: str, after: int = 0) -> Json:
        """The run's lines past seq `after`. A request's messages and tools are named by sha and
        given once per answer in `messages` and `tools`: a long conversation re-sends every
        earlier message on each call, and the page keeps them."""
        if not _NAME.fullmatch(run_id):
            raise FileNotFoundError(run_id)
        path = journal_path(self._roots, run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        tail = read_after(path, after)
        store = blobs_dir(self._roots, run_id)
        out: list[Json] = []
        messages: Json = {}
        tools: Json = {}
        for event in tail.events:
            row = _row(event)
            payload = event.payload
            if isinstance(payload, BoxMessageP):
                text = _blob_text(store, payload.sha)
                row["message"] = _message(text) if text and text.startswith("{") else None
            elif isinstance(payload, ModelCallP) and payload.sent_sha:
                row["sent"] = _sent(store, payload.sent_sha, messages, tools)
            else:
                blobs = {
                    key: text
                    for key, sha in blob_shas(payload)
                    if (text := _blob_text(store, sha)) is not None
                }
                if blobs:
                    row["blobs"] = blobs
            out.append(row)
        damage = str(tail.damage) if tail.damage is not None else None
        return {"events": out, "damage": damage, "messages": messages, "tools": tools}

    # -- a session ------------------------------------------------------------------------

    def session(self, sid: str, after: int = 0) -> Json:
        """The session's lines past seq `after`, each payload parsed; a `prefix` line carries
        Zeus's prompt, or None when its parts were not stored."""
        if not _NAME.fullmatch(sid):
            raise FileNotFoundError(sid)
        root = sessions_root(self._roots)
        if not (session_dir(root, sid) / "session.jsonl").is_file():
            raise FileNotFoundError(sid)
        log = SessionLog.read(root, sid)
        out: list[Json] = []
        for event in log.events():
            if event.seq <= after:
                continue
            raw = log.payload(event).decode("utf-8", errors="replace")
            try:
                payload: Any = json.loads(raw)
            except ValueError:
                payload = {"text": raw}
            row: Json = {
                "seq": event.seq,
                "at": event.at,
                "kind": event.kind,
                "run_id": event.run_id,
                "payload": payload,
            }
            if event.kind == "prefix":
                store = session_dir(root, event.session_id) / "blobs"
                prefix = cast("Json", payload) if isinstance(payload, dict) else {}
                row["prompt"] = _prompt(store, prefix.get("parts", {}))
            out.append(row)
        return {"events": out}

    @staticmethod
    def zeus_template() -> Json:
        """Zeus's template as it is on disk now: what the page shows for a session from before
        D223, labelled as such."""
        path = TEMPLATES / "zeus.md"
        return {"path": str(path), "text": path.read_text(encoding="utf-8")}


def _row(event: JournalEvent) -> Json:
    """`event` as the page reads it, written out as stored."""
    return event.model_dump(mode="json")


def _blob_text(store: Path, sha: object) -> str | None:
    """A blob's text from its flat directory (`durable.blobs`' layout), read as other readers
    read one (`context.project.JournalRecords.blob`): never through a `BlobStore`, which exists
    to write. None for a value that is not a sha or a blob that is not there (pruned)."""
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        return None
    path = store / sha
    if not path.is_file():
        return None
    if path.stat().st_size > BLOB_LIMIT:
        return f"[blob {sha[:12]} is larger than {BLOB_LIMIT} bytes; not inlined]"
    return path.read_bytes().decode("utf-8", errors="replace")


def _message(raw: str) -> Json:
    """A box message's canonical JSON (`boxctx.conversation.message_bytes`), an image named
    by its type and size rather than inlined."""
    body: Json = json.loads(raw)
    content = body.get("content")
    if isinstance(content, list):
        parts: list[Json] = []
        for part in content:  # pyright: ignore[reportUnknownVariableType]
            if "text" in part:
                parts.append({"text": part["text"]})
            else:
                size = len(base64.b64decode(part.get("data", "")))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                parts.append({"image": part.get("media_type"), "bytes": size})  # pyright: ignore[reportUnknownMemberType]
        body["content"] = parts
    return body


def _sent(store: Path, sha: str, messages: Json, tools: Json) -> Json | None:
    """A `model_call`'s request as its transport got it (`boxctx.sent.record_sent`): the
    index, with each message and the tool list it names added to the answer's tables."""
    text = _blob_text(store, sha)
    if text is None:
        return None
    index: Json = json.loads(text)
    for m in index.get("messages", []):
        if m not in messages:
            body = _blob_text(store, m)
            messages[m] = _message(body) if body and body.startswith("{") else None
    t = index.get("tools")
    if t and t not in tools:
        body = _blob_text(store, t)
        tools[t] = json.loads(body) if body else None
    return index


def _prompt(store: Path, parts: Json) -> Json | None:
    """Zeus's system message and tool list from the prefix's part blobs, or None for a
    session that has not stored them."""
    texts: dict[PartName, str] = {}
    for part in PARTS:
        text = _blob_text(store, parts.get(part))
        if text is None:
            return None
        texts[part] = text
    return {"system": system_text(texts), "tools": json.loads(texts["tools"]).get("tools", [])}
