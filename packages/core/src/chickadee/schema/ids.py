"""Identifier formats and the one hash function the contracts use.

Every `*_sha` field in the wire formats is sha256 hex over UTF-8 bytes. Keeping
that in one place matters more than it looks: a stop binds a text's sha (invariant 15), so two
components disagreeing about encoding or digest would silently unbind an approval rather than
fail.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import date

# Task ids: a task is named by its plan, `T3`, and a re-cut's pieces by their parent,
# `T3.1`, `T3.1.2`. The harness checks every id against the pinned plan, or names it by lineage
# where there is no plan, so a model never names a task the harness has not checked. The
# packet, stub and checkpoint ids retired with the packet loop.
TASK_ID_RE = re.compile(r"^T[1-9][0-9]{0,5}(\.[1-9][0-9]{0,3}){0,4}$")

# A task's attempt as the journal keys it: the id for the first attempt, then
# `T3@2`, `T3@3`. Every id is a key, so a journal written in older versions reads unchanged.
TASK_KEY_RE = re.compile(TASK_ID_RE.pattern[:-1] + r"(@([2-9]|[1-9][0-9]{1,3}))?$")

# Run ids: <YYYY-MM-DD>-<8 lowercase hex>.
RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[0-9a-f]{8}$")

# ISO-8601, second precision. Every durable record stamps UTC with a trailing `Z`, one
# clock; a journal written before carries a zone-free local time, which still validates. The
# `Z` is optional so the change is additive: nothing accepted before is refused.
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$")

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def new_run_id(today: date | None = None) -> str:
    """A fresh run id. The date prefix makes `runs/` browsable in filename order."""
    day = today or date.today()
    return f"{day.isoformat()}-{secrets.token_hex(4)}"


def sha256_hex(data: str | bytes) -> str:
    """The project's only digest. Text is hashed as UTF-8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
