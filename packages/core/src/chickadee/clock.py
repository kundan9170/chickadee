"""One clock for every durable record.

Every record that keeps a time writes it as `stamp()` gives it: UTC, second precision, a
trailing `Z`. The journal's `ts`, the session log's `at`, `gc.log`'s `at`, the scout's hints and
the graph index's `indexed_at` all come from here. Ordering is never this: a journal orders by
`seq`, a session log by its own `seq`.

`parse` reads every form a record has ever held, aware: `Z`, `+00:00`, and the zone-free local
time the journal wrote before this decision. That last one is read as the machine's local time,
which is what it was written in, so `gc` compares an old journal and a new one alike.

Not clocks: `init`'s generation directory name (a sortable name, not a record's time) and the
turn frame's `date` (the person's local date, by design).
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["parse", "stamp"]


def stamp() -> str:
    """Now, as every durable record writes it: `YYYY-MM-DDTHH:MM:SSZ`."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(text: str) -> datetime | None:
    """A recorded time, aware; `None` for text that is not one. A zone-free stamp is local
    time, as the journal wrote it in older versions."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone() if parsed.tzinfo is None else parsed
