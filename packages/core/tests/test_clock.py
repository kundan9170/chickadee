"""One clock for every durable record.

`stamp()` is UTC, second precision, with a trailing `Z`, and the journal's contract accepts it.
`parse()` reads every form a record has held, aware: a `Z`, a `+00:00`, and the zone-free
local time the journal wrote in older versions, read as local.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from chickadee import clock
from chickadee.schema.ids import TS_RE


def test_a_stamp_is_utc_to_the_second_with_a_z() -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    text = clock.stamp()
    after = datetime.now(UTC)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text)
    assert TS_RE.match(text), "the journal's contract accepts what the clock writes"
    parsed = clock.parse(text)
    assert parsed is not None and before <= parsed <= after


def test_a_stamp_round_trips() -> None:
    text = clock.stamp()
    parsed = clock.parse(text)
    assert parsed is not None
    assert parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") == text


@pytest.mark.parametrize(
    "text", ["2026-10-04T09:30:00Z", "2026-10-04T09:30:00+00:00", "2026-10-04T11:30:00+02:00"]
)
def test_each_zoned_form_is_the_same_instant(text: str) -> None:
    assert clock.parse(text) == datetime(2026, 10, 4, 9, 30, tzinfo=UTC)


def test_a_zone_free_stamp_is_local_time() -> None:
    """What the journal wrote in older versions: naive, in the machine's local time."""
    parsed = clock.parse("2026-10-04T09:30:00")
    assert parsed is not None and parsed.tzinfo is not None
    assert parsed == datetime(2026, 10, 4, 9, 30).astimezone()


def test_old_and_new_stamps_compare() -> None:
    """`gc` compares a journal's last `ts` with a cutoff, whichever form the journal holds."""
    old = clock.parse((datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S"))
    new = clock.parse(clock.stamp())
    assert old is not None and new is not None and old < new


@pytest.mark.parametrize("text", ["", "yesterday", "2026-13-01T00:00:00Z", "2026-10-04 9:30"])
def test_text_that_is_not_a_time_is_none(text: str) -> None:
    assert clock.parse(text) is None


def test_a_non_utc_offset_keeps_its_zone() -> None:
    parsed = clock.parse("2026-10-04T11:30:00+02:00")
    assert parsed is not None and parsed.utcoffset() == timezone(timedelta(hours=2)).utcoffset(
        None
    )
