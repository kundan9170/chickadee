"""Approval: the command table, the standing preferences and the grants. Zero
model calls.

The work tier's router, its fact rows and the component expansion retired with the packet
loop: the work is approved at the stops (`approval/stops.py`), and a
command's tier is the shell's, per call, from the command table and the person's class rules.
"""

from __future__ import annotations

from chickadee.approval.commands import (
    CommandClass,
    CommandMatch,
    CommandRow,
    CommandTable,
    CommandTableError,
    Destination,
    binding_of,
    profile_rows,
    shipped,
)
from chickadee.approval.prefs import PreferenceError, load_preferences, merge, parse_level
from chickadee.approval.tier import TIER_ORDER, FactHit, Preferences, Tier

__all__ = [
    "TIER_ORDER",
    "CommandClass",
    "CommandMatch",
    "CommandRow",
    "CommandTable",
    "CommandTableError",
    "Destination",
    "FactHit",
    "PreferenceError",
    "Preferences",
    "Tier",
    "binding_of",
    "load_preferences",
    "merge",
    "parse_level",
    "profile_rows",
    "shipped",
]
