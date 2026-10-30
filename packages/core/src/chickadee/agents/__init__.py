"""Agents: a fresh, narrower model Zeus hands one job to.

An agent is a profile plus a skill. The profile (`profiles.py`, built on the roles `reader`
and `reviewer`) says what it may do; the skill says how. `spawn.py` runs one round of it over
the shared tool loop, `conversation.py` journals what it is told and says, and `report.py` is
the one shape its answer takes, its citations checked by code.
"""
