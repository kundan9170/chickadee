"""How a name a person or a model writes refers to a declaration's qualified name.

One rule, for every reader that matches a written name against qualified ones: the auto-packer
choosing a target and the scout checking a pointer. Kept free of the graph's imports so a
reader of plain files can use it: `context/outline.py`'s `resolve_at` still writes the test
itself, because that module is at its D157 size ratchet; it moves here when outline is split.
"""

from __future__ import annotations

__all__ = ["names_qualified"]


def names_qualified(qualified: str, name: str) -> bool:
    """Whether `name` names `qualified`: the whole qualified name, or a suffix of it that
    starts at a dot. `Store.get` names `app.Store.get`; `ore.get` and `get` of `forget` do
    not, because a suffix must begin a segment."""
    return qualified == name or qualified.endswith(f".{name}")
