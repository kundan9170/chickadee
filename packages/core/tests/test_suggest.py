"""The nearest known name: one helper, whose callers keep their cutoffs — a
tool name `Offered` refuses at `difflib`'s 0.6. (The packet worker's parse, which named an
undeclared native call's nearest tool at 0.5, retired with the packet loop: D196, `T-3.157`.)"""

from __future__ import annotations

from chickadee.suggest import nearest


def test_the_nearest_name_is_offered_and_a_far_one_is_not() -> None:
    known = ("high-param", "mid-param", "low-param")
    assert nearest("hgh-param", known) == "high-param"
    assert nearest("nemotron", known) is None
    assert nearest("anything", ()) is None


def test_a_tie_resolves_the_same_way_whatever_order_known_is_in() -> None:
    assert nearest("ab", ["ac", "aa"]) == nearest("ab", ["aa", "ac"])


def test_the_cutoff_decides_what_is_close_enough() -> None:
    # `difflib` scores "grep" against "grep_in" at 0.73.
    assert nearest("grep", ["grep_in"], cutoff=0.8) is None
    assert nearest("grep", ["grep_in"], cutoff=0.6) == "grep_in"
