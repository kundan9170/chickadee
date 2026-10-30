"""Previews (`journal.previews`): what a tool or model record carries for a reader is
redacted before it is clipped, bounded, and marked with what was cut."""

from __future__ import annotations

from chickadee.journal.previews import PREVIEW_CHARS, args_preview, preview

KEY = "AKIAIOSFODNN7EXAMPLE"


def test_a_short_text_is_kept_whole() -> None:
    assert preview("def add(a, b):") == "def add(a, b):"


def test_a_long_text_is_clipped_and_says_how_much_was_cut() -> None:
    shown = preview("x" * (PREVIEW_CHARS + 50))
    assert shown == "x" * PREVIEW_CHARS + "…[+50 chars]"


def test_a_secret_is_redacted_before_the_clip_so_no_half_of_it_is_written() -> None:
    # The key straddles the clip: clipped first, its head would slip past the detector.
    text = "y" * (PREVIEW_CHARS - 11) + " " + KEY
    shown = preview(text)
    assert KEY[:10] not in shown
    assert "[REDACTED:aws_access_key:" in preview(f"key={KEY}")


def test_arguments_are_compact_sorted_json() -> None:
    assert args_preview({"path": "a.py", "anchor": 3}) == '{"anchor": 3, "path": "a.py"}'
    assert KEY not in args_preview({"command": f"echo {KEY}"})


def test_a_secret_named_by_its_key_is_redacted_though_json_quotes_the_key() -> None:
    shown = args_preview({"token": "abcd1234efgh5678abcd", "n": {"api_key": "z" * 16}})
    assert "abcd1234efgh5678abcd" not in shown and "z" * 16 not in shown
    assert shown.count("[REDACTED:assigned_secret:") == 2


def test_a_large_text_is_redacted_only_as_far_as_can_be_shown_and_counts_all_it_cut() -> None:
    shown = preview("x" * 20_000)
    assert shown == "x" * PREVIEW_CHARS + f"…[+{20_000 - PREVIEW_CHARS:,} chars]"
