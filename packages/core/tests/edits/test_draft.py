"""The worker's draft: made over the targets as rollback left them, one
write (`CandidateDraft.apply`) inside one call scope, and what it hands in."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from chickadee.edits.draft import CandidateDraft, DraftMode, DraftRefused, emptied
from chickadee.edits.formats.base import Create, FileEdits, Targets, Whole
from chickadee.edits.lines import Decoded, decode
from chickadee.edits.match import Region

Mode = Literal["edit", "create", "delete"]
E: Mode = "edit"
C: Mode = "create"
D: Mode = "delete"

FILES = {
    "app.py": "def go() -> int:\n    return 1\n",
    "old.py": "x = 1\n",
    "crlf.py": "a = 1\r\nb = 2\r\n",
    "own.py": "# ... existing code ...\nx = 1\n",
    "bin.dat": None,
}


def _targets(*targets: tuple[str, Mode]) -> Targets:
    """The paths a draft is given: the writable ones in order, the deletes, and
    which are new files."""
    return Targets(
        writable=tuple(p for p, m in targets if m != D),
        deletes=tuple(p for p, m in targets if m == D),
        creates=frozenset(p for p, m in targets if m == C),
    )


def _read(path: str) -> Decoded | None:
    text = FILES.get(path)
    return None if text is None else decode(text)


def _denied(why: str) -> DraftRefused:
    """What a scope rule answers: its reason, and what to do instead."""
    return DraftRefused("DENIED", why, do="leave it")


def _draft(*targets: tuple[str, Mode], refused: str | None = None) -> CandidateDraft:
    return CandidateDraft(
        _targets(*(targets or (("app.py", E),))),
        _read,
        lambda path: _denied(path) if refused is not None and path == refused else None,
    )


def _whole(draft: CandidateDraft, path: str, content: str) -> DraftRefused | str:
    with draft.call() as call:
        refused = call.apply(path, None, content)
        return refused if refused is not None else call.answer()


# -- made over the targets as rollback left them ---------------------------------------------


def test_an_edit_target_holds_its_text_a_create_target_is_absent() -> None:
    draft = _draft(("app.py", E), ("new.py", C))
    assert draft.text("app.py") == FILES["app.py"]
    assert draft.text("new.py") is None and draft.absent("new.py")
    assert not draft.absent("app.py")
    assert draft.handed() == FileEdits()


def test_a_crlf_target_is_held_as_lf_and_handed_in_as_crlf() -> None:
    draft = _draft(("crlf.py", E))
    assert draft.text("crlf.py") == "a = 1\nb = 2\n"
    with draft.call() as call:
        assert call.apply("crlf.py", Region(4, 5), "9") is None
        call.answer()
    assert draft.handed() == FileEdits((("crlf.py", Whole("a = 9\r\nb = 2\r\n")),))


# -- the one write: scope, existence, elision -------------------------------------------------


def test_a_path_the_run_may_not_write_is_denied_at_once() -> None:
    draft = _draft(("app.py", E), ("old.py", D), ("bin.dat", E), refused="app.py")
    for path, why in (
        ("old.py", "delete target"),
        ("other.py", "not a TARGETS path"),
        ("app.py", None),
        ("bin.dat", "not UTF-8 text"),
    ):
        refused = _whole(draft, path, "y = 2\n")
        assert isinstance(refused, DraftRefused) and refused.failure == "DENIED", path
        assert why is None or why in refused.reason, (path, refused.reason)


def test_an_absent_create_target_is_written_only_whole() -> None:
    draft = _draft(("new.py", C))
    with draft.call() as call:
        refused = call.apply("new.py", Region(0, 0), "x = 1\n")
        assert isinstance(refused, DraftRefused) and refused.failure == "FILE_ABSENT"
        text = call.text("new.py")
        assert isinstance(text, DraftRefused) and text.failure == "FILE_ABSENT"
    assert _whole(draft, "new.py", "x = 1\n") == "ok: new.py written, 1 lines"
    assert draft.handed() == FileEdits((("new.py", Create("x = 1\n")),))


def test_a_placeholder_for_code_is_refused_naming_its_line() -> None:
    draft = _draft()
    refused = _whole(draft, "app.py", "def go() -> int:\n    # ... existing code ...\n")
    assert isinstance(refused, DraftRefused) and refused.failure == "ELIDED"
    assert "line 2" in refused.reason
    assert draft.text("app.py") == FILES["app.py"]


def test_the_files_own_placeholder_is_its_own() -> None:
    draft = _draft(("own.py", E))
    assert _whole(draft, "own.py", "# ... existing code ...\nx = 2\n") == (
        "ok: own.py written, 2 lines"
    )


# -- one call: every change or none ------------------------------------------------------------


def test_a_call_that_ends_without_its_answer_changes_nothing() -> None:
    draft = _draft(("app.py", E), ("old.py", E))
    with draft.call() as call:
        assert call.apply("app.py", None, "x = 2\n") is None
        assert call.apply("old.py", None, "# ... rest of the file ...\n") is not None
    assert draft.text("app.py") == FILES["app.py"]
    assert draft.handed() == FileEdits()


def test_a_later_edit_in_a_call_sees_the_earlier_and_the_answer_shows_each() -> None:
    draft = _draft()
    with draft.call() as call:
        assert call.apply("app.py", Region(28, 29), "2") is None
        now = call.text("app.py")
        assert now == "def go() -> int:\n    return 2\n"
        assert call.apply("app.py", Region(4, 6), "run") is None
        answer = call.answer()
    assert answer == "ok: app.py\n    return 2\nok: app.py\ndef run() -> int:"
    assert draft.handed() == FileEdits(
        (("app.py", Whole("def run() -> int:\n    return 2\n")),)
    )


def test_an_edit_that_restores_the_text_hands_nothing_in() -> None:
    draft = _draft()
    _whole(draft, "app.py", "x = 1\n")
    _whole(draft, "app.py", FILES["app.py"] or "")
    assert draft.handed() == FileEdits()


# -- the review's pins ---------------------------------------------------------------


def test_a_crlf_write_and_anchor_are_read_as_the_file_is_held() -> None:
    draft = _draft(("crlf.py", E))
    assert draft.crlf("crlf.py") and not _draft().crlf("app.py")
    assert _whole(draft, "crlf.py", "a = 3\r\nb = 4\r\n") == "ok: crlf.py written, 2 lines"
    assert draft.handed() == FileEdits((("crlf.py", Whole("a = 3\r\nb = 4\r\n")),))


def test_each_change_is_answered_where_it_now_is() -> None:
    draft = _draft()
    with draft.call() as call:
        assert call.apply("app.py", Region(4, 6), "run") is None  # def go -> def run
        assert call.apply("app.py", Region(0, 0), "# top\n") is None  # shifts the first
        answer = call.answer()
    assert answer == "ok: app.py\ndef run() -> int:\nok: app.py\n# top"


def test_removing_the_last_line_says_nothing_remains() -> None:
    draft = _draft(("old.py", E))
    with draft.call() as call:
        assert call.apply("old.py", Region(0, 6), "") is None
        assert call.answer() == "ok: old.py\n(nothing remains there)"


# -- an open scope: files taken on as they are first written --------------------------


def _open(path: str) -> DraftMode | DraftRefused:
    return "create" if path.startswith("tests/") else _denied("only a test file")


def test_an_open_scope_takes_on_a_file_and_refuses_another_with_its_reason() -> None:
    draft = CandidateDraft(None, _read, lambda _p: None, opens=_open)
    assert _whole(draft, "tests/test_a.py", "x = 1\n") == "ok: tests/test_a.py written, 1 lines"
    refused = _whole(draft, "app.py", "x = 1\n")
    assert isinstance(refused, DraftRefused) and refused.reason == "only a test file"


def test_a_file_taken_on_by_a_call_that_is_undone_is_still_writable() -> None:
    """Review finding: the call's undo restored the texts from before the file was taken on,
    so the file stayed taken on with no text, and the next edit of it raised `KeyError`."""
    draft = CandidateDraft(None, _read, lambda _p: None, opens=_open)
    with draft.call() as call:
        assert call.apply("tests/test_a.py", None, "x = 1\n") is None
        # Ended without its answer: undone, as a refused call is.
    assert draft.text("tests/test_a.py") is None
    assert _whole(draft, "tests/test_a.py", "x = 2\n") == "ok: tests/test_a.py written, 1 lines"
    assert draft.text("tests/test_a.py") == "x = 2\n"


def test_emptied_is_empty_or_whitespace_only() -> None:
    assert emptied("") and emptied(" \n\t\n")
    assert not emptied("x\n")


def test_refuse_emptied_keeps_a_file_from_being_blanked_but_writes_a_new_file_empty() -> None:
    """The box's edit tool refuses an edit that leaves a non-empty file empty (the Applier's
    `empty_defense` rule, which no gate follows there); a new file may be empty."""
    draft = CandidateDraft(
        _targets(("app.py", E), ("pkg/__init__.py", C)),
        _read,
        lambda _p: None,
        refuse_emptied=True,
    )
    refused = _whole(draft, "app.py", "  \n")
    assert isinstance(refused, DraftRefused) and refused.failure == "DENIED"
    assert "delete it" in refused.do
    with draft.call() as call:
        whole = call.text("app.py")
        assert isinstance(whole, str)
        refused_span = call.apply("app.py", Region(0, len(whole)), "")
    assert isinstance(refused_span, DraftRefused)
    assert _whole(draft, "pkg/__init__.py", "") == "ok: pkg/__init__.py written, 0 lines"


def test_without_refuse_emptied_the_draft_leaves_empty_content_to_the_applier() -> None:
    draft = _draft(("app.py", E))
    assert _whole(draft, "app.py", "\n") == "ok: app.py written, 1 lines"


# -- delete and move ---------------------------------------------------


def _opened(
    removable: Callable[[str], DraftRefused | None] | None,
    answered: list[dict[str, bytes | None]],
) -> CandidateDraft:
    """A draft over nothing, taking on any path as the box's port does: an edit for a file in
    `FILES`, a create otherwise."""

    def opens(path: str) -> str:
        return "edit" if FILES.get(path) is not None else "create"

    return CandidateDraft(
        None,
        _read,
        lambda _p: None,
        opens=opens,  # type: ignore[arg-type]
        removable=removable,
        on_answered=lambda files: answered.append(dict(files)),
    )


def test_a_draft_given_no_removable_refuses_a_delete_and_a_move() -> None:
    draft = _draft(("app.py", E))
    with draft.call() as call:
        refused = call.delete("app.py")
        assert isinstance(refused, DraftRefused) and refused.failure == "DENIED"
        moved = call.move("app.py", "b.py")
        assert isinstance(moved, DraftRefused) and moved.failure == "DENIED"
    assert draft.text("app.py") == FILES["app.py"]


def test_a_delete_is_answered_and_told_as_none() -> None:
    answered: list[dict[str, bytes | None]] = []
    draft = _opened(lambda _p: None, answered)
    with draft.call() as call:
        assert call.delete("old.py") is None
        assert call.answer() == "ok: old.py deleted"
    assert answered == [{"old.py": None}]


def test_removable_says_why_a_file_may_not_go() -> None:
    draft = _opened(lambda p: _denied(f"{p} is ignored, so undo could not bring it back"), [])
    with draft.call() as call:
        refused = call.delete("old.py")
    assert isinstance(refused, DraftRefused) and "undo could not" in refused.reason


def test_a_move_carries_the_text_and_is_one_answer() -> None:
    answered: list[dict[str, bytes | None]] = []
    draft = _opened(lambda _p: None, answered)
    with draft.call() as call:
        assert call.move("old.py", "pkg/new.py") is None
        assert call.answer() == "ok: old.py moved to pkg/new.py"
    assert answered == [{"old.py": None, "pkg/new.py": b"x = 1\n"}]


def test_a_move_never_replaces_a_file_and_a_refused_call_undoes_the_delete() -> None:
    answered: list[dict[str, bytes | None]] = []
    draft = _opened(lambda _p: None, answered)
    with draft.call() as call:
        refused = call.move("old.py", "app.py")
    assert isinstance(refused, DraftRefused) and "never replaces" in refused.reason
    assert answered == []
    with draft.call() as call:
        # Ended unanswered, as a later refusal in the call ends it: nothing it did stands.
        assert call.move("old.py", "new.py") is None
    assert answered == []
    assert draft.text("old.py") == "x = 1\n"


def test_deleting_a_missing_file_says_so() -> None:
    draft = _opened(lambda _p: None, [])
    with draft.call() as call:
        refused = call.delete("nope.py")
    assert isinstance(refused, DraftRefused) and refused.failure == "FILE_ABSENT"
