"""`edits/elision.find_elision`: a placeholder comment that stands for omitted
code is an elision only when it is new. Nothing calls it until `T-3.128` and `T-3.131`."""

from __future__ import annotations

from chickadee.edits.elision import Elision, find_elision, is_placeholder

OLD = "def go():\n    return 1\n\n\ndef stop():\n    return 0\n"


def test_a_new_placeholder_is_an_elision_naming_its_line() -> None:
    new = "def go():\n    return 2\n\n# ... rest of the file unchanged ...\n"
    assert find_elision(new, OLD) == Elision(4, "# ... rest of the file unchanged ...")


def test_each_spelling_in_the_one_list_is_caught() -> None:
    for line in (
        "# ... existing code ...",
        "# ...existing code...",
        "// ... existing code ...",
        "/* rest of the code unchanged */",
        "<!-- ... other methods ... -->",
        "    # (remaining code omitted)",
        "# … unchanged code",
        "-- code omitted for brevity",
        "# rest of the file unchanged",
        "# rest of the class is the same",
        "# existing imports here",
    ):
        assert is_placeholder(line), line
        assert find_elision(f"x = 1\n{line}\n", "x = 0\n") is not None, line


def test_a_comment_about_code_is_not_a_placeholder() -> None:
    """Review: the first rule matched a phrase at a comment's start, so each of these
    was an elision, and under `edit_check`'s refusal a correct edit would have been refused.
    The whole line must be the placeholder, and say so with an ellipsis or a state."""
    for line in (
        "# other functions in this module call it",
        "# existing code paths call this first",
        "# rest of the function handles retries",
        "# same as before, but with retries",
        "// other methods are generated",
        "* rest of the code is in helpers",
        "# the rest of the logic lives in helpers.py",
        "# existing code",
        "# rest of the file",
    ):
        assert not is_placeholder(line), line


def test_a_placeholder_the_file_already_holds_is_its_own_comment() -> None:
    old = "# ... existing code ...\nx = 1\n"
    assert find_elision("# ... existing code ...\nx = 2\n", old) is None


def test_ordinary_code_and_comments_are_not_elisions() -> None:
    for new in (
        "def go():\n    ...\n",  # an Ellipsis body is code
        "# the rest of the logic lives in helpers.py\nx = 1\n",
        "x = 1  # existing behaviour kept\n",
        "",
    ):
        assert find_elision(new, OLD) is None, new
