"""An amended design keeps the Request it was approved with: read whole, a code block
quoted in it included, and the person's new turn added after it."""

from __future__ import annotations

from chickadee.zeus.delegate import amended_request, request_section

DESIGN = """# Login

## Request
Add a login page. Like this:

```python
# the handler
def login(): ...
```
And keep it short.

## Goal
A page.
"""


def test_the_request_is_read_whole_through_a_quoted_code_block() -> None:
    request = request_section(DESIGN)
    assert request is not None
    assert request.startswith("Add a login page.")
    assert request.endswith("And keep it short."), "a `#` line in a code block is not a heading"
    assert "## Goal" not in request


def test_a_design_with_no_request_has_none() -> None:
    assert request_section("# Login\n\n## Goal\nA page.\n") is None


def test_the_new_turn_follows_what_was_asked_first() -> None:
    assert amended_request("Add a page.", "Also a logout.") == (
        "Add a page.\n\nThen:\nAlso a logout."
    )
