"""The Agent Skills standard's fields and Claude Code's agent fields. A skill
written here is a valid standard skill; a field that means nothing where it is written refuses
one of ours and is ignored, with a note, in another tool's."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.errors import ConfigError
from chickadee.skills.frontmatter import (
    COMPATIBILITY_MAX,
    DESCRIPTION_MAX,
    NAME_MAX,
    Parsed,
    parse,
)


def text(name: str = "survey", front: str = "", description: str = "d") -> str:
    return f"---\nname: {name}\ndescription: {description}\n{front}---\nbody\n"


def ours(front: str = "", *, name: str = "survey", description: str = "d") -> Parsed:
    return parse(Path(f"{name}.md"), text(name, front, description), external=False)


def test_the_standard_fields_are_read() -> None:
    front = (
        "license: Apache-2.0\ncompatibility: needs git\nmetadata:\n  author: me\n"
        '  version: "1.0"\n  chickadee-report: Risks, Open questions\n'
        "  chickadee-keywords: grep, search\n"
    )
    parsed = parse(Path("survey.md"), text(front=front), external=False)
    got = parsed.front
    assert (got.license, got.compatibility) == ("Apache-2.0", "needs git")
    assert got.meta("version") == "1.0" and got.meta("author") == "me"
    assert got.report_sections == ("Risks", "Open questions")
    assert got.keywords == ("grep", "search")
    assert not got.is_agent and parsed.ignored == ()


def test_an_agent_skill_reads_its_profile_and_narrowing() -> None:
    front = "context: fork\nagent: reviewer\nallowed-tools: read grep Bash(git add:*)\n"
    got = parse(Path("survey.md"), text(front=front), external=False).front
    assert got.is_agent and got.agent == "reviewer"
    assert got.allowed_tools == ("read", "grep", "Bash(git add:*)")


def test_a_comma_list_of_tools_is_read_too() -> None:
    got = parse(
        Path("survey.md"),
        text(front="context: fork\nallowed-tools: Read, Grep\n"),
        external=False,
    ).front
    assert got.allowed_tools == ("Read", "Grep")


@pytest.mark.parametrize(
    ("front", "says"),
    [
        ("agent: reader\n", "only `context: fork` uses"),
        ("allowed-tools: read\n", "never pre-approves"),
        ("context: inline\n", "the only context is `fork`"),
        ("metadata:\n  tags:\n    - a\n", "a map of strings, one level deep"),
        ("metadata:\n  chickadee-reprot: x\n", "`chickadee-reprot` is not ours"),
        ("model: opus\n", "SK-15"),
    ],
    ids=["agent-on-knowledge", "tools-on-knowledge", "bad-context", "nested", "typo", "model"],
)
def test_ours_refuses_what_means_nothing_where_it_is(front: str, says: str) -> None:
    with pytest.raises(ConfigError, match=says.replace("(", r"\(")):
        ours(front)


@pytest.mark.parametrize(
    ("front", "ignored"),
    [
        ("agent: Explore\n", ("agent",)),
        ("allowed-tools: Read\n", ("allowed-tools",)),
        ("context: inline\n", ("context",)),
        ("metadata:\n  tags:\n    - a\n", ("metadata",)),
        ("model: haiku\nhooks: x\n", ("model", "hooks")),
    ],
    ids=["agent-on-knowledge", "tools-on-knowledge", "bad-context", "nested", "unknown"],
)
def test_theirs_ignores_it_with_a_note(front: str, ignored: tuple[str, ...]) -> None:
    parsed = parse(Path("survey") / "SKILL.md", text(front=front), external=True)
    assert parsed.ignored == ignored


@pytest.mark.parametrize(
    "name",
    ["-lead", "trail-", "dou--ble", "x" * (NAME_MAX + 1)],
    ids=["lead", "trail", "double", "long"],
)
def test_the_standard_name_rule(name: str) -> None:
    with pytest.raises(ConfigError, match="single hyphens"):
        parse(Path(f"{name}.md"), text(name), external=False)


def test_a_name_of_the_longest_length_loads() -> None:
    name = "x" * NAME_MAX
    assert parse(Path(f"{name}.md"), text(name), external=False).front.name == name


@pytest.mark.parametrize(
    "path", [Path("other.md"), Path("other") / "SKILL.md"], ids=["file", "folder"]
)
def test_ours_must_sit_in_a_file_or_folder_of_its_name(path: Path) -> None:
    with pytest.raises(ConfigError, match="same name"):
        parse(path, text("survey"), external=False)


def test_theirs_is_read_as_their_tool_left_it() -> None:
    """Another tool's file is not held to the standard's folder rule or bounds: one broken file
    of theirs must not stop a session."""
    long = "w" * (DESCRIPTION_MAX + 10)
    parsed = parse(Path("folder") / "SKILL.md", text("survey", description=long), external=True)
    assert parsed.front.name == "survey"


def test_ours_is_held_to_the_standard_bounds() -> None:
    with pytest.raises(ConfigError, match=f"at most {DESCRIPTION_MAX}"):
        ours(description="w" * (DESCRIPTION_MAX + 1))
    with pytest.raises(ConfigError, match=f"at most {COMPATIBILITY_MAX}"):
        ours(f"compatibility: {'c' * (COMPATIBILITY_MAX + 1)}\n")
