"""SK-15: `model` in one of our skills refuses that skill by name, and so does `allowed-tools`
on a knowledge skill, where it could only be read as a pre-approval. Neither
is ignored silently: an author who believed it did something must be told. Any other field
outside the set refuses too."""

from __future__ import annotations

from pathlib import Path

import pytest
from skills_helpers import Tree, write_skill

from chickadee.errors import ConfigError
from chickadee.skills.discover import discover
from chickadee.skills.frontmatter import parse


@pytest.mark.parametrize(
    ("front", "named"),
    [
        ("allowed-tools: Bash\n", "`allowed-tools`"),
        ("model: opus\n", "`model`"),
        ("homepage: x\n", "`homepage`"),
    ],
)
def test_a_field_outside_the_closed_set_refuses_our_skill_by_name(
    tmp_path: Path, front: str, named: str
) -> None:
    tree = Tree(tmp_path)
    path = write_skill(tree.project, "deploy", front=front)
    with pytest.raises(ConfigError) as refused:
        discover(tree.roots())
    assert str(path) in str(refused.value) and named in str(refused.value)
    if named != "`homepage`":
        assert "SK-15" in str(refused.value)


def test_the_closed_set_parses_with_folded_and_quoted_values() -> None:
    text = (
        "---\nname: migrations\ndescription: >\n  How this project writes\n  migrations.\n"
        'argument-hint: "<table>"\n---\n# Body\n'
    )
    parsed = parse(Path("migrations.md"), text, external=False)
    assert parsed.front.description == "How this project writes migrations."
    assert parsed.front.argument_hint == "<table>"
    assert parsed.body == "# Body\n"


@pytest.mark.parametrize(
    "text",
    [
        "# no front matter\n",
        "---\nname: x\n",
        "---\ndescription: d\n---\n",
        "---\nname: X Y\ndescription: d\n---\n",
        "---\nname: x\n---\n",
        "---\nname: x\nname: y\ndescription: d\n---\n",
    ],
    ids=["missing", "unclosed", "no-name", "bad-name", "no-description", "twice"],
)
def test_a_malformed_block_refuses_naming_the_file(text: str) -> None:
    with pytest.raises(ConfigError, match=r"bad\.md"):
        parse(Path("bad.md"), text, external=False)
