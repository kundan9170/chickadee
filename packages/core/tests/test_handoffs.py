"""The hand-off templates: the repository's
template first, then the one `[templates]` names, then the shipped one; each recorded by sha;
and the shipped three carry D196 the headings."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chickadee.config import TemplatesPolicy
from chickadee.handoffs import HANDOFFS, REPO_TEMPLATES, SHIPPED, Handoff, resolve
from chickadee.schema.ids import sha256_hex


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", HANDOFFS)
def test_the_shipped_template_when_nothing_overrides_it(tmp_path: Path, kind: Handoff) -> None:
    got = resolve(tmp_path, {}, kind)
    shipped = (SHIPPED / f"{kind}.md").read_text(encoding="utf-8")
    assert (got.kind, got.text, got.sha, got.source, got.warning) == (
        kind,
        shipped,
        sha256_hex(shipped),
        "shipped",
        None,
    )


def test_the_repositorys_template_first_then_the_configured_one(tmp_path: Path) -> None:
    named = _write(tmp_path / "eval" / "plan-b.md", "# configured plan\n")
    got = resolve(tmp_path, {"plan": named}, "plan")
    assert (got.text, got.source, got.sha) == (
        "# configured plan\n",
        "config",
        sha256_hex(got.text),
    )
    _write(tmp_path / REPO_TEMPLATES / "plan.md", "# the repository's plan\n")
    got = resolve(tmp_path, {"plan": named}, "plan")
    assert (got.text, got.source) == ("# the repository's plan\n", "repository")


def test_a_relative_configured_path_is_read_from_the_repository(tmp_path: Path) -> None:
    _write(tmp_path / "templates" / "brief.md", "## Goal\n")
    got = resolve(tmp_path, {"brief": Path("templates/brief.md")}, "brief")
    assert (got.text, got.source) == ("## Goal\n", "config")


def test_a_missing_configured_template_falls_back_with_a_warning(tmp_path: Path) -> None:
    gone = tmp_path / "gone.md"
    got = resolve(tmp_path, {"design": gone}, "design")
    assert got.source == "shipped"
    assert got.warning == f"{gone} does not exist; falling back to the shipped default"


def test_an_unreadable_repository_template_falls_back_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / REPO_TEMPLATES / "brief.md").mkdir(parents=True)
    got = resolve(tmp_path, {}, "brief")
    assert got.source == "shipped"
    assert got.text == (SHIPPED / "brief.md").read_text(encoding="utf-8")
    assert got.warning is not None and "unreadable" in got.warning


def test_templates_config_names_each_kind_with_home_expanded() -> None:
    policy = TemplatesPolicy(plan="~/plans/b.md", brief="briefs/short.md")
    configured = policy.configured()
    assert set(configured) == {"plan", "brief"}
    assert configured["plan"] == Path("~/plans/b.md").expanduser()
    assert configured["brief"] == Path("briefs/short.md")


HEADINGS: dict[Handoff, tuple[str, ...]] = {
    # D204 Amendment 1's table: the design's eight sections (two inserted by
    # the harness, `Goal` and `Files` new), the plan's `Tasks` and `Final check` as sections
    # (were a numbered list and a line), the brief's six as `##` under its `# T<n>:` title
    # (were `###` under `## Task <n> of <total>`).
    "design": (
        "## Request",
        "## Goal",
        "## What changes",
        "## What must not change",
        "## Behaviour",
        "## Outside the repository",
        "## Constraints from our conversation",
        "## Files",
    ),
    "plan": ("# Plan:", "## Tasks", "1. [T1: ", "## Final check"),
    "brief": (
        "# T<n>:",
        "## Goal",
        "## Where to look",
        "## Approach",
        "## Don't",
        "## Done when",
        "## Report",
    ),
}


@pytest.mark.parametrize("kind", HANDOFFS)
def test_the_shipped_templates_follow_the_headings_in_order(kind: Handoff) -> None:
    text = (SHIPPED / f"{kind}.md").read_text(encoding="utf-8")
    missing = [h for h in HEADINGS[kind] if h not in text]
    assert not missing, f"{kind}: missing {missing}"
    at = [text.find(h) for h in HEADINGS[kind]]
    assert at == sorted(at)
    assert not re.search(r"\{\w+\}", text), "a template is prose, never a format string"
