"""Claude Code's and OpenCode's agent files are agent skills. Their
tools narrow ours and never grant one; a file asking for a tool that writes is listed and not
runnable; a primary agent is never a subagent."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, load, new_session, write_skill

from chickadee.skills.discover import discover
from chickadee.skills.index import render_index
from chickadee.skills.load import Skills


def claude(base: Path, name: str, front: str = "", body: str = "Do the job.\n") -> Path:
    path = base / ".claude" / "agents" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: the {name} agent\n{front}---\n{body}")
    return path


def opencode(root: Path, name: str, front: str = "") -> Path:
    path = root / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ndescription: the {name} agent\n{front}---\nDo the job.\n")
    return path


def test_a_shell_makes_it_a_reviewer(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "checker", "tools: Read, Bash(git diff:*)\n")
    (skill,) = discover(tree.roots()).skills
    assert (skill.profile, skill.tools, skill.runnable) == ("reviewer", ("read", "shell"), True)


def test_a_writing_tool_is_listed_and_not_runnable(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "fixer", "tools: Read, Edit, Write\n")
    catalog = discover(tree.roots())
    (skill,) = catalog.skills
    assert not skill.runnable
    assert skill.note == "not runnable: it asks for Edit, Write, and no agent here writes"
    assert "not runnable" in render_index(catalog)


def test_no_tools_named_runs_read_only(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "helper")
    (skill,) = discover(tree.roots()).skills
    assert (skill.profile, skill.tools, skill.runnable) == ("reader", None, True)
    assert skill.note == "named no tools, so runs read-only as reader"


def test_a_tool_with_no_counterpart_is_named(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "searcher", "tools: Read, WebFetch\n")
    (skill,) = discover(tree.roots()).skills
    assert skill.tools == ("read",) and skill.note == "no tool here for WebFetch"


def test_model_and_other_fields_are_ignored_with_a_note(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "styled", "tools: Read\nmodel: haiku\ncolor: blue\n")
    catalog = discover(tree.roots())
    (skill,) = catalog.skills
    assert skill.ignored == ("model", "color")
    assert "(ignored from another tool's format: model, color)" in render_index(catalog)


def test_an_opencode_agent_is_named_by_its_file(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    off = "tools:\n  write: false\n  edit: false\n  patch: false\n  multiedit: false\n"
    opencode(tree.repo / ".opencode" / "agent", "auditor", off + "  bash: false\n")
    (skill,) = discover(tree.roots()).skills
    assert (skill.name, skill.profile, skill.runnable) == ("auditor", "reader", True)
    assert skill.form == "opencode-agent"


def test_an_opencode_agent_that_leaves_writing_on_is_not_runnable(tmp_path: Path) -> None:
    """OpenCode's map switches tools off; every tool it does not name stays on."""
    tree = Tree(tmp_path)
    opencode(tree.repo / ".opencode" / "agents", "builder", "tools:\n  bash: false\n")
    (skill,) = discover(tree.roots()).skills
    assert not skill.runnable and skill.note is not None
    assert skill.note.startswith("not runnable: it asks for edit")


def test_a_primary_agent_is_never_a_subagent(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    opencode(tree.repo / ".opencode" / "agent", "main", "mode: primary\n")
    catalog = discover(tree.roots())
    assert catalog.skills == () and catalog.skipped == ()


def test_the_user_roots_are_read(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.home, "mine")
    opencode(tree.home / ".config" / "opencode" / "agent", "theirs")
    got = {(s.name, s.origin) for s in discover(tree.roots()).skills}
    assert got == {("mine", "user"), ("theirs", "user")}


def test_a_built_in_name_or_a_bad_name_is_skipped_not_refused(tmp_path: Path) -> None:
    """An agent called `plan` is common in Claude Code, and never `/plan` here."""
    tree = Tree(tmp_path)
    claude(tree.repo, "plan")
    claude(tree.repo, "Bad_Name")
    catalog = discover(tree.roots())
    assert catalog.skills == ()
    assert sorted(s.why for s in catalog.skipped)[0].startswith("agent name 'Bad_Name'")
    assert any("`/plan`" in s.why for s in catalog.skipped)


def test_external_false_turns_them_off(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "helper")
    assert discover(tree.roots(external=False)).skills == ()


def test_an_agent_file_loads_as_knowledge_saying_spawn_agent_runs_it(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    claude(tree.repo, "helper", body="Look carefully.\n")
    session = new_session(tmp_path)
    try:
        text = load(session, Skills(discover(tree.roots()), session), "helper", "k1")
    finally:
        session.close()
    assert "an agent skill: `spawn_agent` runs it as reader" in text
    assert "Look carefully." in text


def test_claude_code_agent_types_map_to_our_profiles(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    fork = "context: fork\nagent: {}\n"
    write_skill(tree.external(), "explorer", front=fork.format("Explore"), bundle=True)
    write_skill(tree.external(), "anything", front=fork.format("general-purpose"), bundle=True)
    write_skill(tree.project, "own", front="context: fork\nallowed-tools: read grep_in\n")
    by = {s.name: s for s in discover(tree.roots()).skills}
    assert (by["explorer"].profile, by["explorer"].note) == ("reader", "Explore runs as reader")
    assert by["anything"].note == "runs read-only as reader"
    assert (by["own"].profile, by["own"].tools) == ("reader", ("read", "grep_in"))


def test_an_external_skill_asking_for_a_writer_is_not_runnable(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    front = "context: fork\nallowed-tools: Read Edit\n"
    write_skill(tree.external(), "patcher", front=front, bundle=True)
    (skill,) = discover(tree.roots()).skills
    assert not skill.runnable and skill.tools == ("read",)


def test_a_tools_list_in_any_form_is_read_by_the_one_rule(tmp_path: Path) -> None:
    """A YAML list and a pattern with a space in it read as `allowed-tools` does
    (`frontmatter.tool_names`), never as stray words (the review of #640)."""
    tree = Tree(tmp_path)
    claude(tree.repo, "lister", "tools:\n  - Read\n  - Grep\n")
    claude(tree.repo, "logger", "tools: Bash(git log:*), Read\n")
    lister, logger = sorted(discover(tree.roots()).skills, key=lambda s: s.name)
    assert (lister.tools, lister.note) == (("read", "grep"), None)
    assert (logger.profile, logger.tools, logger.note) == ("reviewer", ("shell", "read"), None)


def test_a_skill_and_an_agent_file_naming_a_shell_both_run_as_reviewer(tmp_path: Path) -> None:
    """Moved (the review of #640): a `context: fork` skill with no `agent:` whose tools run
    commands ran as `reader`, its shell dropped; an agent file naming the same ran as
    `reviewer`. One rule now (`agent_files.runs_as`): both are reviewers."""
    tree = Tree(tmp_path)
    write_skill(tree.project, "ours", front="context: fork\nallowed-tools: read shell\n")
    write_skill(
        tree.external(),
        "theirs",
        front="context: fork\nallowed-tools: Read, Bash\n",
        bundle=True,
    )
    claude(tree.repo, "filed", "tools: Read, Bash\n")
    found = {s.name: s for s in discover(tree.roots()).skills}
    assert {n: (s.profile, s.tools) for n, s in found.items()} == {
        "ours": ("reviewer", ("read", "shell")),
        "theirs": ("reviewer", ("read", "shell")),
        "filed": ("reviewer", ("read", "shell")),
    }
    write_skill(tree.project, "looker", front="context: fork\nallowed-tools: read\n")
    (looker,) = [s for s in discover(tree.roots()).skills if s.name == "looker"]
    assert looker.profile == "reader"


def test_a_skill_and_an_agent_file_written_alike_carry_one_note(tmp_path: Path) -> None:
    """Moved (the second review of #640): a `SKILL.md` and an agent file had their own rules,
    and their notes differed for the same tools. One rule now (`agent_files.runs_as`). A note
    says only what the index row does not: the row already names the profile, so a profile
    its tools decide is no longer noted (`ours` and `looker` above had "runs as reviewer, since
    its tools run commands" and "runs read-only as reader"), and a `SKILL.md` naming no tools
    is noted as an agent file naming none is (it was "runs read-only as reader")."""
    tree = Tree(tmp_path)
    write_skill(tree.project, "bare", front="context: fork\n")
    claude(tree.repo, "filed")
    write_skill(tree.project, "web", front="context: fork\nallowed-tools: read WebFetch\n")
    claude(tree.repo, "fetcher", "tools: Read, WebFetch\n")
    found = {s.name: s.note for s in discover(tree.roots()).skills}
    assert found["bare"] == found["filed"] == "named no tools, so runs read-only as reader"
    assert found["web"] == found["fetcher"] == "no tool here for WebFetch"
