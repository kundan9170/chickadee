"""`SURFACE` is the one list: sixteen entries in five families;
every entry has a template and every template an entry; every `failures` set is a subset of the
vocabulary and carries `INVALID_ARGUMENTS`; the parallel classes; the three doc tables are
renderings of `SURFACE` — the names in every one, and per tool the family, parallel class,
mutation, tier and model calls in `33-tools.md` §3 and the parallel class in `04-zeus.md`
§4.1; and `tools` itself."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from chickadee.client.request import ToolSchema
from chickadee.errors import HarnessBug
from chickadee.roles import role
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.spec import (
    DYNAMIC_STUB_MAX,
    FAMILIES,
    FIND_SKILL_RESULTS,
    LOCATE_MAX_CALLS,
    RESULT_LINES_DEFAULT,
    SURFACE,
    parallel_class_of,
    render_table,
)
from chickadee.tools.validate import (
    InvalidArguments,
    ToolBrief,
    ToolBriefs,
    resident,
    stub,
    tools,
)

ROOT = Path(__file__).resolve().parents[4]
TEMPLATES = (
    ROOT
    / "packages"
    / "core"
    / "src"
    / "chickadee"
    / "adapters"
    / "frontier"
    / "templates"
    / "tools"
)


def _descriptions() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in TEMPLATES.glob("*.md")}


# --- the list --------------------------------------------------------------------------------


def test_twenty_five_entries_in_five_families_in_order() -> None:
    # `grep` and `glob` where `find` was; D204: `doc`, his design's documents; D201:
    # `job`, the jobs a shell command became; D213: `spawn_agent`, `follow_up` and
    # `find_skill`; D230: `worktree`, his moves between the checkout and a worktree; D240:
    # `pause`, the team's run paused while it works in the background, and `tell`, his
    # message to it.
    assert len(SURFACE) == 25
    assert [s.name for s in SURFACE] == [
        "atlas", "grep", "glob", "symbol", "neighbors", "impact", "recall", "locate",
        "spawn_agent", "follow_up", "tools",
        "read", "grep_in", "load_skill", "find_skill",
        "shell", "job", "coder", "doc", "pause", "tell", "worktree",
        "ask_user",
        "note_concern", "note_rejected",
    ]  # fmt: skip
    assert FAMILIES == ("navigate", "read", "act", "talk", "record", "edit", "task")
    # A-E are Zeus's; F, `edit`, is the worker's alone: Zeus edits through `coder`;
    # G, `task`, is Athena's.
    assert {s.family for s in SURFACE} == set(FAMILIES[:5])
    # In family order, as §3 lists them.
    order = [FAMILIES.index(s.family) for s in SURFACE]
    assert order == sorted(order)


def test_every_entry_has_a_template_and_every_template_an_entry() -> None:
    files = {p.name for p in TEMPLATES.glob("*.md")}
    named = {s.description for s in SURFACE}
    assert named == files, (
        f"specs without templates: {named - files}; templates without specs: {files - named}"
    )
    for s in SURFACE:
        assert s.description == f"{s.name}.md"


def test_every_template_first_line_is_a_stub_naming_its_argument() -> None:
    """The first line is what a dynamic tool would be guessed from: what it does and its one
    required argument (or that it has none)."""
    for s in SURFACE:
        text = (TEMPLATES / s.description).read_text(encoding="utf-8")
        first = text.splitlines()[0]
        assert first and not first.startswith("#"), s.name
        required = s.schema.get("required")
        if isinstance(required, list) and required:
            names = [str(r) for r in required]  # pyright: ignore[reportUnknownVariableType]
            assert any(f"`{r}`" in first for r in names), (s.name, first)
        else:
            # A tool whose arguments are a choice (`coder`: a design, or a title and brief)
            # names the choice, in a sentence of its own or within one.
            assert "No arguments" in first or "exactly one of" in first.lower(), (s.name, first)


def test_parallel_classes() -> None:
    # `worktree` holds the repository as `coder` does, when it opens or lands one.
    assert [s.name for s in SURFACE if s.parallel == "exclusive"] == [
        "coder",
        "pause",
        "tell",
        "worktree",
    ]
    assert [s.name for s in SURFACE if s.parallel == "capped"] == [
        "locate",
        "spawn_agent",
        "follow_up",
    ]
    # Agents run at once, each on a thread of its own, and one
    # agent is followed up once a response (decision 2).
    assert [s.name for s in SURFACE if s.threads] == ["spawn_agent", "follow_up"]
    assert [(s.name, s.one_per) for s in SURFACE if s.one_per] == [("follow_up", "agent")]
    assert {s.name for s in SURFACE if s.parallel == "serial"} == {
        "shell",
        "job",
        "doc",
        "ask_user",
    }
    assert all(
        s.parallel == "any"
        for s in SURFACE
        if s.family in ("navigate", "read")
        and s.name not in ("locate", "spawn_agent", "follow_up")
    )


def test_only_coder_doc_and_worktree_mutate_source_and_shell_and_job_the_environment() -> None:
    # `doc` writes only the work folder's `design/`, through the edit port;
    # the edit tools refuse that folder and source stays the coder's. `worktree` lands or
    # discards the coder's committed work, asked of the person, and writes none of its own
    assert [s.name for s in SURFACE if s.mutates == "source"] == ["coder", "doc", "worktree"]
    assert [s.name for s in SURFACE if s.mutates == "environment"] == ["shell", "job"]
    assert [s.name for s in SURFACE if s.tier == "computed"] == ["shell", "coder", "worktree"]
    assert [s.name for s in SURFACE if s.model_calls != "none"] == [
        "locate",
        "spawn_agent",
        "follow_up",
        "coder",
    ]


# --- the docs are renderings --------------------------------------------------------------


def _names_in_design_table(text: str) -> list[str]:
    start = text.index("| Family | Tools | Model calls |")
    names: list[str] = []
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        names.extend(re.findall(r"`([a-z_]+)`", cells[1]))
    return names


def test_parallel_class_of_reads_the_tables_own_vocabulary() -> None:
    assert parallel_class_of("**yes** — any number") == "any"
    assert parallel_class_of("yes, but each costs a model call, so cap the fan-out") == "capped"
    assert parallel_class_of("**never more than one**") == "exclusive"
    assert parallel_class_of("one at a time by default") == "serial"
    assert parallel_class_of("whenever") is None, "an unrecognised cell is reported, not passed"


def test_render_table_yields_the_design_family_table_and_an_arch_row_per_tool() -> None:
    """A table for a reader, self-consistent with the parsers above — not the docs' bytes,
    which carry prose neither style renders (the docstring says so)."""
    design = render_table(SURFACE, style="design")
    assert _names_in_design_table(design) == [s.name for s in SURFACE]
    arch = render_table(SURFACE, style="arch")
    assert len(arch.splitlines()) == 2 + len(SURFACE)
    assert "| `coder` | C | exclusive | source | computed |" in arch


def test_cap_numbers_have_one_home_each() -> None:
    """§3 states `locate`'s 40 (`limits.locate.max_tool_calls`); the rest are starting
    values named once in `spec.py` and read from there — `tools()`'s entry cap is
    `result_cap`'s number, not a second 200."""
    by_name = {s.name: s for s in SURFACE}
    assert by_name["locate"].bound.value == LOCATE_MAX_CALLS == 40
    lines = {s.name for s in SURFACE if s.bound.kind == "inline_lines"} - {"find_skill"}
    assert lines == {"grep", "glob", "tools", "grep_in"}
    assert all(by_name[n].bound.value == RESULT_LINES_DEFAULT for n in lines)
    assert all(by_name[n].bound.config_key == "result_cap" for n in lines)
    # `find_skill`'s rows are its own register row.
    found = by_name["find_skill"].bound
    assert (found.kind, found.value, found.config_key) == (
        "inline_lines",
        FIND_SKILL_RESULTS,
        "limits.skills.find_results",
    )
    assert inspect.signature(tools).parameters["entry_cap"].default == RESULT_LINES_DEFAULT


# --- the resident set and the stub ---------------------------------------------------------


def test_resident_set_ships_full_schemas_and_stub_is_the_first_line() -> None:
    descriptions = _descriptions()
    entries = resident(SURFACE, descriptions)
    assert [e.name for e in entries] == [s.name for s in SURFACE]
    read = next(e for e in entries if e.name == "read")
    assert read.schema == next(s for s in SURFACE if s.name == "read").schema
    assert {"path", "handle"} <= set(next(s for s in SURFACE if s.name == "read").arguments)
    s = next(x for x in SURFACE if x.name == "shell")
    short = stub(s, descriptions["shell.md"])
    assert short == ToolSchema(
        "shell", descriptions["shell.md"].splitlines()[0], {"type": "object"}
    )
    assert DYNAMIC_STUB_MAX == 32


# --- `tools` itself ---------------------------------------------------------------------------


def test_tools_query_returns_matching_names_and_first_lines_and_nothing_else() -> None:
    descriptions = _descriptions()
    result = tools(query="read", descriptions=descriptions)
    assert isinstance(result, ToolBriefs)
    assert result.names == ("read", "grep_in"), result
    assert result.matched == 2 and not result.truncated
    for brief in result.briefs:
        assert isinstance(brief, ToolBrief)
        assert brief.line == descriptions[f"{brief.name}.md"].splitlines()[0]


def test_tools_name_returns_the_full_schema() -> None:
    result = tools(name="shell", descriptions=_descriptions())
    assert isinstance(result, dict)
    assert result["name"] == "shell"
    assert result["parameters"] == next(s for s in SURFACE if s.name == "shell").schema


def test_tools_both_or_neither_is_invalid_arguments() -> None:
    both = tools(query="read", name="shell")
    neither = tools()
    assert isinstance(both, InvalidArguments) and both.tool == "tools"
    assert isinstance(neither, InvalidArguments)
    assert "exactly one of" in neither.violations[0].problem


def test_tools_empty_match_is_an_empty_answer_not_unresolved() -> None:
    nothing = tools(query="no such thing anywhere", descriptions=_descriptions())
    assert isinstance(nothing, ToolBriefs)
    assert nothing == ToolBriefs((), 0) and not nothing.truncated
    assert tools(name="no_such_tool") == {}


def test_tools_matches_case_insensitively_and_a_cut_list_says_so() -> None:
    """Truncation is a typed value. The briefs are in surface order, not relevance
    order, so a cut the model cannot see would hide tools arbitrarily."""
    descriptions = _descriptions()
    upper = tools(query="READ", descriptions=descriptions)
    assert isinstance(upper, ToolBriefs) and upper.names == ("read", "grep_in")
    capped = tools(query="the", descriptions=descriptions, entry_cap=2)
    assert isinstance(capped, ToolBriefs)
    assert len(capped.briefs) == 2 and capped.matched > 2 and capped.truncated
    whole = tools(query="the", descriptions=descriptions)
    assert isinstance(whole, ToolBriefs)
    assert whole.matched == capped.matched and not whole.truncated


def test_tools_refuses_a_surface_without_itself_as_a_harness_bug() -> None:
    without = tuple(s for s in SURFACE if s.name != "tools")
    with pytest.raises(HarnessBug, match="no `tools` entry"):
        tools(query="read", surface=without)


# --- the registry -------------------------------------------------------------------


def test_every_surface_entry_is_a_registered_tool_and_zeus_s_default() -> None:
    """Each `SURFACE` entry is a built-in with a run, and the list is Zeus's default."""
    for s in SURFACE:
        tool = BUILTINS.get(s.name)
        assert tool is not None and tool.spec == s, s.name
    default = role("zeus").tool_defaults
    assert default is not None and default.enabled == tuple(s.name for s in SURFACE)


ZEUS_SRC = ROOT / "packages" / "core" / "src" / "chickadee" / "zeus"
_NAME = "|".join(re.escape(s.name) for s in SURFACE)
_COMPARED = re.compile(rf"""(==|!=)\s*["']({_NAME})["']|["']({_NAME})["']\s*(==|!=)""")


def test_no_tool_name_is_compared_in_zeus() -> None:
    """What Zeus's loop asks of a tool it asks of its spec (`streams`, `model_calls`,
    `mutates`, `family`), so a renamed or reconfigured tool needs no edit here."""
    found = [
        f"{p.relative_to(ROOT)}:{n}: {line.strip()}"
        for p in sorted(ZEUS_SRC.rglob("*.py"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        # A family is a spec fact, and `read` names both a family and a tool.
        if _COMPARED.search(line) and ".family" not in line
    ]
    assert found == []


def test_a_zeus_selection_chooses_what_the_prefix_lists_and_the_loop_dispatches(
    tmp_path: Path,
) -> None:
    """`[tools.zeus]` is read: a tool it leaves out is not offered, and one it
    renames is listed and called under its new name, and run as itself."""
    from dataclasses import replace

    from .zeus_helpers import answer, call, calls, rig

    default = role("zeus").tool_defaults
    assert default is not None
    chosen = replace(
        default,
        enabled=tuple(n for n in default.enabled if n != "atlas"),
        shown_as={"symbol": "lookup"},
    )
    r = rig(
        tmp_path,
        [calls(call("lookup", "s1", name="Retry"), call("atlas", "a1")), answer()],
        policy=chosen,
    )
    r.run("where is the retry?")
    listed = [t.name for t in r.model.requests[0].tools]
    assert "lookup" in listed and "symbol" not in listed and "atlas" not in listed
    assert r.graph.asked == [("symbol", "Retry")]
    state = r.session.state
    assert state.ran("s1") and not state.ran("a1")
    result = state.result("a1")
    assert result is not None and result.text.startswith("UNKNOWN_TOOL{name: 'atlas'")


def test_a_model_switch_brings_its_own_selection(tmp_path: Path) -> None:
    """`[tools.zeus]` may differ per model, so `/rung zeus` hands the loop
    the new model's policy with its prefix, and the next turn dispatches that one."""
    from dataclasses import replace

    from .zeus_helpers import answer, call, calls, rig, zeus_prefix

    default = role("zeus").tool_defaults
    assert default is not None
    r = rig(tmp_path, [answer(), calls(call("lookup", "s1", name="Retry")), answer()])
    r.run("hello")
    narrowed = replace(default, shown_as={"symbol": "lookup"})
    r.loop.reprefix(zeus_prefix(r.workspace, policy=narrowed), "frontier", narrowed)
    r.run("where is the retry?")
    assert r.graph.asked == [("symbol", "Retry")]
    assert r.session.state.ran("s1")
