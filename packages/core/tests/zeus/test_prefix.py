"""The system prefix has one composer: two assemblies with the same
inputs are byte-identical and share a sha; the order is template, schemas, index, env; the
date and git head are absent from the prefix and present in the first turn's frame; every
`SURFACE` tool's schema is present; `zeus.md` names no tool `SURFACE` lacks; changing any one
input changes the sha — and says which part moved."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest
from session_lines import raw

from chickadee.adapters.base import load_template
from chickadee.client.request import text_of
from chickadee.context.lines import FrameLine
from chickadee.context.session import OtherLine, Session
from chickadee.roles import role
from chickadee.schema import sha256_hex
from chickadee.schema.execution import SandboxLevel
from chickadee.tools.meta import FINDINGS
from chickadee.tools.spec import SURFACE
from chickadee.work import DEFAULT_DIR, with_work_dir
from chickadee.zeus.loop import frame_due, frame_text
from chickadee.zeus.prefix import (
    PARTS,
    TEMPLATES,
    Env,
    Prefix,
    assemble,
    render_env,
)

from .zeus_helpers import HEAD, SESSION_ID, TODAY, answer, env, rig, zeus_prefix

SKILLS = "review · project · reads a diff against the design"


def _prefix(tmp_path: Path, **kw: object) -> Prefix:
    templates = kw.pop("templates", TEMPLATES)
    surface = kw.pop("surface", SURFACE)
    skills = kw.pop("skills", SKILLS)
    the_env = dataclasses.replace(env(tmp_path / "ws"), **kw)  # type: ignore[arg-type]
    return assemble(
        role("zeus"),
        surface,  # type: ignore[arg-type]
        skills,  # type: ignore[arg-type]
        the_env,
        templates=templates,  # type: ignore[arg-type]
        work_dir=DEFAULT_DIR,
    )


def test_two_assemblies_with_the_same_inputs_are_byte_identical(tmp_path: Path) -> None:
    a, b = _prefix(tmp_path), _prefix(tmp_path)
    assert a.text == b.text
    assert a.tools == b.tools
    assert a.parts == b.parts
    assert a.sha == b.sha
    assert len(a.sha) == 64


def test_the_order_is_template_schemas_index_env(tmp_path: Path) -> None:
    p = _prefix(tmp_path)
    assert PARTS == ("template", "tools", "skills", "env")
    assert tuple(p.parts) == PARTS
    # As rendered: `{work_dir}` named, the default here.
    template = with_work_dir(
        (TEMPLATES / "zeus.md").read_text(encoding="utf-8"), DEFAULT_DIR
    ).rstrip("\n")
    # The system message is the template, then the index, then the environment block; the
    # schemas are the request's `tools` array, part 2 of the one sha.
    assert p.text.startswith(template)
    assert p.text.index(template) < p.text.index(SKILLS) < p.text.index("<env>")
    assert p.text.rstrip().endswith("</env>")
    assert p.system().role == "system"


def test_the_env_block_names_the_configured_model_and_the_sandbox_facts(tmp_path: Path) -> None:
    block = render_env(env(tmp_path / "ws", model="frontier-model-a"))
    assert "model: frontier-model-a" in block
    assert "sandbox: native · reads_masked: yes\n" in block
    assert f"workspace: {tmp_path / 'ws'}" in block
    assert "repo_trust: first_party" in block


def test_date_and_head_are_absent_from_the_prefix_and_in_the_first_turn_frame(
    tmp_path: Path,
) -> None:
    r = rig(tmp_path, [answer("hello")])
    prefix = zeus_prefix(r.workspace)
    wire = prefix.text + json.dumps([t.schema for t in prefix.tools])
    for fact in (TODAY.isoformat(), HEAD[0][:7], HEAD[0][:12], HEAD[1]):
        assert fact not in wire, f"{fact!r} is in the prefix"
    r.run("what is this repo?")
    messages = r.model.requests[0].messages
    assert messages[0].role == "system"
    assert messages[0].content == prefix.text
    frame, turn = text_of(messages[1]), text_of(messages[2])
    assert TODAY.isoformat() in frame and HEAD[0][:12] in frame and HEAD[1] in frame
    assert frame.startswith("<harness_notice>")
    assert turn.endswith("what is this repo?")
    frames = r.session.state.frames()
    assert [(f.date, f.head, f.branch) for f in frames] == [(TODAY.isoformat(), *HEAD)]


def test_the_frame_is_repeated_only_on_the_first_turn_after_the_date_changes(
    tmp_path: Path,
) -> None:
    days = [TODAY, TODAY, TODAY + timedelta(days=1), TODAY + timedelta(days=1)]
    clock = iter(days)
    r = rig(tmp_path, today=lambda: next(clock))
    for n in range(4):
        r.run(f"turn {n}")
    assert [f.date for f in r.session.state.frames()] == [
        TODAY.isoformat(),
        (TODAY + timedelta(days=1)).isoformat(),
    ]


def test_a_cut_that_took_the_last_frame_is_followed_by_a_new_one(tmp_path: Path) -> None:
    r = rig(tmp_path)
    r.run("first")
    state = r.session.state
    assert not frame_due(state, TODAY, HEAD[1])
    seq = r.session.next_seq
    r.session.append(
        raw(
            "compaction",
            {
                "boundary": seq,
                "previous": 0,
                "block": "[the first turn, summarised]",
                "standing": "[standing]",
                "lifted": [],
                "facts": {},
            },
        )
    )
    assert frame_due(r.session.state, TODAY, HEAD[1])


def test_a_changed_branch_is_followed_by_a_new_frame(tmp_path: Path) -> None:
    """The view's branch names its workspace, so a move into the session's
    worktree, or back, makes a frame due. The same branch with a new head does not."""
    r = rig(tmp_path)
    r.run("first")
    assert not frame_due(r.session.state, TODAY, HEAD[1])
    assert frame_due(r.session.state, TODAY, "chickadee/add-multiply-1a2b3c4d")


def test_frame_text_without_git_is_the_date_alone() -> None:
    assert frame_text(date(2026, 1, 2), None, None) == "2026-01-02"
    assert frame_text(date(2026, 1, 2), "abc", None) == "2026-01-02 · head abc"


def test_every_surface_tool_schema_is_present(tmp_path: Path) -> None:
    p = _prefix(tmp_path)
    by_name = {t.name: t for t in p.tools}
    assert list(by_name) == [s.name for s in SURFACE], "the resident set is SURFACE, in order"
    for spec in SURFACE:
        shown = by_name[spec.name]
        props = dict(shown.schema["properties"])
        # Findings alone beside the tool's own, last; no purpose line.
        assert list(props)[-1] == FINDINGS, spec.name
        assert not set(props) & {"why", "intent"}, spec.name
        own = {k: v for k, v in props.items() if k != FINDINGS}
        assert own == dict(spec.schema["properties"]), spec.name  # type: ignore[arg-type]
        assert shown.schema.get("required") == spec.schema.get("required"), spec.name
        rest = {k: v for k, v in shown.schema.items() if k not in ("properties", "required")}
        assert rest == {
            k: v for k, v in spec.schema.items() if k not in ("properties", "required")
        }
        text = (TEMPLATES / "tools" / spec.description).read_text(encoding="utf-8").strip()
        assert shown.description == text


def test_zeus_md_names_no_tool_surface_lacks() -> None:
    text = (TEMPLATES / "zeus.md").read_text(encoding="utf-8")
    named = set(re.findall(r"\b([a-z_]+)\(", text))
    assert named, "the template names the tools it discusses in `name(` form"
    assert named <= {s.name for s in SURFACE}, (
        f"not in SURFACE: {named - {s.name for s in SURFACE}}"
    )
    # The sixteenth tool joins SURFACE with T-3.19; until then the template may not name it,
    # and `outline` was folded into `read(depth=)` by D78.
    for absent in ("note_constraint", "outline(", "write(", "edit("):
        assert absent not in text


def test_zeus_md_has_its_sections_in_order() -> None:
    """D197 amended D193 decision 1: a role-based prompt with no cast, the job first, the rules
    after it (the owner's direction)."""
    text = (TEMPLATES / "zeus.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert headings == [
        "What is being asked",
        "Understanding the requirement",
        "Designing a change",
        "Handing the work over",
        "Continuing a work",
        "While the team works",
        "Reading the report",
        "Using the tools",
        "Rules",
        "Asking and recording",
        "Writing to the user",
        "Reading a failure",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace", "/elsewhere"),
        ("platform", "darwin"),
        ("sandbox", SandboxLevel.CONTAINER),
        ("reads_masked", False),
        ("model", "frontier-model-b"),
        ("repo_trust", "third_party"),
    ],
)
def test_changing_one_env_fact_changes_the_sha_and_only_the_env_part(
    tmp_path: Path, field: str, value: object
) -> None:
    base = _prefix(tmp_path)
    moved = _prefix(tmp_path, **{field: value})
    assert moved.sha != base.sha
    assert [p for p in PARTS if moved.parts[p] != base.parts[p]] == ["env"]


def _copy_templates(tmp_path: Path) -> Path:
    copy = tmp_path / "templates"
    shutil.copytree(TEMPLATES, copy)
    return copy


def test_changing_the_template_changes_the_sha_and_says_so(tmp_path: Path) -> None:
    base = _prefix(tmp_path)
    copy = _copy_templates(tmp_path)
    (copy / "zeus.md").write_text(
        (copy / "zeus.md").read_text(encoding="utf-8") + "\nOne more sentence.\n",
        encoding="utf-8",
    )
    moved = _prefix(tmp_path, templates=copy)
    assert moved.sha != base.sha
    assert [p for p in PARTS if moved.parts[p] != base.parts[p]] == ["template"]


def test_changing_a_tool_description_or_the_surface_moves_the_tools_part(
    tmp_path: Path,
) -> None:
    base = _prefix(tmp_path)
    copy = _copy_templates(tmp_path)
    (copy / "tools" / "read.md").write_text("Read a file.\n", encoding="utf-8")
    described = _prefix(tmp_path, templates=copy)
    fewer = _prefix(tmp_path, surface=tuple(s for s in SURFACE if s.name != "locate"))
    for moved in (described, fewer):
        assert moved.sha != base.sha
        assert [p for p in PARTS if moved.parts[p] != base.parts[p]] == ["tools"]


def test_changing_the_skills_index_changes_the_sha(tmp_path: Path) -> None:
    base = _prefix(tmp_path)
    moved = _prefix(tmp_path, skills="")
    assert moved.sha != base.sha
    assert [p for p in PARTS if moved.parts[p] != base.parts[p]] == ["skills"]
    assert SKILLS not in moved.text


def test_the_prefix_sha_goes_to_the_session_log_once_per_prefix(tmp_path: Path) -> None:
    r = rig(tmp_path)
    r.run("one")
    r.run("two")

    def prefix_lines(session: Session) -> list[OtherLine]:
        return [
            rec
            for _, rec in session.state.lines()
            if isinstance(rec, OtherLine) and rec.kind == "prefix"
        ]

    assert len(prefix_lines(r.session)) == 1
    r.session.close()
    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    again = rig(tmp_path, session=reopened, workspace=r.workspace)
    again.run("three")
    assert len(prefix_lines(reopened)) == 1, (
        "a resumed session with the same prefix writes nothing"
    )
    again.loop.reprefix(zeus_prefix(r.workspace, model="frontier-model-b"), model="frontier-b")
    again.run("four")
    assert len(prefix_lines(reopened)) == 2, "a /rung switch reassembles, and its sha is logged"
    # The frame opens only the first turn of the day, however many prefixes there are.
    assert sum(isinstance(rec, FrameLine) for _, rec in reopened.state.lines()) == 1
    reopened.close()


def test_the_prefix_is_zeus_only(tmp_path: Path) -> None:
    from chickadee.errors import HarnessBug

    with pytest.raises(HarnessBug):
        assemble(role("worker"), SURFACE, "", env(tmp_path), work_dir=DEFAULT_DIR)
    assert isinstance(_prefix(tmp_path), Prefix)
    assert Env is not None


def test_tools_answers_with_the_briefs_and_the_schema_the_model_is_shown(
    tmp_path: Path,
) -> None:
    """`tools(name=)` is the way back to a schema compaction evicted, so it returns the one
    resident in the prefix — meta arguments included — and `tools(query=)` the stub lines."""
    from .zeus_helpers import call, calls

    r = rig(
        tmp_path, [calls(call("tools", "t1", name="read"), call("tools", "t2", query="grep"))]
    )
    r.run("which tools read?")
    state = r.session.state
    by_name = state.result("t1")
    assert by_name is not None and by_name.handle is not None
    shown = json.loads(r.session.blob(by_name.handle.content_sha))
    resident = {t.name: t for t in zeus_prefix(r.workspace).tools}["read"]
    assert shown["parameters"] == resident.schema
    first = (TEMPLATES / "tools" / "grep_in.md").read_text(encoding="utf-8").splitlines()[0]
    assert f"grep_in — {first}" in state.result("t2").text  # type: ignore[union-attr]


def test_the_templates_zeus_writes_to_follow_his_instructions(tmp_path: Path) -> None:
    """Told to write a design and a brief to their templates, he is shown them, after
    the instructions and before the index, and a change to either moves the template part's
    sha. Without them the brief he wrote was one sentence."""
    from chickadee.handoffs import ZEUS_HANDOFFS
    from chickadee.handoffs import resolve as resolve_handoff

    shown = tuple(
        (title, resolve_handoff(tmp_path, {}, kind).text) for kind, title in ZEUS_HANDOFFS
    )
    bare = _prefix(tmp_path)
    p = assemble(
        role("zeus"),
        SURFACE,
        SKILLS,
        env(tmp_path / "ws"),
        handoffs=shown,
        work_dir=DEFAULT_DIR,
    )
    # As rendered: `{work_dir}` named, the default here.
    template = with_work_dir(
        (TEMPLATES / "zeus.md").read_text(encoding="utf-8"), DEFAULT_DIR
    ).rstrip("\n")
    assert p.text.startswith(template)
    for title, text in shown:
        assert f"## {title}\n\n{text.strip()}" in p.text
    assert p.text.index("## The design template") < p.text.index(SKILLS)
    assert "## Where to look" in p.text and "## What must not change" in p.text
    assert p.parts["template"] != bare.parts["template"]
    assert p.parts["tools"] == bare.parts["tools"]


# -- T-3.175: the configured work.dir --------------------------------------------------------


def test_the_prefix_names_the_configured_work_dir(tmp_path: Path) -> None:
    """Zeus is told where his first `doc` opens a folder: the configured."""
    the_env = env(tmp_path / "ws")
    mine = assemble(role("zeus"), SURFACE, SKILLS, the_env, work_dir="notes/designs")
    default = assemble(role("zeus"), SURFACE, SKILLS, the_env, work_dir=DEFAULT_DIR)
    assert "opens its folder under `notes/designs/`" in mine.text
    assert "docs/chickadee" not in mine.text and "{work_dir}" not in mine.text
    assert "opens its folder under `docs/chickadee/`" in default.text
    assert mine.sha != default.sha, "a changed work.dir is a new prefix"


def test_with_the_default_work_dir_the_template_is_todays_bytes() -> None:
    """The pin of the default's bytes: moved by D208, its `why` line removed; by D214, a
    design that describes the code and never holds it; by D201, a command that
    runs on as a job and the exit that is told; and by D230, the team working in
    this conversation's workspace, the checkout unless a worktree was asked for,
    which `worktree(` moves between; and by D240, continuing a work where it stopped,
    its `RESUMED` and `COMPLETE` answers; and by D240's background run, "While the
    team works", `RESUMED` gone; and by its `tell(`, and how a tell differs from an
    amendment; and by the agents line, `spawn_agent(` with a profile, never the
    researcher as a skill."""
    text = with_work_dir(load_template(TEMPLATES, "zeus.md"), DEFAULT_DIR)
    assert (
        sha256_hex(text) == "6f5cc3d361a14e88a5afab109c90a8f51e969037168c66c85c90c655b5e0fb79"
    )
