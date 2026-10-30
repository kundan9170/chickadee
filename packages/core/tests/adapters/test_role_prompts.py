"""Each role is told about the tools it holds, in plain terms.

What these pin: a role prompt's optional blocks are kept or dropped by what the run offers; no
role's prompt, or a description of a tool it holds, names a built-in tool that role is not
offered (the worker's prompt and `find` sent it to a shell no sandbox gave it, the scout's
`find` to a shell it never has); the role prompts carry none of the project's own names; the box
roles' `shell` withholds `effects`, which their profile refuses. Zeus's prompt and descriptions
name only its own tools; the strings the box answers its roles with in code name only tools the
role holds, and none of the project's names, nor do the reasons a report gives Zeus.
"""

from __future__ import annotations

import re
from collections.abc import Collection

import pytest

from chickadee.adapters.base import (
    BOX_DESCRIPTIONS,
    SHARED_TEMPLATES,
    load_template,
    select_blocks,
    tool_descriptions,
)
from chickadee.adapters.frontier import TEMPLATES
from chickadee.roles import ATHENA_TASK, TASK_WORKER
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.policy import RolePolicy

SCOUT_TOOLS = ("symbol", "grep", "glob", "neighbors", "impact", "read", "grep_in")
"""The scout's offered set with escapes (`scout/tools`): `request_exec` is named only by its
own description, which the scout reads as offered."""

JARGON = ("Zeus", "Athena", "the worker", "the cast", "the box", "stop 1", "stop 2", "tier")
"""The project's own names for its parts: meaningless to a model, so never in a role prompt."""


def _named(text: str) -> set[str]:
    """Built-in tool names the text refers to: in backticks, or called as `name(`."""
    tools = set(BUILTINS.names())
    marked = re.findall(r"`([a-z_]+)(?:\(|`)", text) + re.findall(r"\b([a-z_]+)\(", text)
    return {n for n in marked if n in tools}


def _described(
    policy: RolePolicy, offered: Collection[str], present: Collection[str] = ()
) -> dict[str, str]:
    specs = tuple(t.spec for n in offered if (t := BUILTINS.get(n)) is not None)
    by_template = tool_descriptions(specs, *BOX_DESCRIPTIONS, present=present)
    return {s.name: by_template[s.description] for s in specs}


# -- blocks -----------------------------------------------------------------------------------


def test_a_block_is_kept_or_dropped_by_what_is_offered() -> None:
    text = "a\n{{#shell}}\nwith\n{{/shell}}\n{{^shell}}\nwithout\n{{/shell}}\nz\n"
    assert select_blocks(text, {"shell"}) == "a\nwith\nz\n"
    assert select_blocks(text, set()) == "a\nwithout\nz\n"


@pytest.mark.parametrize(
    "bad", ["{{#shell}}\nx\n", "{{/shell}}\n", "{{#a}}\n{{#b}}\n{{/b}}\n{{/a}}\n"]
)
def test_a_block_that_does_not_close_or_nests_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        select_blocks(bad, set())


# -- the box roles ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "prompt"), [(TASK_WORKER, "worker.md"), (ATHENA_TASK, "athena.md")]
)
@pytest.mark.parametrize("shell", [True, False])
def test_a_box_role_is_told_only_of_the_tools_it_holds(
    policy: RolePolicy, prompt: str, shell: bool
) -> None:
    offered = tuple(n for n in policy.enabled if shell or n != "shell")
    text = select_blocks(load_template(SHARED_TEMPLATES, prompt), {"shell"} if shell else ())
    assert _named(text) <= set(offered), f"{prompt} names {_named(text) - set(offered)}"
    for name, description in _described(policy, offered).items():
        stray = _named(description) - set(offered)
        assert not stray, f"{name}'s description names {stray}, which {policy.role} lacks"


def test_without_a_shell_a_box_role_is_told_nothing_runs() -> None:
    for prompt in ("worker.md", "athena.md"):
        text = select_blocks(load_template(SHARED_TEMPLATES, prompt), ())
        assert "`shell`" not in text and "No command can run" in text, prompt
    # Nor is Athena told to ask the agent for commands, or to look for ones that ran.
    hers = select_blocks(load_template(SHARED_TEMPLATES, "athena.md"), ())
    assert "commands to run" not in hers and "commands run" not in hers
    assert "ask for none" in hers


@pytest.mark.parametrize(
    ("policy", "prompt"), [(TASK_WORKER, "worker.md"), (ATHENA_TASK, "athena.md")]
)
def test_a_shell_held_unsealed_is_said_and_a_sealed_one_is_not(
    policy: RolePolicy, prompt: str
) -> None:
    """Held without a sandbox, the prompt and the shell's description say commands
    run on the user's machine, may each need approval, and must not touch the version-control
    directory or files outside the repository; neither claims the sandbox's protections."""
    template = load_template(SHARED_TEMPLATES, prompt)
    # One line per paragraph, so a phrase is found wherever the template wraps it.
    sealed = " ".join(select_blocks(template, {"shell", "sealed"}).split())
    unsealed = " ".join(select_blocks(template, {"shell", "unsealed"}).split())
    assert "No sandbox runs here" in unsealed and "No sandbox runs here" not in sealed
    assert "the version-control directory is not" in sealed
    assert "the version-control directory is not" not in unsealed
    assert "outside the repository" in unsealed and "approval" in unsealed
    offered = tuple(policy.enabled)
    held = " ".join(_described(policy, offered, present={"shell", "unsealed"})["shell"].split())
    assert "No sandbox runs here" in held and "read-only" not in held
    sealed_held = " ".join(_described(policy, offered)["shell"].split())
    assert "No sandbox runs here" not in sealed_held and "The network is off" in sealed_held
    # Unsealed, nothing closes the network, so the description never says it is off.
    assert "network is off" not in held and "reaching the network" in held
    for text in (unsealed, held):
        assert _named(text) <= set(offered)
        assert not any(word in text for word in JARGON), text


def test_a_shell_withheld_for_want_of_a_sandbox_says_why() -> None:
    for prompt in ("worker.md", "athena.md"):
        text = select_blocks(load_template(SHARED_TEMPLATES, prompt), {"no_sandbox"})
        assert "settings require one" in text and "chickadee setup" in text, prompt
        assert "`shell`" not in text
        assert "settings require one" not in select_blocks(
            load_template(SHARED_TEMPLATES, prompt), ()
        )


@pytest.mark.parametrize("policy", [TASK_WORKER, ATHENA_TASK])
def test_no_shell_shows_effects(policy: RolePolicy) -> None:
    """`effects` retired from every shell, Zeus's too; the box roles were never
    shown it."""
    shell = BUILTINS.get("shell")
    assert shell is not None
    shown = policy.resolved(shell.spec).schema
    assert {"command", "network"} <= set(shown["properties"])  # type: ignore[arg-type]
    assert "why" not in shown["properties"]  # type: ignore[operator]
    assert "effects" not in shown["properties"]  # type: ignore[operator]
    assert "effects" not in shell.spec.schema["properties"]  # type: ignore[operator]


# -- the scout --------------------------------------------------------------------------------


def test_the_scout_is_never_sent_to_a_shell() -> None:
    specs = tuple(t.spec for n in SCOUT_TOOLS if (t := BUILTINS.get(n)) is not None)
    for template, description in tool_descriptions(specs, TEMPLATES).items():
        assert "shell(" not in description, template


# -- every role prompt ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "prompt"),
    [(TEMPLATES, "zeus.md"), (SHARED_TEMPLATES, "athena.md"), (SHARED_TEMPLATES, "worker.md")],
)
def test_a_role_prompt_carries_none_of_the_projects_names(root: object, prompt: str) -> None:
    text = load_template(root, prompt)  # type: ignore[arg-type]
    found = [w for w in JARGON if re.search(rf"\b{re.escape(w)}\b", text)]
    assert not found, f"{prompt} says {found}"


@pytest.mark.parametrize("policy", [TASK_WORKER, ATHENA_TASK])
def test_a_withheld_argument_is_refused_naming_what_the_tool_takes(policy: RolePolicy) -> None:
    """The call is validated against the schema the role is shown (the dispatcher's pending
    spec is the resolved one), so `effects=` from a box role is `INVALID_ARGUMENTS`."""
    from chickadee.tools.validate import Arguments, validate

    shell = BUILTINS.get("shell")
    assert shell is not None
    spec = policy.resolved(shell.spec)
    ok = validate(spec, {"command": "pytest -q"})
    assert isinstance(ok, Arguments)
    refused = validate(spec, {"command": "git commit -am x", "effects": [".git"]})
    assert not isinstance(refused, Arguments)
    assert [v.field for v in refused.violations] == ["effects"]
    assert "effects" not in refused.schema["properties"]  # type: ignore[operator]


# -- Zeus -------------------------------------------------------------------------------------


def test_zeus_is_told_only_of_the_tools_he_holds() -> None:
    from chickadee.tools.spec import SURFACE

    held = {s.name for s in SURFACE}
    text = load_template(TEMPLATES, "zeus.md")
    assert _named(text) <= held, f"zeus.md names {_named(text) - held}"
    for template, description in tool_descriptions(SURFACE, TEMPLATES).items():
        stray = _named(description) - held
        assert not stray, f"{template} names {stray}, which Zeus lacks"


# -- what the box says in code ----------------------------------------------------------------


def _box_strings() -> dict[str, tuple[str, frozenset[str]]]:
    """Each string the box answers a role with from code, and the tools that role may hold
    without a shell, since none of these is rendered from what the run offers."""
    from chickadee.box import desk, loop
    from chickadee.boxctx import athena, worker
    from chickadee.hooks.builtins import review_owed

    hers = frozenset(n for n in ATHENA_TASK.enabled if n != "shell")
    its = frozenset(n for n in TASK_WORKER.enabled if n != "shell")
    return {
        "PLAN_TEMPLATE": (athena.PLAN_TEMPLATE, hers),
        "BRIEF_TEMPLATE": (athena.BRIEF_TEMPLATE, hers),
        "TASK_DESIGN": (athena.TASK_DESIGN, hers),
        "ONE_TASK": (athena.ONE_TASK, hers),
        "FIRST_ROUND": (athena.FIRST_ROUND, hers),
        "REVIEW_FIRST": (athena.REVIEW_FIRST, hers),
        "TAKEN_OVER": (desk.TAKEN_OVER, hers),
        "REVIEW_OWED": (review_owed("T1", "x", taken_over=False), hers),
        "REVIEW_OWED_HELD": (review_owed("T1", "x", taken_over=True), hers),
        "NO_TASK_HELD": (desk.NO_TASK_HELD.render(), hers),
        "ENDED": (desk.ENDED.render(), hers),
        "DONE_SO_FAR": (worker.DONE_SO_FAR, its),
        "TASKS_OMITTED": (worker.TASKS_OMITTED, its),
        **{f"NO_REPORT[{k}]": (v, frozenset()) for k, v in loop.NO_REPORT.items()},
    }


@pytest.mark.parametrize("name", sorted(_box_strings()))
def test_a_string_the_box_answers_with_names_only_what_its_role_holds(name: str) -> None:
    text, holds = _box_strings()[name]
    assert _named(text) <= holds, f"{name} names {_named(text) - holds}"
    found = [w for w in JARGON if re.search(rf"\b{re.escape(w)}\b", text)]
    assert not found, f"{name} says {found}"


def test_the_reasons_a_report_gives_say_no_project_names() -> None:
    from chickadee.box.desk import Ended

    for why in (Ended("not now").why, Ended(None, bound="three rounds").why):
        found = [w for w in (*JARGON, "stop") if re.search(rf"\b{re.escape(w)}\b", why)]
        assert not found, f"{why!r} says {found}"


def test_the_guard_fails_on_a_planted_bad_description() -> None:
    """Done when: the guard is shown to fail, not only to pass."""
    planted = "List files with `shell(` instead; or ask `coder` to."
    assert _named(planted) - set(TASK_WORKER.enabled) == {"coder"}
    assert _named(planted) - {n for n in TASK_WORKER.enabled if n != "shell"} == {
        "shell",
        "coder",
    }


def test_no_answer_built_in_code_shows_read_by_a_name_it_no_longer_takes() -> None:
    """`read` takes `path` or `handle`, by name. An answer that shows it called with
    `target=`, or with a bare positional argument, teaches the model a call that is refused
    (`symbol`'s `body:` line did, after the rename). Parked and retired code is not served."""
    from pathlib import Path

    import chickadee

    root = Path(chickadee.__file__).parent
    bad = re.compile(r"\bread\((?:target=|\{|h_)")
    found = [
        f"{p.relative_to(root)}:{n}"
        for p in root.rglob("*.py")
        if not {"parked", "retired"} & set(p.relative_to(root).parts)
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if bad.search(line)
    ]
    assert not found, found
