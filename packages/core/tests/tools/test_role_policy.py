"""A role is a ceiling and a policy: each role's ceiling equals D164 decision 7's table; each
default sits within it; a
selection naming an unknown tool, one above the ceiling, one needing a port outside it, wider
files, a name shown twice, or a confirmation without a person is refused, every reason
named; and `[tools.<role>]` is refused until its role's loop reads it (Zeus's is read)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from chickadee.config import EngineConfig
from chickadee.errors import ConfigError
from chickadee.roles import DEFAULT_RUNGS, ROLES, role
from chickadee.schema.authorization import Tier
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import AskWhen, Valid
from chickadee.tools.policy import Ceiling, FileRule, RolePolicy, Selection, resolve
from chickadee.tools.rules import Pipeline
from chickadee.tools.spec import SURFACE
from chickadee.tools.tool import PORTS

OWN = {s.name: s for s in SURFACE}

MODEL: dict[str, Any] = {
    "base_url": "http://localhost:8000/v1",
    "model_id": "m",
    "context_window": 65_536,
    "max_output_tokens": 16_384,
}


def raw(**tools: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": {f"{name}-model": dict(MODEL) for name in DEFAULT_RUNGS},
        "rungs": {name: f"{name}-model" for name in DEFAULT_RUNGS},
        "spend": {"hard_cost_usd": 5.0},
        "tools": tools,
    }


ALL = FileRule("repository", "visible")

CEILINGS = {
    "zeus": Ceiling("source", Tier.T2, PORTS, ALL),
    # While she verifies, her edits land on the worktree and her sealed shell runs
    # them (`exec`), so her ceiling mutates the environment; it was "nothing" before. The
    # `packet` port they first landed through is gone with the packet loop. `source`
    # and `edits` since the edit tools write through `edits` and say they mutate source (D196
    # §10). T2 and `desk`: in the task loop her shell asks
    # above T0, and the desk is her plan, delegations and reviews.
    "athena": Ceiling(
        "source",
        Tier.T2,
        frozenset(
            {
                "desk",
                "docs",
                "files",
                "edits",
                "exec",
                "navigator",
                "graph",
                "handles",
                "journal",
                "design_notes",
                "exec_request",
                "mcp_request",
                "search",
            }
        ),
        ALL,
    ),
    # The task worker edits the worktree and its shell asks above T0.
    # `search`: the scout's file listing, offered to both box roles.
    "worker": Ceiling(
        "source",
        Tier.T2,
        frozenset(
            {"files", "navigator", "graph", "handles", "exec", "edits", "search", "finish"}
        ),
        ALL,
    ),
    "locate": Ceiling(
        "nothing",
        Tier.T0,
        frozenset({"files", "navigator", "handles", "search", "exec_request", "mcp"}),
        ALL,
    ),
    "critic": Ceiling("nothing", Tier.T0, frozenset({"files", "navigator", "handles"}), ALL),
    # The agents read the repository; the reviewer also runs T0 commands.
    "reader": Ceiling(
        "nothing", Tier.T0, frozenset({"files", "navigator", "handles", "search"}), ALL
    ),
    "reviewer": Ceiling(
        "environment",
        Tier.T0,
        frozenset({"files", "navigator", "handles", "search", "exec"}),
        ALL,
    ),
    # The researcher reads the web, and nothing of the repository.
    "researcher": Ceiling("nothing", Tier.T0, frozenset({"web"}), FileRule("packet", "hidden")),
    "residue": Ceiling("nothing", Tier.T0, frozenset(), FileRule("packet", "hidden")),
    "checkpoint": Ceiling("nothing", Tier.T0, frozenset(), FileRule("packet", "hidden")),
    # A session's name, from the words it is given, and nothing else.
    "title": Ceiling("nothing", Tier.T0, frozenset(), FileRule("packet", "hidden")),
}


def test_each_roles_ceiling_is_the_table() -> None:
    assert {r.name: r.tool_ceiling for r in ROLES} == CEILINGS


@pytest.mark.parametrize("name", [r.name for r in ROLES if r.tool_defaults is not None])
def test_each_roles_default_is_within_its_ceiling(name: str) -> None:
    r = role(name)
    assert r.tool_defaults is not None
    _, refusals = resolve(
        r.tool_defaults, r.tool_ceiling, None, model=None, known=BUILTINS, own=OWN
    )
    assert refusals == ()


def test_the_defaults_are_todays_lists() -> None:
    assert role("zeus").tool_defaults.enabled == tuple(s.name for s in SURFACE)  # type: ignore[union-attr]
    worker = role("worker").tool_defaults
    assert worker is not None
    # The task worker's list, native calls only; it was the
    # packet worker's five, shown under its packet tools' short names, over the packet's
    # graph-visible files one hop deep, on `prompt_simple`.
    assert [worker.shown(n) for n in worker.enabled] == [
        "read",
        "grep_in",
        "grep",
        "glob",
        "symbol",
        "neighbors",
        "impact",
        "tests_for",
        "edit",
        "create",
        "delete",
        "move",
        "shell",
        "finish",
    ], "D74's shell, on by default (T-3.10, owner 2026-10-08)"
    assert worker.files == ALL, "D196 §9.1: the worktree's files, tests readable (D167)"
    assert worker.graph_depth is None
    assert role("zeus").tool_defaults.pipeline.slots == (  # type: ignore[union-attr]
        "findings",
        "excess_coder",
        "repetition",
    )


def _refusals(
    name: str, selection: Selection, ceiling: Ceiling | None = None
) -> tuple[str, ...]:
    r = role(name)
    assert r.tool_defaults is not None
    return resolve(
        r.tool_defaults,
        ceiling or r.tool_ceiling,
        selection,
        model=None,
        known=BUILTINS,
        own=OWN,
    )[1]


def test_an_unknown_name_is_refused() -> None:
    (why,) = _refusals("locate", Selection(enabled=("read", "ripgrep")))
    assert "'ripgrep' is not a tool" in why


def test_a_tool_that_mutates_beyond_the_ceiling_is_refused() -> None:
    why = _refusals("locate", Selection(enabled=("shell",)))
    # Every refusal is returned, not the first: registered since T-3.115,
    # `shell` is also refused for the `exec` port the scout is never given.
    assert any("mutates environment" in w for w in why)


def test_a_fixed_tier_above_the_ceiling_is_refused_and_a_computed_one_fits() -> None:
    from dataclasses import replace

    from chickadee.tools.policy import within

    assert not within(Tier.T1, Tier.T0)
    assert within(Tier.T0, Tier.T0)
    assert within("computed", Tier.T0), "the worker's shell: refused above T0 per call (D74)"
    spec = replace(OWN["read"], name="publish", tier=Tier.T1)
    r = role("locate")
    assert r.tool_defaults is not None
    _, why = resolve(
        replace(r.tool_defaults, enabled=("publish",)),
        r.tool_ceiling,
        None,
        model=None,
        known=BUILTINS,
        own={"publish": spec},
    )
    assert any("is tier T1" in w for w in why)


def test_a_tool_needing_a_port_outside_the_ceiling_is_refused() -> None:
    narrow = Ceiling("nothing", Tier.T0, frozenset({"navigator"}), ALL)
    (why,) = _refusals("locate", Selection(enabled=("read",)), narrow)
    assert "needs ports ['files']" in why


def test_files_wider_than_the_ceiling_are_refused() -> None:
    narrow = Ceiling("nothing", Tier.T0, PORTS, FileRule("graph_visible", "visible"))
    (why,) = _refusals("locate", Selection(files=ALL), narrow)
    assert "wider than the role may see" in why


def test_two_tools_shown_under_one_name_are_refused() -> None:
    why = _refusals(
        "locate", Selection(enabled=("grep", "grep_in"), shown_as={"grep": "grep_in"})
    )
    assert any("shown as ['grep_in']" in w for w in why)


def test_a_shown_name_for_a_tool_not_enabled_is_refused() -> None:
    (why,) = _refusals("locate", Selection(enabled=("grep",), shown_as={"read": "cat"}))
    assert "not enabled" in why


def test_a_confirmation_without_a_person_is_refused() -> None:
    r = role("worker")
    assert r.tool_defaults is not None
    asks = Pipeline(admit=(AskWhen(lambda s: True, lambda p: Tier.T1), Valid()))
    policy = RolePolicy(
        "worker", r.tool_defaults.enabled, {}, r.tool_defaults.files, 1, asks, "prompt_simple"
    )
    _, why = resolve(policy, r.tool_ceiling, None, model=None, known=BUILTINS, own=OWN)
    assert any("no user port" in w for w in why)


def test_a_models_section_replaces_only_the_keys_it_names() -> None:
    r = role("locate")
    assert r.tool_defaults is not None
    chosen = Selection(
        enabled=("read", "grep"),
        files=ALL,
        models={"small": Selection(enabled=("read",), shown_as={"read": "cat"})},
    )
    small, _ = resolve(
        r.tool_defaults, r.tool_ceiling, chosen, model="small", known=BUILTINS, own=OWN
    )
    other, _ = resolve(
        r.tool_defaults, r.tool_ceiling, chosen, model="large", known=BUILTINS, own=OWN
    )
    assert (small.enabled, small.shown("read"), small.files) == (("read",), "cat", ALL)
    assert (other.enabled, other.shown("read")) == (("read", "grep"), "read")


# -- the config -----------------------------------------------------------------------------


def test_a_selection_is_refused_until_its_roles_loop_reads_it() -> None:
    with pytest.raises(
        ConfigError, match=r"`\[tools\.critic\]` is not read yet.*the critic task"
    ):
        EngineConfig.with_defaults(raw(critic={"enabled": []}))


def test_athena_s_selection_is_read() -> None:
    """T-4.2 binds her loop to the dispatcher, so `[tools.athena]` loads and is her
    policy; a `files` rule is refused, since no port of hers narrows by it."""
    config = EngineConfig.with_defaults(raw(athena={"enabled": ["read", "note"]}))
    assert config.tool_policy("athena").enabled == ("read", "note")
    with pytest.raises(ConfigError, match="athena"):
        EngineConfig.with_defaults(raw(athena={"files": {"scope": "packet"}}))


def test_the_worker_s_selection_is_read() -> None:
    """T-3.111 composes the worker in production (`box.compose.authoring`), so
    `[tools.worker]` loads and is the run's policy: what is offered, parsed and listed."""
    config = EngineConfig.with_defaults(raw(worker={"enabled": ["read", "edit"]}))
    policy = config.tool_policy("worker")
    assert policy.enabled == ("read", "edit")
    assert policy.shown_names == ("read", "edit")


def test_zeus_s_selection_is_read() -> None:
    """T-3.115 moved Zeus's loop onto the dispatcher, so `[tools.zeus]` loads and is its
    policy: the prefix lists it and the loop dispatches it."""
    config = EngineConfig.with_defaults(
        raw(zeus={"enabled": ["read", "grep"], "shown_as": {"grep": "search"}})
    )
    policy = config.tool_policy("zeus")
    assert [s.name for s in policy.surface(BUILTINS)] == ["read", "search"]


@pytest.mark.parametrize(
    "chosen",
    [
        {"files": {"scope": "graph_visible", "tests": "visible"}},
        {"models": {"high-param-model": {"files": {"scope": "repository", "tests": "hidden"}}}},
    ],
)
def test_a_zeus_files_rule_is_refused_because_no_port_narrows_it(
    chosen: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError, match=r"`tools\.zeus\.files` is not read"):
        EngineConfig.with_defaults(raw(zeus=chosen))


def test_the_locate_selection_is_read() -> None:
    """`scout_locator` hands the scout `config.tool_policy("locate")`, so
    `[tools.locate]` loads (the scout's side is `tests/scout/test_tools.py`)."""
    config = EngineConfig.with_defaults(raw(locate={"enabled": ["read", "glob"]}))
    assert config.tool_policy("locate").enabled == ("read", "glob")


@pytest.mark.parametrize("name", ["worker", "locate"])
def test_a_files_rule_is_refused_where_no_port_narrows_it(name: str) -> None:
    with pytest.raises(ConfigError, match=rf"`tools\.{name}\.files` is not read"):
        EngineConfig.with_defaults(
            raw(**{name: {"files": {"scope": "graph_visible", "tests": "hidden"}}})
        )


def test_a_bad_arm_is_refused_for_its_own_reason_first() -> None:
    with pytest.raises(ConfigError, match=r"tools\.locate: 'shell' mutates environment"):
        EngineConfig.with_defaults(raw(locate={"enabled": ["read", "shell"]}))


def test_a_models_section_must_name_a_catalog_model() -> None:
    with pytest.raises(ConfigError, match=r"tools\.locate\.models names \['nope'\]"):
        EngineConfig.with_defaults(raw(locate={"models": {"nope": {"enabled": ["read"]}}}))


def test_a_models_section_is_checked_on_its_own() -> None:
    bad = {"models": {"mid-param-model": {"enabled": ["shell"]}}}
    with pytest.raises(
        ConfigError, match=r'tools\.locate\.models\."mid-param-model": \'shell\''
    ):
        EngineConfig.with_defaults(raw(locate=bad))


def test_a_role_with_no_tools_has_nothing_to_choose() -> None:
    with pytest.raises(ConfigError, match="has no tools to choose"):
        EngineConfig.with_defaults(raw(residue={"enabled": []}))


def test_an_unchosen_table_loads_and_the_policy_is_the_default() -> None:
    config = EngineConfig.with_defaults(raw(worker={}))
    assert config.tool_policy("worker") == role("worker").tool_defaults


def test_narrowing_the_list_drops_the_defaults_names_for_tools_it_leaves_out() -> None:
    """A default that shows `grep_in` as `search`; an
    arm without it is not refused for a name it never asked for."""
    r = role("worker")
    assert r.tool_defaults is not None
    default = replace(r.tool_defaults, shown_as={"read": "cat", "grep_in": "search"})
    policy, why = resolve(
        default,
        r.tool_ceiling,
        Selection(enabled=("read",)),
        model=None,
        known=BUILTINS,
        own=OWN,
    )
    assert why == () and dict(policy.shown_as) == {"read": "cat"}


# -- the worker's edit choices ------------------

WORKER_MODEL = f"{role('worker').default_rung}-model"
EDIT = ["read", "edit"]
LINES = ["read", "replace_lines"]
PATCH = ["read", "apply_patch"]
READS = ["read"]
"""An arm with no edit tool: `edit_check` is read in none of its views."""


def _worker(**keys: Any) -> EngineConfig:
    return EngineConfig.with_defaults(raw(worker=keys))


def _worker_on(endpoint: dict[str, Any], **keys: Any) -> EngineConfig:
    loaded = raw(worker=keys)
    loaded["models"][WORKER_MODEL].update(endpoint)
    return EngineConfig.with_defaults(loaded)


def test_the_edit_choices_default_to_the_table() -> None:
    """Moved: the protocol and the reply format are no longer choices, since every
    role calls its tools natively and every edit is a tool call."""
    policy = _worker().tool_policy("worker")
    assert (policy.line_anchors, policy.edit_check, policy.constrain) == (
        "number",
        "report",
        (),
    )


def test_each_edit_choice_loads_where_something_offered_reads_it() -> None:
    # T-3.159, a moved answer: `hash` was the choice this loaded; it is now refused
    # (below), so `number`, set explicitly, is what a line tool's table loads.
    lines = _worker(enabled=LINES, line_anchors="number", edit_check="report")
    policy = lines.tool_policy("worker")
    assert (policy.line_anchors, policy.edit_check) == ("number", "report")
    constrained = _worker_on({"grammar_tools": True}, enabled=PATCH, constrain=["apply_patch"])
    assert constrained.tool_policy("worker").constrain == ("apply_patch",)


@pytest.mark.parametrize(
    ("keys", "said"),
    [
        (
            {"enabled": LINES, "line_anchors": "sha"},
            r"line_anchors 'sha' is not a known name",
        ),
        ({"edit_check": "warn"}, r"edit_check"),
    ],
)
def test_a_name_no_registry_holds_is_refused(keys: dict[str, Any], said: str) -> None:
    with pytest.raises(ConfigError, match=said):
        _worker(**keys)


@pytest.mark.parametrize(
    ("key", "value", "why"),
    [
        ("tool_protocol", "native", "calls its tools natively"),
        ("reply_format", "whole_file", "every edit is an edit-tool call"),
    ],
)
@pytest.mark.parametrize("name", ["worker", "zeus"])
def test_a_key_the_rebuild_retired_is_refused(
    name: str, key: str, value: str, why: str
) -> None:
    """D196 §5.1, `T-3.157` item 7: refused with the reason, on any role's table or section,
    never loaded and ignored."""
    with pytest.raises(ConfigError, match=rf"tools\.{name}\.{key}` is no longer read: .*{why}"):
        EngineConfig.with_defaults(raw(**{name: {key: value}}))
    section = {"models": {"mid-param-model": {key: value}}}
    with pytest.raises(ConfigError, match=r"since the rebuild"):
        EngineConfig.with_defaults(raw(**{name: section}))


def test_a_model_a_tool_role_runs_on_without_native_tools_is_refused() -> None:
    """No prompt protocol reads a call from text any more, so such a model could
    call no tool. Refused naming the model and the roles."""
    with pytest.raises(ConfigError, match=rf"'{WORKER_MODEL}' \(for worker\)"):
        _worker_on({"native_tools": False})


def test_hash_anchors_are_refused_for_the_worker_naming_why() -> None:
    """The worker's `read` shows numbers, never hash tags, so a hash anchor
    could not be copied from a view. Refused at load, on the table and in a section."""
    with pytest.raises(ConfigError, match=r"line_anchors 'hash' is refused: .*line numbers"):
        _worker(enabled=LINES, line_anchors="hash")
    with pytest.raises(ConfigError, match=r'models\."mid-param-model": line_anchors \'hash\''):
        _worker(models={"mid-param-model": {"enabled": LINES, "line_anchors": "hash"}})


def test_line_anchors_are_refused_while_no_line_tool_is_offered() -> None:
    with pytest.raises(ConfigError, match=r"line_anchors is set and nothing offered reads"):
        _worker(enabled=EDIT, line_anchors="hash")


def test_an_edit_check_is_refused_while_no_edit_tool_is_offered() -> None:
    with pytest.raises(ConfigError, match=r"edit_check is set and nothing offered reads"):
        _worker(enabled=READS, edit_check="off")


def test_constrain_names_an_offered_tool_with_a_grammar() -> None:
    with pytest.raises(ConfigError, match=r"'apply_patch', which is not offered"):
        _worker_on({"grammar_tools": True}, constrain=["apply_patch"])
    with pytest.raises(ConfigError, match=r"'edit', whose spec carries no grammar"):
        _worker_on({"grammar_tools": True}, enabled=EDIT, constrain=["edit"])


def test_constrain_needs_a_model_with_grammar_tools() -> None:
    with pytest.raises(ConfigError, match=r"constrain needs grammar_tools"):
        _worker(enabled=PATCH, constrain=["apply_patch"])


def test_a_capability_is_never_refused_for_being_unused() -> None:
    _worker_on({"native_tools": True, "grammar_tools": True})


def test_every_refusal_names_its_key_at_once() -> None:
    with pytest.raises(ConfigError) as caught:
        _worker(enabled=READS, edit_check="off", line_anchors="hash")
    assert "edit_check is set" in str(caught.value)
    assert "line_anchors is set" in str(caught.value)


def test_a_models_section_merges_its_edit_choices_key_by_key() -> None:
    config = _worker(
        enabled=LINES,
        line_anchors="number",
        models={WORKER_MODEL: {"edit_check": "off"}},
    )
    policy = config.tool_policy("worker")
    assert (policy.line_anchors, policy.edit_check) == ("number", "off")


def test_a_models_section_is_judged_against_its_own_models_capabilities() -> None:
    section = {"enabled": PATCH, "constrain": ["apply_patch"]}
    with pytest.raises(
        ConfigError, match=r'tools\.worker\.models\."mid-param-model": constrain needs'
    ):
        _worker(models={"mid-param-model": section})


def test_a_tables_choice_read_only_by_a_models_section_is_not_refused() -> None:
    """The role's table offers no line tool, but the section for another model does: a
    `line_anchors` there is the section's default, and read."""
    config = _worker(line_anchors="number", models={"mid-param-model": {"enabled": LINES}})
    assert config.tool_policy("worker").line_anchors == "number"


def test_a_tables_choice_every_reader_replaces_is_refused() -> None:
    """Review: the one section that reads `line_anchors` sets its own, so the table's is read
    nowhere."""
    section = {"enabled": LINES, "line_anchors": "number"}
    with pytest.raises(ConfigError, match=r"tools\.worker: line_anchors is set"):
        _worker(line_anchors="hash", models={"mid-param-model": section})


def test_the_tables_view_is_judged_against_the_model_its_rung_runs() -> None:
    with pytest.raises(ConfigError, match=r"tools\.worker: constrain needs grammar_tools"):
        _worker_on({"grammar_tools": False}, enabled=PATCH, constrain=["apply_patch"])
    # The running model's own section is the view that runs; the table is not judged by it.
    _worker_on(
        {"grammar_tools": False},
        enabled=PATCH,
        constrain=["apply_patch"],
        models={WORKER_MODEL: {"constrain": []}},
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("line_anchors", "number"),
        ("edit_check", "off"),
        ("force_tool_call", True),
        ("calling", "python"),
    ],
)
def test_the_edit_choices_are_refused_on_a_role_that_does_not_edit(
    key: str, value: object
) -> None:
    with pytest.raises(ConfigError, match=rf"tools\.locate\.{key} is not read"):
        EngineConfig.with_defaults(raw(locate={key: value}))
    in_section = {"models": {"mid-param-model": {key: value}}}
    with pytest.raises(ConfigError, match=rf'tools\.locate\.models\."mid-param-model"\.{key}'):
        EngineConfig.with_defaults(raw(locate=in_section))


def test_a_new_tool_is_judged_by_its_spec_with_no_change_to_the_rule() -> None:
    """The rule reads `family`, `anchor` and `grammar` from what is offered: a tool it has
    never heard of is judged by the same three."""
    from dataclasses import replace

    from chickadee.tools.policy import Capabilities, Choices, choice_refusals

    base = BUILTINS.get("replace_lines")
    assert base is not None
    novel = replace(base, spec=replace(base.spec, name="splice_lines"))
    known = BUILTINS.with_dynamic([novel])
    worker = role("worker").tool_defaults
    assert worker is not None
    choices = Choices(styles=frozenset({"number"}))
    chosen = frozenset({"line_anchors"})
    offered = replace(worker, enabled=("splice_lines",))
    assert choice_refusals(offered, chosen, known, choices, Capabilities(False)) == ()
    absent = replace(worker, enabled=("read",))
    assert choice_refusals(absent, chosen, known, choices, Capabilities(False)) == (
        "worker: line_anchors is set and nothing offered reads it (it is read only when an "
        "offered tool has anchor 'line')",
    )


def test_athenas_task_policy_is_within_her_ceiling() -> None:
    # The reads, the map, the journal, her
    # note, her three task tools, the edit tools and her shell, all inside ATHENA_CEILING.
    from chickadee.roles import ATHENA_CEILING, ATHENA_TASK
    from chickadee.tools.builtin import BUILTINS
    from chickadee.tools.policy import resolve

    policy, refusals = resolve(ATHENA_TASK, ATHENA_CEILING, None, model=None, known=BUILTINS)
    assert refusals == ()
    assert {"plan", "delegate", "review", "edit", "shell", "note"} <= set(policy.shown_names)


def test_athena_is_told_of_a_repeated_call_and_her_tool_use_never_ends_on_it() -> None:
    """T-3.159 (a REPL run): she ran `find "test"` three times and the same shell command
    three times, told nothing. Her repetition rule notes the second identical call and refuses
    from the fourth (`BOUND_REACHED`), but never ends her tool use, which would strand an open
    task, as the worker's `on_cap="stop"` does."""
    from chickadee.roles import ATHENA_TASK
    from chickadee.tools.catalogue import Repetition

    [rule] = [r for r in ATHENA_TASK.pipeline.admit if isinstance(r, Repetition)]
    assert (rule.key, rule.on_cap) == ("athena.repeat", "refuse")


def test_the_registry_says_which_ports_a_set_of_tools_reaches() -> None:
    """What decides an agent's shell, and an agent file's profile: the `exec` port a
    tool needs, never its name."""
    assert "exec" in BUILTINS.needs(("read", "shell"))
    assert "exec" not in BUILTINS.needs(("read", "grep", "tests_for"))
    assert BUILTINS.needs(("no-such-tool",)) == frozenset()


def test_force_tool_call_loads_on_a_model_without_forced_tools() -> None:
    """Never refused for the capability; such a model runs unforced."""
    policy = _worker_on({"forced_tools": False}, force_tool_call=True).tool_policy("worker")
    assert policy.force_tool_call is True
    assert _worker().tool_policy("worker").force_tool_call is False


# -- how the worker calls its tools ----------------------------------------


def test_the_worker_calls_natively_unless_the_config_says_python() -> None:
    assert _worker().tool_policy("worker").calling == "native"
    assert _worker(calling="python").tool_policy("worker").calling == "python"


def test_calling_is_chosen_per_model() -> None:
    """A small model gets the Python arm while the role's table keeps native calls."""
    on_this = _worker(models={WORKER_MODEL: {"calling": "python"}})
    assert on_this.tool_policy("worker").calling == "python"
    elsewhere = _worker(models={"high-param-model": {"calling": "python"}})
    assert elsewhere.tool_policy("worker").calling == "native"


@pytest.mark.parametrize("name", ["zeus", "athena"])
def test_calling_python_is_the_workers_alone(name: str) -> None:
    with pytest.raises(ConfigError, match=rf"tools\.{name}\.calling is not read"):
        EngineConfig.with_defaults(raw(**{name: {"calling": "python"}}))


def test_calling_python_with_a_forced_call_is_refused() -> None:
    """The arm sends no native tool, so there is none to force."""
    with pytest.raises(ConfigError, match="force_tool_call has nothing to force"):
        _worker(calling="python", force_tool_call=True)
    _worker(calling="native", force_tool_call=True)


def test_a_block_with_no_sandbox_runs_only_where_the_person_said() -> None:
    assert EngineConfig.with_defaults(raw()).sandbox.allow_unsandboxed_python is False
    loaded = {**raw(), "sandbox": {"allow_unsandboxed_python": True}}
    assert EngineConfig.with_defaults(loaded).sandbox.allow_unsandboxed_python is True
