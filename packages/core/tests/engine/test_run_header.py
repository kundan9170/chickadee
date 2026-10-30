"""The run header: where a session runs, and every guarantee it runs
without, stated from recorded facts only — and only while each fact holds.

Sessions and run journals are written here through their own writers, exactly as the loop and
the box write them, because the header is a projection of those records and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
from repos import init_repo

import chickadee.context.store as store
import chickadee.isolation.base as isolation_base
import chickadee.run_header as run_header
from chickadee.config import EngineConfig
from chickadee.context.lines import EpisodeLine
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.engine import IN_PLACE_NO_BOOTSTRAP, NO_BOOTSTRAP, PacketEngine
from chickadee.errors import SessionError
from chickadee.journal.runs import RunProjections
from chickadee.journal.writer import Journal
from chickadee.run_header import (
    BOOTSTRAP_FAILED,
    HeaderFacts,
    HeaderLine,
    RunHeader,
    header,
    project,
)
from chickadee.schema import EventKind, ToolCall
from chickadee.schema.adapter import Usage
from chickadee.schema.execution import SandboxLevel
from chickadee.schema.journal import (
    EffectCompletedP,
    EffectIntendedP,
    ExecP,
    RunStartP,
    SecretSuspectedP,
)
from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots

SHA = "a" * 64
RUN_A = "2026-09-30-0000000a"
RUN_B = "2026-09-30-0000000b"
SID = "S-header"


# -- fixtures -------------------------------------------------------------------------------


@pytest.fixture
def where(tmp_path: Path) -> Roots:
    base = tmp_path / "state"
    return Roots(
        project=base / "project",
        repo_local=base / "repo_local",
        checkout_local=base / "checkout_local",
        user=base / "user",
    )


@pytest.fixture
def config(raw: dict[str, Any]) -> EngineConfig:
    return EngineConfig.with_defaults(raw)


@pytest.fixture
def session(where: Roots) -> Iterator[Session]:
    held = Session.create(sessions_root(where), SID)
    yield held
    held.close()


def worktree_ref(**over: Any) -> WorkspaceRef:
    fields: dict[str, Any] = {
        "mode": "worktree",
        "lease_id": "repo-1:4242",
        "source_ref": "main",
        "source_head": "abc123def4567890",
        "branch": "chickadee/fix-fetcher-0000000a",
        "worktree_path": "/repo/.git/chickadee/worktrees/2026-09-30-0000000a",
        "handoff": "pending",
    }
    fields.update(over)
    return WorkspaceRef(**fields)


def in_place_ref() -> WorkspaceRef:
    return worktree_ref(mode="in_place", branch=None, worktree_path=None)


def open_episode(session: Session, run_id: str) -> None:
    """A `coder` call dispatched, and the episode it opened."""
    session.record_user(f"do {run_id}")
    call = ToolCall(
        id=f"c-{run_id}", name="coder", arguments={"title": run_id, "brief": run_id}
    )
    turn = session.record_assistant("", [call])
    session.record_episode(EpisodeLine(run_id, "planned", None, SHA, (), turn.seq))


def run_start() -> RunStartP:
    return RunStartP(
        goal="fix the fetcher",
        work_sha=SHA,
        config_sha=SHA,
        repo_head=SHA,
        engine_version="0",
        schema_version=2,
        isolation="worktree",
        hard_cost_cap_usd=5.0,
    )


def bootstrap_intended(run_id: str) -> EffectIntendedP:
    return EffectIntendedP(
        operation_id=f"bootstrap:{run_id}",
        effect_kind="bootstrap",
        approved_sha=SHA,
        preconditions=(),
        idempotent=True,
        declared_surface=(),
    )


def exec_event(*, sandbox: str = "native", network: bool = False) -> ExecP:
    return ExecP(
        operation_id="op-1",
        policy="worktree",
        label="pytest",
        argv_sha=SHA,
        cwd=".",
        exit_code=0,
        wall_ms=1,
        timed_out=False,
        sandbox=sandbox,
        network=network,
        stdout_sha=SHA,
        stderr_sha=SHA,
        violations=(),
    )


def suspected(path: str = "config/prod.env", line: int = 3) -> SecretSuspectedP:
    return SecretSuspectedP(path=path, line=line, detector="aws_access_key")


def journal(where: Roots, run_id: str, *events: tuple[EventKind, Any]) -> None:
    with Journal.create(where, run_id, {}) as j:
        j.append(EventKind.RUN_START, run_start())
        for kind, payload in events:
            j.append(kind, payload)


def append(where: Roots, run_id: str, kind: EventKind, payload: Any) -> None:
    """Write one more event to an existing run, as the box would mid-episode."""
    j = Journal.open_for_resume(where, run_id)
    with j:
        j.append(kind, payload)


def at(
    where: Roots,
    config: EngineConfig,
    level: SandboxLevel = SandboxLevel.NATIVE,
    *,
    accepted: bool = False,
) -> RunHeader:
    """The header over the list the engine would resolve: `accepted` stands for the loader's
    answer to the trust question (`Resolved.trust_accepted`), which the header never asks."""
    listed = config.secrets.denylist(accepted=accepted)
    return header(
        where,
        config,
        SID,
        level=level,
        denylist=listed,
        folds=ReplayCache(sessions_root(where)),
        runs=RunProjections(where),
        commands=config.command_table(accepted=accepted),
    )


def kinds(h: RunHeader) -> list[str]:
    return [line.kind for line in h.lines]


def only(h: RunHeader, kind: str) -> list[HeaderLine]:
    return [line for line in h.lines if line.kind == kind]


# -- the clean case -------------------------------------------------------------------------


def test_a_clean_worktree_session_at_a_native_sandbox_states_where_and_nothing_else(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EFFECT_INTENDED, bootstrap_intended(RUN_A)))

    h = at(where, config)

    assert kinds(h) == ["workspace", "sandbox", "models"]
    (ws,) = only(h, "workspace")
    assert "/repo/.git/chickadee/worktrees/2026-09-30-0000000a" in ws.text
    assert "chickadee/fix-fetcher-0000000a" in ws.text
    assert "repo-1:4242" in ws.text  # the lease owner
    assert "abc123def456" in ws.text  # the base commit
    (sb,) = only(h, "sandbox")
    assert sb.text.startswith("sandbox: native")
    assert all(line.source for line in h.lines)


def test_before_any_episode_the_workspace_line_says_none_is_taken_yet(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["workspace"] = {"mode": "worktree"}
    session.record_user("hello")
    h = at(where, EngineConfig.with_defaults(raw))
    assert kinds(h) == ["workspace", "sandbox", "models"]
    assert "none taken yet" in only(h, "workspace")[0].text
    assert "dedicated worktree" in only(h, "workspace")[0].text


def test_before_any_episode_in_place_the_header_says_the_checkout(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    """In place is the default, and its header still says the isolation is
    reduced (verification depth never overstated)."""
    session.record_user("hello")
    h = at(where, config)
    assert "your checkout, in place" in only(h, "workspace")[0].text
    assert "isolation" in kinds(h)


# -- isolation and bootstrap ----------------------------------------------------------------


def test_an_in_place_session_adds_isolation_and_bootstrap(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(in_place_ref())
    h = at(where, config)
    assert kinds(h) == ["workspace", "isolation", "sandbox", "bootstrap", "models"]
    assert "in place" in only(h, "workspace")[0].text
    assert only(h, "bootstrap")[0].text == IN_PLACE_NO_BOOTSTRAP


def test_in_place_config_states_both_before_any_workspace_is_taken(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["workspace"] = {"mode": "in_place"}
    h = at(where, EngineConfig.with_defaults(raw))
    assert "isolation" in kinds(h)
    assert only(h, "bootstrap")[0].text == IN_PLACE_NO_BOOTSTRAP


def test_a_worktree_bootstrap_that_ran_nothing_is_stated_once_the_episode_is_past_it(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A)  # only run_start: the bootstrap may still be about to run
    assert "bootstrap" not in kinds(at(where, config))

    append(where, RUN_A, EventKind.EXEC, exec_event())
    assert only(at(where, config), "bootstrap")[0].text == NO_BOOTSTRAP


def completed(run_id: str, outcome: Literal["succeeded", "failed"]) -> EffectCompletedP:
    return EffectCompletedP(
        operation_id=f"bootstrap:{run_id}", outcome=outcome, observed_surface=()
    )


def test_a_bootstrap_still_running_is_not_yet_a_bootstrap_that_ran_nothing(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EFFECT_INTENDED, bootstrap_intended(RUN_A)))
    assert "bootstrap" not in kinds(at(where, config))


def test_a_failed_bootstrap_says_the_worktree_was_not_prepared(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(
        where,
        RUN_A,
        (EventKind.EFFECT_INTENDED, bootstrap_intended(RUN_A)),
        (EventKind.EFFECT_COMPLETED, completed(RUN_A, "failed")),
    )
    assert only(at(where, config), "bootstrap")[0].text == BOOTSTRAP_FAILED


def test_a_bootstrap_in_any_episode_prepared_the_session_worktree(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(
        where,
        RUN_A,
        (EventKind.EFFECT_INTENDED, bootstrap_intended(RUN_A)),
        (EventKind.EFFECT_COMPLETED, completed(RUN_A, "succeeded")),
    )
    session.record_tool(f"c-{RUN_A}", f"run {RUN_A} SOLVED", run_id=RUN_A)
    open_episode(session, RUN_B)
    journal(where, RUN_B, (EventKind.EXEC, exec_event()))
    assert "bootstrap" not in kinds(at(where, config))


# -- sandbox --------------------------------------------------------------------------------


def test_sandbox_none_says_reads_are_unmasked_and_escapes_are_off(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["sandbox"] = {"require_sandbox": False}
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "reads unmasked" in sb.text
    assert "escapes are off" in sb.text
    assert "EX-2" in sb.source


def test_sandbox_none_with_unsandboxed_escapes_says_they_run_unsandboxed(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """The person let the scout's escapes run with no sandbox, and the header is
    where they are told, every time — the model never is."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_escapes": True}
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "reads unmasked" in sb.text
    assert "escapes run unsandboxed" in sb.text
    assert "escapes are off" not in sb.text
    assert "D134" in sb.source


def test_unsandboxed_escapes_are_not_claimed_where_a_sandbox_now_wraps_them(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """An earlier command ran at `none`, so the line says so; but the machine has a sandbox
    now, and an escape would run under it — the line must not say it runs unsandboxed."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_escapes": True}
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event(sandbox="none")))
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NATIVE), "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "run unsandboxed" not in sb.text


def test_unsandboxed_escapes_are_not_claimed_where_require_sandbox_refuses_them(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """A command ran at `none` earlier, and the config now sets `require_sandbox`: every
    command, escapes included, is refused, so the line must not say they run."""
    raw["sandbox"] = {"require_sandbox": True, "allow_unsandboxed_escapes": True}
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event(sandbox="none")))
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert "run unsandboxed" not in sb.text


def test_unsandboxed_escapes_add_nothing_where_a_sandbox_wraps_them(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_escapes": True}
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NATIVE), "sandbox")
    assert "escapes" not in sb.text


def test_sandbox_none_under_require_sandbox_says_commands_refuse_to_run(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    """EX-2's default: nothing runs unwrapped, so the header must not say everything does."""
    (sb,) = only(at(where, config, SandboxLevel.NONE), "sandbox")
    assert "require_sandbox" in sb.text
    assert "refuses to run" in sb.text
    assert "reads unmasked" not in sb.text


@pytest.mark.parametrize(
    ("sandbox", "says", "never"),
    [
        (
            {"require_sandbox": True},
            "every command refuses to run",
            "Zeus's shell",
        ),
        (
            {"require_sandbox": False},
            "Zeus's, the worker's and Athena's shells ask before every command, and run none "
            "nobody confirms",
            "unprotected",
        ),
        (
            {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}},
            "Zeus's shell runs commands unprotected (sandbox.allow_unsandboxed_shell.zeus); "
            "the worker's and Athena's shells ask before every command",
            "Zeus's shell asks",
        ),
        (
            {"require_sandbox": False, "allow_unsandboxed_shell": {"worker": True}},
            "the worker's shell runs commands unprotected "
            "(sandbox.allow_unsandboxed_shell.worker); Zeus's and Athena's shells ask",
            "Zeus's shell runs",
        ),
        (
            {"require_sandbox": False, "never_ask": True},
            "every shell runs commands unprotected and unasked (sandbox.never_ask)",
            "asks before",
        ),
    ],
    ids=["require_sandbox", "asks", "lifted for zeus", "lifted for the worker", "never_ask"],
)
def test_sandbox_none_names_what_holds_for_each_shell(
    where: Roots,
    raw: dict[str, Any],
    session: Session,
    sandbox: dict[str, object],
    says: str,
    never: str,
) -> None:
    """D137 decision 5, per role since item 8: at `none` the person is told what each
    role's shell will do, every time."""
    raw["sandbox"] = sandbox
    lines = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    sb = lines[-1]
    assert sb.text.startswith("sandbox: none")
    assert says in sb.text
    assert never not in sb.text


def test_never_ask_is_said_first_whatever_the_sandbox(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """D199 item 8b: the line that says nothing is asked leads the sandbox lines, with what
    still refuses."""
    raw["sandbox"] = {"never_ask": True}
    first, *_ = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NATIVE), "sandbox")
    assert first.text.startswith("sandbox: no shell command is asked (sandbox.never_ask)")
    assert "deny rules and secret paths still refuse" in first.text
    raw["sandbox"] = {}
    (only_line,) = only(
        at(where, EngineConfig.with_defaults(raw), SandboxLevel.NATIVE), "sandbox"
    )
    assert "never_ask" not in only_line.text


def test_a_command_ran_unwrapped_and_require_sandbox_now_says_every_command_refuses(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """The first of D137's three where a command already ran at `none`: the line says that,
    and what the next command does, which is refuse."""
    raw["sandbox"] = {"require_sandbox": True, "allow_unsandboxed_shell": {"zeus": True}}
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event(sandbox="none")))
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "every command now refuses to run (require_sandbox)" in sb.text
    assert "Zeus's shell" not in sb.text


def test_zeus_shell_is_not_claimed_unprotected_where_a_sandbox_now_wraps_it(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """An earlier command ran at `none`, so the line says so; the machine has a sandbox now,
    and Zeus's next command runs under it — sealed, unasked, and not unprotected."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event(sandbox="none")))
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NATIVE), "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "Zeus's shell" not in sb.text


def test_the_level_commands_ran_under_wins_over_the_level_detected_now(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event(sandbox="native")))
    (sb,) = only(at(where, config, SandboxLevel.CONTAINER), "sandbox")
    assert sb.text.startswith("sandbox: native")


@pytest.mark.parametrize(("backend", "asked"), [("auto", None), ("native", "native")])
def test_the_detected_level_is_the_configured_backends(
    monkeypatch: pytest.MonkeyPatch, backend: str, asked: str | None
) -> None:
    """The header reads `sandbox.backend` as every engine spawn site does, through
    `isolation.base.configured`, so it names the backend the commands run under."""
    seen: list[object] = []

    class Native:
        def available(self) -> bool:
            return True

        def level(self) -> SandboxLevel:
            return SandboxLevel.NATIVE

    def fake_select(preferred: str | None = None) -> Native:
        seen.append(preferred)
        return Native()

    monkeypatch.setattr(isolation_base, "select", fake_select)
    assert run_header.detected_level(backend) is SandboxLevel.NATIVE
    assert seen == [asked]


def test_backend_none_says_none_even_where_a_sandbox_runs(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """With no level handed in, the header reads the configured backend: `none` is the
    passthrough whatever this machine has."""
    raw["sandbox"] = {"backend": "none", "require_sandbox": False}
    config = EngineConfig.with_defaults(raw)
    h = header(
        where,
        config,
        SID,
        denylist=config.secrets.denylist(accepted=False),
        folds=ReplayCache(sessions_root(where)),
        runs=RunProjections(where),
    )
    (sb,) = only(h, "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "asked for" not in sb.text, "none was asked for and none runs"


def test_a_backend_asked_for_and_not_available_says_what_runs_instead(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """`native` asked for where it does not run falls through to `none`, and the line
    names both, never silently."""
    raw["sandbox"] = {"backend": "native", "require_sandbox": False}
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert sb.text.startswith("sandbox: none")
    assert "asked for native; none runs" in sb.text


def test_auto_names_no_preference(where: Roots, config: EngineConfig, session: Session) -> None:
    (sb,) = only(at(where, config, SandboxLevel.NATIVE), "sandbox")
    assert "asked for" not in sb.text


def test_a_command_that_ran_unwrapped_is_stated_whatever_is_detected_now(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event(sandbox="none")))
    (sb,) = only(at(where, config, SandboxLevel.NATIVE), "sandbox")
    assert sb.text.startswith("sandbox: none")


def test_a_command_run_with_the_network_on_is_stated_only_once_one_ran(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EXEC, exec_event()))
    assert len(only(at(where, config), "sandbox")) == 1

    # A command the pre-filter refused never spawned, so it opened no network.
    refused = exec_event(network=True).model_copy(
        update={"exit_code": 126, "wall_ms": 0, "violations": (".env",)}
    )
    append(where, RUN_A, EventKind.EXEC, refused)
    assert len(only(at(where, config), "sandbox")) == 1

    append(where, RUN_A, EventKind.EXEC, exec_event(network=True))
    lines = only(at(where, config), "sandbox")
    assert len(lines) == 2
    assert lines[1].text == "sandbox: 1 command(s) ran with the network on, each approved"


# -- deny list ------------------------------------------------------------------------------


def test_a_config_removing_npmrc_names_it_and_agrees_with_the_list_in_both_states(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["secrets"] = {"deny_remove": [".npmrc"]}
    config = EngineConfig.with_defaults(raw)
    # Unaccepted, the run's list still denies it, and the header says so —
    # named anyway, so the person who wrote the removal learns why it did nothing.
    assert not config.secrets.denylist(accepted=False).readable(".npmrc")
    (inert,) = only(at(where, config), "deny_list")
    assert ".npmrc" in inert.text
    assert "inert until" in inert.text
    assert "still denied" in inert.text

    assert config.secrets.denylist(accepted=True).readable(".npmrc")
    (line,) = only(at(where, config, accepted=True), "deny_list")
    assert ".npmrc" in line.text
    assert "inert" not in line.text
    assert line.source == "D118 item 5"


def test_allow_secret_read_is_stated_loudly_only_when_in_force(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["secrets"] = {"allow_secret_read": True}
    config = EngineConfig.with_defaults(raw)
    (line,) = only(at(where, config, accepted=True), "deny_list")
    assert "allow_secret_read" in line.text
    assert line.source == "SE-8"
    # The sandbox line must not claim a mask the list no longer asks for.
    (sb,) = only(at(where, config, accepted=True), "sandbox")
    assert "nothing masked" in sb.text
    assert "deny-listed paths masked" not in sb.text

    # Unaccepted: the list is whole, so the mask holds and the request is marked inert.
    (inert,) = only(at(where, config), "deny_list")
    assert "inert until" in inert.text
    assert "deny-listed paths masked" in only(at(where, config), "sandbox")[0].text


def test_the_header_never_asks_the_trust_question_itself(
    where: Roots, raw: dict[str, Any], session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It reads the run's list; a second answer to the trust question could disagree."""
    import chickadee.state.trust as trust

    def refuse(*_args: object) -> bool:
        raise AssertionError("the header asked is_active itself")

    monkeypatch.setattr(trust, "is_active", refuse)
    assert not hasattr(run_header, "is_active")  # no name imported to call behind the patch
    raw["secrets"] = {"deny_remove": [".npmrc"]}
    at(where, EngineConfig.with_defaults(raw), accepted=True)


def test_the_repositorys_command_rows_are_named_inert_until_accepted(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """A repository's `[[commands]]` row is executable config. Unaccepted, the run's
    table does not hold it and the header names it beside the command that accepts it; the
    person's own row is never inert."""
    raw["commands"] = [
        {"prefix": ["deploy.sh"], "class": "push", "layer": "project"},
        {"prefix": ["reset-db.sh"], "class": "destructive"},
    ]
    config = EngineConfig.with_defaults(raw)
    (line,) = only(at(where, config), "commands")
    assert "deploy.sh" in line.text and "reset-db.sh" not in line.text
    assert "`chickadee trust`" in line.text
    assert line.source == "RB-27, 31-state.md §6"
    assert only(at(where, config, accepted=True), "commands") == []


def test_a_repeated_inert_removal_is_named_once(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    raw["secrets"] = {"deny_remove": [".npmrc", ".npmrc"]}
    (inert,) = only(at(where, EngineConfig.with_defaults(raw)), "deny_list")
    assert inert.text.count(".npmrc") == 1


# -- secrets --------------------------------------------------------------------------------


def test_a_secret_suspected_mid_session_is_on_the_next_call(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.EFFECT_INTENDED, bootstrap_intended(RUN_A)))
    assert "secrets" not in kinds(at(where, config))

    append(where, RUN_A, EventKind.SECRET_SUSPECTED, suspected())
    append(where, RUN_A, EventKind.SECRET_SUSPECTED, suspected())  # the same finding again
    (line,) = only(at(where, config), "secrets")
    assert "config/prod.env:3" in line.text
    assert line.source == "SE-11"


# -- the header is a projection -------------------------------------------------------------


def snapshot(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_header_adds_no_journal_event_and_writes_nothing(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    session.record_workspace(worktree_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.SECRET_SUSPECTED, suspected()))
    session.sync()
    before = snapshot(where.repo_local)
    at(where, config)
    at(where, config)
    assert snapshot(where.repo_local) == before


def test_unverified_carries_every_header_line(
    where: Roots, config: EngineConfig, session: Session
) -> None:
    """What `handoff().unverified` will carry: every line, in order, as text."""
    session.record_workspace(in_place_ref())
    open_episode(session, RUN_A)
    journal(where, RUN_A, (EventKind.SECRET_SUSPECTED, suspected()))
    h = at(where, config, SandboxLevel.NONE)
    assert h.unverified() == tuple(line.text for line in h.lines)
    assert len(h.unverified()) == len(h.lines) >= 5


def test_the_projection_is_a_function_of_the_facts_alone() -> None:
    facts = HeaderFacts(mode="worktree", level=SandboxLevel.CONTAINER)
    assert project(SID, facts) == project(SID, facts)
    assert kinds(project(SID, facts)) == ["workspace", "sandbox"]


def test_an_unknown_session_is_a_session_error(where: Roots, config: EngineConfig) -> None:
    with pytest.raises(SessionError, match="S-nope"):
        at_session = config.secrets.denylist(accepted=False)
        header(
            where,
            config,
            "S-nope",
            level=SandboxLevel.NATIVE,
            denylist=at_session,
            folds=ReplayCache(sessions_root(where)),
            runs=RunProjections(where),
        )


# -- through the facade ---------------------------------------------------------------------


def test_engine_header_reads_the_repositorys_session(
    tmp_path: Path, config: EngineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_repo(tmp_path / "repo", commit=None)
    where = resolve_roots(repo)
    held = Session.create(sessions_root(where), SID)
    try:
        held.record_workspace(worktree_ref())
    finally:
        held.close()
    monkeypatch.setattr(run_header, "detected_level", lambda _asked="auto": SandboxLevel.NATIVE)

    h = PacketEngine(repo, config).header(SID)

    assert h.session_id == SID
    assert kinds(h) == ["workspace", "sandbox", "models"]


def test_engine_header_outside_a_repository_is_a_session_error(
    tmp_path: Path, config: EngineConfig
) -> None:
    with pytest.raises(SessionError, match="not a git repository"):
        PacketEngine(tmp_path, config).header(SID)


# -- a lifted cap --------------------------------------------------------------------


def test_the_spend_line_is_projected_only_from_an_uncapped_fact() -> None:
    capped = HeaderFacts(mode="worktree", level=SandboxLevel.NATIVE)
    assert only(project(SID, capped), "spend") == []

    lifted = HeaderFacts(mode="worktree", level=SandboxLevel.NATIVE, uncapped=(SID, 7, 3.3))
    (line,) = only(project(SID, lifted), "spend")
    assert line.text == "spend: uncapped (since seq 7) — $3.30 spent"
    assert line.source == "D144"

    from_fork = HeaderFacts(
        mode="worktree", level=SandboxLevel.NATIVE, uncapped=("S-child", 4, 0.0)
    )
    (line,) = only(project(SID, from_fork), "spend")
    assert line.text == "spend: uncapped (since seq 4 in S-child) — $0.00 spent"


def test_engine_header_says_uncapped_while_the_cap_is_lifted_and_only_then(
    tmp_path: Path, config: EngineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_repo(tmp_path / "repo", commit=None)
    where = resolve_roots(repo)
    with Session.create(sessions_root(where), SID) as held:
        held.record_spend(
            "high-param-model", Usage(prompt_tokens=10, completion_tokens=1), 1.25, "high-param"
        )
    monkeypatch.setattr(run_header, "detected_level", lambda _asked="auto": SandboxLevel.NATIVE)
    # One second per line, so the family's latest `cap` line is decided by when it was written.
    clock = iter(f"2026-10-01T12:00:{s:02d}Z" for s in range(60))
    monkeypatch.setattr(store, "_now", lambda: next(clock))
    engine = PacketEngine(repo, config)

    assert only(engine.header(SID), "spend") == []

    lifted = engine.set_cap(SID, None)
    (line,) = only(engine.header(SID), "spend")
    assert line.text == f"spend: uncapped (since seq {lifted.cap_seq}) — $1.25 spent"

    engine.set_cap(SID, 4.0)
    assert only(engine.header(SID), "spend") == []


# -- a lifted rung -------------------------------------------------------------------


def test_a_lifted_rung_is_projected_with_its_own_line() -> None:
    facts = HeaderFacts(
        mode="worktree",
        level=SandboxLevel.NATIVE,
        uncapped_rungs=(("coder", SID, 9, 1.5), ("scout", "S-child", 4, 0.0)),
    )
    lines = only(project(SID, facts), "spend")
    assert [line.text for line in lines] == [
        "spend: coder uncapped (since seq 9) — $1.50 spent",
        "spend: scout uncapped (since seq 4 in S-child) — $0.00 spent",
    ]
    assert {line.source for line in lines} == {"D145"}


def test_engine_header_names_a_lifted_rung_only_while_it_is_lifted(
    tmp_path: Path, raw: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rung with a cap configured and never lifted says nothing, and nor does one with no
    cap at all: only a lift is news. The charge was written on `coder` in older versions, and
    counts
    on its successor."""
    repo = init_repo(tmp_path / "repo", commit=None)
    where = resolve_roots(repo)
    with Session.create(sessions_root(where), SID) as held:
        held.record_spend(
            "low-param-model", Usage(prompt_tokens=1, completion_tokens=1), 0.75, "coder"
        )
    monkeypatch.setattr(run_header, "detected_level", lambda _asked="auto": SandboxLevel.NATIVE)
    clock = iter(f"2026-10-02T12:00:{s:02d}Z" for s in range(60))
    monkeypatch.setattr(store, "_now", lambda: next(clock))
    raw["spend"] = {"hard_cost_usd": 5.0, "rungs": {"low-param": 2.0, "high-param": 3.0}}
    engine = PacketEngine(repo, EngineConfig.with_defaults(raw))

    assert only(engine.header(SID), "spend") == []

    lifted = engine.set_cap(SID, None, rung="low-param")
    (line,) = only(engine.header(SID), "spend")
    seq = lifted.rungs["low-param"].cap_seq
    assert line.text == f"spend: low-param uncapped (since seq {seq}) — $0.75 spent"

    engine.set_cap(SID, 2.0, rung="low-param")
    assert only(engine.header(SID), "spend") == []


def test_unsandboxed_python_blocks_are_named_where_they_run(
    where: Roots, raw: dict[str, Any], session: Session
) -> None:
    """A lift that runs the worker's blocks with no sandbox is said here,
    as every other `[sandbox]` lift is."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_python": True}
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert "Python blocks run unsandboxed (sandbox.allow_unsandboxed_python)" in sb.text
    raw["sandbox"] = {"require_sandbox": True, "allow_unsandboxed_python": True}
    (sb,) = only(at(where, EngineConfig.with_defaults(raw), SandboxLevel.NONE), "sandbox")
    assert "Python blocks" not in sb.text
