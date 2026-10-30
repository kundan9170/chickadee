"""Construction wires what exists, and refuses what it cannot honour."""

from __future__ import annotations

import pathlib
import threading
from typing import Any

import pytest

from chickadee.config import EngineConfig
from chickadee.engine import NonInteractive, PacketEngine
from chickadee.errors import HarnessBug
from chickadee.schema import InteractionPort
from chickadee.schema.interaction import AnswerSource, ApprovalStop, StopAnswer, StopDecision


@pytest.fixture
def config(raw: dict[str, Any]) -> EngineConfig:
    # `with_defaults`, not `model_validate`: the raw fixture omits D48's per-rung
    # tool-call caps, and a config missing them is rejected at load by design.
    return EngineConfig.with_defaults(raw)


def test_construction_wires_and_keeps_its_inputs(
    config: EngineConfig, tmp_path: pathlib.Path
) -> None:
    engine = PacketEngine(tmp_path, config)
    assert engine.repo == tmp_path
    assert engine.config is config


def test_absent_interaction_port_means_unattended(
    config: EngineConfig, tmp_path: pathlib.Path
) -> None:
    """The default is `NonInteractive`, not "prompt somehow". A host that
    embeds the engine and never passes a port must get today's semantics, because the
    alternative is a library that blocks on stdin inside someone else's server."""
    assert isinstance(PacketEngine(tmp_path, config).interaction, NonInteractive)


def test_a_supplied_port_is_used_as_given(config: EngineConfig, tmp_path: pathlib.Path) -> None:
    class Scripted:
        def ask(self, q: Any, *, cancel: Any = None) -> Any: ...
        def confirm_plan(self, dag: Any, est: Any) -> Any: ...

        def confirm_stop(
            self, stop: ApprovalStop, *, cancel: threading.Event | None = None
        ) -> StopDecision:
            return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

        def confirm_action(
            self, approval: Any, *, cancel: threading.Event | None = None
        ) -> Any: ...

    port = Scripted()
    assert PacketEngine(tmp_path, config, port).interaction is port


def test_noninteractive_satisfies_the_frozen_port() -> None:
    """The default must actually be an `InteractionPort` — a default that does not satisfy
    the contract it stands in for fails only when a host finally supplies its own."""
    assert isinstance(NonInteractive(), InteractionPort)


def test_a_wrong_config_type_is_a_harness_bug(tmp_path: pathlib.Path) -> None:
    """Not `ConfigError`: that means the user's config is bad. Handing the engine a dict
    is our caller misusing the API, and the two must not read the same to a host."""
    with pytest.raises(HarnessBug):
        PacketEngine(tmp_path, {"models": {}, "rungs": {}})  # type: ignore[arg-type]


def test_unbuilt_surfaces_refuse_rather_than_return_empty(
    config: EngineConfig, tmp_path: pathlib.Path
) -> None:
    """An unimplemented method must not look like a successful empty answer. `preview()`
    returning an empty DAG would read as "nothing to do" and pass straight through a
    caller's success path."""
    engine = PacketEngine(tmp_path, config)
    # `init_repo` is deliberately absent: T-0.7 implemented it, and leaving it here would
    # assert that a built surface is still unbuilt.
    for call in (
        lambda: engine.expand_turn("s1", 1),
        lambda: engine.session_context("s1", object(), 0),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_the_surface_is_the_one_the_spec_declares(
    config: EngineConfig, tmp_path: pathlib.Path
) -> None:
    """D77 deleted seven surfaces, and a skeleton is a promise about shape — it is what
    hosts are written against before anything works, so a stale one invites a host to be
    built against methods that will never exist.

    Checked in both directions: a name here that `18-engine.md` §2 does not declare is a
    promise the specs do not make, and a name §2 declares that is missing is one a host
    cannot write against yet.
    """
    engine = PacketEngine(tmp_path, config)
    declared = {
        "init_repo",
        # The index built beside the person, the host starting it.
        "index_in_background",
        "reindex",
        "locate",
        # What the REPL's `@` completes, and what its history keeps out.
        "files",
        "holds_secret",
        "coder",
        "resume",
        # The person's `/pause` and `/drop` of the team's run.
        "pause_run",
        "drop_run",
        "create_session",
        # The repository's open work, which a new session may continue.
        "open_works",
        "open_session",
        "latest_session",
        "close_session",
        "session_context",
        "expand_turn",
        "read",
        "journal",
        "handoff",
        "header",
        "set_cap",
        "budget",
        # D128, T-5.10 builds them; declared unbuilt so the wire carries their types.
        "models",
        "switch_model",
        "turn",
        "interrupt",
        # What the researcher's evaluation reaches through the facade.
        "spawn_agent",
        "ask_model",
        "attach_caps",
        "preview_attachment",
        "metrics",
        # What a run's records name, for a frontend.
        "run_blob",
        "run_diff",
        # One workspace file `files` lists, for the CODE pane.
        "file_text",
        # The person's name for the session.
        "rename",
        "finalize",
        "close",
        # The person's undo, between turns.
        "jobs",
        "kill_job",
        # Where the session's open work stands, each task across its runs.
        "work",
        "undoable",
        "undo",
        "redo",
        "undo_task",
        "undo_operation",
        # A fork that restores or keeps the code.
        "fork",
        # What a frontend reads without opening a session.
        "sessions",
        "session_info",
        "hidden",
        "liveness",
    }
    public = {
        name
        for name in dir(engine)
        if not name.startswith("_") and callable(getattr(engine, name, None))
    }
    assert public == declared

    gone = {
        "route_goal",
        "plan",
        "discover",
        "run_goal",
        "run_dag",
        "run_packet",
        "ask",
        "investigate",
        # The work tier and the packet loop's surfaces went with it.
        "tier_for",
        "preview",
        "pack_context",
        "undo_packet",
    }
    assert not (public & gone), "a surface D77 deleted is still reachable"


def test_close_is_idempotent(config: EngineConfig, tmp_path: pathlib.Path) -> None:
    engine = PacketEngine(tmp_path, config)
    engine.close()
    engine.close()
