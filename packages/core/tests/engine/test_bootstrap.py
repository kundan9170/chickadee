"""D109 / T-2.30 — a worktree episode bootstraps its worktree before the first packet.

A worktree is a checkout of tracked files and nothing else. The profile's
`[toolchain].bootstrap` runs once at episode start, journaled as the first operation, and
a bootstrap that fails means the episode does not start — never a packet's failure.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.bootstrap import (
    BootstrapReport,
    bootstrap_env,
    bootstrap_worktree,
)
from chickadee.errors import InfraError
from chickadee.isolation.none import Passthrough
from chickadee.journal import Journal, paths, project, read_events
from chickadee.process.supervisor import NETWORK_ENV, Completed, ProcessSupervisor
from chickadee.profiles.model import LanguageProfile, Template, ToolchainSpec
from chickadee.run_header import IN_PLACE_NO_BOOTSTRAP, NO_BOOTSTRAP
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import acquire

RUN = "2026-09-20-0000b007"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"kept.txt": "original\n"}, commit="first")


def profile(*argv: str) -> LanguageProfile:
    """A python profile whose bootstrap is `argv`; none when `argv` is empty."""
    bootstrap = Template(source=" ".join(argv), argv=tuple(argv)) if argv else None
    return LanguageProfile(
        name="python",
        extensions=(".py",),
        tree_sitter_grammar=((".py", "python"),),
        toolchain=ToolchainSpec(bootstrap=bootstrap),
    )


def _journal(repo: Path) -> Journal:
    return Journal.create(resolve_roots(repo), RUN, {})


def _events(repo: Path) -> list[tuple[str, dict[str, object]]]:
    return [
        (ev.kind.value, dict(ev.payload))
        for ev in read_events(paths.journal_path(resolve_roots(repo), RUN))
    ]


def test_a_worktree_episode_runs_the_bootstrap_once_and_journals_it(repo: Path) -> None:
    """The first operation of the episode: intended, started, completed, in that order,
    under one operation_id, before anything a packet would do."""
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        report = bootstrap_worktree(
            ws, profile("touch", "bootstrapped.marker"), journal, backend=Passthrough()
        )
        assert report.ran
        assert (ws.root / "bootstrapped.marker").exists(), "ran in the worktree"
        assert not (repo / "bootstrapped.marker").exists(), "not in the source checkout"
        assert report.operation_id == f"bootstrap:{RUN}"
        assert report.argv == ("touch", "bootstrapped.marker")
        assert "touch bootstrapped.marker" in report.header
        assert report.output_sha is not None

    kinds = [k for k, _ in _events(repo)]
    assert kinds[:3] == ["effect_intended", "effect_started", "effect_completed"]
    intended, started, completed = (p for _, p in _events(repo)[:3])
    assert intended["operation_id"] == started["operation_id"] == f"bootstrap:{RUN}"
    assert intended["effect_kind"] == "bootstrap"
    assert completed["outcome"] == "succeeded"
    assert completed["result_sha"] == report.output_sha
    state = project(read_events(paths.journal_path(resolve_roots(repo), RUN)))
    assert state.operations[f"bootstrap:{RUN}"].phase == "completed"


def test_the_output_is_a_blob_the_handle_names(repo: Path) -> None:
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        report = bootstrap_worktree(
            ws, profile("echo", "installed 3 packages"), journal, backend=Passthrough()
        )
        assert report.output_sha is not None
        assert b"installed 3 packages" in journal.get_blob(report.output_sha)


def test_a_non_zero_exit_is_an_infra_error_and_the_episode_does_not_start(
    repo: Path,
) -> None:
    """Named the command and the output handle; journaled as a failed effect, so a resume
    sees the worktree was never prepared rather than assuming it was."""
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        with pytest.raises(InfraError, match=r"bootstrap 'false' exited 1") as info:
            bootstrap_worktree(ws, profile("false"), journal, backend=Passthrough())
        assert "Output handle:" in str(info.value)
    completed = next(p for k, p in _events(repo) if k == "effect_completed")
    assert completed["outcome"] == "failed"


def test_a_timeout_is_an_infra_error_too(repo: Path) -> None:
    with (
        acquire(repo, run_id=RUN, goal="g") as ws,
        _journal(repo) as journal,
        pytest.raises(InfraError, match="timed out after 1s"),
    ):
        bootstrap_worktree(
            ws, profile("sleep", "5"), journal, backend=Passthrough(), timeout_s=1
        )


def test_the_bootstrap_env_keeps_the_network_and_still_scrubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap is the one harness spawn that needs the network. `scrub_env()`'s
    keep-list drops the proxy and index variables, so behind a proxy `uv sync` could not
    reach anything and the episode refused to start; the passthrough carries them, and
    only them — a credential-named variable is scrubbed on either list."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("UV_INDEX_URL", "https://pypi.example/simple")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_not_for_the_bootstrap")
    monkeypatch.setenv("EDITOR", "vi")  # the shell's, on neither list
    env = bootstrap_env()
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["UV_INDEX_URL"] == "https://pypi.example/simple"
    assert "GITHUB_TOKEN" not in env
    assert "EDITOR" not in env
    assert "PATH" in env, "scrub_env's keep-list is the base"
    assert "GITHUB_TOKEN" not in NETWORK_ENV


def test_a_passthrough_name_the_scrub_would_drop_is_still_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The passthrough is a list of names, and the scrub judges names; a passthrough entry
    that ever acquired a credential marker would travel only if the scrub let it, and it
    does not — asserted here so the list cannot quietly grow one."""
    for name in NETWORK_ENV:
        monkeypatch.setenv(name, "set")
    env = bootstrap_env()
    assert all(name in env for name in NETWORK_ENV)
    monkeypatch.setattr(
        "chickadee.process.supervisor.NETWORK_ENV",
        (*NETWORK_ENV, "UV_TOKEN"),
    )
    monkeypatch.setenv("UV_TOKEN", "leak")
    assert "UV_TOKEN" not in bootstrap_env()


def test_the_bootstrap_command_sees_the_proxy(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("NPM_TOKEN", "leak")
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        bootstrap_worktree(
            ws,
            profile("sh", "-c", 'printf "%s|%s" "$HTTPS_PROXY" "$NPM_TOKEN" > env.marker'),
            journal,
            backend=Passthrough(),
        )
        assert (ws.root / "env.marker").read_text() == "http://proxy.example:3128|"


def test_in_place_runs_nothing_and_says_so(repo: Path) -> None:
    from chickadee.config import WorkspacePolicy

    ws = acquire(repo, run_id=RUN, goal="g", policy=WorkspacePolicy(mode="in_place"))
    try:
        with _journal(repo) as journal:
            report = bootstrap_worktree(
                ws, profile("touch", "never.marker"), journal, backend=Passthrough()
            )
    finally:
        ws.release()
    assert report == BootstrapReport(ran=False, header=IN_PLACE_NO_BOOTSTRAP)
    assert not (repo / "never.marker").exists()
    assert _events(repo) == []


def test_a_profile_without_a_bootstrap_runs_nothing_and_the_header_says_so(
    repo: Path,
) -> None:
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        report = bootstrap_worktree(ws, profile(), journal, backend=Passthrough())
    assert report == BootstrapReport(ran=False, header=NO_BOOTSTRAP)
    assert "no bootstrap" in report.header
    assert _events(repo) == []


def test_a_required_sandbox_that_is_missing_refuses_rather_than_running_unwrapped(
    repo: Path,
) -> None:
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        with pytest.raises(InfraError, match="needs a sandbox"):
            bootstrap_worktree(
                ws,
                profile("touch", "never.marker"),
                journal,
                backend=Passthrough(),
                require_sandbox=True,
            )
        assert not (ws.root / "never.marker").exists()


class _Raising(ProcessSupervisor):
    """A supervisor that raises `error` instead of running anything: an exception after
    `effect_started`, from below the bootstrap."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        raise self.error


def test_an_exception_after_the_start_closes_the_operation_ambiguous(repo: Path) -> None:
    """An exception from the supervisor may come after the
    command changed the worktree, so the operation is closed `ambiguous`, never left
    started, and the exception reaches the caller unchanged."""
    error = RuntimeError("the supervisor broke mid-bootstrap")
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        with pytest.raises(RuntimeError) as info:
            bootstrap_worktree(
                ws,
                profile("touch", "x.marker"),
                journal,
                backend=Passthrough(),
                supervisor=_Raising(error),
            )
        assert info.value is error
    events = _events(repo)
    assert [k for k, _ in events] == ["effect_intended", "effect_started", "effect_completed"]
    completed = events[2][1]
    assert completed["operation_id"] == f"bootstrap:{RUN}"
    assert completed["outcome"] == "ambiguous"
    assert completed["observed_surface"] == ()
    state = project(read_events(paths.journal_path(resolve_roots(repo), RUN)))
    assert state.operations[f"bootstrap:{RUN}"].outcome == "ambiguous"


def test_an_interrupt_after_the_start_writes_nothing_more(repo: Path) -> None:
    """A `BaseException` that is not an `Exception` is treated as a kill, which cannot
    write: the operation stays started, for recovery to read as such."""
    with (
        acquire(repo, run_id=RUN, goal="g") as ws,
        _journal(repo) as journal,
        pytest.raises(KeyboardInterrupt),
    ):
        bootstrap_worktree(
            ws,
            profile("touch", "x.marker"),
            journal,
            backend=Passthrough(),
            supervisor=_Raising(KeyboardInterrupt()),
        )
    assert [k for k, _ in _events(repo)] == ["effect_intended", "effect_started"]


# -- D153 decision 8: every field of the bootstrap's effect lines, pinned --------


def _bootstrap_fields(repo: Path, root: Path) -> list[object]:
    """The effect lines as their fields, the worktree path and the shas pinned by shape."""
    out: list[object] = []
    for kind, payload in _events(repo):
        if not kind.startswith("effect_"):
            continue
        fields = {
            key: (
                "<sha>"
                if isinstance(value, str) and len(value) == 64
                else tuple(str(v).replace(str(root), "<root>") for v in value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in payload.items()
        }
        out.append((kind, fields))
    return out


BOOTSTRAP_PINS: dict[str, list[object]] = {
    "ok": [
        (
            "effect_intended",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "effect_kind": "bootstrap",
                "approved_sha": "<sha>",
                "preconditions": ("worktree <root> exists",),
                "idempotent": True,
                "declared_surface": ("<root>",),
                "preimages_sha": None,
            },
        ),
        ("effect_started", {"operation_id": "bootstrap:2026-09-20-0000b007"}),
        (
            "effect_completed",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "outcome": "succeeded",
                "observed_surface": ("<root>",),
                "post_state_sha": None,
                "result_sha": "<sha>",
            },
        ),
    ],
    "exits": [
        (
            "effect_intended",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "effect_kind": "bootstrap",
                "approved_sha": "<sha>",
                "preconditions": ("worktree <root> exists",),
                "idempotent": True,
                "declared_surface": ("<root>",),
                "preimages_sha": None,
            },
        ),
        ("effect_started", {"operation_id": "bootstrap:2026-09-20-0000b007"}),
        (
            "effect_completed",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "outcome": "failed",
                "observed_surface": ("<root>",),
                "post_state_sha": None,
                "result_sha": "<sha>",
            },
        ),
    ],
    "refused": [
        (
            "effect_intended",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "effect_kind": "bootstrap",
                "approved_sha": "<sha>",
                "preconditions": ("worktree <root> exists",),
                "idempotent": True,
                "declared_surface": ("<root>",),
                "preimages_sha": None,
            },
        ),
        ("effect_started", {"operation_id": "bootstrap:2026-09-20-0000b007"}),
        (
            "effect_completed",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "outcome": "failed",
                "observed_surface": ("<root>",),
                "post_state_sha": None,
                "result_sha": None,
            },
        ),
    ],
    "raises": [
        (
            "effect_intended",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "effect_kind": "bootstrap",
                "approved_sha": "<sha>",
                "preconditions": ("worktree <root> exists",),
                "idempotent": True,
                "declared_surface": ("<root>",),
                "preimages_sha": None,
            },
        ),
        ("effect_started", {"operation_id": "bootstrap:2026-09-20-0000b007"}),
        (
            "effect_completed",
            {
                "operation_id": "bootstrap:2026-09-20-0000b007",
                "outcome": "ambiguous",
                "observed_surface": (),
                "post_state_sha": None,
                "result_sha": None,
            },
        ),
    ],
}


@pytest.mark.parametrize("name", ["ok", "exits", "refused", "raises"])
def test_the_bootstraps_effect_lines_are_pinned(repo: Path, name: str) -> None:
    argv = ("false",) if name == "exits" else ("touch", "x.marker")
    supervisor = _Raising(RuntimeError("broke")) if name == "raises" else None
    with acquire(repo, run_id=RUN, goal="g") as ws, _journal(repo) as journal:
        with contextlib.suppress(InfraError, RuntimeError):
            bootstrap_worktree(
                ws,
                profile(*argv),
                journal,
                backend=Passthrough(),
                supervisor=supervisor,
                require_sandbox=name == "refused",
            )
        root = ws.root
    assert _bootstrap_fields(repo, root) == BOOTSTRAP_PINS[name]
