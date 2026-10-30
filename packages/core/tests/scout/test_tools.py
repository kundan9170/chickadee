"""The scout's surface is the ladder's shared tools plus `search` and
`request_exec`; caps and dedup are applied; `search` reaches files the graph does not index."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from scout_helpers import FileReader, GraphNavigate, ListSink

from chickadee.adapters.frontier import TEMPLATES
from chickadee.errors import HarnessBug
from chickadee.graph.queries import Graph
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.limits import ResolvedLimit, limit
from chickadee.process.supervisor import Completed, ProcessSupervisor
from chickadee.roles import role
from chickadee.schema.journal import EventKind, ScoutToolP
from chickadee.scout.escapes import REQUEST_EXEC
from chickadee.scout.search import Search
from chickadee.scout.snapshot import SNAPSHOT_TOKENS
from chickadee.scout.tools import LocateCaps, ScoutTools
from chickadee.tools.spec import spec

SHARED = ("symbol", "grep", "glob", "neighbors", "impact", "read", "grep_in")
"""Rungs 1-4 of the ladder, and `grep` and `glob`: the `locate` role's
enabled list in `ROLES`, ahead of the scout's own `request_exec`."""
NEVER = frozenset(
    {"atlas", "recall", "locate", "shell", "coder", "ask_user", "note_concern", "note_rejected"}
)
"""The "not on the surface"."""


def test_the_surface_is_the_ladder_plus_request_exec(tools: ScoutTools) -> None:
    names = tools.names(offer_exec=True)
    assert names == (*SHARED, REQUEST_EXEC)
    assert [s.name for s in tools.schemas(offer_exec=True)] == list(names)


@pytest.mark.parametrize("offer_exec", [True, False])
def test_the_surface_never_offers_atlas_recall_locate_shell_coder_ask_user_or_a_record_tool(
    tools: ScoutTools, offer_exec: bool
) -> None:
    offered = set(tools.names(offer_exec=offer_exec))
    assert not offered & NEVER
    assert not {s.name for s in tools.schemas(offer_exec=offer_exec)} & NEVER


def test_request_exec_is_offered_only_with_escapes_and_the_offer(
    graph: Graph, repo: Path, tools: ScoutTools
) -> None:
    assert REQUEST_EXEC not in tools.names(offer_exec=False)
    bare = ScoutTools(GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None))
    assert REQUEST_EXEC not in bare.names(offer_exec=True)


def test_the_scout_offers_the_policy_it_is_given(graph: Graph, repo: Path) -> None:
    """`[tools.locate]` as the config resolves it: what the scout offers and
    describes is that policy, not the cast's default."""
    default = role("locate").tool_defaults
    assert default is not None
    narrowed = replace(default, enabled=("read", "glob"))
    tools = ScoutTools(
        GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None), policy=narrowed
    )
    assert tools.names(offer_exec=False) == ("read", "glob")
    assert [s.name for s in tools.schemas(offer_exec=False)] == ["read", "glob"]


def test_the_shared_tools_are_zeus_s_schemas_described_for_the_scout(tools: ScoutTools) -> None:
    """One tool implementation, not three forks: the same schema, and since the
    same words, which name no shell the scout never has."""
    by_name = {s.name: s for s in tools.schemas(offer_exec=True)}
    for name in SHARED:
        zeus = spec(name)
        assert zeus is not None
        assert by_name[name].schema == zeus.schema
        source = TEMPLATES / "tools" / f"{name}.md"
        assert by_name[name].description == source.read_text(encoding="utf-8")
        assert "shell(" not in by_name[name].description, f"{name} names a shell"
    assert by_name[REQUEST_EXEC].description.strip()


def test_caps_are_the_register_s() -> None:
    caps = LocateCaps.for_window(100_000)
    assert caps.max_tool_calls == limit("locate.max_tool_calls").value == 40
    assert caps.max_result_tokens == int(
        float(limit("locate.max_cumulative_result_tokens").value) * 100_000
    )
    resolved = (
        ResolvedLimit("locate.max_tool_calls", "locate", "scout", 12, "config"),
        ResolvedLimit(
            "locate.max_cumulative_result_tokens", "locate", "scout", 9_000, "default"
        ),
        ResolvedLimit("locate.max_tool_calls", "zeus", "zeus", 64, "default"),
    )
    with_row = (
        *resolved,
        ResolvedLimit("locate.snapshot_tokens", "locate", "scout", 2_000, "config"),
    )
    assert LocateCaps.from_resolved(with_row) == LocateCaps(12, 9_000, snapshot_tokens=2_000)
    # The row is in the register, so a table without it is malformed, as a table
    # missing any other `locate` row is; it no longer falls back to the default.
    with pytest.raises(HarnessBug, match=r"locate\.snapshot_tokens"):
        LocateCaps.from_resolved(resolved)
    assert LocateCaps.for_window(100_000).snapshot_tokens == SNAPSHOT_TOKENS


def test_a_repeated_call_is_answered_with_a_marker_and_not_re_run(
    graph: Graph, repo: Path, sink: ListSink
) -> None:
    nav = GraphNavigate(graph)
    tools = ScoutTools(nav, FileReader(repo), Search(root=repo, rg=None))
    first = tools.call("symbol", {"name": "HttpClient"}, sink, offer_exec=False)
    again = tools.call("symbol", {"name": "HttpClient"}, sink, offer_exec=False)
    assert not first.dedup_hit and again.dedup_hit
    assert again.text.startswith("[dedup:") and "call #1" in again.text
    assert nav.calls == ["symbol HttpClient"], "the tool ran once"
    records = [p for p in sink.of(EventKind.SCOUT_TOOL) if isinstance(p, ScoutToolP)]
    assert [r.dedup_hit for r in records] == [False, True]
    assert records[0].args_sha == records[1].args_sha
    assert tools.dedup_streak == 1


def test_a_different_call_resets_the_dedup_streak(tools: ScoutTools, sink: ListSink) -> None:
    for _ in range(2):
        tools.call("symbol", {"name": "HttpClient"}, sink, offer_exec=False)
    assert tools.dedup_streak == 1
    tools.call("symbol", {"name": "fetch"}, sink, offer_exec=False)
    assert tools.dedup_streak == 0


def test_an_unoffered_tool_is_answered_not_raised(tools: ScoutTools, sink: ListSink) -> None:
    answer = tools.call("atlas", {}, sink, offer_exec=True)
    assert answer.text.startswith("UNKNOWN_TOOL{name: 'atlas'")
    exec_answer = tools.call(
        REQUEST_EXEC, {"command": "ls", "reason": "r"}, sink, offer_exec=False
    )
    assert exec_answer.text.startswith("UNKNOWN_TOOL")
    assert tools.calls == 2


def test_invalid_arguments_are_typed_and_nothing_runs(
    graph: Graph, repo: Path, sink: ListSink
) -> None:
    nav = GraphNavigate(graph)
    tools = ScoutTools(nav, FileReader(repo), Search(root=repo, rg=None))
    answer = tools.call("symbol", {"q": "x"}, sink, offer_exec=False)
    assert answer.text.startswith("INVALID_ARGUMENTS: symbol")
    assert nav.calls == []


# -- search ------------------------------------------------------------------------------


def test_search_hits_files_the_graph_does_not_index(repo: Path, graph: Graph) -> None:
    assert not graph.file_known("config/app.yaml")
    text = Search(root=repo, rg=None)("timeout", kind="text")
    assert "config/app.yaml:2:" in text
    assert "pkg/net/client.py" in text
    paths = Search(root=repo, rg=None)("*.yaml", kind="path")
    assert "config/app.yaml" in paths.splitlines()


def test_search_honours_gitignore_and_never_returns_a_deny_listed_file(repo: Path) -> None:
    text = Search(root=repo, rg=None)("SECRET_WORD", kind="text")
    assert "ignored_dir" not in text and "UNRESOLVED" in text
    hits = Search(root=repo, rg=None)("timeout", kind="text")
    assert ".env" not in hits and "hunter2" not in hits


def test_search_is_capped_by_hits_and_bytes_and_names_the_rest(repo: Path) -> None:
    by_hits = Search(root=repo, rg=None, max_hits=1)("timeout", kind="text")
    assert "[truncated:" in by_hits and "result_cap" in by_hits
    assert len([ln for ln in by_hits.splitlines() if ":" in ln and "truncated" not in ln]) == 1
    by_bytes = Search(root=repo, rg=None, max_bytes=60)("timeout", kind="text")
    assert "[truncated:" in by_bytes


def test_search_paths_limit_where_it_looks(repo: Path) -> None:
    text = Search(root=repo, rg=None)("timeout", kind="text", paths=["config/*"])
    assert "config/app.yaml" in text and "pkg/" not in text


@pytest.mark.writes_worktree("app.yaml")
def test_the_description_s_own_glob_finds_a_file_at_the_root(repo: Path) -> None:
    (repo / "app.yaml").write_text("name: demo\n", encoding="utf-8")
    hits = Search(root=repo, rg=None)("**/*.yaml", kind="path").splitlines()
    assert "app.yaml" in hits and "config/app.yaml" in hits


def test_the_fallback_matches_case_sensitively_as_ripgrep_does(repo: Path) -> None:
    # `timeout` and `DEFAULT_TIMEOUT` are in the repository; `Timeout` is not.
    assert "UNRESOLVED" in Search(root=repo, rg=None)("Timeout", kind="text")
    assert "pkg/net/client.py" in Search(root=repo, rg=None)("DEFAULT_TIMEOUT", kind="text")


@pytest.mark.writes_worktree("docs", "docs/notes.txt")
def test_the_fallback_refuses_a_link_whose_target_is_deny_listed(repo: Path) -> None:
    """The deny list is asked of what the read reaches, not only of the name it went by."""
    link = repo / "docs" / "notes.txt"
    link.parent.mkdir()
    link.symlink_to(repo / ".env")
    text = Search(root=repo, rg=None)("hunter2", kind="text")
    assert "UNRESOLVED" in text and "docs/notes.txt" not in text
    assert "timeout_token" not in text, "no line of the target reaches the model"


class _Backend:
    def __init__(self) -> None:
        self.seen: list[Isolation] = []

    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        self.seen.append(isolation)
        return ("sandboxed", *isolation.argv)


class _Supervisor(ProcessSupervisor):
    def __init__(self, stdout: bytes, exit_code: int = 0, stderr: bytes = b"") -> None:
        self.stdout, self.exit_code, self.stderr = stdout, exit_code, stderr
        self.argv: tuple[str, ...] = ()

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
        self.argv = argv
        return Completed(self.stdout, self.stderr, self.exit_code, False)


def test_ripgrep_runs_as_harness_argv_sealed_and_deny_listed_hits_are_dropped(
    repo: Path,
) -> None:
    """Through Isolation and the supervisor, network off, the repository read-only. A
    deny-listed hit is dropped even when the backend let `rg` read it."""
    backend = _Backend()
    sup = _Supervisor(b"./config/app.yaml:2:  timeout: 30\n./.env:1:timeout_token=hunter2\n")
    search = Search(root=repo, backend=backend, supervisor=sup, rg="/usr/bin/rg")
    text = search("timeout")
    (iso,) = backend.seen
    assert iso.network_off and not iso.repo_writable and iso.cwd == repo
    assert iso.argv[0] == "/usr/bin/rg" and "timeout" in iso.argv
    # The fallback's file set: hidden files in, the indexer's always-skipped directories out.
    assert "--hidden" in iso.argv and "--glob=!.git/" in iso.argv
    assert sup.argv[0] == "sandboxed"
    assert "config/app.yaml:2:" in text and "ripgrep" in text.splitlines()[0]
    assert ".env" not in text and "hunter2" not in text


def test_a_pattern_ripgrep_rejects_is_invalid_arguments(repo: Path) -> None:
    sup = _Supervisor(b"", exit_code=2, stderr=b"regex parse error: unclosed group")
    search = Search(root=repo, backend=_Backend(), supervisor=sup, rg="/usr/bin/rg")
    text = search("(")
    assert text.startswith("INVALID_ARGUMENTS: grep") and "unclosed group" in text


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed here")
def test_real_ripgrep_finds_a_non_code_file(repo: Path) -> None:
    text = Search(root=repo)("timeout", kind="text")
    assert "config/app.yaml:2:" in text
    assert ".env" not in text and "ignored_dir" not in text
