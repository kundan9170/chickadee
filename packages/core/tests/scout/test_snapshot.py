"""The snapshot is field-complete, never over `locate.snapshot_tokens` (by
construction, D133; default 4096) and using most of it, gitignore-aware, merges promoted
hints, and is reused while the generation holds."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from scout_helpers import make_repo

from chickadee.graph.queries import Graph
from chickadee.limits import tokens_for_chars
from chickadee.profiles.model import LanguageProfile
from chickadee.scout import snapshot as snapshot_mod
from chickadee.scout.snapshot import (
    SNAPSHOT_TOKENS,
    Snapshot,
    SnapshotCache,
    build_snapshot,
    field_names,
)


def _probe(binary: str) -> str:
    return f"{binary} 1.0"


SPEC_FIELDS = {
    # repo
    "root", "source_ref", "head", "execution_mode", "dirty_count", "dirty_first",
    "recent_commits",
    # tree
    "dir_counts", "small_dir_files",
    # languages
    "languages", "profile_name",
    # manifests
    "manifests",
    # tests
    "test_dirs", "test_runner", "test_count", "test_naming",
    # toolchain
    "toolchain", "has_make", "has_just", "has_docker",
    # graph
    "files_indexed", "symbols_indexed", "edge_classes", "index_truncated",
    # the rest
    "entrypoints", "promoted", "tokens", "omitted",
}  # fmt: skip

ROW_LABELS = (
    "repo:", "dirty:", "recent commits:", "tree:", "languages:", "manifests:", "tests:",
    "toolchain:", "graph:", "entrypoints:", "promoted:",
)  # fmt: skip


def test_the_type_is_the_spec_table() -> None:
    """The type, field for field: a field added or dropped fails until the spec says so."""
    assert set(field_names()) == SPEC_FIELDS


def test_every_field_is_filled_from_the_repository(
    repo: Path, python: LanguageProfile, graph: Graph
) -> None:
    snap = build_snapshot(
        repo, [python], graph, promoted={"make help": "targets: help test"}, probe=_probe
    )
    assert snap.root == str(repo.resolve())
    assert len(snap.head) == 40 and snap.source_ref not in ("", "unknown")
    assert snap.execution_mode == "worktree"
    assert snap.dirty_count == 0 and snap.dirty_first == ()
    assert snap.recent_commits == ("first commit",)
    assert {".", "pkg", "pkg/net", "pkg/api", "config", "tests"} <= set(snap.dir_counts)
    assert snap.small_dir_files["config"] == ("app.yaml",)
    assert snap.languages == ("python",) and snap.profile_name == "python"
    assert snap.manifests == {"pyproject.toml": ("httpx", "pydantic")}
    assert snap.test_dirs == ("tests",) and snap.test_count == 1
    assert snap.test_runner == "pytest" and snap.test_naming == "test_*.py, *_test.py"
    assert snap.toolchain == {"python3": "python3 1.0", "uv": "uv 1.0", "pytest": "pytest 1.0"}
    assert snap.has_make and not snap.has_just and not snap.has_docker
    assert snap.files_indexed >= 4 and snap.symbols_indexed >= 3 and snap.edge_classes
    assert not snap.index_truncated
    assert snap.entrypoints == ("pkg/__main__.py",)
    assert snap.promoted == {"make help": "targets: help test"}
    text = snap.render()
    for label in ROW_LABELS:
        assert label in text, label


def test_an_empty_row_says_none_rather_than_disappearing(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    snap = build_snapshot(bare, [], None, probe=_probe)
    text = snap.render()
    for label in ROW_LABELS:
        assert label in text, label
    assert "graph: 0 files, 0 symbols" in text
    assert "entrypoints: none" in text and "promoted: none" in text
    assert snap.head == "" and "no commits" in text


def test_within_the_token_budget_and_measured(
    repo: Path, python: LanguageProfile, graph: Graph
) -> None:
    snap = build_snapshot(repo, [python], graph, probe=_probe)
    assert snap.tokens == tokens_for_chars(len(snap.render()))
    assert snap.tokens <= SNAPSHOT_TOKENS


def test_a_large_tree_is_trimmed_to_the_budget_and_keeps_every_row(
    tmp_path: Path, python: LanguageProfile
) -> None:
    big = tmp_path / "big"
    for d in range(40):
        for s in range(3):
            folder = big / f"component_{d:02d}" / f"sub_{s}" / "leaf"
            folder.mkdir(parents=True)
            for f in range(12):
                (folder / f"module_with_a_long_name_{f:02d}.py").write_text("x = 1\n")
    promoted = {f"tool-{i} --list": "\n".join(f"line {n}" for n in range(50)) for i in range(5)}
    snap = build_snapshot(big, [python], None, promoted=promoted, probe=_probe)
    assert snap.tokens <= SNAPSHOT_TOKENS
    text = snap.render()
    for label in ROW_LABELS:
        assert label in text, label
    assert snap.dir_counts["."] == 40 * 3 * 12, "the root's count survives the trim"


def test_lists_that_outrun_every_trim_step_are_cut_with_a_marker(
    tmp_path: Path, python: LanguageProfile
) -> None:
    """The budget holds by construction. Many top-level directories, manifests and
    promoted commands, and long commit subjects: none of them had a count cap of its own."""
    wide = tmp_path / "wide"
    for d in range(120):
        pkg = wide / f"service_with_a_long_descriptive_name_{d:03d}"
        pkg.mkdir(parents=True)
        (pkg / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = ["alpha", "beta", "gamma", "delta"]\n'
        )
    promoted = {f"tool-{i} --list --with-a-long-flag": "one line" for i in range(40)}
    for budget in (SNAPSHOT_TOKENS, 300):
        snap = build_snapshot(
            wide, [python], None, promoted=promoted, probe=_probe, budget=budget
        )
        assert snap.tokens == tokens_for_chars(len(snap.render())) <= budget, budget
        text = snap.render()
        for label in ROW_LABELS:
            assert label in text, label
        assert snap.omitted, "what was cut is counted"
        assert " more)" in text, "and the render says so"
    assert snap.dir_counts["."] == 120, "the root's count still covers everything cut"


def test_a_large_budget_is_used_not_thrown_away(
    tmp_path: Path, python: LanguageProfile
) -> None:
    """The last trim step keeps the most entries that fit. A fixed ladder sized for a small
    budget (8, 4, 2, 1, 0) cut a 5,800-token map to about 540 tokens under a 4,096 budget."""
    wide = tmp_path / "wide"
    for d in range(200):
        pkg = wide / f"service_with_a_long_descriptive_name_{d:03d}"
        pkg.mkdir(parents=True)
        (pkg / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = ["alpha", "beta", "gamma", "delta"]\n'
        )
    snap = build_snapshot(wide, [python], None, probe=_probe, budget=4096)
    assert snap.omitted, "the fixture does not fit whole"
    assert 4096 * 0.9 < snap.tokens <= 4096


def test_the_budget_is_the_caller_s(repo: Path, python: LanguageProfile, graph: Graph) -> None:
    """`locate.snapshot_tokens` reaches the build as a parameter, resolved by the caller."""
    roomy = build_snapshot(repo, [python], graph, probe=_probe, budget=4_000)
    tight = build_snapshot(repo, [python], graph, probe=_probe, budget=150)
    assert tight.tokens <= 150 < roomy.tokens
    assert roomy.omitted == {}


def test_gitignore_and_the_always_skipped_are_honoured(
    repo: Path, python: LanguageProfile, graph: Graph
) -> None:
    snap = build_snapshot(repo, [python], graph, probe=_probe)
    assert "ignored_dir" not in snap.dir_counts
    assert "ignored_dir" not in snap.render()
    assert not any(d.startswith(".git") for d in snap.dir_counts)


def test_a_capped_index_is_reported_as_truncated(
    repo: Path, python: LanguageProfile, graph: Graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 20: an index at the cap may be partial, and the snapshot says so."""
    monkeypatch.setattr(snapshot_mod, "MAX_FILES", 1)
    snap = build_snapshot(repo, [python], graph, probe=_probe)
    assert snap.index_truncated
    assert "INDEX TRUNCATED" in snap.render()


def test_promoted_hints_are_merged(repo: Path, python: LanguageProfile, graph: Graph) -> None:
    snap = build_snapshot(
        repo, [python], graph, promoted={"make help": "targets: help test"}, probe=_probe
    )
    assert "$ make help" in snap.render()
    assert "targets: help test" in snap.render()


def test_reused_while_the_generation_holds_and_rebuilt_when_it_moves(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "gen")
    generation = [1]
    cache = SnapshotCache(
        lambda _cancel: build_snapshot(root, [], None, probe=_probe), lambda: generation[0]
    )
    first = cache.get()
    assert cache.get() is first and cache.builds == 1
    generation[0] = 2
    second = cache.get()
    assert second is not first and cache.builds == 2
    assert cache.get() is second and cache.builds == 2


def test_a_stopped_build_is_not_kept(tmp_path: Path) -> None:
    """A build the session's stop cut short returns nothing and is not kept: it is
    missing what the stop cut, so the next call builds afresh. The stop reaches the build."""
    root = make_repo(tmp_path / "stop")
    seen: list[threading.Event | None] = []
    stop = threading.Event()

    def build(cancel: threading.Event | None) -> Snapshot | None:
        seen.append(cancel)
        if cancel is not None and cancel.is_set():
            return None
        return build_snapshot(root, [], None, probe=_probe)

    cache = SnapshotCache(build, lambda: 1)
    stop.set()
    assert cache.get(stop) is None and cache.builds == 0
    fresh = cache.get()
    assert fresh is not None and cache.builds == 1
    assert seen == [stop, None]
