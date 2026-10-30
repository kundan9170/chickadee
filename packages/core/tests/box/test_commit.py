"""The task commit, moved with `box/commit.py` from
`tests/orchestrator/test_projector.py`: the commit's message and trailer, exactly the
changed files staged and reindexed, no commit in place, and the trailer lookup resume
trusts — over a real git repository. The bodies are the packet loop's pins, unchanged but
for `commit(subject,...)` in place of `commit_packet(packet, subject,...)`, and the repository
fake they read, which lived in the orchestrator's helpers."""

from __future__ import annotations

import pathlib
import sqlite3
from collections.abc import Sequence
from typing import Any

import pytest
from indexers import RecordingIndexer
from repos import git, git_command, init_repo

from chickadee.box.commit import Projector, commit_message
from chickadee.errors import HarnessBug, InfraError
from chickadee.executor.reconcile import TRAILER
from chickadee.process import Completed, ProcessSupervisor
from chickadee.workspace import SNAPSHOT_IDENTITY

SUBJECT = "chickadee: T1 Make the thing work"
BASE = "def f() -> None:\n    return None\n"


def scenario_repo(root: pathlib.Path) -> pathlib.Path:
    """Two python modules and the harness's state directory ignored, in one commit `base`."""
    files = {"src/m.py": BASE, "src/n.py": BASE, ".gitignore": ".chickadee/\n"}
    return init_repo(root, files, commit="base")


def projector(tmp_path: pathlib.Path, *, in_place: bool = False) -> Projector:
    return Projector(scenario_repo(tmp_path / "repo"), RecordingIndexer(), in_place=in_place)


def edit(root: pathlib.Path, rel: str, text: str = "def f() -> None:\n    return 1\n") -> None:
    (root / rel).write_text(text, encoding="utf-8")


def test_the_caller_chooses_the_subject_and_the_projector_adds_the_trailer() -> None:
    # The desk chooses the subject (the task and its title); the projector adds the
    # trailer beneath it. The packet machine's `slug` retired with it.
    assert commit_message(SUBJECT, "task-commit:T1:1") == (
        f"{SUBJECT}\n\n{TRAILER}: task-commit:T1:1\n"
    )


def test_a_subject_is_one_line() -> None:
    with pytest.raises(HarnessBug, match="one line"):
        commit_message("two\nlines", "commit:p1:1:1")


def test_the_commit_carries_the_subject_and_a_trailer_git_reads_as_one(
    tmp_path: pathlib.Path,
) -> None:
    proj = projector(tmp_path)
    edit(proj.root, "src/m.py")
    sha = proj.commit(SUBJECT, "commit:p1:2:5", ("src/m.py",))
    assert sha == proj.head() == git(proj.root, "rev-parse", "HEAD").strip()
    assert git(proj.root, "log", "-1", "--format=%s").strip() == SUBJECT
    trailer = git(proj.root, "log", "-1", f"--format=%(trailers:key={TRAILER},valueonly)")
    assert trailer.strip() == "commit:p1:2:5"
    author = git(proj.root, "log", "-1", "--format=%an <%ae>").strip()
    pairs = (arg.partition("=") for arg in SNAPSHOT_IDENTITY if "=" in arg)
    identity = {key: value for key, _, value in pairs}
    assert author == f"{identity['user.name']} <{identity['user.email']}>"


def test_only_the_changed_files_are_committed(tmp_path: pathlib.Path) -> None:
    # `git add -A -- <changed>`, never `git add -A`: the person's stray file and an edit
    # the packet did not make stay out of the packet's commit.
    proj = projector(tmp_path)
    edit(proj.root, "src/m.py")
    edit(proj.root, "src/n.py", "def f() -> None:\n    return 2\n")
    edit(proj.root, "notes.txt", "mine\n")
    proj.commit(SUBJECT, "commit:p1:2:5", ("src/m.py",))
    committed = git(proj.root, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["src/m.py"]
    status = git(proj.root, "status", "--porcelain").splitlines()
    assert sorted(line.strip() for line in status) == ["?? notes.txt", "M src/n.py"]


def test_a_finder_file_is_never_committed_even_in_a_named_folder(
    tmp_path: pathlib.Path,
) -> None:
    # A design approved at a stop commits its folder whole, and a Finder file the
    # person's browsing left in it is no one's work. Named on its own, it is dropped too.
    proj = projector(tmp_path)
    (proj.root / "docs/w").mkdir(parents=True)
    edit(proj.root, "docs/w/design.md", "# d\n")
    (proj.root / "docs/w/.DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (proj.root / "Thumbs.db").write_bytes(b"\x00")
    proj.commit(SUBJECT, "commit:p1:2:5", ("docs/w", "Thumbs.db"))
    committed = git(proj.root, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["docs/w/design.md"]
    assert (proj.root / "docs/w/.DS_Store").exists()


def test_a_new_file_the_packet_created_is_committed(tmp_path: pathlib.Path) -> None:
    proj = projector(tmp_path)
    edit(proj.root, "src/new.py", "X = 1\n")
    proj.commit(SUBJECT, "commit:p1:1:2", ("src/new.py",))
    assert git(proj.root, "show", "--name-only", "--format=", "HEAD").split() == ["src/new.py"]


def test_a_packet_that_changed_nothing_still_has_its_rollback_unit(
    tmp_path: pathlib.Path,
) -> None:
    proj = projector(tmp_path)
    before = proj.head()
    sha = proj.commit(SUBJECT, "commit:p1:1:2", ())
    assert sha != before
    assert proj.landed("commit:p1:1:2") == sha


def test_the_persons_hooks_do_not_run_on_the_harness_commit(tmp_path: pathlib.Path) -> None:
    proj = projector(tmp_path)
    hook = proj.root / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    edit(proj.root, "src/m.py")
    proj.commit(SUBJECT, "commit:p1:2:5", ("src/m.py",))
    assert git(proj.root, "log", "-1", "--format=%s").strip() == SUBJECT


def test_in_place_never_commits_the_persons_work(tmp_path: pathlib.Path) -> None:
    proj = projector(tmp_path, in_place=True)
    edit(proj.root, "src/m.py")
    with pytest.raises(HarnessBug, match="D39"):
        proj.commit(SUBJECT, "commit:p1:1:2", ("src/m.py",))
    assert git(proj.root, "log", "--format=%s").split("\n")[0] == "base"


def test_reindex_is_called_with_exactly_the_changed_files(tmp_path: pathlib.Path) -> None:
    proj = projector(tmp_path)
    proj.reindex(("src/m.py", "src/new.py"))
    indexer = proj.indexer
    assert isinstance(indexer, RecordingIndexer)
    assert indexer.calls == [(proj.root / "src/m.py", proj.root / "src/new.py")]


def test_head_is_none_before_the_first_commit(tmp_path: pathlib.Path) -> None:
    root = init_repo(tmp_path / "empty", commit=None)
    assert Projector(root, RecordingIndexer()).head() is None


def test_landed_finds_the_operation_exactly_and_nothing_else(tmp_path: pathlib.Path) -> None:
    proj = projector(tmp_path)
    edit(proj.root, "src/m.py")
    five = proj.commit(SUBJECT, "commit:p1:1:5", ("src/m.py",))
    edit(proj.root, "src/m.py", BASE)
    fifty_one = proj.commit(SUBJECT, "commit:p1:1:51", ("src/m.py",))
    assert proj.landed("commit:p1:1:5") == five
    assert proj.landed("commit:p1:1:51") == fifty_one
    # A prefix of an operation that landed is an operation that did not.
    assert proj.landed("commit:p1:1:1") is None
    assert proj.landed("commit:p1:1:") is None


def test_landed_refuses_to_guess_when_git_cannot_answer(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "not-a-repo"
    root.mkdir()
    with pytest.raises(InfraError, match="cannot tell whether"):
        Projector(root, RecordingIndexer()).landed("commit:p1:1:1")


def test_an_actions_snapshot_parks_the_tree_without_moving_head(
    tmp_path: pathlib.Path,
) -> None:
    proj = projector(tmp_path)
    before = proj.head()
    edit(proj.root, "src/m.py")
    tree = proj.snapshot("exec:a1:1:1")
    # The tree as it is, uncommitted edit included — the undo unit for the action.
    assert (
        git(proj.root, "show", f"{tree}:src/m.py").strip() == "def f() -> None:\n    return 1"
    )
    assert proj.head() == before
    with pytest.raises(InfraError, match="snapshots once"):
        proj.snapshot("exec:a1:1:1")


def test_landed_looks_only_at_the_runs_own_commits(tmp_path: pathlib.Path) -> None:
    # Operation ids are unique within a run, not across runs. An earlier run's packet p1
    # committed under the same id, and its commit is in the history this run starts from.
    proj = projector(tmp_path)
    edit(proj.root, "src/m.py")
    earlier = proj.commit(SUBJECT, "commit:p1:1:2", ("src/m.py",))
    start = proj.head()  # this run begins here
    assert proj.landed("commit:p1:1:2") == earlier  # unscoped, the other run's answers
    assert proj.landed("commit:p1:1:2", since=start) is None
    edit(proj.root, "src/m.py", BASE)
    mine = proj.commit(SUBJECT, "commit:p1:1:2", ("src/m.py",))
    assert proj.landed("commit:p1:1:2", since=start) == mine


# -- the lookup is `reconcile.trailer_commit`: three defects the old copy had ------


def commit_raw(root: pathlib.Path, message: str) -> str:
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


@pytest.mark.writes_worktree("HEAD")
def test_a_file_named_head_in_the_worktree_does_not_stop_the_lookup(
    tmp_path: pathlib.Path,
) -> None:
    # A bare `HEAD` with no `--` was ambiguous to git: `landed` raised, and resume escalated.
    proj = projector(tmp_path)
    sha = commit_raw(proj.root, f"p1\n\n{TRAILER}: commit:p1:1:2\n")
    (proj.root / "HEAD").write_text("the person's file\n", encoding="utf-8")
    assert proj.landed("commit:p1:1:2") == sha
    assert proj.landed("commit:p1:1:3") is None


def test_show_signature_in_the_persons_config_does_not_become_the_sha(
    tmp_path: pathlib.Path,
) -> None:
    # Git prints the verifier's words before each signed commit; the old lookup returned
    # them as the sha, and resume recorded them. The verifier is the test's own, so those
    # words are the same on every machine, and the commit is signed by hand: no key needed.
    proj = projector(tmp_path)
    verifier = tmp_path / "verify.sh"
    verifier.write_text("#!/bin/sh\necho 'gpg: the test verifier' >&2\nexit 1\n", "utf-8")
    verifier.chmod(0o755)
    obj = tmp_path / "commit.txt"
    obj.write_text(
        f"tree {git(proj.root, 'rev-parse', 'HEAD^{tree}')}\n"
        f"parent {git(proj.root, 'rev-parse', 'HEAD')}\n"
        "author t <t@t> 1700000000 +0000\n"
        "committer t <t@t> 1700000000 +0000\n"
        "gpgsig -----BEGIN PGP SIGNATURE-----\n"
        " bm90IGEgc2lnbmF0dXJl\n"
        " -----END PGP SIGNATURE-----\n"
        f"\np1\n\n{TRAILER}: commit:p1:1:2\n",
        encoding="utf-8",
    )
    sha = git(proj.root, "hash-object", "-t", "commit", "-w", str(obj))
    git(proj.root, "update-ref", "HEAD", sha)
    git(proj.root, "config", "log.showSignature", "true")
    git(proj.root, "config", "gpg.program", str(verifier))
    git(proj.root, "config", "gpg.openpgp.program", str(verifier))
    assert proj.landed("commit:p1:1:2") == sha


def test_the_trailer_inside_a_sentence_is_not_the_trailer(tmp_path: pathlib.Path) -> None:
    # `splitlines` broke at U+2028, which git does not: the words inside the sentence read
    # as a line of their own, a false "landed", and the packet's commit was never made.
    proj = projector(tmp_path)
    landed = commit_raw(proj.root, f"p1\n\n{TRAILER}: commit:p1:1:1\n")
    commit_raw(
        proj.root,
        f"revert\n\nThe commit that said\u2028{TRAILER}: commit:p1:1:2\u2028was rolled back.\n",
    )
    assert proj.landed("commit:p1:1:2") is None
    assert proj.landed("commit:p1:1:1") == landed, "the same history still answers"


def test_a_start_the_history_was_rewritten_under_cannot_answer(tmp_path: pathlib.Path) -> None:
    # The run's starting head is no longer behind HEAD, so "after it" would take in the
    # rewritten copy of an earlier run's commit. Unknown, not "did not land".
    proj = projector(tmp_path)
    earlier = commit_raw(proj.root, f"p1\n\n{TRAILER}: commit:p1:1:2\n")
    start = commit_raw(proj.root, "this run starts here")
    git(proj.root, "checkout", "-q", "--orphan", "rewritten")
    commit_raw(proj.root, "base, reworded")
    git(proj.root, "cherry-pick", "--allow-empty", earlier)
    with pytest.raises(InfraError, match="rewritten"):
        proj.landed("commit:p1:1:2", since=start)


class RecordingSupervisor(ProcessSupervisor):
    """Every argv and cwd the projector hands the one spawn, then the real run."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[str, ...], pathlib.Path]] = []

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Completed:
        self.calls.append((argv, kwargs["cwd"]))
        return super().run(argv, **kwargs)


def test_landed_hands_its_root_since_and_supervisor_to_the_lookup(
    tmp_path: pathlib.Path,
) -> None:
    root = scenario_repo(tmp_path / "repo")
    start = git(root, "rev-parse", "HEAD")
    sha = commit_raw(root, f"p1\n\n{TRAILER}: commit:p1:1:2\n")
    sup = RecordingSupervisor()
    proj = Projector(root, RecordingIndexer(), supervisor=sup)
    assert proj.landed("commit:p1:1:2", since=start) == sha
    logs = [(argv, cwd) for argv, cwd in sup.calls if git_command(argv)[:2] == ("git", "log")]
    assert len(logs) == 1 and logs[0][1] == root
    assert f"^{start}" in logs[0][0], "the run's own commits only"
    assert any(
        git_command(argv)[:3] == ("git", "merge-base", "--is-ancestor") for argv, _ in sup.calls
    )


class _LockedIndex:
    """The graph database another process holds: sqlite's own error, not ours."""

    def reindex(self, changed: Sequence[pathlib.Path] | None = None) -> object:
        raise sqlite3.OperationalError("database is locked")


def test_an_index_that_cannot_take_the_reindex_is_infra(tmp_path: pathlib.Path) -> None:
    # The graph unavailable, as the packer reports it: InfraError, so the
    # run's §8 handler ends the run ABORTED instead of a raw sqlite3 error passing it by.
    proj = Projector(scenario_repo(tmp_path / "repo"), _LockedIndex())
    with pytest.raises(InfraError, match="database is locked") as raised:
        proj.reindex(["src/m.py"])
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
