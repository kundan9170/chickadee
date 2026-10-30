"""`by_operation_trailer` finds the operation's own trailer line, among the commits it is asked
about. A false `YES` is the dangerous answer: it forbids the retry, so an effect
that never landed is never redone. Over a real git repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repos import git, git_command, init_repo

from chickadee.errors import HarnessBug
from chickadee.executor.reconcile import TRAILER, Landed, by_operation_trailer, trailer_commit
from chickadee.process import Completed, ProcessSupervisor


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", commit="base")


def commit(root: Path, message: str) -> str:
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def packet(operation_id: str) -> str:
    return f"chickadee: p1 make-it-work\n\n{TRAILER}: {operation_id}\n"


def test_a_prefix_of_an_operation_that_landed_is_an_operation_that_did_not(
    repo: Path,
) -> None:
    sha = commit(repo, packet("commit:p1:1:51"))
    assert by_operation_trailer(repo, "commit:p1:1:5").landed is Landed.NO
    found = by_operation_trailer(repo, "commit:p1:1:51")
    assert found.landed is Landed.YES
    assert sha in found.evidence


def test_the_trailer_text_inside_a_sentence_is_not_the_trailer(repo: Path) -> None:
    commit(
        repo,
        f"revert the retry\n\nThis undoes the commit that said {TRAILER}: commit:p1:1:5 "
        "before it was rolled back.\n",
    )
    assert by_operation_trailer(repo, "commit:p1:1:5").landed is Landed.NO


@pytest.mark.parametrize("separator", ["\u2028", "\u0085", "\x0c", "\x1c"])
def test_a_line_is_what_git_calls_a_line(repo: Path, separator: str) -> None:
    # Python's `splitlines` breaks at each of these; git does not, so the words still sit
    # inside a sentence.
    commit(
        repo,
        f"revert\n\nThe commit that said{separator}{TRAILER}: op-1{separator}"
        "was rolled back.\n",
    )
    assert by_operation_trailer(repo, "op-1").landed is Landed.NO


@pytest.mark.writes_worktree("HEAD")
def test_a_file_named_head_in_the_worktree_changes_nothing(repo: Path) -> None:
    sha = commit(repo, packet("op-1"))
    (repo / "HEAD").write_text("the person's file\n", encoding="utf-8")
    found = by_operation_trailer(repo, "op-1")
    assert found.landed is Landed.YES
    assert sha in found.evidence
    assert by_operation_trailer(repo, "op-2").landed is Landed.NO


def signed_commit(root: Path, message: str, scratch: Path) -> str:
    """A commit carrying a signature header, written as an object: making one needs no key,
    only verifying it does."""
    obj = scratch / "commit.txt"
    obj.write_text(
        f"tree {git(root, 'rev-parse', 'HEAD^{tree}')}\n"
        f"parent {git(root, 'rev-parse', 'HEAD')}\n"
        "author t <t@example.invalid> 1700000000 +0000\n"
        "committer t <t@example.invalid> 1700000000 +0000\n"
        "gpgsig -----BEGIN PGP SIGNATURE-----\n"
        " bm90IGEgc2lnbmF0dXJl\n"
        " -----END PGP SIGNATURE-----\n"
        f"\n{message}",
        encoding="utf-8",
    )
    sha = git(root, "hash-object", "-t", "commit", "-w", str(obj))
    git(root, "update-ref", "HEAD", sha)
    return sha


def test_show_signature_in_the_persons_config_does_not_hide_the_commit(
    repo: Path, tmp_path: Path
) -> None:
    # `log.showSignature` prints the verifier's words before each signed commit. The
    # verifier here is the test's own, so those words are the same on every machine.
    verifier = tmp_path / "verify.sh"
    verifier.write_text("#!/bin/sh\necho 'gpg: the test verifier' >&2\nexit 1\n", "utf-8")
    verifier.chmod(0o755)
    sha = signed_commit(repo, packet("op-1"), tmp_path)
    git(repo, "config", "log.showSignature", "true")
    git(repo, "config", "gpg.program", str(verifier))
    git(repo, "config", "gpg.openpgp.program", str(verifier))
    found = by_operation_trailer(repo, "op-1")
    assert found.landed is Landed.YES
    assert f"commit {sha}" in found.evidence


def test_since_leaves_out_an_earlier_runs_commit_under_the_same_id(repo: Path) -> None:
    # Operation ids are unique within a run, not across runs: the earlier run's p1
    # committed under the id this run's p1 uses, in the history this run starts from.
    commit(repo, packet("commit:p1:1:2"))
    start = commit(repo, "the person's own commit, after that run and before this one")
    assert by_operation_trailer(repo, "commit:p1:1:2").landed is Landed.YES  # unscoped
    assert by_operation_trailer(repo, "commit:p1:1:2", since=start).landed is Landed.NO
    mine = commit(repo, packet("commit:p1:1:2"))
    found = by_operation_trailer(repo, "commit:p1:1:2", since=start)
    assert found.landed is Landed.YES
    assert mine in found.evidence


def test_a_trailer_far_back_is_found_with_since_and_without(repo: Path) -> None:
    # A run past fifty commits: its first commit is behind every count a cap might pick,
    # and "did not land" would make it a second time.
    start = git(repo, "rev-parse", "HEAD")
    first = commit(repo, packet("commit:p1:1:1"))
    for n in range(2, 53):
        git(repo, "commit", "-q", "--allow-empty", "-m", packet(f"commit:p{n}:1:1"))
    for since in (None, start):
        found = by_operation_trailer(repo, "commit:p1:1:1", since=since)
        assert found.landed is Landed.YES, since
        assert first in found.evidence, since


class ArgvRecorder(ProcessSupervisor):
    """Every argv handed to the one spawn, then the real run."""

    def __init__(self) -> None:
        super().__init__()
        self.argvs: list[tuple[str, ...]] = []

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Completed:
        self.argvs.append(argv)
        return super().run(argv, **kwargs)


def test_git_hands_back_only_the_commits_that_hold_the_words(repo: Path) -> None:
    # Unfiltered, a history of any length crosses into memory whole, every message of it;
    # the answer is the same, so only the argv shows the difference.
    sha = commit(repo, packet("commit:p1:1:2"))
    sup = ArgvRecorder()
    assert trailer_commit(repo, "commit:p1:1:2", supervisor=sup) == sha
    (log,) = [argv for argv in sup.argvs if git_command(argv)[:2] == ("git", "log")]
    assert "--fixed-strings" in log
    assert f"--grep={TRAILER}: commit:p1:1:2" in log


@pytest.mark.parametrize("pattern_type", ["basic", "extended", "perl"])
def test_the_persons_grep_config_does_not_make_the_id_a_pattern(
    repo: Path, pattern_type: str
) -> None:
    # Read as a pattern, `+` and `[1]` do not match themselves: git would filter out the
    # one commit searched for, and the answer would be `NO`.
    sha = commit(repo, packet("op+[1]"))
    git(repo, "config", "grep.patternType", pattern_type)
    found = by_operation_trailer(repo, "op+[1]")
    assert found.landed is Landed.YES
    assert sha in found.evidence


def test_no_says_what_was_searched(repo: Path) -> None:
    start = git(repo, "rev-parse", "HEAD")
    unscoped = by_operation_trailer(repo, "commit:p1:1:1")
    assert unscoped.landed is Landed.NO
    assert unscoped.evidence.endswith("in HEAD's history")
    scoped = by_operation_trailer(repo, "commit:p1:1:1", since=start)
    assert scoped.landed is Landed.NO
    assert scoped.evidence.endswith(f"in the commits after {start}")


def test_a_since_the_history_was_rewritten_under_is_unknown(repo: Path) -> None:
    # The earlier run's commit, then this run's start. The person then rewrites the
    # history, and a copy of that commit is no longer behind `since`.
    earlier = commit(repo, packet("commit:p1:1:2"))
    start = commit(repo, "this run starts here")
    git(repo, "checkout", "-q", "--orphan", "rewritten")
    commit(repo, "base, reworded")
    git(repo, "cherry-pick", "--allow-empty", earlier)
    found = by_operation_trailer(repo, "commit:p1:1:2", since=start)
    assert found.landed is Landed.UNKNOWN
    assert "rewritten" in found.evidence
    assert not found.may_retry


def test_a_since_ending_in_a_dot_reaches_no_commit_head_cannot(repo: Path) -> None:
    # Pasted into a range, `other.` makes `other...HEAD`, which also takes in the commits
    # of `other` that never reached HEAD.
    git(repo, "checkout", "-q", "-b", "other")
    commit(repo, packet("op-9"))
    git(repo, "checkout", "-q", "-")
    found = by_operation_trailer(repo, "op-9", since="other.")
    assert found.landed is Landed.UNKNOWN
    assert not found.may_retry


def test_a_since_git_does_not_know_is_unknown_never_no(repo: Path) -> None:
    gone = "0" * 40
    found = by_operation_trailer(repo, "commit:p1:1:2", since=gone)
    assert found.landed is Landed.UNKNOWN
    assert not found.may_retry


def test_a_directory_that_is_not_a_repository_is_unknown_never_no(tmp_path: Path) -> None:
    # A `.git` that points nowhere, so git cannot fall back to a repository around the
    # test's temporary directory.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / ".git").write_text("gitdir: /nonexistent\n", encoding="utf-8")
    found = by_operation_trailer(plain, "commit:p1:1:2")
    assert found.landed is Landed.UNKNOWN
    assert not found.may_retry


@pytest.mark.parametrize("since", ["", "--output=log.txt", "-n1", "ab\x00c"])
def test_a_since_that_names_no_revision_is_refused(repo: Path, since: str) -> None:
    # `..HEAD` is an empty range, `--output=` an option git obeys, and a NUL an argv the
    # spawn refuses: each would end in `NO`, or in an error nobody catches.
    with pytest.raises(HarnessBug, match="must name a revision"):
        by_operation_trailer(repo, "commit:p1:1:2", since=since)
    assert not list(repo.glob("log.txt*"))


@pytest.mark.parametrize("operation_id", ["", " op-1", "op-1 ", "op-1\n", "op 1"])
def test_an_operation_id_that_is_not_one_token_is_refused(
    repo: Path, operation_id: str
) -> None:
    # Matched against stripped lines, such an id never matches: `NO`, the verdict that
    # permits a retry of an effect that may have landed.
    commit(repo, f"squash\n\n{TRAILER}: {operation_id}\n")
    with pytest.raises(HarnessBug, match="one token"):
        by_operation_trailer(repo, operation_id)


def test_an_operation_id_holding_a_nul_is_refused(repo: Path) -> None:
    # The id is git's filter, and an argv with a NUL is refused by the spawn itself, with
    # an error nothing on the way out catches.
    with pytest.raises(HarnessBug, match="one token"):
        by_operation_trailer(repo, "op\x00-1")
