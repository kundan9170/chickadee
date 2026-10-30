"""`/handoff` at the terminal: with no choice, the
session's worktree against its base, its diff and every claim the session could not make, then
the choices the engine accepts; leaving the REPL with work not yet handed off asks the same, and
Enter leaves it pending. A choice the engine refuses is never offered, and says why.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from engines import scripted
from hosting import SID, repository, short_root
from repl_driver import Script, Typed, printed
from repos import git

from chickadee.context.questions import open_reclaimed
from chickadee.context.store import sessions_root
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.state.roots import roots
from chickadee_cli.commands import LEAVE_PENDING, RUNNING_HINT
from chickadee_cli.render.diff import handoff_lines, shown_line
from chickadee_cli.render.style import Style
from chickadee_cli.repl import Repl
from chickadee_serve.frames import ErrorBody, ErrorCode
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.host import host_in_process
from chickadee_wire.models import HandoffSummary

BRANCH = "chickadee/review-S-wire"


def _session_with_a_worktree(monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A repository on `main` with one commit, and session `SID` owning a worktree whose
    branch changed `app.py` in one commit: what one `coder` call leaves."""
    repo = repository(monkeypatch, short_root())
    (repo / ".git" / "info" / "exclude").write_text(".chickadee/\n")
    (repo / "app.py").write_text("print('old')\n")
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "seed")
    base = git(repo, "rev-parse", "HEAD")
    worktree = repo / ".git" / "chickadee-worktrees" / SID
    git(repo, "worktree", "add", "-q", "-b", BRANCH, str(worktree))
    (worktree / "app.py").write_text("print('new')\n")
    git(worktree, "commit", "-q", "-am", "task T1")
    with open_reclaimed(sessions_root(roots(repo)), SID) as session:
        session.record_workspace(
            WorkspaceRef(
                mode="worktree",
                lease_id="lease",
                source_ref="main",
                source_head=base,
                branch=BRANCH,
                worktree_path=str(worktree),
                handoff="pending",
            )
        )
    assert git(repo, "status", "--porcelain") == "", "the checkout is clean"
    return repo, worktree


class _Prompted(Script):
    """The person's lines, and each prompt they were shown: a question's prompt is not in
    the output, it is the line they type on."""

    def __init__(self, lines: Sequence[Typed]) -> None:
        super().__init__(lines)
        self.prompts: list[str] = []

    def read(self, prompt_text: str, default: str = "") -> str:
        self.prompts.append(prompt_text)
        return super().read(prompt_text, default)


ASKED = "(Enter leaves it pending) "


@contextmanager
def _repl(repo: Path, lines: Sequence[Typed]) -> Iterator[_Prompted]:
    source = _Prompted(lines)
    with host_in_process(repo) as host, host.client() as client:
        Repl(client, SID, source=source, write=source.out.append, style=Style(tty=False)).run()
    yield source


def test_handoff_reviews_the_diff_and_carries_out_the_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, worktree = _session_with_a_worktree(monkeypatch)
    scripted(monkeypatch, ())
    lines = [
        Typed("/handoff", after=printed("Ctrl-D to leave")),
        Typed("apply", after=printed("+print('new')")),
    ]
    with _repl(repo, lines) as source:
        out = "\n".join(source.out)
        joined = " ".join(line.strip() for line in source.out)  # a pipe wraps at 80
    assert source.prompts.count(f"bring, apply, squash, keep, discard? {ASKED}") == 1
    assert f"worktree {BRANCH} at {worktree}" in joined
    assert "1 commit(s) from" in out and "onto main" in out
    assert "-print('old')" in out and "+print('new')" in out, "the diff, verbatim"
    assert "unverified" not in out and "not verified" not in out, (
        "the owner's call: the review shows what bears on the choice, not the record's claims"
    )
    assert "apply: fast-forwarded main" in joined and "nothing was pushed" in out
    assert (repo / "app.py").read_text() == "print('new')\n"
    assert sum(ASKED in p for p in source.prompts) == 1, "nothing is left to ask on leaving"


def test_leaving_with_work_not_handed_off_asks_and_enter_leaves_it_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _worktree = _session_with_a_worktree(monkeypatch)
    scripted(monkeypatch, ())
    with _repl(repo, [Typed("/exit", after=printed("Ctrl-D to leave")), Typed("")]) as source:
        out = "\n".join(source.out)
    assert source.prompts[-1].endswith(ASKED)
    assert "+print('new')" in out, "the review is shown on the way out"
    assert LEAVE_PENDING in out
    assert (repo / "app.py").read_text() == "print('old')\n"


def test_ctrl_d_at_the_question_leaves_it_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _worktree = _session_with_a_worktree(monkeypatch)
    scripted(monkeypatch, ())
    with _repl(repo, []) as source:  # Ctrl-D at the prompt, and again at the question
        out = "\n".join(source.out)
    assert LEAVE_PENDING in out
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_a_refused_choice_is_not_offered_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """The person's branch moved after the worktree was made: `apply` and `squash` would not
    fast-forward, so `bring`, `keep` and `discard` are offered
    (`HandoffPreview.refusal`)."""
    repo, worktree = _session_with_a_worktree(monkeypatch)
    (repo / "notes.txt").write_text("mine\n")
    git(repo, "add", "notes.txt")
    git(repo, "commit", "-q", "-m", "moved on")
    scripted(monkeypatch, ())
    lines = [
        Typed("/handoff", after=printed("Ctrl-D to leave")),
        Typed("keep", after=printed("squash refused")),
        Typed("/handoff push", after=printed("keep: kept")),
    ]
    with _repl(repo, lines) as source:
        out = " ".join(source.out)  # a pipe wraps at 80
    assert "apply refused: the source branch moved during the run" in out
    assert f"bring, keep, discard? {ASKED}" in source.prompts
    assert "/handoff [bring|apply|squash|keep|discard]" in out
    assert worktree.is_dir()
    assert sum(ASKED in p for p in source.prompts) == 1, "a kept worktree is not asked again"


def test_discard_is_offered_over_uncommitted_work_and_says_what_it_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Work no commit holds is listed, every choice says what it does with it, and
    `discard` deletes it with the worktree and the branch."""
    repo, worktree = _session_with_a_worktree(monkeypatch)
    (worktree / "scratch.txt").write_text("not committed\n")
    scripted(monkeypatch, ())
    lines = [
        Typed("/handoff", after=printed("Ctrl-D to leave")),
        Typed("discard", after=printed("+print('new')")),
    ]
    with _repl(repo, lines) as source:
        out = " ".join(line.strip() for line in source.out)  # a pipe wraps at 80
    assert f"bring, apply, squash, keep, discard? {ASKED}" in source.prompts
    assert "no commit does, which no choice lands: scratch.txt" in out
    assert "the uncommitted work stays in the worktree" in out
    assert f"discard: delete the worktree and branch {BRANCH}" in out
    assert "deleted work no commit held: scratch.txt" in out
    assert not worktree.exists()
    assert (repo / "app.py").read_text() == "print('old')\n", "main is untouched"


def test_a_handoff_while_work_runs_says_how_to_end_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine refuses a handoff while a turn runs (`EngineBusy`, 409 `BUSY`): the person is
    told the way out, and the REPL goes on."""
    repo, worktree = _session_with_a_worktree(monkeypatch)
    scripted(monkeypatch, ())

    def busy(*_args: object, **_kwargs: object) -> object:
        raise WireError(409, ErrorBody(code=ErrorCode.BUSY, message="run R-1 is in flight"))

    monkeypatch.setattr(Client, "finalize", busy)
    lines = [
        Typed("/handoff apply", after=printed("Ctrl-D to leave")),
        Typed("/exit"),
        Typed(""),
    ]
    with _repl(repo, lines) as source:
        out = " ".join(line.strip() for line in source.out)  # a pipe wraps at 80
    assert "/handoff: run R-1 is in flight" in out
    assert RUNNING_HINT in out
    assert worktree.is_dir() and (repo / "app.py").read_text() == "print('old')\n"


def test_a_diffs_control_characters_are_shown_not_sent_to_the_terminal() -> None:
    """The diff's bytes are the coder's: an escape sequence in them would move the cursor or
    hide a line on the one screen the person decides from."""
    summary = HandoffSummary.model_validate(
        {
            "session_id": SID,
            "git": None,
            "spend": {
                "scope": "session",
                "tokens_in": 0,
                "tokens_out": 0,
                "cached_in": 0,
                "cost_usd": 0.0,
                "wall_ms": 0,
            },
            "episodes": [],
            "last_outcome": None,
            "unverified": [],
            "diff": "+evil\x1b[1A\x1b[2K\tok",
        }
    )
    assert shown_line("+evil\x1b[1A\x1b[2K\tok") == "+evil\\x1b[1A\\x1b[2K\tok"
    assert all("\x1b" not in line for line in handoff_lines(summary, Style(tty=False)))


def test_a_server_that_sends_no_words_for_the_choices_shows_their_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final review of 2026-10-25: a server one minor version older sends no `offers`, and
    each choice printed as `name: ` with nothing after it."""
    repo, _worktree = _session_with_a_worktree(monkeypatch)
    scripted(monkeypatch, ())
    with host_in_process(repo) as host, host.client() as client:
        summary = client.handoff(SID).model_copy(update={"offers": None})

    shown = [line.strip() for line in handoff_lines(summary, Style(tty=False))]

    assert "bring" in shown and "discard" in shown
    assert not any(line.endswith(":") for line in shown)
