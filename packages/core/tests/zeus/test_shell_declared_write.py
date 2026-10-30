"""Zeus's shell writes the worktree, every command recorded (D199 amendment 1, items 11-16; it
was D188 decision 6's declared write).

Every Zeus command that is not a pure read runs with the worktree writable, and is an undo unit:
a snapshot ref before `effect_started`, its D40 lifecycle on the session log
(`shell_effect` lines), and `/undo <operation_id>`. In the session's worktree a command that
left changes is committed as its own commit, so no later packet folds it in. A tracked source
file it changed is put back, unless the line is git's own operation; a path the person denies
is put back too; a change to git's `config` or `hooks` fails the call and puts everything back.

The moved answers, each pinned below: nothing is declared (`effects=` is refused,
naming its retirement); a write runs at its class's tier, so `echo > uv.lock` is T0 and unasked
where a declaration made it T1; a write is no longer held to a declaration, so a stray file
stays and is recorded; a command that exits unsuccessfully keeps what it wrote, where a
declared write was put back.

The shell here runs over a real repository with no sandbox, the person having lifted the
unsandboxed floor, so the step, not the wall, is what is under test.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from repos import git, init_repo
from tool_args import checked

from chickadee.approval.prefs import merge, parse_level
from chickadee.approval.tier import Preferences
from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import InvalidArgument, ToolResult
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.executor.policy import isolation_for
from chickadee.init.detect import resolve
from chickadee.init.lease import take_write_lease
from chickadee.isolation.none import Passthrough
from chickadee.profiles.loader import ProfileSet
from chickadee.shell import Shell, ShellClaim, ShellCommand, WriteDenied
from chickadee.tools.spec import spec
from chickadee.tools.validate import InvalidArguments, validate
from chickadee.workspace import ops_ref, ref_exists, restore_operation
from chickadee.zeus.writes import ShellWrites, shell_effects

SID = "S-write"


class WriteConfirmer:
    """The person, answering every confirmation the same way, and keeping what was asked."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return self.answer


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": "x = 1\n", "uv.lock": "old\n"})


def shell(
    tmp_path: Path,
    root: Path,
    authority: WriteConfirmer | None,
    *,
    commits: bool = True,
    writes: bool = True,
    prefs: Preferences | None = None,
) -> tuple[Shell, Session]:
    session = Session.create(tmp_path / "sessions", SID)
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=root, session_blob=session.blob),
        vision=Vision("z"),
    )
    recorded = ShellWrites(
        session,
        root,
        commits=commits,
        profiles=ProfileSet.load(repo=root).all(),
        prefs=None if prefs is None else (lambda: prefs),
    )
    built = Shell(
        Executor(Passthrough()),
        root,
        lambda data: reader.store("output", data),
        authority=authority,
        allow_unsandboxed_shell=True,
        writes=recorded if writes else None,
    )
    return built, session


def run(sh: Shell, command: str, op: str = "S:c1") -> object:
    return sh.run(checked("shell", {"command": command}), operation_id=op)


def effects_of(session: Session) -> tuple[object, ...]:
    return shell_effects(session.state, lambda e: session.blob(e.payload_sha))


# -- nothing is declared ---------------------------------------------------------------------


def test_effects_is_refused_naming_its_retirement() -> None:
    found = spec("shell")
    assert found is not None
    got = validate(found, {"command": "x", "effects": ["uv.lock"]})
    assert isinstance(got, InvalidArguments)
    [(field, problem)] = [(v.field, v.problem) for v in got.violations]
    assert field == "effects" and problem.startswith("effects is retired")


def test_a_shell_with_no_place_to_record_a_write_runs_every_command_sealed(
    tmp_path: Path, repo: Path
) -> None:
    """Without a `ShellWrites` (a shell built for no session) there is nowhere to record a
    write, so every request keeps the repository read-only."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), writes=False)
    parsed = sh.parse(checked("shell", {"command": "echo y > uv.lock"}))
    assert isinstance(parsed, ShellCommand) and not parsed.writable
    assert isolation_for(sh.request(parsed)).repo_writable is False
    session.close()


def test_a_pure_read_runs_sealed_and_records_nothing(tmp_path: Path, repo: Path) -> None:
    """D199's cost, paid only where a command may write: `git status` takes no snapshot."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    got = run(sh, "git status --short")
    assert isinstance(got, ToolResult)
    assert effects_of(session) == ()
    parsed = sh.parse(checked("shell", {"command": "echo y > uv.lock"}))
    assert isinstance(parsed, ShellCommand) and parsed.writable, "a redirect writes"
    session.close()


# -- the write -------------------------------------------------------------------------------


def test_a_write_runs_at_its_tier_and_its_lifecycle_is_on_the_session_log(
    tmp_path: Path, repo: Path
) -> None:
    """A redirect into a lock file is T0, so nobody is asked; it was T1 as a declared write."""
    asked = WriteConfirmer(True)
    sh, session = shell(tmp_path, repo, asked, commits=False)
    got = run(sh, "echo new > uv.lock")
    assert isinstance(got, ToolResult), got
    assert asked.claims == []
    assert (repo / "uv.lock").read_text() == "new\n"
    (effect,) = effects_of(session)
    assert effect.operation_id == "S:c1"  # type: ignore[attr-defined]
    assert effect.surface == ()  # type: ignore[attr-defined]
    assert effect.written == ("uv.lock",)  # type: ignore[attr-defined]
    assert effect.events == ("effect_intended", "effect_started", "effect_completed")  # type: ignore[attr-defined]
    assert effect.outcome == "succeeded"  # type: ignore[attr-defined]
    assert ref_exists(repo, ops_ref(repo, "S:c1", run_id=SID)), "D107: parked before it ran"
    session.close()


def test_a_write_is_undone_from_its_snapshot(tmp_path: Path, repo: Path) -> None:
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    run(sh, "echo new > uv.lock")
    restore_operation(repo, "S:c1", ("uv.lock",), run_id=SID)
    assert (repo / "uv.lock").read_text() == "old\n"
    session.close()


@pytest.mark.writes_worktree("gen.txt", "app.py", "uv.lock")
def test_a_source_file_a_command_changed_is_put_back_and_the_rest_kept(
    tmp_path: Path, repo: Path
) -> None:
    """`app.py` is tracked Python, so the coder's; the lock file and a new file
    are not, and stay, committed on their own."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    got = run(sh, "echo y > app.py && echo new > uv.lock && echo data > gen.txt")
    assert isinstance(got, ToolResult)
    assert "put back app.py: source is coder's" in got.render()
    assert (repo / "app.py").read_text() == "x = 1\n"
    assert (repo / "uv.lock").read_text() == "new\n"
    (effect,) = effects_of(session)
    assert effect.written == ("gen.txt", "uv.lock")  # type: ignore[attr-defined]
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == [
        "gen.txt",
        "uv.lock",
    ]
    session.close()


@pytest.mark.writes_worktree("backend/app/main.py", "README.md", "web/main.js")
def test_a_new_source_file_a_command_wrote_is_removed_and_named(
    tmp_path: Path, repo: Path
) -> None:
    """D202 decision 1, the owner's run: Zeus wrote a whole app with `cat > file << EOF` into a
    repository where nothing was tracked yet. A new file of any installed profile's language
    is the coder's, removed and named; a new file of no language is kept."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    got = run(
        sh,
        "mkdir -p backend/app web && echo 'app = 1' > backend/app/main.py"
        " && echo '# hi' > README.md && echo 'x()' > web/main.js",
    )
    assert isinstance(got, ToolResult)
    assert "put back backend/app/main.py, web/main.js: source is coder's" in got.render()
    assert "a new file included" in got.render()
    assert not (repo / "backend/app/main.py").exists()
    assert not (repo / "web/main.js").exists()
    assert (repo / "README.md").read_text() == "# hi\n"
    (effect,) = effects_of(session)
    assert effect.written == ("README.md",)  # type: ignore[attr-defined]
    session.close()


@pytest.mark.writes_worktree("README.md")
def test_a_new_source_file_is_the_coders_where_no_profile_is_detected(tmp_path: Path) -> None:
    """An empty repository detects no profile, which judged nothing; every
    installed profile judges a file the repository does not hold yet."""
    bare = init_repo(tmp_path / "bare", {"README.md": "# x\n"})
    sh, session = shell(tmp_path, bare, WriteConfirmer(True))
    got = run(sh, "printf 'print(1)\\n' > main.py")
    assert isinstance(got, ToolResult) and "put back main.py" in got.render()
    assert not (bare / "main.py").exists()
    session.close()


@pytest.mark.writes_worktree("node_modules/pkg/index.js", "node_modules/pkg/package.json")
def test_an_installs_files_are_kept_though_they_are_javascript(
    tmp_path: Path, repo: Path
) -> None:
    """`node_modules/**/*.js` is an install's output, not source anyone
    wrote, even in a repository with no `.gitignore` to hide it."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(
        sh,
        "mkdir -p node_modules/pkg && echo 'x' > node_modules/pkg/index.js"
        " && echo '{}' > node_modules/pkg/package.json",
    )
    assert isinstance(got, ToolResult) and "put back" not in got.render()
    assert (repo / "node_modules/pkg/index.js").exists()
    session.close()


@pytest.mark.writes_worktree("app.py")
def test_git_working_on_its_history_keeps_the_source_it_brings(
    tmp_path: Path, repo: Path
) -> None:
    """A switch brings source as a commit holds it: git's own operation, kept (item 14)."""
    git(repo, "switch", "-q", "-c", "other")
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "other")
    git(repo, "switch", "-q", "-")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git switch -q other")
    assert isinstance(got, ToolResult) and "put back" not in got.render()
    assert (repo / "app.py").read_text() == "x = 2\n"
    session.close()


def _other_branch(repo: Path) -> None:
    """A branch `other` whose `app.py` is `x = 2`, the repository left on the first."""
    git(repo, "switch", "-q", "-c", "other")
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "other")
    git(repo, "switch", "-q", "-")


@pytest.mark.writes_worktree("app.py")
@pytest.mark.parametrize(
    "line",
    [
        "cd . && git switch -q other",
        "git switch -q other 2>&1 | tail -5",
        "git switch -q other 2>/dev/null",
        "bash -c 'git switch -q other'",
        "git switch -q other || true",
        "timeout 30 git checkout -q other && git log --oneline -1",
    ],
)
def test_how_a_git_line_is_joined_does_not_put_back_what_git_brought(
    tmp_path: Path, repo: Path, line: str
) -> None:
    """Zeus could not check out: a `cd`, a pipe or a wrapper beside the switch made the line
    not git's, so the source the switch brought was put back over the branch it reached."""
    _other_branch(repo)
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, line)
    assert isinstance(got, ToolResult) and "put back" not in got.render()
    assert (repo / "app.py").read_text() == "x = 2\n"
    assert git(repo, "status", "--porcelain", "--", "app.py") == ""
    session.close()


@pytest.mark.writes_worktree("app.py", "build.log")
def test_git_beside_a_writing_command_keeps_what_git_brought(
    tmp_path: Path, repo: Path
) -> None:
    """A switch beside a command that may write (a test run, a build): the source that holds
    what the commit git reached holds is git's, and kept."""
    _other_branch(repo)
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git switch -q other && touch build.log")
    assert isinstance(got, ToolResult) and "put back" not in got.render()
    assert (repo / "app.py").read_text() == "x = 2\n"
    assert (repo / "build.log").exists()
    session.close()


@pytest.mark.writes_worktree("app.py")
def test_a_write_over_what_git_brought_is_put_back_to_what_git_brought(
    tmp_path: Path, repo: Path
) -> None:
    """The command's own change is undone, git's is not: the file holds the branch it reached,
    never the one it left, which would leave it dirty on the new branch."""
    _other_branch(repo)
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git switch -q other && echo 'x = 9' > app.py")
    assert isinstance(got, ToolResult) and "put back app.py" in got.render()
    assert (repo / "app.py").read_text() == "x = 2\n"
    assert git(repo, "status", "--porcelain", "--", "app.py") == ""
    session.close()


@pytest.mark.writes_worktree("app.py", "step.sh")
@pytest.mark.parametrize(
    "script",
    [
        "git merge -q --no-ff -m m other\n",
        "git merge -q --no-ff -m m other\ngit fetch -q . HEAD:refs/remotes/o/x\n",
    ],
)
def test_a_commit_a_program_made_unseen_is_not_kept_and_is_named(
    tmp_path: Path, repo: Path, script: str
) -> None:
    """Git the line does not show (a script) made the commit `HEAD` is on,
    even fetched into a remote-tracking ref of its own. Nothing could refuse it before it ran:
    its source is put back on disk and in the index, and the answer names the commit."""
    _other_branch(repo)
    (repo / "step.sh").write_text(script)
    before = git(repo, "rev-parse", "HEAD")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "sh step.sh")
    assert isinstance(got, ToolResult)
    text = got.render()
    assert "put back app.py" in text and "a commit this command made" in text
    assert f"git reset --soft {before[:8]}" in text
    assert (repo / "app.py").read_text() == "x = 1\n"
    assert git(repo, "show", ":app.py") == "x = 1"
    session.close()


@pytest.mark.writes_worktree("app.py", "step.sh")
def test_a_switch_a_program_made_unseen_keeps_what_git_brought(
    tmp_path: Path, repo: Path
) -> None:
    """A script that switches moves `HEAD` to history, so the line is read as
    git beside a write, and the branch it reached stays whole."""
    _other_branch(repo)
    (repo / "step.sh").write_text("git switch -q other\n")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "sh step.sh")
    assert isinstance(got, ToolResult) and "put back" not in got.render()
    assert (repo / "app.py").read_text() == "x = 2\n"
    assert git(repo, "status", "--porcelain", "--", "app.py") == ""
    session.close()


def _patch(repo: Path) -> None:
    """`p.diff`, a patch the model wrote that makes `app.py` `x = 9`; the tree left clean."""
    (repo / "app.py").write_text("x = 9\n")
    (repo / "p.diff").write_text(git(repo, "diff") + "\n")
    git(repo, "checkout", "--", "app.py")


@pytest.mark.writes_worktree("app.py", "p.diff", "b.log")
@pytest.mark.parametrize(
    "line",
    [
        "git apply --cached p.diff",
        "git apply --index p.diff",
        "git rm -q app.py",
        "git update-index --force-remove app.py",
        "git checkout -q other -- app.py && touch b.log",
    ],
)
def test_source_staged_by_a_command_is_put_back_in_the_index(
    tmp_path: Path, repo: Path, line: str
) -> None:
    """A commit takes source from the index, so a model's write staged there
    is put back as one on disk is, whatever staged it. Disk, index and `HEAD` agree after."""
    _other_branch(repo)
    _patch(repo)
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, line)
    assert isinstance(got, ToolResult) and "put back the staged app.py" in got.render()
    assert (repo / "app.py").read_text() == "x = 1\n"
    assert git(repo, "status", "--porcelain", "--", "app.py") == ""
    session.close()


@pytest.mark.writes_worktree("app.py", "notes.md")
def test_a_staged_change_git_history_holds_is_kept(tmp_path: Path, repo: Path) -> None:
    """Unstaging back to what `HEAD` holds is git's, beside a write or not."""
    (repo / "app.py").write_text("x = 5\n")
    git(repo, "add", "app.py")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git restore --staged app.py && touch notes.md")
    assert isinstance(got, ToolResult) and "put back the staged" not in got.render()
    assert git(repo, "show", ":app.py") == "x = 1"
    assert (repo / "app.py").read_text() == "x = 5\n"
    session.close()


@pytest.mark.parametrize(
    ("line", "recorder"),
    [
        ("echo 'x = 9' > app.py && git commit -qam evil", "git commit"),
        ("sed -i s/1/9/ app.py && git add app.py", "git add"),
        ("echo 'x = 9' > app.py; git stash", "git stash"),
        ("git am fix.patch", "git am"),
        ("git add tmp.txt && git mv -f tmp.txt app.py", "git add"),
        ("git merge -q --no-ff -m m other && touch b.log", "git merge"),
        ("git hash-object -w app.py", "git hash-object"),
    ],
)
def test_git_recording_a_write_beside_it_is_refused_before_it_runs(
    tmp_path: Path, repo: Path, line: str, recorder: str
) -> None:
    """The put-back restores the file on disk and never in git, so a line whose
    git would keep the model's write in git's history is refused, nothing spawned."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    head = git(repo, "rev-parse", "HEAD")
    got = run(sh, line)
    assert isinstance(got, InvalidArgument) and got.field == "command"
    assert recorder in got.why and got.do
    assert (repo / "app.py").read_text() == "x = 1\n"
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""
    session.close()


@pytest.mark.writes_worktree("app.py", "notes.md")
def test_git_recording_on_its_own_still_commits(tmp_path: Path, repo: Path) -> None:
    """A line that is only git records what the tree already holds."""
    (repo / "notes.md").write_text("n\n")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git add -A && git commit -qm wip && git log --oneline -1")
    assert isinstance(got, ToolResult) and "put back" not in got.render()
    assert git(repo, "log", "-1", "--format=%s") == "wip"
    session.close()


@pytest.mark.writes_worktree("app.py", "fix.diff")
def test_a_patch_applied_with_git_is_put_back_like_any_source_edit(
    tmp_path: Path, repo: Path
) -> None:
    """`git apply` writes a patch a model wrote: not git's own operation (`PATCHING_GIT`)."""
    (repo / "app.py").write_text("x = 9\n")
    (repo / "fix.diff").write_text(git(repo, "diff") + "\n")
    git(repo, "checkout", "--", "app.py")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git apply fix.diff")
    assert isinstance(got, ToolResult) and "put back app.py" in got.render()
    assert (repo / "app.py").read_text() == "x = 1\n"
    session.close()


@pytest.mark.writes_worktree("app.py", "tmp.txt")
def test_a_git_subcommand_that_writes_model_content_is_put_back(
    tmp_path: Path, repo: Path
) -> None:
    """Review of #592: `git mv -f` moves a file the model wrote over tracked source; only
    `HISTORY_GIT` keeps what it brings. Its `git add` beside it is D228's refusal, pinned
    with the other recorders above; on its own the move is put back."""
    (repo / "tmp.txt").write_text("evil = 1\n")
    git(repo, "add", "tmp.txt")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "git mv -f tmp.txt app.py")
    assert isinstance(got, ToolResult) and "put back app.py" in got.render()
    assert (repo / "app.py").read_text() == "x = 1\n"
    session.close()


def test_only_a_git_command_gets_gits_directories_writable(tmp_path: Path, repo: Path) -> None:
    """Review of #592: in a linked worktree a non-git command never gets the repository's
    objects and refs, which lie outside the worktree and outside its snapshot."""
    tree = tmp_path / "tree"
    git(repo, "worktree", "add", "-q", "-b", "side", str(tree))
    sh, session = shell(tmp_path, tree, WriteConfirmer(True))
    common = (repo / ".git").resolve()

    def writable(line: str) -> tuple[Path, ...]:
        parsed = sh.parse(checked("shell", {"command": line}))
        assert isinstance(parsed, ShellCommand)
        return sh.request(parsed).also_writable

    assert writable("rm -rf build") == (tree.resolve(),)
    assert common / "refs" in writable("git switch -c other")
    session.close()


@pytest.mark.writes_worktree("infra", "infra/main.tf", "uv.lock")
def test_a_path_the_person_denies_is_put_back(tmp_path: Path, repo: Path) -> None:
    denied = merge([parse_level({"approval": {"deny": ["infra/**"]}}, "user")])
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), prefs=denied)
    got = run(sh, "mkdir -p infra && echo s > infra/main.tf && echo new > uv.lock")
    assert isinstance(got, ToolResult)
    assert "put back infra/main.tf: your approval.toml denies" in got.render()
    assert not (repo / "infra" / "main.tf").exists()
    assert (repo / "uv.lock").read_text() == "new\n"
    session.close()


def test_a_write_in_the_sessions_worktree_is_committed_on_its_own(
    tmp_path: Path, repo: Path
) -> None:
    """Committed right after the command, under the harness's identity and the operation's
    trailer, so no later packet folds it into its commit."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    head = git(repo, "rev-parse", "HEAD")
    got = run(sh, "echo new > uv.lock")
    assert isinstance(got, ToolResult)
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "rev-parse", "HEAD~1") == head
    body = git(repo, "log", "-1", "--format=%an%n%B")
    assert body.splitlines()[0] == "chickadee"
    assert "Chickadee-Operation: S:c1" in body
    assert "committed as" in got.render() and "/undo S:c1" in got.render()
    (effect,) = effects_of(session)
    assert effect.post_state == git(repo, "rev-parse", "HEAD")  # type: ignore[attr-defined]
    session.close()


def test_a_command_that_changed_nothing_commits_nothing(tmp_path: Path, repo: Path) -> None:
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    head = git(repo, "rev-parse", "HEAD")
    run(sh, "true")
    assert git(repo, "rev-parse", "HEAD") == head
    session.close()


def test_a_write_in_the_persons_checkout_is_never_committed(tmp_path: Path, repo: Path) -> None:
    """Before the move the write lands in the person's checkout: their tree, not the harness's
    to commit. The first `coder` call's dirty-checkout refusal answers for it."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    head = git(repo, "rev-parse", "HEAD")
    run(sh, "echo new > uv.lock")
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == "M uv.lock"
    session.close()


@pytest.mark.writes_worktree("notes.txt")
def test_only_what_the_command_wrote_is_committed(tmp_path: Path, repo: Path) -> None:
    """Anything dirty before it (a cache an earlier check left, a file the person dropped
    there) is not this command's, and stays out of its commit."""
    (repo / "notes.txt").write_text("not this write's\n")
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    run(sh, "echo new > uv.lock")
    assert git(repo, "show", "--name-only", "--format=", "HEAD") == "uv.lock"
    assert git(repo, "status", "--porcelain") == "?? notes.txt"
    session.close()


def test_a_linked_worktree_binds_what_git_writes_and_never_its_config_or_hooks(
    tmp_path: Path, repo: Path
) -> None:
    """Design 4a: a linked worktree's HEAD and index are in its own git directory, and its
    objects, refs and reflogs in the common one, outside it, so a git command there needs those
    writable. Never the common directory whole: `config` and `hooks/` run later, in the
    person's own git calls, outside any sandbox."""
    tree = tmp_path / "tree"
    git(repo, "worktree", "add", "-q", "-b", "side", str(tree))
    common = (repo / ".git").resolve()
    session = Session.create(tmp_path / "sessions", SID)
    assert ShellWrites(session, tree, commits=True).git_dirs() == (
        common / "objects",
        common / "refs",
        common / "logs",
        common / "worktrees" / "tree",
    )
    assert ShellWrites(session, repo, commits=False).git_dirs() == ()
    session.close()


def test_a_command_that_fails_keeps_what_it_wrote(tmp_path: Path, repo: Path) -> None:
    """Moved answer: a declared write that exited unsuccessfully was put back. A command's
    write is now its own result, recorded and undoable like any other, and the exit says it
    failed."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(sh, "echo half > uv.lock && false")
    assert isinstance(got, ToolResult) and got.header.startswith("exit 1")
    assert (repo / "uv.lock").read_text() == "half\n"
    (effect,) = effects_of(session)
    assert effect.outcome == "succeeded" and effect.written == ("uv.lock",)  # type: ignore[attr-defined]
    session.close()


def test_a_command_that_changes_git_config_or_hooks_fails_and_is_put_back(
    tmp_path: Path, repo: Path
) -> None:
    """In the person's checkout the whole `.git` lies inside the writable workspace. A hook or
    a config line would run on the person's next git call, outside any sandbox, so the guard
    fails the call and puts every change back."""
    config = (repo / ".git" / "config").read_bytes()
    hook = repo / ".git" / "hooks" / "pre-commit"
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    got = run(
        sh,
        "echo new > uv.lock && git config core.fsmonitor evil "
        "&& printf '#!/bin/sh\\n' > .git/hooks/pre-commit",
    )
    assert isinstance(got, ToolResult)
    assert "git runs outside any sandbox" in got.render()
    assert (repo / ".git" / "config").read_bytes() == config
    assert not hook.exists()
    assert (repo / "uv.lock").read_text() == "old\n"
    (effect,) = effects_of(session)
    assert effect.outcome == "failed"  # type: ignore[attr-defined]
    session.close()


def test_an_approved_push_keeps_the_upstream_it_set(tmp_path: Path, repo: Path) -> None:
    """`git push -u` writes the branch's upstream into `config`. It names a
    remote the person already has, so it stands, and the push answers as the push it was. The
    guard used to put it back and fail a push that had happened."""
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    asked = WriteConfirmer(True)
    sh, session = shell(tmp_path, repo, asked, commits=False)
    got = run(sh, "git push -u origin main")
    assert isinstance(got, ToolResult), got
    assert asked.claims, "a push is asked"
    assert "git runs outside any sandbox" not in got.render()
    assert git(repo, "config", "branch.main.remote") == "origin"
    assert git(repo, "config", "branch.main.merge") == "refs/heads/main"
    (effect,) = effects_of(session)
    assert effect.outcome == "succeeded"  # type: ignore[attr-defined]
    session.close()


def test_a_hook_path_or_a_remote_url_is_put_back_whoever_approved_it(
    tmp_path: Path, repo: Path
) -> None:
    """Only an upstream to a remote the config names stands. `core.hooksPath`
    (what `husky` sets) makes git run the worktree's scripts, and a URL as a branch's remote
    sends a bare push anywhere: both are put back, and the call fails."""
    git(repo, "remote", "add", "origin", "https://example.com/x.git")
    config = (repo / ".git" / "config").read_bytes()
    sh, session = shell(tmp_path, repo, WriteConfirmer(True), commits=False)
    for n, line in enumerate(
        (
            "git config core.hooksPath .husky",
            "git config branch.main.remote https://elsewhere.example/x.git",
        )
    ):
        got = run(sh, line, op=f"S:c{n}")
        assert isinstance(got, ToolResult) and "git runs outside any sandbox" in got.render()
        assert (repo / ".git" / "config").read_bytes() == config, line
    session.close()


def test_a_write_is_refused_while_a_run_holds_the_lease(tmp_path: Path, repo: Path) -> None:
    """A run's worktree holds the repository WRITE lease for each `coder` call. A write
    meanwhile would land between its packets uncounted, so it is refused, unrun."""
    sh, session = shell(tmp_path, repo, WriteConfirmer(True))
    with take_write_lease(resolve(repo)):
        got = run(sh, "echo new > uv.lock")
    assert isinstance(got, WriteDenied) and "lease" in got.why
    assert (repo / "uv.lock").read_text() == "old\n"
    assert effects_of(session) == ()
    session.close()
