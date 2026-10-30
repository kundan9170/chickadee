"""The `[[commands]]` table: one row classifies an argv and templates its grant.

Three columns. `prefix` is the tokens that form the meaningful unit — OpenCode's arity, as
tokens (`["git", "push"]`, `["npm", "run"]`, `["cat"]`). `class` is one of
the command classes, each of which has a tier and a lowerability.
`destination` names an extractor from a **closed set implemented here** — chosen by name in
config, never an expression.

**The classifier is the table, and `classify` is its one entry.** Every
caller that asks *what kind of command is this* asks it: Zeus's shell for its tier, the
router's fact rows 1 and 3, the scout and a DISCOVERY run for *is this a read*, and the
risk reasons. A line is taken apart into the simple commands it runs (`shell_line`: `ls; rm x`
runs `rm`, and so do `sudo rm`, `xargs rm`, `sh -c "rm x"` and `$(rm x)`), and each is matched
by **longest prefix**. A few things are not commands and are seen by a pattern anywhere in the
line: a credential named in an argument, a write by redirection, an SQL `DROP`. Those are
`PATTERNS`, rows with a regex in place of a prefix. A pattern is code, never config, and every
table has the shipped ones.

The answer is a `Classification`: every row that matched, and the one with the highest floor
(`top`), whose class is the `prefilter_floor` in `max(declared_floor, prefilter_floor,
hard_floor)` and names the tier it set. A line no row names is *unrecognised*: it runs
sealed, T0, and has no class, so the prompt can offer *once* and never *always*.
Recognition can only raise.

A grant (`[[approval.grants]]`, "always for this repo") binds **class, prefix and
destination** — all three — so `git push origin other-branch` matches a grant for
`git push origin` and `git push --force origin main` does not: the longest-prefix row is
`destructive`, a different class, and a hard floor nothing could have granted anyway.

The table is config under the state rules: shipped defaults, the profile's declared
commands generated in at load, project rows keyed by `prefix`. A project row may add a prefix
or **raise** a shipped row's class; one that would lower it refuses the load — lowering is
what the trust digest is for, and it still cannot lower a hard floor.
"""

from __future__ import annotations

import functools
import posixpath
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import urlsplit

from chickadee.paths import within
from chickadee.profiles.model import LanguageProfile
from chickadee.schema.authorization import TIER_ORDER, Tier
from chickadee.schema.interaction import CommandBinding
from chickadee.shell_line import (
    GIT_GLOBAL_OPTIONS,
    PRINTERS,
    RUNS_ITS_ARGUMENT,
    first_operand,
    may_be,
    read_argv,
    read_line,
    runs_its_body,
)

__all__ = [
    "CLASS_TIERS",
    "CLASS_WORDS",
    "COMMITTING_GIT",
    "EXCLUDED",
    "FAMILY_HEADS",
    "FORGING_GIT",
    "GIT_STATE",
    "HISTORY_GIT",
    "NETWORK_CLASSES",
    "NEVER_GRANTED",
    "OUTSIDE_THE_REPOSITORY",
    "PATTERNS",
    "READS",
    "RECORDING_GIT",
    "SHIPPED",
    "Classification",
    "CommandClass",
    "CommandMatch",
    "CommandPattern",
    "CommandRow",
    "CommandTable",
    "CommandTableError",
    "Destination",
    "GitLine",
    "GitRefusal",
    "binding_of",
    "exact_binding_of",
    "parse_row",
    "profile_rows",
    "repo_any_arguments",
    "repo_offered",
    "shipped",
]


class CommandTableError(ValueError):
    """A table that cannot be used. Raised at load, never at match."""


class CommandClass(StrEnum):
    """The command classes — the rows of the tier table below."""

    READ_ONLY = "read_only"
    WORKTREE_WRITE = "worktree_write"
    """Writes the worktree or the repository's history, undoably: git and the operation
    snapshot take it back. T0 like a read, but not a read: the scout and a DISCOVERY
    run refuse it."""
    TESTS = "tests"
    BUILD = "build"
    DEV_SERVER = "dev_server"
    LOCKFILE_INSTALL = "lockfile_install"
    EXTERNAL_FETCH = "external_fetch"
    PUSH = "push"
    DESTRUCTIVE = "destructive"
    CREDENTIAL = "credential"
    SCOPE_EXPANDING = "scope_expanding"


CLASS_TIERS: Mapping[CommandClass, tuple[Tier, bool]] = {
    # (floor, lowerable). Undoable → T0; pinned and repo-declared → T1, lowerable;
    # irreversible → T2, hard. One row per class.
    CommandClass.READ_ONLY: (Tier.T0, False),
    CommandClass.WORKTREE_WRITE: (Tier.T0, False),
    CommandClass.TESTS: (Tier.T0, False),
    CommandClass.BUILD: (Tier.T0, False),
    # Asked, and approved it runs outside the sandbox: sealed, a server cannot be
    # reached, by the person's browser or by a check. Not lowerable: lowered is sealed.
    CommandClass.DEV_SERVER: (Tier.T1, False),
    CommandClass.LOCKFILE_INSTALL: (Tier.T1, True),
    CommandClass.EXTERNAL_FETCH: (Tier.T2, False),
    CommandClass.PUSH: (Tier.T2, False),
    CommandClass.DESTRUCTIVE: (Tier.T2, False),
    CommandClass.CREDENTIAL: (Tier.T2, False),
    CommandClass.SCOPE_EXPANDING: (Tier.T2, False),
}

NEVER_GRANTED: Final = frozenset({CommandClass.DESTRUCTIVE})
"""Classes no *always* covers: *"`destructive` is a hard floor, so nothing could have
granted it anyway"*. Every such call is confirmed on its own. Not `lowerable`: that is whether a
preference may lower a floor, and `push`, which is grantable, is not lowerable either. Here,
beside the tiers, because the table names a destructive match first when floors tie
(`Classification.top`); `approval/grants.py` reads it from here."""

READS: frozenset[CommandClass] = frozenset(
    {CommandClass.READ_ONLY, CommandClass.TESTS, CommandClass.BUILD}
)
"""The classes that answer a question rather than act. The scout's `request_exec` and a
DISCOVERY run ask only these; every other class is an
action, and actions are Zeus's, with approval. A build is a read here because it runs sealed
in a read-only repository: what it would write, the sandbox refuses."""

NETWORK_CLASSES: Final = frozenset(
    {CommandClass.PUSH, CommandClass.EXTERNAL_FETCH, CommandClass.LOCKFILE_INSTALL}
)
"""The classes whose command cannot do its work with the network off: approved,
it runs with the network on, as if it had said `network: true`."""


def needs_network(found: Classification | None) -> bool:
    """Whether a line's class says it needs the network (`NETWORK_CLASSES`)."""
    return found is not None and any(m.cls in NETWORK_CLASSES for m in found.matches)


def serves(found: Classification | None) -> bool:
    """Whether a line starts a server (`dev_server`): a `server` job by default, asked, and run
    outside the sandbox once approved."""
    return found is not None and any(m.cls is CommandClass.DEV_SERVER for m in found.matches)


def reads_only(line: str, found: Classification | None) -> bool:
    """Whether a line only reads: every command a read the table names (`git status`, `ls`,
    `cat`), and no output redirect (`echo x > f` writes). Such a line runs sealed, with no
    snapshot around it (its cost)."""
    return (
        found is not None
        and all(m.cls is CommandClass.READ_ONLY for m in found.matches)
        and not read_line(line).writes_to
    )


HISTORY_GIT: Final = frozenset(
    {
        "switch",
        "checkout",
        "restore",
        "stash",
        "merge",
        "rebase",
        "cherry-pick",
        "revert",
        "reset",
        "pull",
        "commit",
        "branch",
        "tag",
        "status",
        "log",
        "diff",
        "show",
        "fetch",
        "push",
        "add",
    }
)
"""The git subcommands that work on git's own history, bringing a file as a commit, the index or
a stash holds it: a source file such a line changes is kept. Every other git
subcommand (`apply`, `am`, `mv`, `checkout-index`, `update-index`, an unrecognised one) may
write content a model wrote, so a source file it changes is put back like any other. A list,
never a rule: a subcommand nobody listed is the cautious answer."""


class GitLine(StrEnum):
    """What git's history accounts for in a line that may write: which of the
    source files it changed are kept rather than put back (`CommandTable.git_line`)."""

    NONE = "none"
    """It runs no git it shows: a source file it changed is put back, unless a program it ran
    moved `HEAD`, when the effect is read as `BESIDE`."""
    HISTORY = "history"
    """Every command is git working on its own history (`HISTORY_GIT`), or changes no file
    (a `cd`, a read the table names, a printer, a shell's `-c` wrapper, a wrapper that only
    runs its argument, output to `/dev/null`), and the line was read whole: every source
    change is git's and is kept."""
    BESIDE = "beside"
    """It runs git beside something that may write (a test run, a build, a redirect to a file,
    `git apply`): a source file is kept only where it now holds what the commit git left
    `HEAD` on holds, and that commit is history (`workspace.history_head`)."""


RECORDING_GIT: Final = frozenset({"add", "commit", "stash", "am"})
"""The git subcommands that record the tree into git: the index, a commit, a stash, a
patch file's commit. Beside a write, what they record is the model's write, which the put-back
after the command restores on disk and never in git; such a line is refused before it runs."""

COMMITTING_GIT: Final = frozenset({"merge", "rebase", "cherry-pick", "revert", "pull"})
"""The git subcommands that build a commit from commits. git keeps a write
beside them out of the commit, but the put-back then leaves the files behind the commit the
line made, so beside a write they are refused too."""

FORGING_GIT: Final = frozenset(
    {"commit-tree", "mktree", "update-ref", "fast-import", "replace"}
)
"""git plumbing that builds objects from any content or points a ref at one by hand, which a
later switch brings to disk as git's history. Refused in Zeus's writing
shell, alone or not; `hash-object` is one too when it writes (`-w`)."""


@dataclass(frozen=True)
class GitRefusal:
    """Why a line's git is refused before it runs, as `INVALID_ARGUMENTS` on
    `command`: the git command, what it would do, and the call to send instead."""

    command: str
    why: str
    do: str


_STASH_READS: Final = frozenset({"pop", "apply", "list", "show", "drop", "clear", "branch"})
"""The `git stash` forms that save nothing: every other form, bare included, records."""

_ALONE: Final = (
    "send the {command} part as a shell call of its own, after the call that writes, with "
    "nothing beside it but other git commands"
)
"""The instruction for a git command refused beside a write."""


_HARMLESS_TARGETS: Final = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
"""Redirect targets that are no file of the worktree: `git switch main 2>/dev/null` writes
nothing a source check needs to see."""


def runs_git(line: str) -> bool:
    """Whether any command of `line` is git: only such a line needs git's objects, refs and
    index writable beside the worktree."""
    return any(
        argv and argv[0].rsplit("/", 1)[-1] == "git" for argv in read_line(line).commands
    )


GIT_STATE: Final = "git state"
OUTSIDE_THE_REPOSITORY: Final = "outside the repository"
"""What `CommandTable.beyond_the_worktree` answers."""

CLASS_WORDS: Mapping[CommandClass, str] = {
    CommandClass.READ_ONLY: "reads",
    CommandClass.WORKTREE_WRITE: "writes files",
    CommandClass.TESTS: "runs tests",
    CommandClass.BUILD: "builds",
    CommandClass.DEV_SERVER: "starts a server or a process that stays",
    CommandClass.LOCKFILE_INSTALL: "installs or changes dependencies",
    CommandClass.EXTERNAL_FETCH: "reaches the network",
    CommandClass.PUSH: "publishes beyond this machine",
    CommandClass.DESTRUCTIVE: "is destructive",
    CommandClass.CREDENTIAL: "reads or names a credential",
    CommandClass.SCOPE_EXPANDING: "changes privileges",
}
"""What each class does, in the words a refusal uses."""


class Destination(StrEnum):
    """The closed extractor set. Chosen by name; what each one reads is fixed here."""

    REMOTE = "remote"
    """The first non-flag token after the prefix — `git push origin main` → `origin`."""
    URL_HOST = "url_host"
    """The host of the first URL in the argv — `curl https://x.io/a` → `x.io`."""
    INDEX_HOST = "index_host"
    """The project's package index, from config, never the argv (`lockfile_install`)."""
    PATH_ROOTS = "path_roots"
    """The non-flag tokens after the prefix, sorted — `rm -rf build dist` → `build dist`."""
    NONE = "none"


@dataclass(frozen=True)
class CommandRow:
    prefix: tuple[str, ...]
    cls: CommandClass
    destination: Destination
    source: str = "shipped"
    """`shipped`, `profile`, `project` or `user` — where the row came from, for the prompt."""

    @property
    def floor(self) -> Tier:
        return CLASS_TIERS[self.cls][0]

    @property
    def lowerable(self) -> bool:
        return CLASS_TIERS[self.cls][1]


@dataclass(frozen=True)
class CommandPattern:
    """A row matched by a regex anywhere in the line rather than by a prefix: what is not a
    command, but says what a command does (a credential named in an argument, a `>`)."""

    cls: CommandClass
    pattern: re.Pattern[str]

    @property
    def source(self) -> str:
        """Always `shipped`: a pattern is code, never a config layer's."""
        return "shipped"

    @property
    def floor(self) -> Tier:
        return CLASS_TIERS[self.cls][0]

    @property
    def lowerable(self) -> bool:
        return CLASS_TIERS[self.cls][1]


@dataclass(frozen=True)
class CommandMatch:
    """One row that matched: a prefix row and the destination it extracted, or a pattern row
    and the text it matched."""

    row: CommandRow | CommandPattern
    destination: str | None
    """The extractor's value for this argv, or `None` for `none` / nothing to extract, and
    for every pattern row."""
    matched: str = ""
    """The text a pattern row matched; a prefix row's match is its prefix."""
    grantable: bool = True
    """Whether a repository grant may cover this command: false for a git command
    pointed at another repository (an option before its subcommand, or a `cd` before it)."""
    in_shape: bool = True
    """Whether the command fits its row's grant shape: every option listed and
    every operand admitted. A row with no shape takes any arguments."""

    @property
    def cls(self) -> CommandClass:
        return self.row.cls

    @property
    def prefix(self) -> tuple[str, ...]:
        """The row's prefix; empty for a pattern row, which no grant can bind."""
        return self.row.prefix if isinstance(self.row, CommandRow) else ()

    @property
    def floor(self) -> Tier:
        return self.row.floor

    @property
    def lowerable(self) -> bool:
        return self.row.lowerable

    @property
    def text(self) -> str:
        """What was recognised, as a prompt or a refusal shows it."""
        return self.matched or " ".join(self.prefix)


@dataclass(frozen=True)
class Classification:
    """What a line or an argv is: every row that matched, in order — each simple command's
    prefix row, then the pattern rows — and the one that sets the tier; whether the line is
    plain (`shell_line`), and the simple commands no row named."""

    matches: tuple[CommandMatch, ...]
    plain: bool = True
    """Every word is the one the shell runs: only a plain line can be granted."""
    unread: tuple[tuple[str, ...], ...] = ()
    """The simple commands no row names. A grant given for the command beside one never
    covers it: `git push origin main &&./deploy.sh` is not the push alone."""

    def __post_init__(self) -> None:
        if not self.matches:
            raise ValueError("a classification has at least one match; unrecognised is None")

    @property
    def top(self) -> CommandMatch:
        """The match with the highest floor; among equals, one whose class is never granted,
        so a destructive command names the prompt's class whenever it is in the line; then the
        first, so a prefix row (which carries a destination) names the tier
        before a pattern row of the same floor."""
        best = self.matches[0]
        for match in self.matches[1:]:
            if _lower(best.floor, match.floor) or (
                best.floor is match.floor
                and match.cls in NEVER_GRANTED
                and best.cls not in NEVER_GRANTED
            ):
                best = match
        return best

    @property
    def cls(self) -> CommandClass:
        return self.top.cls

    @property
    def floor(self) -> Tier:
        return self.top.floor

    @property
    def lowerable(self) -> bool:
        return self.top.lowerable

    @property
    def prefix(self) -> tuple[str, ...]:
        return self.top.prefix

    @property
    def destination(self) -> str | None:
        return self.top.destination

    @property
    def classes(self) -> frozenset[CommandClass]:
        return frozenset(m.cls for m in self.matches)

    def action(self) -> CommandMatch | None:
        """The first match that is not a read (`READS`), or `None` when every one is."""
        return next((m for m in self.matches if m.cls not in READS), None)


def binding_of(classification: Classification) -> CommandBinding | None:
    """The one grant that could cover this line: the class, prefix and destination every
    match in it shares, or `None` when two make it up or no prefix row matched. A pattern row
    of the line's own class binds nothing of its own (it has no prefix) and is covered; one of
    another class is a second binding. `None` is the prompt's *once* only: no single
    grant could ever match the line, so offering *always* for it would promise nothing."""
    first = _one_binding(classification)
    if first is None or not all(m.grantable and m.in_shape for m in classification.matches):
        return None
    return CommandBinding(
        cls=first.cls.value, prefix=first.prefix, destination=first.destination
    )


def exact_binding_of(
    classification: Classification, words: Sequence[str]
) -> CommandBinding | None:
    """What *this session, exact* would bind: the line's every word as its prefix,
    so it covers that line and nothing else. A shape and a git line pointed elsewhere do not
    bar it, since the line is the one the person saw; a line that is not plain or holds an
    unread command does (C), since its text is not what runs."""
    first = _one_binding(classification)
    if first is None or not words:
        return None
    return CommandBinding(
        cls=first.cls.value, prefix=tuple(words), destination=first.destination
    )


def repo_offered(classification: Classification) -> bool:
    """Whether the prompt offers *always for this repo* for the line its binding covers (E.3):
    a row with a grant shape, or one whose effects stay on this machine. A row whose effects
    leave the machine and has no shape is granted for the repository only by hand, in
    `approval.toml`."""
    if binding_of(classification) is None:
        return False
    return all(
        _shape(m.row) is not None or not _leaves_the_machine(m.row)
        for m in classification.matches
        if isinstance(m.row, CommandRow)
    )


def repo_any_arguments(classification: Classification) -> bool:
    """Whether the repository grant covers any arguments after the command words: its row has
    no grant shape, so the prompt says so in plain words (E.1)."""
    return any(
        _shape(m.row) is None for m in classification.matches if isinstance(m.row, CommandRow)
    )


def _one_binding(classification: Classification) -> CommandMatch | None:
    """The match every command of a plain, wholly read line shares (class, prefix and
    destination), or `None`."""
    rows = [m for m in classification.matches if isinstance(m.row, CommandRow)]
    if not rows or not classification.plain or classification.unread:
        return None
    first = rows[0]
    for match in classification.matches:
        if match.cls is not first.cls:
            return None
        if isinstance(match.row, CommandRow) and (
            match.prefix != first.prefix or match.destination != first.destination
        ):
            return None
    return first


# -- the pattern rows ----------------------------------------------------------------------

# What is not a command but says what a command does, seen anywhere in the line. Commands
# themselves are the prefix rows' (`ls; rm x`, `sudo rm`, `sh -c "rm x"` are each split into
# the `rm` they run), so a pattern never stands in for one: `\brm\b` read `docker run --rm`
# and `echo rm` as destructive, and the lower-case words `token` and `auth` made
# `grep token src` and `gh auth status` credential-bearing.
_NAME = r"[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)"
_CREDENTIAL_NAME = re.compile(rf"\$\{{?{_NAME}[A-Z0-9_]*\b|\b{_NAME}[A-Z0-9_]*=")
"""An environment variable credentials are kept under, read (`$GITHUB_TOKEN`,
`"${AWS_SECRET_ACCESS_KEY}"`) or set (`GITHUB_TOKEN=x`). Upper-case only, because file names
are lower-case by convention (`tests/test_secret_path_gate.py` names a test), and only as a
variable, because a bare name is a word: `grep -r SECRET_PATH src` searches for a
constant."""
_CREDENTIAL_ARGUMENT = re.compile(
    r"--(?:password|passwd|token|api[-_]?key|secret|client[-_]secret|access[-_]key|"
    r"private[-_]key|auth[-_]token)\b"
    r"|\b(?:authorization|x-api-key)\s*:|\bbearer\s+[A-Za-z0-9._~+/-]{8,}",
    re.IGNORECASE,
)
"""A credential passed as an argument: a flag that takes one, or an HTTP header carrying one."""
_REDIRECT_WRITE = re.compile(
    r"(?<![<>&=-])(?:\d*>>?|&>>?|>\|)(?!&)\s*(?!/dev/(?:null|stdout|stderr|tty)\b)[^\s;&|<>)]"
)
"""A `>` or `>>` into a file. Into a descriptor (`2>&1`) or `/dev/null` writes nothing."""
_SQL_DESTRUCTIVE = re.compile(
    r"\b(?:drop\s+(?:table|database|schema)|truncate\s+table)\b", re.I
)
"""SQL that cannot be taken back, wherever a client is handed it."""

PATTERNS: tuple[CommandPattern, ...] = (
    CommandPattern(CommandClass.CREDENTIAL, _CREDENTIAL_NAME),
    CommandPattern(CommandClass.CREDENTIAL, _CREDENTIAL_ARGUMENT),
    CommandPattern(CommandClass.WORKTREE_WRITE, _REDIRECT_WRITE),
    CommandPattern(CommandClass.DESTRUCTIVE, _SQL_DESTRUCTIVE),
)
"""The shipped pattern rows. Every table carries them; config cannot add or remove one."""


# -- the table -----------------------------------------------------------------------------


_RO = CommandClass.READ_ONLY
_W = CommandClass.WORKTREE_WRITE
_T = CommandClass.TESTS
_B = CommandClass.BUILD
_DEV = CommandClass.DEV_SERVER
_LOCK = CommandClass.LOCKFILE_INSTALL
_FETCH = CommandClass.EXTERNAL_FETCH
_PUSH = CommandClass.PUSH
_DESTR = CommandClass.DESTRUCTIVE
_CRED = CommandClass.CREDENTIAL
_SCOPE = CommandClass.SCOPE_EXPANDING
_NONE = Destination.NONE
_REMOTE = Destination.REMOTE
_URL = Destination.URL_HOST
_INDEX = Destination.INDEX_HOST
_PATHS = Destination.PATH_ROOTS

# One line per row: the prefix as it is typed, its class, its destination extractor.
# Grouped by the head command, in the order of OpenCode's dictionary, with the
# class rows and the pattern rows' commands folded in where they belong. The
# rule for an infrastructure CLI (aws, gcloud, kubectl, helm, …) is: the head is `push` —
# it mutates shared state by default and that is not undoable — and the read verbs are
# listed as `read_only` so the longest prefix lets them through. A database shell is
# `destructive`. A toolchain manager that downloads is `external_fetch`.
_TABLE: tuple[tuple[str, CommandClass, Destination], ...] = (
    # --- the coreutils and shell tools ---
    ("cat", _RO, _NONE),
    ("cd", _RO, _NONE),
    ("chmod", _SCOPE, _PATHS),
    ("chown", _SCOPE, _PATHS),
    # Worktree edits by a command: undoable through git and the operation snapshot.
    ("cp", _W, _PATHS),
    ("echo", _RO, _NONE),
    # Prints or sets the environment, which is where credentials live.
    ("env", _CRED, _NONE),
    ("printenv", _CRED, _NONE),
    ("grep", _RO, _NONE),
    ("rg", _RO, _NONE),
    ("find", _RO, _NONE),
    # `find … -delete` removes what it finds; `shell_line` reads it as this command.
    ("find -delete", _DESTR, _PATHS),
    ("head", _RO, _NONE),
    ("tail", _RO, _NONE),
    ("wc", _RO, _NONE),
    ("sed", _RO, _NONE),
    ("sed -i", _W, _NONE),
    ("sed --in-place", _W, _NONE),
    ("perl -i", _W, _NONE),
    ("perl -pi", _W, _NONE),
    ("tee", _W, _PATHS),
    ("patch", _W, _NONE),
    ("awk", _RO, _NONE),
    ("sort", _RO, _NONE),
    ("diff", _RO, _NONE),
    ("ln", _W, _PATHS),
    ("ls", _RO, _NONE),
    ("mkdir", _W, _PATHS),
    ("mv", _DESTR, _PATHS),
    ("ps", _RO, _NONE),
    ("pwd", _RO, _NONE),
    ("rm", _DESTR, _PATHS),
    ("rm -r", _DESTR, _PATHS),
    ("rm -rf", _DESTR, _PATHS),
    ("rmdir", _DESTR, _PATHS),
    ("shred", _DESTR, _PATHS),
    ("dd", _DESTR, _NONE),
    ("mkfs", _DESTR, _NONE),
    ("truncate", _DESTR, _PATHS),
    ("sleep", _RO, _NONE),
    ("touch", _W, _PATHS),
    ("which", _RO, _NONE),
    ("sudo", _SCOPE, _NONE),
    ("apt install", _SCOPE, _NONE),
    ("apt-get install", _SCOPE, _NONE),
    ("curl", _FETCH, _URL),
    ("wget", _FETCH, _URL),
    ("ssh", _FETCH, _REMOTE),
    ("nc", _FETCH, _NONE),
    ("ncat", _FETCH, _NONE),
    ("telnet", _FETCH, _NONE),
    ("ftp", _FETCH, _NONE),
    ("scp", _PUSH, _REMOTE),
    ("rsync", _PUSH, _PATHS),
    # --- cloud CLIs: the head mutates shared infrastructure; the reads are listed ---
    ("aws", _PUSH, _NONE),
    ("aws s3 ls", _RO, _NONE),
    ("aws s3 cp", _PUSH, _PATHS),
    ("aws s3 sync", _PUSH, _PATHS),
    ("aws s3 rm", _DESTR, _PATHS),
    ("aws sts get-caller-identity", _RO, _NONE),
    ("az", _PUSH, _NONE),
    ("az account show", _RO, _NONE),
    ("az storage blob list", _RO, _NONE),
    # System package managers install into the system prefix, as `apt install` does.
    ("dnf install", _SCOPE, _NONE),
    ("dnf remove", _SCOPE, _NONE),
    ("yum install", _SCOPE, _NONE),
    ("yum remove", _SCOPE, _NONE),
    ("apk add", _SCOPE, _NONE),
    ("apk del", _SCOPE, _NONE),
    # Build tools and bundlers; the ones that serve are dev servers.
    ("just", _B, _NONE),
    ("ninja", _B, _NONE),
    ("ant", _B, _NONE),
    ("rollup", _B, _NONE),
    ("esbuild", _B, _NONE),
    ("webpack", _B, _NONE),
    ("webpack serve", _DEV, _NONE),
    ("vite", _DEV, _NONE),
    ("vite build", _B, _NONE),
    ("vite preview", _DEV, _NONE),
    ("next dev", _DEV, _NONE),
    ("next start", _DEV, _NONE),
    ("astro dev", _DEV, _NONE),
    ("bazel build", _B, _NONE),
    ("bazel test", _T, _NONE),
    ("bazel run", _B, _NONE),
    ("bazel query", _RO, _NONE),
    ("bazel clean", _DESTR, _NONE),
    # Homebrew installs into the system prefix: scope-expanding, not a lockfile install.
    ("brew install", _SCOPE, _NONE),
    ("brew uninstall", _SCOPE, _NONE),
    ("brew upgrade", _SCOPE, _NONE),
    ("brew list", _RO, _NONE),
    ("brew info", _RO, _NONE),
    ("bun install", _LOCK, _INDEX),
    ("bun add", _LOCK, _INDEX),
    ("bun remove", _LOCK, _INDEX),
    ("bun test", _T, _NONE),
    ("bun build", _B, _NONE),
    ("bun run dev", _DEV, _NONE),
    ("bun dev", _DEV, _NONE),
    ("bun run start", _DEV, _NONE),
    ("bun run build", _B, _NONE),
    ("bun run test", _T, _NONE),
    ("bun run lint", _B, _NONE),
    # `bun x <pkg>`: whatever follows is fetched and run.
    ("bun x", _FETCH, _NONE),
    ("cargo build", _B, _NONE),
    ("cargo check", _B, _NONE),
    ("cargo clippy", _B, _NONE),
    ("cargo fmt", _B, _NONE),
    ("cargo test", _T, _NONE),
    ("cargo run", _B, _NONE),
    ("cargo add", _LOCK, _INDEX),
    ("cargo remove", _LOCK, _INDEX),
    ("cargo fetch", _LOCK, _INDEX),
    ("cargo update", _LOCK, _INDEX),
    ("cargo publish", _PUSH, _INDEX),
    ("cargo clean", _DESTR, _NONE),
    ("cdk", _PUSH, _NONE),
    ("cdk diff", _RO, _NONE),
    ("cdk synth", _B, _NONE),
    ("cdk ls", _RO, _NONE),
    ("cf", _PUSH, _NONE),
    ("cf apps", _RO, _NONE),
    ("cf logs", _RO, _NONE),
    ("cmake", _B, _NONE),
    # The C++ profile's runner: runs the tests the build registered.
    ("ctest", _T, _NONE),
    ("composer install", _LOCK, _INDEX),
    ("composer require", _LOCK, _INDEX),
    ("composer update", _LOCK, _INDEX),
    ("composer show", _RO, _NONE),
    ("composer run", _B, _NONE),
    ("php -S", _DEV, _NONE),
    ("php artisan serve", _DEV, _NONE),
    ("symfony serve", _DEV, _NONE),
    ("symfony server:start", _DEV, _NONE),
    ("consul", _PUSH, _NONE),
    ("consul members", _RO, _NONE),
    ("consul kv get", _RO, _NONE),
    ("consul kv put", _PUSH, _NONE),
    ("consul kv delete", _DESTR, _NONE),
    ("crictl", _PUSH, _NONE),
    ("crictl ps", _RO, _NONE),
    ("crictl images", _RO, _NONE),
    ("crictl logs", _RO, _NONE),
    # The C# profile's commands.
    ("dotnet build", _B, _NONE),
    ("dotnet test", _T, _NONE),
    ("dotnet restore", _LOCK, _INDEX),
    ("dotnet add package", _LOCK, _INDEX),
    ("deno run", _B, _NONE),
    ("deno test", _T, _NONE),
    ("deno lint", _B, _NONE),
    ("deno fmt", _B, _NONE),
    ("deno task dev", _DEV, _NONE),
    ("deno task start", _DEV, _NONE),
    ("deno task build", _B, _NONE),
    ("deno task test", _T, _NONE),
    ("doctl", _PUSH, _NONE),
    ("doctl kubernetes cluster list", _RO, _NONE),
    ("docker run", _DEV, _NONE),
    ("docker build", _B, _NONE),
    ("docker exec", _DEV, _NONE),
    ("docker pull", _FETCH, _URL),
    ("docker push", _PUSH, _URL),
    ("docker rm", _DESTR, _NONE),
    ("docker rmi", _DESTR, _NONE),
    ("docker ps", _RO, _NONE),
    ("docker images", _RO, _NONE),
    ("docker logs", _RO, _NONE),
    ("docker inspect", _RO, _NONE),
    ("docker builder prune", _DESTR, _NONE),
    ("docker compose up", _DEV, _NONE),
    ("docker compose down", _DESTR, _NONE),
    ("docker compose build", _B, _NONE),
    ("docker compose logs", _RO, _NONE),
    ("docker compose ps", _RO, _NONE),
    ("docker container ls", _RO, _NONE),
    ("docker container rm", _DESTR, _NONE),
    ("docker container prune", _DESTR, _NONE),
    ("docker image ls", _RO, _NONE),
    ("docker image rm", _DESTR, _NONE),
    ("docker image prune", _DESTR, _NONE),
    ("docker network ls", _RO, _NONE),
    ("docker network inspect", _RO, _NONE),
    ("docker network rm", _DESTR, _NONE),
    ("docker volume ls", _RO, _NONE),
    ("docker volume rm", _DESTR, _NONE),
    ("docker volume prune", _DESTR, _NONE),
    ("docker system prune", _DESTR, _NONE),
    ("eksctl", _PUSH, _NONE),
    ("eksctl get", _RO, _NONE),
    ("eksctl create cluster", _PUSH, _NONE),
    ("eksctl delete", _DESTR, _NONE),
    ("firebase", _PUSH, _NONE),
    ("firebase emulators", _DEV, _NONE),
    ("firebase projects", _RO, _NONE),
    ("flyctl", _PUSH, _NONE),
    ("flyctl logs", _RO, _NONE),
    ("flyctl status", _RO, _NONE),
    ("gcloud", _PUSH, _NONE),
    ("gcloud auth list", _RO, _NONE),
    ("gcloud config list", _RO, _NONE),
    ("gcloud compute instances list", _RO, _NONE),
    ("gh", _PUSH, _NONE),
    ("gh pr list", _RO, _NONE),
    ("gh pr view", _RO, _NONE),
    ("gh pr create", _PUSH, _REMOTE),
    ("gh pr merge", _PUSH, _REMOTE),
    ("gh issue list", _RO, _NONE),
    ("gh issue view", _RO, _NONE),
    ("gh issue create", _PUSH, _REMOTE),
    ("gh release create", _PUSH, _REMOTE),
    ("gh repo view", _RO, _NONE),
    ("gh auth status", _RO, _NONE),
    ("gh run list", _RO, _NONE),
    ("gh run view", _RO, _NONE),
    # --- git: read-only verbs, the reflog-recoverable writes, and the verbs that are not ---
    ("git status", _RO, _NONE),
    ("git log", _RO, _NONE),
    ("git diff", _RO, _NONE),
    ("git show", _RO, _NONE),
    ("git blame", _RO, _NONE),
    ("git branch", _RO, _NONE),
    ("git branch -D", _DESTR, _NONE),
    # A branch deleted, renamed or copied is the repository's to undo; forced over another
    # branch (`-M`, `-C`), it loses that one, as `-D` does.
    ("git branch -d", _W, _NONE),
    ("git branch --delete", _W, _NONE),
    ("git branch -m", _W, _NONE),
    ("git branch --move", _W, _NONE),
    ("git branch -c", _W, _NONE),
    ("git branch --copy", _W, _NONE),
    ("git branch -M", _DESTR, _NONE),
    ("git branch -C", _DESTR, _NONE),
    ("git ls-files", _RO, _NONE),
    ("git rev-parse", _RO, _NONE),
    ("git add", _W, _NONE),
    ("git commit", _W, _NONE),
    ("git merge", _W, _NONE),
    ("git tag", _W, _NONE),
    ("git switch", _W, _NONE),
    ("git worktree", _W, _NONE),
    ("git am", _W, _NONE),
    ("git apply", _W, _NONE),
    ("git cherry-pick", _W, _NONE),
    ("git revert", _W, _NONE),
    ("git mv", _W, _PATHS),
    ("git rm", _W, _PATHS),
    ("git init", _W, _NONE),
    # `git checkout <branch>` is reversible; `git checkout.` discards the worktree and is
    # not, and a class cannot see the tree — so the verb is classified by what it *can* do
    # (invariant 18). `switch` refuses to discard, so it is an undoable write.
    ("git checkout", _DESTR, _PATHS),
    ("git checkout --", _DESTR, _PATHS),
    ("git restore", _DESTR, _PATHS),
    ("git reset --hard", _DESTR, _NONE),
    ("git clean", _DESTR, _PATHS),
    ("git rebase", _DESTR, _NONE),
    ("git filter-branch", _DESTR, _NONE),
    ("git push", _PUSH, _REMOTE),
    # Longest prefix wins, so these beat `git push` and reclassify it.
    ("git push --force", _DESTR, _REMOTE),
    ("git push -f", _DESTR, _REMOTE),
    ("git push --force-with-lease", _DESTR, _REMOTE),
    # A push that deletes is as unrecoverable as one that forces; an option row
    # matches wherever its option is (`CommandTable._best`).
    ("git push --delete", _DESTR, _REMOTE),
    ("git push -d", _DESTR, _REMOTE),
    ("git push --mirror", _DESTR, _REMOTE),
    ("git push --prune", _DESTR, _REMOTE),
    ("git fetch", _FETCH, _REMOTE),
    ("git pull", _FETCH, _REMOTE),
    ("git clone", _FETCH, _URL),
    ("git config", _W, _NONE),
    ("git remote", _W, _NONE),
    ("git remote remove", _DESTR, _NONE),
    ("git stash", _W, _NONE),
    ("git stash drop", _DESTR, _NONE),
    ("git stash clear", _DESTR, _NONE),
    ("go build", _B, _NONE),
    ("go vet", _B, _NONE),
    ("go run", _B, _NONE),
    ("go test", _T, _NONE),
    ("go get", _LOCK, _INDEX),
    ("go mod", _LOCK, _INDEX),
    ("go mod download", _LOCK, _INDEX),
    ("go mod tidy", _B, _NONE),
    ("go clean", _DESTR, _NONE),
    ("gradlew", _B, _NONE),
    ("gradlew test", _T, _NONE),
    ("gradlew bootRun", _DEV, _NONE),
    ("gradle build", _B, _NONE),
    ("gradle test", _T, _NONE),
    ("gradle clean", _DESTR, _NONE),
    ("gradle publish", _PUSH, _INDEX),
    ("gradle bootRun", _DEV, _NONE),
    ("helm", _PUSH, _NONE),
    ("helm list", _RO, _NONE),
    ("helm status", _RO, _NONE),
    ("helm template", _B, _NONE),
    ("helm lint", _B, _NONE),
    ("helm uninstall", _DESTR, _NONE),
    ("heroku", _PUSH, _NONE),
    ("heroku logs", _RO, _NONE),
    ("heroku apps", _RO, _NONE),
    ("hugo", _B, _NONE),
    ("hugo server", _DEV, _NONE),
    ("hugo serve", _DEV, _NONE),
    # Network configuration is root's: the reads are listed, the rest expands scope.
    ("ip", _SCOPE, _NONE),
    ("ip addr show", _RO, _NONE),
    ("ip link show", _RO, _NONE),
    ("ip route show", _RO, _NONE),
    ("ip netns list", _RO, _NONE),
    ("ip netns exec", _SCOPE, _NONE),
    ("kind get", _RO, _NONE),
    ("kind delete", _DESTR, _NONE),
    ("kind create cluster", _DEV, _NONE),
    ("kubectl", _PUSH, _NONE),
    ("kubectl get", _RO, _NONE),
    ("kubectl describe", _RO, _NONE),
    ("kubectl logs", _RO, _NONE),
    ("kubectl exec", _DEV, _NONE),
    ("kubectl apply", _PUSH, _NONE),
    ("kubectl delete", _DESTR, _NONE),
    ("kubectl kustomize", _B, _NONE),
    ("kubectl rollout status", _RO, _NONE),
    ("kubectl rollout restart", _PUSH, _NONE),
    ("kubectl rollout undo", _PUSH, _NONE),
    ("kustomize build", _B, _NONE),
    ("make", _B, _NONE),
    ("make test", _T, _NONE),
    # Installs into the system prefix, as `brew install` does.
    ("make install", _SCOPE, _NONE),
    ("mc", _PUSH, _NONE),
    ("mc ls", _RO, _NONE),
    ("mc cp", _PUSH, _PATHS),
    ("mc rm", _DESTR, _PATHS),
    ("mc admin info", _RO, _NONE),
    ("minikube", _DEV, _NONE),
    ("minikube status", _RO, _NONE),
    ("minikube delete", _DESTR, _NONE),
    # Database shells: what they can do is drop the database.
    ("mongosh", _DESTR, _NONE),
    ("mysql", _DESTR, _NONE),
    ("psql", _DESTR, _NONE),
    ("redis-cli", _DESTR, _NONE),
    ("redis-cli ping", _RO, _NONE),
    ("dropdb", _DESTR, _NONE),
    ("mvn", _B, _NONE),
    ("mvn test", _T, _NONE),
    # Downloads every artifact the build needs (the Java profile's bootstrap).
    ("mvn dependency:go-offline", _LOCK, _INDEX),
    ("mvn deploy", _PUSH, _INDEX),
    ("mvn clean", _DESTR, _NONE),
    ("mvn spring-boot:run", _DEV, _NONE),
    ("ng", _B, _NONE),
    ("ng serve", _DEV, _NONE),
    ("ng test", _T, _NONE),
    ("npm install", _LOCK, _INDEX),
    ("npm ci", _LOCK, _INDEX),
    ("npm uninstall", _LOCK, _INDEX),
    ("npm update", _LOCK, _INDEX),
    ("npm test", _T, _NONE),
    ("npm start", _DEV, _NONE),
    ("npm publish", _PUSH, _INDEX),
    ("npm ls", _RO, _NONE),
    ("npm exec", _FETCH, _NONE),
    ("npm init", _FETCH, _NONE),
    ("npm run dev", _DEV, _NONE),
    ("npm run start", _DEV, _NONE),
    ("npm run serve", _DEV, _NONE),
    ("npm run preview", _DEV, _NONE),
    ("npm run build", _B, _NONE),
    ("npm run test", _T, _NONE),
    ("npm run lint", _B, _NONE),
    ("npm run typecheck", _B, _NONE),
    ("npx", _FETCH, _NONE),
    ("npx jest", _T, _NONE),
    ("npx vitest", _T, _NONE),
    ("npx tsc", _B, _NONE),
    ("npx eslint", _B, _NONE),
    ("npx prettier", _B, _NONE),
    ("nx build", _B, _NONE),
    ("nx test", _T, _NONE),
    ("nx lint", _B, _NONE),
    ("nx serve", _DEV, _NONE),
    ("nx run", _B, _NONE),
    ("openssl", _CRED, _NONE),
    ("openssl req", _CRED, _NONE),
    ("openssl x509", _RO, _NONE),
    ("openssl version", _RO, _NONE),
    ("pip install", _LOCK, _INDEX),
    ("pip uninstall", _LOCK, _INDEX),
    ("pip download", _FETCH, _INDEX),
    ("pip list", _RO, _NONE),
    ("pip show", _RO, _NONE),
    ("pip freeze", _RO, _NONE),
    ("pip3 install", _LOCK, _INDEX),
    ("pip3 uninstall", _LOCK, _INDEX),
    ("pip3 download", _FETCH, _INDEX),
    ("pip3 list", _RO, _NONE),
    ("pip3 show", _RO, _NONE),
    ("pip3 freeze", _RO, _NONE),
    # pipx installs an application outside the project, as `brew install` does; `run`
    # fetches one and runs it.
    ("pipx install", _SCOPE, _NONE),
    ("pipx run", _FETCH, _NONE),
    ("pipx list", _RO, _NONE),
    ("pdm add", _LOCK, _INDEX),
    ("pdm remove", _LOCK, _INDEX),
    ("pdm install", _LOCK, _INDEX),
    ("pdm sync", _LOCK, _INDEX),
    ("pdm lock", _LOCK, _INDEX),
    ("pdm update", _LOCK, _INDEX),
    ("pdm run", _B, _NONE),
    ("conda install", _LOCK, _INDEX),
    ("conda remove", _LOCK, _INDEX),
    ("conda update", _LOCK, _INDEX),
    ("conda create", _LOCK, _INDEX),
    ("conda env create", _LOCK, _INDEX),
    ("conda list", _RO, _NONE),
    ("pipenv install", _LOCK, _INDEX),
    ("pipenv run", _B, _NONE),
    ("pipenv lock", _LOCK, _INDEX),
    ("pnpm install", _LOCK, _INDEX),
    ("pnpm add", _LOCK, _INDEX),
    ("pnpm remove", _LOCK, _INDEX),
    ("pnpm test", _T, _NONE),
    ("pnpm publish", _PUSH, _INDEX),
    ("pnpm dlx", _FETCH, _NONE),
    ("pnpm exec", _B, _NONE),
    ("pnpm run dev", _DEV, _NONE),
    ("pnpm dev", _DEV, _NONE),
    ("pnpm start", _DEV, _NONE),
    ("pnpm run start", _DEV, _NONE),
    ("pnpm run serve", _DEV, _NONE),
    ("pnpm run preview", _DEV, _NONE),
    ("pnpm run build", _B, _NONE),
    ("pnpm run test", _T, _NONE),
    ("pnpm run lint", _B, _NONE),
    ("poetry install", _LOCK, _INDEX),
    ("poetry add", _LOCK, _INDEX),
    ("poetry remove", _LOCK, _INDEX),
    ("poetry lock", _LOCK, _INDEX),
    ("poetry run", _B, _NONE),
    ("poetry build", _B, _NONE),
    ("poetry publish", _PUSH, _INDEX),
    ("podman run", _DEV, _NONE),
    ("podman build", _B, _NONE),
    ("podman pull", _FETCH, _URL),
    ("podman push", _PUSH, _URL),
    ("podman rm", _DESTR, _NONE),
    ("podman rmi", _DESTR, _NONE),
    ("podman ps", _RO, _NONE),
    ("podman images", _RO, _NONE),
    ("podman logs", _RO, _NONE),
    ("podman container ls", _RO, _NONE),
    ("podman container rm", _DESTR, _NONE),
    ("podman container prune", _DESTR, _NONE),
    ("podman image ls", _RO, _NONE),
    ("podman image rm", _DESTR, _NONE),
    ("podman image prune", _DESTR, _NONE),
    ("pulumi", _PUSH, _NONE),
    ("pulumi preview", _RO, _NONE),
    ("pulumi destroy", _DESTR, _NONE),
    ("pulumi stack output", _RO, _NONE),
    ("pulumi stack ls", _RO, _NONE),
    ("pulumi stack rm", _DESTR, _NONE),
    # Toolchain managers download; their reads are listed.
    ("pyenv", _FETCH, _NONE),
    ("pyenv versions", _RO, _NONE),
    # `local` writes `.python-version` in the worktree, `global` the person's own default.
    ("pyenv local", _W, _NONE),
    ("pyenv global", _SCOPE, _NONE),
    ("pyenv which", _RO, _NONE),
    ("rbenv", _FETCH, _NONE),
    ("rbenv versions", _RO, _NONE),
    ("rbenv local", _W, _NONE),
    ("rustup", _FETCH, _NONE),
    ("rustup show", _RO, _NONE),
    ("volta", _FETCH, _NONE),
    ("volta list", _RO, _NONE),
    # `python <anything>`: runs sealed, so its class is what a build is; the
    # `-m pytest` unit is a test run.
    ("python", _B, _NONE),
    ("python -m pytest", _T, _NONE),
    ("python -m pip install", _LOCK, _INDEX),
    ("python3 -m pip install", _LOCK, _INDEX),
    ("python3", _B, _NONE),
    ("python3 -m pytest", _T, _NONE),
    # Servers the person opens. Those under `python3` are longer, so they only raise.
    ("python -m http.server", _DEV, _NONE),
    ("python3 -m http.server", _DEV, _NONE),
    ("python manage.py runserver", _DEV, _NONE),
    ("python3 manage.py runserver", _DEV, _NONE),
    ("flask run", _DEV, _NONE),
    ("uvicorn", _DEV, _NONE),
    ("gunicorn", _DEV, _NONE),
    ("streamlit run", _DEV, _NONE),
    ("jupyter notebook", _DEV, _NONE),
    ("jupyter lab", _DEV, _NONE),
    ("mkdocs serve", _DEV, _NONE),
    ("node", _B, _NONE),
    ("ruby", _B, _NONE),
    ("pytest", _T, _NONE),
    # The PHP profile's runner, as Composer installs it and on the PATH.
    ("phpunit", _T, _NONE),
    ("vendor/bin/phpunit", _T, _NONE),
    ("ruff", _B, _NONE),
    ("pyright", _B, _NONE),
    ("mypy", _B, _NONE),
    ("eslint", _B, _NONE),
    ("tsc", _B, _NONE),
    ("uv sync", _LOCK, _INDEX),
    ("uv add", _LOCK, _INDEX),
    ("uv remove", _LOCK, _INDEX),
    ("uv lock", _LOCK, _INDEX),
    ("uv pip install", _LOCK, _INDEX),
    ("uv pip sync", _LOCK, _INDEX),
    ("uv run", _B, _NONE),
    ("uv publish", _PUSH, _INDEX),
    ("bundle install", _LOCK, _INDEX),
    ("bundle exec", _B, _NONE),
    ("bundle exec rspec", _T, _NONE),
    ("bundle exec rails server", _DEV, _NONE),
    ("bundle exec jekyll serve", _DEV, _NONE),
    ("rails server", _DEV, _NONE),
    ("rails s", _DEV, _NONE),
    ("jekyll serve", _DEV, _NONE),
    ("jekyll s", _DEV, _NONE),
    ("bundle add", _LOCK, _INDEX),
    ("rspec", _T, _NONE),
    ("gem install", _LOCK, _INDEX),
    ("rake", _B, _NONE),
    ("rake test", _T, _NONE),
    ("rake db:migrate", _DESTR, _NONE),
    ("rake db:drop", _DESTR, _NONE),
    ("rake db:reset", _DESTR, _NONE),
    ("serverless", _PUSH, _NONE),
    ("serverless logs", _RO, _NONE),
    ("serverless info", _RO, _NONE),
    ("sls", _PUSH, _NONE),
    ("sls logs", _RO, _NONE),
    ("sls info", _RO, _NONE),
    ("sfdx", _PUSH, _NONE),
    ("sfdx force:org:list", _RO, _NONE),
    ("skaffold", _PUSH, _NONE),
    ("skaffold dev", _DEV, _NONE),
    ("skaffold build", _B, _NONE),
    ("sst", _PUSH, _NONE),
    ("sst dev", _DEV, _NONE),
    ("sst remove", _DESTR, _NONE),
    ("swift build", _B, _NONE),
    ("swift test", _T, _NONE),
    ("swift run", _B, _NONE),
    ("swift package", _B, _NONE),
    # Fetches the packages Package.resolved names (the Swift profile's bootstrap).
    ("swift package resolve", _LOCK, _INDEX),
    # The Scala profile's commands.
    ("sbt test", _T, _NONE),
    ("sbt compile", _B, _NONE),
    ("systemctl", _SCOPE, _NONE),
    ("systemctl status", _RO, _NONE),
    ("terraform", _PUSH, _NONE),
    ("terraform plan", _RO, _NONE),
    ("terraform validate", _B, _NONE),
    ("terraform fmt", _B, _NONE),
    ("terraform init", _FETCH, _NONE),
    ("terraform destroy", _DESTR, _NONE),
    ("terraform workspace list", _RO, _NONE),
    ("terraform workspace show", _RO, _NONE),
    ("terraform workspace select", _W, _NONE),  # writes `.terraform/environment`
    ("terraform workspace delete", _DESTR, _NONE),
    # A terminal multiplexer: a process that stays, asked as a server is.
    ("tmux", _DEV, _NONE),
    ("turbo run", _B, _NONE),
    ("turbo run build", _B, _NONE),
    ("turbo run test", _T, _NONE),
    ("turbo run lint", _B, _NONE),
    ("turbo run dev", _DEV, _NONE),
    ("ufw", _SCOPE, _NONE),
    ("ufw status", _RO, _NONE),
    # Vault holds credentials; reading one is credential-bearing (CREDENTIAL, T2).
    ("vault", _CRED, _NONE),
    ("vault status", _RO, _NONE),
    ("vault auth list", _RO, _NONE),
    ("vault kv get", _CRED, _NONE),
    ("vault kv put", _CRED, _NONE),
    ("vault kv list", _RO, _NONE),
    ("vercel", _PUSH, _NONE),
    ("vercel dev", _DEV, _NONE),
    ("vercel logs", _RO, _NONE),
    ("vercel ls", _RO, _NONE),
    ("wp", _PUSH, _NONE),
    ("wp plugin list", _RO, _NONE),
    ("yarn install", _LOCK, _INDEX),
    ("yarn add", _LOCK, _INDEX),
    ("yarn remove", _LOCK, _INDEX),
    ("yarn test", _T, _NONE),
    ("yarn publish", _PUSH, _INDEX),
    ("yarn dlx", _FETCH, _NONE),
    ("yarn run dev", _DEV, _NONE),
    ("yarn dev", _DEV, _NONE),
    ("yarn start", _DEV, _NONE),
    ("yarn serve", _DEV, _NONE),
    ("yarn run start", _DEV, _NONE),
    ("yarn run serve", _DEV, _NONE),
    ("yarn run build", _B, _NONE),
    ("yarn run test", _T, _NONE),
    ("yarn run lint", _B, _NONE),
)

SHIPPED: tuple[CommandRow, ...] = tuple(
    CommandRow(prefix=tuple(line.split()), cls=cls, destination=to) for line, cls, to in _TABLE
)
"""The shipped defaults: the classes, the pattern rows' commands as
prefixes, and OpenCode's `BashArity` dictionary — every entry the author could
classify by hand. **Coverage against that dictionary is asserted by
`test_commands.py`**, whose fixture is the dictionary verbatim: every entry has a row here,
a family of rows under it, or a line in `EXCLUDED` saying why not."""

EXCLUDED: Mapping[str, str] = {
    "export": "a shell builtin, not an argv the executor spawns (EX-1: argv, never a shell)",
    "source": "a shell builtin, not an argv the executor spawns (EX-1)",
    "unset": "a shell builtin, not an argv the executor spawns (EX-1)",
    "nvm": "a shell function, not an executable; there is no argv to classify",
    "kill": (
        "a process signal: under the sandbox's pid namespace it reaches nothing outside the "
        "command, and outside one it is not the harness's to classify — unrecognised, sealed"
    ),
    "killall": "as `kill`",
    "npm view": (
        "a registry read: neither an install nor a fetch-and-run, and the sandbox already "
        "answers it (sealed, it fails typed; declared `network=`, it is row 1) — unrecognised"
    ),
}
"""Arity entries left out, each with the reason. An entry the author cannot classify is not
given a made-up class: it is unrecognised, sealed, T0, *once* only."""

FAMILY_HEADS: Mapping[str, str] = {
    "aws": "the head mutates shared infrastructure by default; the reads are listed",
    "az": "as `aws`",
    "gcloud": "as `aws`",
    "doctl": "as `aws`",
    "sfdx": "as `aws`",
    "gh": "as `aws` — the GitHub CLI's default verbs create and merge; the reads are listed",
    "cdk": "as `aws`",
    "cf": "as `aws`",
    "consul": "as `aws`",
    "crictl": "as `aws`",
    "eksctl": "as `aws`",
    "firebase": "as `aws`",
    "flyctl": "as `aws`",
    "helm": "as `aws`",
    "heroku": "as `aws`",
    "kubectl": "as `aws`",
    "mc": "as `aws`",
    "pulumi": "as `aws`",
    "serverless": "as `aws`",
    "sls": "as `aws`",
    "skaffold": "as `aws`",
    "sst": "as `aws`",
    "terraform": "as `aws`",
    "vercel": "as `aws`",
    "wp": "as `aws`",
    "cmake": "every subcommand configures or builds; one class fits",
    "hugo": "builds the site; `server` is the one exception and is listed",
    "make": "runs a build target; `test` and `install` are listed",
    "minikube": "a local cluster: contained; `delete` is listed",
    "mongosh": "a database shell: what it can do is drop the database",
    "mysql": "as `mongosh`",
    "psql": "as `mongosh`",
    "redis-cli": "as `mongosh`; `ping` is listed",
    "mvn": "builds; `test`, `deploy` and `clean` are listed",
    "ng": "builds; `serve` and `test` are listed",
    "openssl": "generates keys: credential-bearing; the inspections are listed",
    "openssl req": "a certificate request carries a key: credential-bearing",
    "openssl x509": "inspects a certificate; one class fits",
    "python": "runs sealed, so its class is a build's; `-m pytest` is listed",
    "pyenv": "a toolchain manager downloads; its reads are listed",
    "rbenv": "as `pyenv`",
    "rustup": "as `pyenv`",
    "volta": "as `pyenv`",
    "rake": "runs a task; the database tasks are listed as destructive",
    "systemctl": "system services are root's; `status` is listed",
    "ip": "network configuration is root's; the `show` verbs are listed",
    "ufw": "the firewall is root's; `status` is listed",
    "tmux": "a process that stays; one class fits",
    "vault": "holds credentials; the reads that do not return one are listed",
    "bun x": "whatever follows is fetched and run",
    "npm exec": "as `bun x`",
    "npm init": "as `bun x` — `npm init <x>` fetches `create-<x>` and runs it",
    "pnpm dlx": "as `bun x`",
    "pnpm exec": "runs a local binary",
    "yarn dlx": "as `bun x`",
    "cargo add": "the crate is the destination extractor's, not the prefix's",
    "cargo run": "runs the crate; one class fits",
    "git config": "writes `.git/config`; harmless whichever key",
    "git remote": "manages remotes; `remove` is listed",
    "git stash": "reflog-recoverable; `drop` and `clear` are listed",
    "kubectl kustomize": "renders manifests; one class fits",
}
"""Dictionary families (an entry whose arity exceeds its own token count) that also carry a
row at the *head*, and why one class fits the head. Without a reason here a head row is a
mistake: it would give every subcommand a class the author never looked at, which is the
wildcard the table refuses."""


@functools.cache
def shipped() -> CommandTable:
    """The shipped table, built once: a table is never changed after it is built
    (`with_rows` returns a new one), so every caller can share it."""
    return CommandTable(SHIPPED)


def profile_rows(profile: LanguageProfile) -> tuple[CommandRow, ...]:
    """The profile's declared commands as rows, so the language profile's vocabulary and this
    table are one thing. The prefix is the template's argv up to its first placeholder."""
    rows: list[CommandRow] = []
    declared = (
        (profile.tests.runner, CommandClass.TESTS, Destination.NONE),
        (profile.build.compile_cmd, CommandClass.BUILD, Destination.NONE),
        (profile.build.import_smoke, CommandClass.BUILD, Destination.NONE),
        (profile.deps.install_cmd, CommandClass.LOCKFILE_INSTALL, Destination.INDEX_HOST),
        (profile.toolchain.bootstrap, CommandClass.LOCKFILE_INSTALL, Destination.INDEX_HOST),
    )
    for template, cls, destination in declared:
        if template is None:
            continue
        prefix: list[str] = []
        for token in template.argv:
            if "{" in token:
                break
            prefix.append(token)
        if prefix:
            rows.append(CommandRow(tuple(prefix), cls, destination, source="profile"))
    return tuple(rows)


class CommandTable:
    """Rows keyed by prefix, longest prefix winning, and the shipped pattern rows. Never
    changed once built."""

    def __init__(self, rows: Iterable[CommandRow]) -> None:
        self._rows: dict[tuple[str, ...], CommandRow] = {}
        for row in rows:
            if not row.prefix or any(not t for t in row.prefix):
                raise CommandTableError(f"a command row needs a non-empty prefix: {row}")
            self._rows[row.prefix] = row
        # Option rows by their command words: `git push --force` under `("git", "push")`.
        self._option_rows: dict[tuple[str, ...], list[CommandRow]] = {}
        for row in self._rows.values():
            if _is_option_row(row):
                self._option_rows.setdefault(row.prefix[:-1], []).append(row)

    @property
    def rows(self) -> tuple[CommandRow, ...]:
        return tuple(self._rows.values())

    def row_for(self, prefix: Sequence[str]) -> CommandRow | None:
        return self._rows.get(tuple(prefix))

    def classify(self, argv: Sequence[str]) -> Classification | None:
        """An argv as the executor would start it: its own row, the rows of every command
        it starts (`sudo rm`, `sh -c "…"`), and the pattern rows over its words. `None` is
        *unrecognised*: no class, sealed, T0."""
        words = tuple(argv)
        reading = read_argv(words)
        return self._classify(reading.commands, " ".join(words), plain=reading.plain)

    def classify_line(self, line: str) -> Classification | None:
        """A shell line, never evaluated: every simple command in it (`shell_line`), and the
        pattern rows over the whole line. A line the grammar cannot read is not plain; the
        commands it could read still raise its tier, and the pattern rows read its text."""
        reading = read_line(line)
        return self._classify(reading.commands, line, plain=reading.plain)

    def git_line(self, line: str) -> GitLine:
        """How much of what `line` changes in source git's own history accounts for, read from
        the commands it runs, never from how they are joined: `cd x && git
        switch y`, `git switch y 2>&1 | tail -5` and `bash -c "git switch y"` are the switch
        alone. Asked by the table, so a read is what the table says reads, and a row that
        raises a command's class also takes it out of `HISTORY`. A line it cannot be sure of
        is `BESIDE`, never `HISTORY`: there what is kept is checked against git's history."""
        reading = read_line(line)
        history = False
        alone = reading.whole and all(t in _HARMLESS_TARGETS for t in reading.writes_to)
        for argv in reading.commands:
            head = argv[0].rsplit("/", 1)[-1] if argv else ""
            if head == "git":
                words = argv[first_operand(argv, _GLOBAL_OPTIONS["git"]) :]
                if words and words[0] in HISTORY_GIT:
                    history = True
                    continue
            if (head in RUNS_ITS_ARGUMENT and self._match(argv) is None) or runs_its_body(argv):
                # Its part starts the command after it, which is read on its own.
                continue
            if head in PRINTERS:
                # `|| true`, `; echo done`: prints or succeeds; a redirect is seen above.
                continue
            match = self._match(argv)
            if match is None or match.cls is not CommandClass.READ_ONLY:
                alone = False
        if history and alone:
            return GitLine.HISTORY
        if any(argv and argv[0].rsplit("/", 1)[-1] == "git" for argv in reading.commands):
            return GitLine.BESIDE
        return GitLine.NONE

    def git_refusal(self, line: str) -> GitRefusal | None:
        """Why `line`'s git is refused before it runs in a shell that records writes, or
        `None`. git that forges objects or moves a ref by hand (`FORGING_GIT`), anywhere; and,
        in a line `git_line` does not read as `HISTORY`, git that records the tree
        (`RECORDING_GIT`) or builds a commit (`COMMITTING_GIT`). `git add -A &&
        git commit -m wip` records only what the tree already held, and is `None`."""
        found: list[GitRefusal] = []
        history = self.git_line(line) is GitLine.HISTORY
        for argv in read_line(line).commands:
            if not argv or argv[0].rsplit("/", 1)[-1] != "git":
                continue
            sub, *rest = argv[first_operand(argv, _GLOBAL_OPTIONS["git"]) :] or ("",)
            command = f"git {sub}"
            if sub in FORGING_GIT or (sub == "hash-object" and "-w" in rest):
                return GitRefusal(
                    command,
                    f"{command} builds git objects or moves a ref from content git's history "
                    "does not hold, which a later switch would bring to disk unchecked (D229)",
                    "leave git's objects and refs to git's own commands: git commit to commit, "
                    "git switch, git reset or git branch to move a branch",
                )
            if history:
                continue
            if sub in RECORDING_GIT and not (
                sub == "stash" and rest and rest[0] in _STASH_READS
            ):
                found.append(
                    GitRefusal(
                        command,
                        f"{command} would record into git a file this same command writes, "
                        "and a source file a command writes is put back on disk but not in "
                        "git (D228)",
                        _ALONE.format(command=command),
                    )
                )
            elif sub in COMMITTING_GIT:
                found.append(
                    GitRefusal(
                        command,
                        f"{command} makes a commit, and a file this same command writes is put "
                        "back on disk but not in that commit, leaving the branch and the files "
                        "apart (D229)",
                        _ALONE.format(command=command),
                    )
                )
        return found[0] if found else None

    def beyond_the_worktree(self, line: str, root: Path | None = None) -> str | None:
        """What a line reaches beyond the files of the worktree, as the line itself names it,
        or `None`. `git state`: a git command no read-only row names (a commit,
        a reset, `update-ref`, a config change, an unrecognised subcommand), or any path under
        a `.git` directory. `outside the repository`: an absolute path (unless within `root`),
        `~` or a path climbing out with `..`, named as a classified destination, an output
        redirect's target, a `cd`'s operand, or an operand of a command that may write (one
        no row names, or whose class is not a read). Unsealed, nothing stops either, so the
        lift of the floor does not cover them. It reads only what the line names: a write a
        command makes on its own, to a path it never shows, is not seen."""
        reading = read_line(line)
        named: list[str] = list(reading.writes_to)
        for argv in reading.commands:
            head = argv[0].rsplit("/", 1)[-1] if argv else ""
            match = self._match(argv)
            if head == "git" and (match is None or match.cls is not CommandClass.READ_ONLY):
                return GIT_STATE
            if match is not None and match.destination is not None:
                named.extend(match.destination.split())
            if head == "cd":
                # A bare `cd` goes home: every command after it runs outside.
                named.extend(_operands_named(argv) or ("~",))
            elif match is None or match.cls is not CommandClass.READ_ONLY:
                # A build or a test run is a read only where it runs sealed (`READS`); this
                # check runs unsealed, where it writes where its operands say.
                named.extend(_operands_named(argv))
        for path in named:
            where = _beyond(path, root)
            if where is not None:
                return where
        return None

    def _classify(
        self, commands: Iterable[tuple[str, ...]], text: str, *, plain: bool
    ) -> Classification | None:
        matches: list[CommandMatch] = []
        unread: list[tuple[str, ...]] = []
        after_cd = False
        for argv in commands:
            head = argv[0].rsplit("/", 1)[-1] if argv else ""
            match = self._match(argv)
            if match is None and head not in RUNS_ITS_ARGUMENT:
                # A wrapper that only runs its argument is no command of its own.
                unread.append(argv)
            if match is not None:
                # B.5: after a `cd`, `origin` and the hooks may be another repository's.
                if after_cd and head == "git":
                    match = replace(match, grantable=False)
                matches.append(match)
            after_cd = after_cd or head == "cd"
        for row in PATTERNS:
            found = row.pattern.search(text)
            if found is not None:
                matches.append(CommandMatch(row, None, found.group(0).strip()))
        if not matches:
            return None
        return Classification(tuple(matches), plain=plain, unread=tuple(unread))

    def _match(self, argv: tuple[str, ...]) -> CommandMatch | None:
        """One simple command's longest-prefix row. An option between the head and its
        subcommand (`git -C sub push`) is not part of the prefix, a fetch from loopback is
        the dev-server row, not an external fetch, and a command asked only for its
        version or its help (`terraform --version`) reads, whatever its head does."""
        for tokens in _spellings(argv):
            row = self._longest(tokens)
            rest = tokens[len(row.prefix) :] if row is not None else ()
            if row is not None and rest and set(rest) <= _ABOUT_ITSELF:
                row = CommandRow(
                    row.prefix, CommandClass.READ_ONLY, Destination.NONE, row.source
                )
            else:
                row = self._best(tokens, row)
            if row is not None:
                destination = _extract(row, tokens)
                if (
                    row.cls is CommandClass.EXTERNAL_FETCH
                    and row.destination is Destination.URL_HOST
                    and destination in _LOOPBACK
                    and _hosts({_host_of(t) for t in tokens}) == destination
                ):
                    row = CommandRow(
                        row.prefix, CommandClass.DEV_SERVER, row.destination, row.source
                    )
                refined = _git_refined(self, CommandMatch(row, destination), argv)
                shape = _shape(refined.row)
                if shape is not None and not shape.fits(tokens[len(_words(row)) :]):
                    refined = replace(refined, in_shape=False)
                return refined
        return None

    def _best(
        self, tokens: tuple[str, ...], positional: CommandRow | None
    ) -> CommandRow | None:
        """The positional row, or an option row whose option is anywhere in the arguments:
        the longest wins, and among equals the higher floor."""
        best = positional
        for n in range(1, len(tokens) + 1):
            for row in self._option_rows.get(tokens[:n], ()):
                if not _has_option(tokens[n:], row.prefix[-1]):
                    continue
                if (
                    best is None
                    or len(row.prefix) > len(best.prefix)
                    or (len(row.prefix) == len(best.prefix) and _lower(best.floor, row.floor))
                ):
                    best = row
        return best

    def _longest(self, tokens: tuple[str, ...]) -> CommandRow | None:
        for length in range(len(tokens), 0, -1):
            row = self._rows.get(tokens[:length])
            if row is not None:
                return row
        return None

    def with_rows(self, rows: Iterable[CommandRow], *, source: str) -> CommandTable:
        """Overlay rows by prefix (a keyed merge).

        A row from `project` or `user` may add a prefix or **raise** a shipped row's class;
        one whose class has a lower floor than the row it replaces refuses the load. A
        shipped row's floor is the tier the harness stands behind for that prefix, and a
        row that lowers it is a lowering nobody accepted.

        "The row it replaces" is the row that **matches** the new prefix today — the longest
        shipped prefix of it — not a row with the same key. `match` resolves by longest
        prefix, so a project row `git push --quiet -> read_only` would win over the shipped
        `git push` (T2, hard) for every `git push --quiet …` argv, and a guard that looked
        for an exact `git push --quiet` key would not see it. The table cannot enumerate
        every flag; the chain walk does not have to.
        """
        merged = dict(self._rows)
        for row in rows:
            existing = _matching(merged, row.prefix)
            if existing is not None and _lower(row.floor, existing.floor):
                raise CommandTableError(
                    f"[[commands]] row for {list(row.prefix)} ({source}) classifies it "
                    f"{row.cls.value} ({row.floor.value}), below the {existing.source} row "
                    f"{list(existing.prefix)} that matches it today, "
                    f"{existing.cls.value} ({existing.floor.value}); a row may add or "
                    "raise, never lower (D106)"
                )
            merged[row.prefix] = CommandRow(row.prefix, row.cls, row.destination, source)
        return CommandTable(merged.values())

    def with_config(self, rows: Sequence[Mapping[str, Any]], *, source: str) -> CommandTable:
        """`[[commands]]` rows from a config layer, validated: three known keys, a known
        class and destination, a token-list prefix. Unknown keys refuse."""
        return self.with_rows((parse_row(r) for r in rows), source=source)

    def row(self, prefix: tuple[str, ...]) -> CommandRow | None:
        """The row this table holds for exactly `prefix`, or None: which layer's row is in
        force for it, as the run header asks."""
        return self._rows.get(prefix)

    def layered(self, layers: Sequence[tuple[str, Sequence[CommandRow]]]) -> CommandTable:
        """Each layer's rows over this table by prefix, in order, so a later layer's row wins
        a prefix both name (the repository's, over the person's). Each layer is checked against
        this table alone (`with_rows`): no row lowers
        a shipped row, and the repository may name differently a prefix the person named."""
        merged = dict(self._rows)
        for source, rows in layers:
            self.with_rows(rows, source=source)
            for row in rows:
                merged[row.prefix] = CommandRow(row.prefix, row.cls, row.destination, source)
        return CommandTable(merged.values())


def _lower(a: Tier, b: Tier) -> bool:
    return TIER_ORDER.index(a) < TIER_ORDER.index(b)


_GLOBAL_OPTIONS: Mapping[str, frozenset[str]] = {"git": GIT_GLOBAL_OPTIONS}
"""Heads whose options may come before the subcommand, and which of those take a separate
value. `git -C sub push` is `git push`: without this, every flag written first would hide
the subcommand from the table."""

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_ABOUT_ITSELF = frozenset({"--version", "--help", "-h"})

_VALUE_OPTIONS: Mapping[tuple[str, ...], frozenset[str]] = {
    ("git", "push"): frozenset({"-o", "--push-option", "--repo", "--receive-pack", "--exec"}),
    ("git", "fetch"): frozenset(
        {"-o", "--server-option", "--upload-pack", "--depth", "--deepen", "-j", "--jobs"}
    ),
    ("git", "pull"): frozenset(
        {"-o", "--server-option", "--upload-pack", "--depth", "--deepen", "-j", "--jobs"}
    ),
    ("ssh",): frozenset(
        {"-B", "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m", "-O"}
        | {"-o", "-P", "-p", "-Q", "-R", "-S", "-W", "-w"}
    ),
    ("scp",): frozenset({"-c", "-D", "-F", "-i", "-J", "-l", "-o", "-P", "-S", "-X"}),
    ("curl",): frozenset(
        {"-A", "-b", "-C", "-c", "-d", "-E", "-e", "-F", "-H", "-K", "-m", "-o", "-r", "-T"}
        | {"-u", "-w", "-X", "-x", "-Y", "-y", "-z", "--config", "--cookie", "--cookie-jar"}
        | {"--data", "--form", "--header", "--max-time", "--output", "--proxy", "--range"}
        | {"--referer", "--request", "--upload-file", "--user", "--user-agent", "--write-out"}
    ),
    ("wget",): frozenset(
        {"-a", "-e", "-i", "-O", "-o", "-P", "-T", "-t", "-U", "--body-data", "--body-file"}
        | {"--header", "--method", "--output-document", "--output-file", "--password"}
        | {"--post-data", "--post-file", "--timeout", "--tries", "--user", "--user-agent"}
    ),
    ("docker", "pull"): frozenset({"--platform"}),
    ("podman", "pull"): frozenset({"--platform", "--arch", "--os", "--creds", "--authfile"}),
    ("podman", "push"): frozenset({"--creds", "--authfile", "--format"}),
}
"""Options a command's arguments carry a separate value for, so the value is not read as the
destination: in `git push -o origin upstream main` the push goes to `upstream`,
and in `ssh -p 2222 good.com` to `good.com`, not `2222`. Beside `_GLOBAL_OPTIONS`, in its
shape. A value option missing here is read as an operand, which only adds a word to the
destination, so a grant covers less, never more."""


def _any_value(value: str) -> bool:
    del value
    return True


@dataclass(frozen=True)
class GrantShape:
    """What a repository grant for a row covers: an allowlist, never a list of
    the bad ones. `flags` and `letters` are the options it covers as long and one-letter
    options; `long_values` and `letter_values` those that take a value, covered with a value
    `value_ok` accepts; `operands` how many words that are not options may follow (`None`:
    any number), each one `operand_ok` accepts."""

    flags: frozenset[str] = frozenset()
    letters: frozenset[str] = frozenset()
    long_values: frozenset[str] = frozenset()
    letter_values: frozenset[str] = frozenset()
    value_ok: Callable[[str], bool] = _any_value
    operands: int | None = None
    operand_ok: Callable[[str], bool] = _any_value

    def fits(self, args: tuple[str, ...]) -> bool:
        operands = 0
        i = 0
        while i < len(args):
            arg = args[i]
            i += 1
            if arg == "--":
                rest = args[i:]
                if not all(self.operand_ok(word) for word in rest):
                    return False
                operands += len(rest)
                break
            if arg.startswith("--"):
                name, eq, attached = arg.partition("=")
                if name in self.long_values:
                    value = attached if eq else (args[i] if i < len(args) else "")
                    i += 0 if eq else 1
                    if not self.value_ok(value):
                        return False
                elif name not in self.flags:
                    return False
            elif arg.startswith("-") and len(arg) > 1:
                for at, letter in enumerate(arg[1:], start=1):
                    if letter in self.letter_values:
                        attached = arg[at + 1 :]
                        value = attached or (args[i] if i < len(args) else "")
                        i += 0 if attached else 1
                        if not self.value_ok(value):
                            return False
                        break
                    if letter not in self.letters:
                        return False
            elif self.operand_ok(arg):
                operands += 1
            else:
                return False
        return self.operands is None or operands <= self.operands


_PUSH_SHAPE: Final = GrantShape(
    flags=frozenset(
        {
            "--set-upstream",
            "--quiet",
            "--verbose",
            "--dry-run",
            "--tags",
            "--follow-tags",
            "--porcelain",
            "--progress",
            "--no-progress",
        }
    ),
    letters=frozenset("uqvn"),
)
"""`git push`'s (B.3): every other option, one nobody listed included, makes the push one no
repository grant covers, so it is asked. A list of the bad ones was what failed: `--exec=` and
`--receive-pack=` were on no one's list."""

_FETCH_SHAPE: Final = GrantShape(
    flags=frozenset(
        {
            "--quiet",
            "--verbose",
            "--prune",
            "--tags",
            "--all",
            "--ff-only",
            "--rebase",
            "--no-rebase",
        }
    ),
    letters=frozenset("qvp"),
)


def _operands_named(argv: Sequence[str]) -> tuple[str, ...]:
    """The words after a command's head that may name a path: each operand, the value of an
    `--option=value` or `NAME=value`, and a short option's attached value (`-o/tmp/x`). A URL is
    not a path here; a destination says it."""
    out: list[str] = []
    for word in argv[1:]:
        value = word.split("=", 1)[1] if "=" in word else word
        if word.startswith("-") and not word.startswith("--") and "=" not in word:
            # `-o/tmp/x`: a value glued to its flag, when it reads as a path.
            value = word[2:] if len(word) > 2 and word[2] in "/~.$" else ""
        if not value or "://" in value:
            continue
        out.append(value)
    return tuple(out)


_COMPUTED_PATH: Final = re.compile(
    r"\$(?:\{?(?:HOME|TMPDIR|TMP|TEMP|XDG_[A-Z_]+)\}?$|\(.*\)/|\{?\w+\}?/)"
)
"""A word the shell computes into a path: `$HOME`, `${TMPDIR}`, or a variable or substitution
followed by a path (`$HOME/.bashrc`, `$(pwd)/..`). A bare `$VALUE` is an argument's value."""


def _beyond(path: str, root: Path | None) -> str | None:
    """`GIT_STATE` for a path under a `.git` directory, `OUTSIDE_THE_REPOSITORY` for one that
    leaves the worktree, or `None`."""
    if ".git" in path.split("/"):
        return GIT_STATE
    if path.startswith("~") or _COMPUTED_PATH.match(path):
        # Home, or a path the shell will compute (`$HOME/x`, `${TMPDIR}`, `$(pwd)/..`): not
        # known to stay inside, so the cautious answer. A bare `$VALUE` is a value, not a path.
        return OUTSIDE_THE_REPOSITORY
    normal = posixpath.normpath(path)
    if path.startswith("/"):
        inside = root is not None and within(root, normal) is not None
        return None if inside else OUTSIDE_THE_REPOSITORY
    return OUTSIDE_THE_REPOSITORY if normal == ".." or normal.startswith("../") else None


def _in_the_repository(path: str) -> bool:
    """A path that names a file in the working tree: relative, never above it, and no URL or
    home directory (`-r https://…`, `-r /tmp/x.txt`, `-r../x.txt`)."""
    parts = path.split("/")
    return bool(path) and not (
        path.startswith(("/", "~", "-")) or "://" in path or ".." in parts or "\\" in path
    )


def _not_a_file(value: str) -> bool:
    """A curl value given as itself, not read from a file: `@file` sends the file."""
    return not value.startswith("@")


_LOCKFILE_SHAPE: Final = GrantShape(
    flags=frozenset(
        {
            "--quiet",
            "--silent",
            "--verbose",
            "--frozen",
            "--locked",
            "--frozen-lockfile",
            "--no-dev",
            "--no-audit",
            "--no-fund",
            "--all-extras",
            "--all-groups",
            "--sync",
            "--deploy",
        }
    ),
    letters=frozenset("qv"),
    long_values=frozenset(
        {"--only", "--group", "--extra", "--with", "--without", "--requirement"}
    ),
    letter_values=frozenset("r"),
    value_ok=_in_the_repository,
    operands=0,
)
"""An install a lockfile or a requirements file decides: no package named, and every value
(`-r`, `--group`, …) a word that could name a file in the working tree, so what installs is what
the repository pinned. A requirements file from a URL, an absolute path or above the working
tree is outside it."""

_READ_SHAPE: Final = GrantShape(
    flags=frozenset({"--silent", "--show-error", "--fail", "--head", "--compressed"}),
    letters=frozenset("sSfI"),
    long_values=frozenset({"--header"}),
    letter_values=frozenset("H"),
    value_ok=_not_a_file,
    operands=1,
)
"""`curl` reading one URL to stdout. Anything that sends data, uploads, writes a file, reads its
options or a header from a file, follows a redirect to another host or changes the method
(`-d`, `-F`, `-T`, `-X`, `-o`, `-O`, `-K`, `--config`, `-L`, `-H @file`) is outside it, and so
is a second URL: curl takes a bare `evil.com` as one, so the grant's host could not bind it. A
header that carries a secret is still raised by the credential pattern row."""

_WGET_SHAPE: Final = GrantShape(
    flags=frozenset({"--quiet", "--no-verbose", "--server-response", "--spider"}),
    letters=frozenset("qS"),
    long_values=frozenset({"--header", "--output-document"}),
    letter_values=frozenset("O"),
    operands=1,
)
"""`wget` reading one URL (E.2). Anything that posts or uploads (`--post-file`, `--body-file`,
`--method`), reads options or URLs from a file (`-i`, `-e`, `--config`), follows links
(`-r`, `-m`, `-p`) or names a second URL is outside it. Where it writes is the call's declared
effects, which the grant binds, not its options."""

_SYNC_SHAPE: Final = replace(_LOCKFILE_SHAPE, operands=None, operand_ok=_in_the_repository)
"""`uv pip sync <file>…`: a lockfile install whose files are operands, each in the working
tree."""

_SHAPES: Mapping[tuple[str, ...], GrantShape] = {
    ("git", "push"): _PUSH_SHAPE,
    ("git", "fetch"): _FETCH_SHAPE,
    ("git", "pull"): _FETCH_SHAPE,
    **{
        tuple(words.split()): _LOCKFILE_SHAPE
        for words in (
            "pip install",
            "pip3 install",
            "uv sync",
            "npm ci",
            "npm install",
            "pnpm install",
            "yarn install",
            "bun install",
            "poetry install",
            "pdm install",
            "pipenv install",
            "bundle install",
            "composer install",
            "go mod download",
        )
    },
    ("uv", "pip", "sync"): _SYNC_SHAPE,
    ("curl",): _READ_SHAPE,
    ("wget",): _WGET_SHAPE,
}
"""The rows a person grants in ordinary work, by their command words (E.2). Every other row a
grant can cover has none: its repository grant covers any arguments, offered at the prompt
only where its effects stay on this machine (`repo_offered`)."""

_LEAVES_THE_MACHINE_CLASSES: Final = frozenset(
    {CommandClass.PUSH, CommandClass.CREDENTIAL, CommandClass.SCOPE_EXPANDING}
)
_ACTS_ON_ANOTHER_HOST: Final = frozenset({"ssh", "nc", "ncat", "telnet", "ftp"})
"""`external_fetch` rows that run something on another host rather than read from it."""


def _shape(row: CommandRow | CommandPattern) -> GrantShape | None:
    return _SHAPES.get(_words(row)) if isinstance(row, CommandRow) else None


def _leaves_the_machine(row: CommandRow | CommandPattern) -> bool:
    """E.3: an effect beyond this machine that cannot be taken back."""
    if row.cls in _LEAVES_THE_MACHINE_CLASSES:
        return True
    return isinstance(row, CommandRow) and row.prefix[0] in _ACTS_ON_ANOTHER_HOST


_PUSH_DESTRUCTIVE_KEY = re.compile(r"^remote\.[^=]*\.(mirror|push)$", re.IGNORECASE)
"""A `git -c` key that turns any push into a mirror or a forced refspec."""


def _is_option_row(row: CommandRow) -> bool:
    """A row whose prefix ends in one option after its command words (`git push --force`,
    `sed -i`), which matches wherever its option is."""
    return len(row.prefix) > 1 and row.prefix[-1].startswith("-")


def _words(row: CommandRow) -> tuple[str, ...]:
    """A row's command words: its prefix without the option an option row ends in."""
    return row.prefix[:-1] if _is_option_row(row) else row.prefix


def _has_option(args: tuple[str, ...], option: str) -> bool:
    """Whether `option` is among `args`, the arguments after a row's command words: as itself; a
    long option also as `--long=value` and as any abbreviation of it, as git
    and GNU `getopt_long` accept one (`--del` is `--delete`); a one-letter option also as a
    letter of a single-dash word (`-uf`, `-ni`, `-i.bak`). An abbreviation another option shares
    and a value read as a letter are misreadings that only ever raise. A multi-letter
    single-dash option (`find`'s `-delete`) is a whole word. Nothing after `--` is an option."""
    for arg in args:
        if arg == option:
            return True
        if arg == "--":
            return False
        if may_be(arg, option):
            # A glob the shell may expand into the option (`-del*` names a file `-delete`).
            return True
        name = arg.split("=", 1)[0]
        long_form = (
            option.startswith("--")
            and name.startswith("--")
            and len(name) > 2
            and option.startswith(name)
        )
        in_cluster = (
            len(option) == 2
            and arg.startswith("-")
            and not arg.startswith("--")
            and option[1] in arg[1:]
        )
        if long_form or in_cluster:
            return True
    return False


def _split_git(argv: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A git argv's options before its subcommand, and the subcommand onwards."""
    i = first_operand(argv, _GLOBAL_OPTIONS["git"])
    return argv[1:i], argv[i:]


def _config_keys(options: tuple[str, ...]) -> tuple[str, ...]:
    """The keys `git -c key=value` and `--config-env key=ENV` set."""
    keys: list[str] = []
    for i, option in enumerate(options):
        if option in ("-c", "--config-env") and i + 1 < len(options):
            keys.append(options[i + 1].split("=", 1)[0])
        elif option.startswith("--config-env="):
            keys.append(option.removeprefix("--config-env=").split("=", 1)[0])
    return tuple(keys)


def _operands(args: tuple[str, ...], takes_value: frozenset[str]) -> tuple[str, ...]:
    """`args` without their options and the values of those that take one."""
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == "--":
            continue
        elif arg.startswith("-") and len(arg) > 1:
            skip = arg in takes_value
        else:
            out.append(arg)
    return tuple(out)


def _git_refined(
    table: CommandTable, match: CommandMatch, argv: tuple[str, ...]
) -> CommandMatch:
    """What git's own arguments say beyond the row (B.3, B.5, B.6): a push that
    forces or deletes by refspec or by `-c` is destructive, a push outside its allowlist and a
    git command pointed at another repository are not grantable, and a branch deleted by force
    in any spelling is destructive."""
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return match
    options, rest = _split_git(argv)
    if options:
        match = replace(match, grantable=False)
    sub, args = rest[:1], rest[1:]
    if sub == ("push",):
        operands = _operands(args, _VALUE_OPTIONS[("git", "push")])
        if any(_PUSH_DESTRUCTIVE_KEY.match(k) for k in _config_keys(options)) or any(
            o.startswith(("+", ":")) for o in operands
        ):
            match = replace(
                match,
                row=CommandRow(("git", "push"), CommandClass.DESTRUCTIVE, Destination.REMOTE),
            )
    elif sub == ("branch",):
        forced_delete = table.row_for(("git", "branch", "-D"))
        if (
            forced_delete is not None
            and match.cls is not CommandClass.DESTRUCTIVE
            and (_has_option(args, "--delete") or _has_option(args, "-d"))
            and (_has_option(args, "--force") or _has_option(args, "-f"))
        ):
            match = replace(match, row=forced_delete, destination=None)
    return match


def _spellings(argv: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """The ways an argv names its command, most exact first: `/usr/bin/rm` is `rm`, and a
    dotted head is its family when the head itself has no row (`mkfs.ext4` is `mkfs`,
    `python3.12` is `python3`)."""
    if not argv:
        return
    head = argv[0].rsplit("/", 1)[-1]
    heads = [head] if "." not in head.lstrip(".") else [head, head.lstrip(".").split(".")[0]]
    for name in heads:
        if name:
            yield _without_global_options((name, *argv[1:]))


def _without_global_options(argv: tuple[str, ...]) -> tuple[str, ...]:
    takes_value = _GLOBAL_OPTIONS.get(argv[0]) if argv else None
    if takes_value is None:
        return argv
    return (argv[0], *argv[first_operand(argv, takes_value) :])


def _matching(
    rows: Mapping[tuple[str, ...], CommandRow], prefix: tuple[str, ...]
) -> CommandRow | None:
    """The row `_match` would return for `prefix` as an argv: the longest key that is a
    prefix of it, the key itself included."""
    for length in range(len(prefix), 0, -1):
        row = rows.get(prefix[:length])
        if row is not None:
            return row
    return None


_ROW_KEYS = frozenset({"prefix", "class", "destination"})


def parse_row(raw: Mapping[str, Any]) -> CommandRow:
    """One `[[commands]]` row, validated: three known keys, a known class and
    destination, a token-list prefix whose first token is a command's name. Refused naming
    the problem, never loaded and left to not match."""
    unknown = sorted(set(raw) - _ROW_KEYS)
    if unknown:
        raise CommandTableError(
            f"[[commands]] row has unknown key(s) {unknown}; known: {sorted(_ROW_KEYS)}"
        )
    prefix_raw: object = raw.get("prefix")
    if (
        not isinstance(prefix_raw, list)
        or not prefix_raw
        or not all(isinstance(t, str) and t for t in prefix_raw)  # pyright: ignore[reportUnknownVariableType]
    ):
        raise CommandTableError(
            f"[[commands]] prefix must be a non-empty list of tokens, got {prefix_raw!r}"
        )
    try:
        cls = CommandClass(str(raw.get("class")))
    except ValueError:
        raise CommandTableError(
            f"[[commands]] class {raw.get('class')!r} is not one of "
            f"{[c.value for c in CommandClass]}"
        ) from None
    try:
        destination = Destination(str(raw.get("destination", "none")))
    except ValueError:
        raise CommandTableError(
            f"[[commands]] destination {raw.get('destination')!r} is not one of "
            f"{[d.value for d in Destination]}; extractors are a closed set (D106)"
        ) from None
    tokens = tuple(cast("list[str]", prefix_raw))
    if "/" in tokens[0]:
        # The table matches a command by its name at any path (`_spellings`), so a row
        # naming a directory would never match: `./deploy.sh` is `deploy.sh` here.
        name = tokens[0].rsplit("/", 1)[-1]
        raise CommandTableError(
            f"[[commands]] prefix {list(tokens)} names a directory; a row matches a command "
            f"by its name at any path, so write {[name, *tokens[1:]]}"
        )
    return CommandRow(tokens, cls, destination)


# -- the extractors: a closed set --------------------------------------------------------


def _extract(row: CommandRow, argv: tuple[str, ...]) -> str | None:
    # After the row's command words, so an option row matched anywhere (B.1) reads the same
    # arguments as the row it outranks: `git push origin main --force` binds `origin`.
    words = _words(row)
    rest = argv[len(words) :]
    match row.destination:
        case Destination.REMOTE:
            operands = _operands(rest, _VALUE_OPTIONS.get(words, frozenset()))
            if words == ("scp",):
                # The hosts it copies to or from (`host:path`), never a local file's name.
                return _hosts({_remote_host(o) for o in operands})
            return operands[0] if operands else None
        case Destination.URL_HOST:
            # Every host the line names, so a grant for one host never covers a line that also
            # reaches another (`curl https://good.com https://evil.com`), however the host is
            # written: a URL, curl's and wget's bare `example.com`, git's `git@host:path`, an
            # image's registry.
            operands = _operands(rest, _VALUE_OPTIONS.get(words, frozenset()))
            if words in _IMAGE_ROWS:
                return _hosts({_registry(o, words[0]) for o in operands})
            bare = words in (("curl",), ("wget",))
            return _hosts(
                {_host_of(t) for t in argv} | {_operand_host(o, bare=bare) for o in operands}
            )
        case Destination.INDEX_HOST:
            # From the project's index config, never the argv. Until config
            # names one, the destination is the class's own word, so a grant still binds
            # something stable rather than the argv's spelling.
            return "index"
        case Destination.PATH_ROOTS:
            roots = sorted(t for t in rest if not t.startswith("-"))
            return " ".join(roots) if roots else None
        case Destination.NONE:
            return None


_IMAGE_ROWS: Final = frozenset(
    {("docker", "pull"), ("docker", "push"), ("podman", "pull"), ("podman", "push")}
)


def _hosts(found: set[str | None]) -> str | None:
    hosts = sorted(h for h in found if h)
    return " ".join(hosts) if hosts else None


def _remote_host(operand: str) -> str | None:
    """The host of scp's and git's `[user@]host:path`, or of a URL; `None` for a local path."""
    if "://" in operand:
        return _host_of(operand)
    head, colon, _path = operand.partition(":")
    if not colon or not head or "/" in head:
        return None
    return head.rsplit("@", 1)[-1].lower()


def _operand_host(operand: str, *, bare: bool) -> str | None:
    """The host an operand names: a URL's, `host:path`'s, and, where `bare`, a word curl or
    wget takes as a URL without its scheme (`curl example.com`)."""
    host = _remote_host(operand)
    if host is None and bare:
        host = _host_of(f"http://{operand}")
    return host


def _registry(reference: str, head: str) -> str:
    """The registry an image reference names: its first component when that is a host
    (`good.io/x`, `localhost:5000/x`), else the default an unqualified name resolves to:
    Docker Hub for docker, and for podman whatever its search list says, named as such."""
    first, slash, _rest = reference.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        return first.lower()
    return "docker.io" if head == "docker" else "unqualified"


def _host_of(token: str) -> str | None:
    if "://" not in token:
        return None
    try:
        host = urlsplit(token).hostname
    except ValueError:
        return None
    return host or None
