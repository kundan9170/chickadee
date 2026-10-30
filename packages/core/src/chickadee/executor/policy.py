"""Per-class rules as data.

A table rather than branches, so "what may a VERIFICATION command do" is one row a reviewer
can read, and adding a class is a row rather than an edit to every call site.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from chickadee.approval.commands import CLASS_WORDS, CommandTable, shipped
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.isolation.types import Isolation
from chickadee.secrets.denylist import Denial, Denylist, name_denied
from chickadee.shell_line import split


@dataclass(frozen=True)
class ClassRules:
    repo_writable: bool
    network_default_off: bool
    effects_enforced: bool
    """`True` hard-fails on an undeclared write; `False` logs it.

    The worktree class is advisory because legitimate test runs touch caches, `__pycache__`
    and coverage files unpredictably, and forcing a declaration there produces constant false
    failures. Actions, being deliberate and approved, get the strict check."""
    needs_approval: bool


RULES: dict[PolicyClass, ClassRules] = {
    PolicyClass.DISCOVERY: ClassRules(False, True, True, False),
    PolicyClass.ACTION: ClassRules(True, True, True, True),
    PolicyClass.WORKTREE: ClassRules(True, True, False, False),
    PolicyClass.HUMAN: ClassRules(True, False, False, False),
}


def rules_for(policy: PolicyClass) -> ClassRules:
    return RULES[policy]


def command_argv(request: ExecRequest) -> tuple[str, ...]:
    """What spawns for a request, before any wrap: a shell line under `/bin/sh -c`, or argv."""
    return ("/bin/sh", "-c", request.shell_line) if request.shell_line else request.argv


def isolation_for(
    request: ExecRequest,
    *,
    deny_read: tuple[Path, ...] = (),
    denylist: Denylist | None = None,
    scratch: tuple[Path, ...] = (),
    tmp: Path | None = None,
) -> Isolation:
    """Map proposed-command policy onto wrap input. Isolation does not know PolicyClass.

    **Undeclared runs sealed.** A request that declares no network gets none, and in
    a class whose writes are held to its declaration (`effects_enforced`), one that declares
    no effects gets a read-only repository, whatever the command is. Under that profile
    there is nothing to undo, which is what makes an undeclared call T0 without a
    classifier. VERIFICATION and HUMAN keep the writable rule: a test run writes its
    caches whatever it declared, and a person's own command is theirs.

    **Every class runs masked.** `deny_read` is the caller's `resolve()` of the same
    `denylist`, done once per session. The list's bare-name half goes to backends that
    match names anywhere on disk (`Isolation.deny_patterns`). `None` is the
    built-in list, as it is for `prefilter`. Under `allow_secret_read` both are empty: one
    switch, every consumer.

    **And it does not break ordinary work.** A certificate pattern never
    masks a trust store or a CA bundle the environment names; `scratch` (the repository's tool
    cache) and `tmp` (the session's) are writable whatever the class.
    """
    argv = command_argv(request)
    rules = rules_for(request.policy)
    listed = denylist if denylist is not None else Denylist()
    spared = listed.trust_stores()
    declared_writes = bool(request.effects) or not rules.effects_enforced
    writable = rules.repo_writable and declared_writes
    return Isolation(
        argv=argv,
        cwd=request.cwd,
        network_off=request.network.off,
        repo_writable=writable,
        # Only where the workspace is writable anyway: beside a declared write, or in a class
        # whose writes are not held to one (the box shell's worktree root). A request
        # cannot open a path through this alone.
        writable=request.also_writable if writable else (),
        read_only=request.read_only if writable else (),
        deny_read=() if listed.allow_secret_read else deny_read,
        deny_patterns=listed.name_patterns(),
        spared_dirs=spared.dirs,
        spared_paths=spared.paths,
        spared_patterns=spared.patterns,
        scratch=scratch,
        tmp=tmp,
    )


SECRET_PATH = "SECRET_PATH"
"""The typed refusal name a harness read returns for a deny-listed path
(`ApplyReason.SECRET_PATH`), reused here so a caller matches one name at every read
surface. It prefixes the string `prefilter` returns; `secret_path_denial` is the typed form."""


@dataclass(frozen=True)
class SecretPathDenial:
    """The pre-filter saw a deny-listed path *named* in an argv.

    Typed so a caller journals `secret_denied{path, rung, surface: "shell"}` from fields
    rather than parsing a message, and refuses before anything spawns. This is the
    message, not the boundary: the sandbox mask (`Isolation.deny_read`) is what stops a
    path the pre-filter did not see.
    """

    token: str
    """The argv token as written — `.env`, `~/.aws/credentials`."""
    denial: Denial
    """The deny-list match, carrying the resolved path and the pattern that named it."""

    @property
    def reason(self) -> str:
        return f"{SECRET_PATH}: {self.denial.reason}"


def secret_path_denial(
    request: ExecRequest,
    denylist: Denylist | None = None,
    *,
    home: Path | None = None,
) -> SecretPathDenial | None:
    """The first argv token that names a deny-listed path, for **every** policy class.

    A `.env` in an ACTION's argv is not made safe by approval, and a HUMAN `!cat.env` is
    the person asking the harness to put a secret into a transcript the model reads. A
    shell line is split with `shell_line.split` so `sh -c "cat.env"` is seen through the `-c`;
    `~` is expanded the way the shell would. Each token is a name the process will open and
    this will not, so it is judged by `secrets.name_denied`: every tail of it, so
    `secrets/prod.pem` and `prod.pem` are the same refusal, and `~/.docker/config.json` and
    `sub/.docker/config.json` are the entry `.docker/config.json` as `read` refuses them. A
    token that is a *directory* is judged as one, so `pytest tests/secrets/` is not a
    credential.
    False-positive-tolerant on purpose, but only where the tolerance costs a rephrasing:
    refusing a repository's own lint and test commands is not tolerance, it is a broken
    harness, which is why the directory question is asked here and not left to `*secret*`.
    """
    denylist = denylist if denylist is not None else Denylist()
    if denylist.allow_secret_read:
        return None
    for token in _tokens(request):
        candidate = _expand(token, home)
        if not _looks_like_a_path(candidate, request.cwd):
            # `grep -r secret_key src/` names a WORD, not a file; `*secret*` would refuse
            # it. A token is a path when it is shaped like one or exists as one.
            continue
        denial = name_denied(candidate, denylist, is_dir=_is_dir(candidate, request.cwd))
        if denial is not None:
            return SecretPathDenial(token=token, denial=denial)
    return None


def _is_dir(text: str, cwd: Path) -> bool:
    """Whether the token names a directory on disk. A trailing separator says so without
    asking the filesystem, which is what an argv naming a tree the command will create or
    walk (`pytest tests/secrets/`) usually carries."""
    if text.endswith("/"):
        return True
    try:
        return (cwd / text).is_dir()
    except OSError:
        return False


def _looks_like_a_path(text: str, cwd: Path) -> bool:
    if "/" in text or text.startswith((".", "~")) or "." in text:
        return True
    try:
        return (cwd / text).exists()
    except OSError:
        return False


def _tokens(request: ExecRequest) -> list[str]:
    """Every word the command would see, shell lines included, `-c` bodies re-split."""
    words = list(request.argv) if request.shell_line is None else split(request.shell_line)
    out: list[str] = []
    for word in words:
        out.append(word)
        # `sh -c "cat.env"` carries its real argv inside one token; look inside it.
        if " " in word:
            out.extend(split(word))
    return out


def _expand(token: str, home: Path | None) -> str:
    text = token.strip("'\"")
    if text.startswith("~"):
        root = str(home) if home is not None else os.path.expanduser("~")
        text = root + text[1:] if text == "~" or text.startswith("~/") else text
    # Posix separators for the matcher, whatever the host wrote.
    return text.replace("\\", "/")


def prefilter(
    request: ExecRequest,
    denylist: Denylist | None = None,
    commands: CommandTable | None = None,
) -> str | None:
    """Reject the obvious early, with a readable reason. Returns the reason, or None.

    Two checks, in the order that keeps the more expensive one honest. **A deny-listed
    path named in the argv** refuses in every class — the string is prefixed
    `SECRET_PATH:` and `secret_path_denial` gives the same fact typed. **A DISCOVERY command
    that is not a read** refuses in that class only: the command table says what it is,
    and a line any of whose commands is not one of `READS` acts. **A pass
    here proves nothing** — it is a convenience for the caller, and the sandbox is what makes
    read-only and unreadable true.

    `denylist` is the list the run resolved — a repository's additions and named removals
    and SE-8's `allow_secret_read` included. `None` is
    the built-in list, which is right for a caller with no config in hand and wrong for
    `Executor.run`, which is to pass its own: a pre-spawn check that ignored the configured
    list would refuse what the config removed and allow what it added. `commands` is the
    `[[commands]]` table; `None` is the shipped table.
    """
    secret = secret_path_denial(request, denylist)
    if secret is not None:
        return secret.reason
    if request.policy is not PolicyClass.DISCOVERY:
        return None
    table = commands if commands is not None else shipped()
    found = (
        table.classify_line(request.shell_line)
        if request.shell_line
        else table.classify(request.argv)
    )
    action = found.action() if found is not None else None
    return (
        f"DISCOVERY is read-only and this looks mutating ({action.text!r} "
        f"{CLASS_WORDS[action.cls]}); the sandbox would deny it anyway, this is the earlier "
        "and clearer message"
        if action is not None
        else None
    )
