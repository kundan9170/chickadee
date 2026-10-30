"""Executable configuration is declared by the project and trusted by the person.

Making project config committed creates a problem the old excluded-directory layout did not
have, and it must be faced rather than inherited: `[[hooks]].command`,
`[[mcp.servers]].command` and `[executor.env].passthrough` mean **cloning a repository would
otherwise be enough to run its author's commands**. `[[commands]]` rows are the same kind of
line: a row that names `deploy.sh` a read would let it run unasked.

So the executable subset is inert until its digest appears in a **repo-local** trust file.
The record is never project-scoped: a repository that could ship its own acceptance would be
asserting its own trustworthiness.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from chickadee.paths import within

TRUST_FILE = "trusted.toml"
"""§8, repo-local. Plain text on purpose: a person must be able to read what they accepted."""

SHRINKS_THE_DENY_LIST = ("secrets.deny_remove", "secrets.allow_secret_read")
"""A removal a repository *ships* is a credential decision made on a teammate's
behalf — D91 the argument about `[[approval.grants]]`, the same act with a different
spelling — so it is inert on a fresh clone until that clone's owner has read the digest.
Only the half that **shrinks** the list: `secrets.deny_add` can only tighten, and an inert
addition would make a clone looser than its project, the wrong direction for inert. A value
that shrinks nothing — `false`, `[]` — is left out of the subset, so a config that spells
the default out loud is not asked about. "Shrinks" is judged the way the config reads the
value (`_shrinks`), never by truthiness: a quoted `"false"` shrinks nothing either."""

EXECUTABLE_PATHS = (
    "commands",
    "hooks",
    "mcp.servers",
    "executor.env",
    "toolchain.bootstrap",
    *SHRINKS_THE_DENY_LIST,
)
"""Identified by name, never guessed. A subset inferred by looking for things that *seem*
executable would silently stop covering a field someone adds later.

`toolchain.bootstrap` is not a `config.toml` key: it is what a *project profile
override* declares, in the shape `profiles.loader.executable_overrides` returns, merged
into the mapping by the caller. A command that runs at worktree creation with the network
up is executable config whichever file it came from. The deny list's shrinking keys are
`SHRINKS_THE_DENY_LIST`'s."""


@dataclasses.dataclass(frozen=True)
class Executable:
    """The subset of a config that can cause something to run, and its digest."""

    subset: Mapping[str, Any]
    digest: str

    @property
    def is_empty(self) -> bool:
        return not self.subset


def _at(config: Mapping[str, Any], dotted: str) -> Any:
    cursor: object = config
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping):
            return None
        table = cast("Mapping[str, object]", cursor)
        if part not in table:
            return None
        cursor = table[part]
    return cursor


def executable_subset(
    config: Mapping[str, Any], root: pathlib.Path | None = None
) -> Executable:
    """The subset of `config` that can cause something to run, and its digest. With `root`,
    the repository, each `[[hooks]]` script's own bytes join it (`hook_scripts`): a `git pull`
    that changes a script changes the digest, so accepting a hook
    once never accepts its replacement, whether its line changed or its file did."""
    from chickadee.state.resolve import canonical

    subset = {
        p: v
        for p in EXECUTABLE_PATHS
        if (v := _at(config, p)) is not None
        and (p not in SHRINKS_THE_DENY_LIST or _shrinks(p, v))
    }
    hooks = subset.get("hooks")
    if root is not None and isinstance(hooks, list | tuple):
        scripts = hook_scripts(cast("Sequence[object]", hooks), root)
        if scripts:
            subset[HOOK_SCRIPTS] = scripts
    return Executable(
        subset=subset,
        digest=hashlib.sha256(canonical(subset).encode("utf-8")).hexdigest(),
    )


HOOK_SCRIPTS: Final = "hooks.scripts"
"""The subset's key for the bytes of the files `[[hooks]]` commands name."""

PROJECT_DIR: Final = "$CHICKADEE_PROJECT_DIR"
"""What a `[[hooks]]` command writes for the repository's root."""


def script_argv(command: Sequence[str], root: pathlib.Path) -> tuple[str, ...]:
    """A `[[hooks]]` command as it runs: `~` and `PROJECT_DIR` resolved, nothing else. The one
    reading, so the digest covers the files the script runner runs (`hooks/script.argv_of`)."""
    return tuple(
        str(pathlib.Path(a).expanduser())
        if a.startswith("~")
        else a.replace(PROJECT_DIR, str(root))
        for a in command
    )


def hook_scripts(hooks: Sequence[object], root: pathlib.Path) -> dict[str, str]:
    """Each repository file a `[[hooks]]` entry's `command` names, by its path, and its
    bytes' sha. A path outside the repository is the person's machine, not the repository's
    to ship, and is not read."""
    out: dict[str, str] = {}
    for entry in hooks:
        if not isinstance(entry, Mapping):
            continue
        command = cast("Mapping[str, object]", entry).get("command")
        if not isinstance(command, list | tuple):
            continue
        written = [a for a in cast("Sequence[object]", command) if isinstance(a, str)]
        for arg in script_argv(written, root):
            if (found := _repository_file(arg, root)) is not None:
                out[found[0]] = hashlib.sha256(found[1]).hexdigest()
    return out


def script_digest(entry: str, argv: Sequence[str], root: pathlib.Path) -> str:
    """One hook's digest: its entry, and the bytes of every repository file its argv names.
    What a bound script is checked against again before each run."""
    digest = hashlib.sha256(entry.encode("utf-8"))
    for arg in argv:
        found = _repository_file(arg, root)
        if found is not None:
            digest.update(f"\0{found[0]}\0".encode())
            digest.update(found[1])
    return digest.hexdigest()


def _repository_file(arg: str, root: pathlib.Path) -> tuple[str, bytes] | None:
    """`arg` as a file inside `root`, with its bytes, or None."""
    inside = within(root, arg)
    if inside is None or not inside.is_file():
        return None
    return inside.relative_to(root.resolve()).as_posix(), inside.read_bytes()


def _shrinks(path: str, value: object) -> bool:
    """Whether a deny-list key shrinks the list, read the way `config.SecretsPolicy` reads it.

    Truthiness is not that reading. `allow_secret_read = "false"` (a quoted string) is a
    non-empty string, so it is truthy, while the policy's bool coercion reads it as false.
    Judged by truthiness it joined the subset and moved the digest for a line that changes
    nothing, leaving every accepted hook on the clone inert until someone re-accepted it.
    Asking the policy itself means the two can never disagree. A value the policy cannot
    read counts as shrinking: that direction asks the owner rather than assuming.
    """
    from chickadee.config import SecretsPolicy

    key = path.removeprefix("secrets.")
    try:
        policy = SecretsPolicy.model_validate({key: value})
    except ValueError:
        return True
    if key == "allow_secret_read":
        return policy.allow_secret_read
    return bool(policy.deny_remove)


def trusted_digests(repo_local: pathlib.Path) -> set[str]:
    path = repo_local / TRUST_FILE
    if not path.is_file():
        return set()
    with path.open("rb") as fh:
        body = tomllib.load(fh)
    return {str(d) for d in body.get("accepted", [])}


def is_active(
    config: Mapping[str, Any], repo_local: pathlib.Path, root: pathlib.Path | None = None
) -> bool:
    """True when the executable subset may run.

    An empty subset is active by construction — there is nothing to accept. A change to any
    executable line produces a new digest and returns to inactive: **accepting a hook once
    never accepts its replacement.**
    """
    ex = executable_subset(config, root)
    if ex.is_empty:
        return True
    return ex.digest in trusted_digests(repo_local)


def acceptance_record(
    config: Mapping[str, Any], repo_local: pathlib.Path, root: pathlib.Path | None = None
) -> tuple[str, str]:
    """Return `(digest, file_body)` for accepting the current executable subset.

    **Pure: it computes, it does not write.** Acceptance is a person's act, performed
    through the CLI after the commands have been shown itemised — so the surface that
    asked the question is the one that records the answer. Keeping the write out of here
    also keeps this package free of disk writes entirely, which invariant 1's writer
    enumeration is there to make someone think about.

    The body is append-only: an older digest stays, so a person who accepts A, then B, then
    edits back to A is not asked again about something they already read.

    `root` is the repository's, as `is_active` is asked with it: the digest then covers the
    bytes of the scripts `[[hooks]]` commands name, and the two agree.
    """
    digest = executable_subset(config, root).digest
    return digest, acceptance_body(repo_local, digest)


def acceptance_body(repo_local: pathlib.Path, *digests: str) -> str:
    """`trusted.toml`'s body with `digests` accepted beside every digest already there: the
    one spelling of the file, for the config's executable subset and for the repository's
    `[[approval.grants]]` (`approval/grants.py`). Pure, as `acceptance_record` is."""
    have = trusted_digests(repo_local)
    have.update(digests)
    lines = [
        "# Executable config accepted for this clone.",
        "accepted = [",
        *[f'  "{d}",' for d in sorted(have)],
        "]",
    ]
    return "\n".join(lines) + "\n"
