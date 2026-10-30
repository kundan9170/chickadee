"""Toolchain probe, shared by `init` and the run-header degradation banner.

Core-side rather than CLI-side: knowing that A5 will skip is engine knowledge, and a
surface may import only the facade. `chickadee_cli/doctor.py` renders what this
returns.

Absence is **never** an error here. A missing linter means gate A4 will record an honest
skip at run time, and telling the user at `init` that A4 will always skip is
strictly better than letting them find out in a handoff — by which point they have paid
for a run whose verification was quietly thinner than the report implied.

A missing *grammar* is different and is not this module's call: without it the file cannot
be parsed at all, so there is nothing to skip, and `init` exits non-zero.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.errors import InfraError
from chickadee.graph.grammars import GRAMMARS
from chickadee.process import VERSION_TIMEOUT_S, run_harness
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.tools import find_tool

# Which profile fields name an executable, and the check each one feeds. The check is in the
# message because "node not found" is a fact and "the syntax check will skip" is the
# consequence the user needs. The gates A4 and A5 retired with the packet loop.
PROBES: tuple[tuple[str, str, str], ...] = (
    ("syntax", "cmd", "the syntax check"),
    ("tests", "runner", "the test runner"),
    ("build", "compile_cmd", "the build check"),
    ("build", "import_smoke", "the import check"),
)

# The servers kept beside the index: found where the index looks for them
# (`find_tool`, Chickadee's own environment included) and never run here, since a language
# server started with `--version` starts a server, or fails, rather than answering.
SERVERS: tuple[tuple[str, str, str], ...] = (
    ("lsp", "server", "each file's diagnostics and resolved edges"),
    ("lint", "server", "lint findings"),
)


def _binary(profile: object, section: str, field: str) -> str | None:
    """The executable a profile command runs, from its already-parsed `argv`.

    Read from `argv` rather than re-split from the source string: the profile loader has
    done the quoting once, and splitting the raw text again would disagree with it on any
    command containing a quoted argument.
    """
    spec = getattr(profile, section, None)
    template = getattr(spec, field, None) if spec is not None else None
    argv = getattr(template, "argv", None) if template is not None else None
    return str(argv[0]) if argv else None


def version_of(binary: str, root: Path | None = None) -> str | None:
    """The tool's own version string, or None when it is absent or will not answer. Found as
    the project's commands find it (`find_tool`): the repository's own `.venv` first,
    then `PATH`, `python` as `python3` where only that is installed; never Chickadee's own
    environment, which holds the harness's tools, not the project's. One
    found by an alternate name says so."""
    where = find_tool(binary, root)
    if where is None:
        return None
    alias = "" if where.name == binary else f" (as {where.name})"
    try:
        out = run_harness(
            (str(where), "--version"),
            cwd=root if root is not None else Path.cwd(),
            timeout_s=VERSION_TIMEOUT_S,
        )
    except (OSError, InfraError):
        return f"present (version unknown){alias}"
    if out.timed_out:
        return f"present (version unknown){alias}"
    line = (out.stdout or out.stderr).decode("utf-8", "replace").strip().splitlines()
    return (line[0][:80] if line else "present (version unknown)") + alias


def probe(
    profile_names: tuple[str, ...], profiles: ProfileSet, root: Path | None = None
) -> tuple[dict[str, dict[str, str | None]], tuple[str, ...]]:
    """Per profile: binary → version or None, plus one plain sentence per degradation. A
    server's entry is where it was found; `root`'s `.venv` is looked in first, as the index
    looks."""
    found: dict[str, dict[str, str | None]] = {}
    degraded: list[str] = []
    for name in profile_names:
        p = profiles.by_name(name)
        if p is None:
            continue
        found[name] = {}
        for section, field, consequence in PROBES:
            binary = _binary(p, section, field)
            if binary is None:
                continue
            version = version_of(binary, root)
            found[name][binary] = version
            if version is None:
                degraded.append(f"{name}: {binary} not found — {consequence} will skip")
        for section, field, consequence in SERVERS:
            binary = _binary(p, section, field)
            if binary is None:
                continue
            where = find_tool(binary, root, own=True)
            found[name][binary] = None if where is None else f"server at {where}"
            if where is None:
                degraded.append(
                    f"{name}: {binary} not found — {consequence} will be unavailable; "
                    "install it, or name another in the profile's override"
                )
    return found, tuple(degraded)


def missing_grammars(profile_names: tuple[str, ...], profiles: ProfileSet) -> tuple[str, ...]:
    """Profiles naming a tree-sitter grammar the code graph cannot parse with (`GRAMMARS`),
    or no profile at all.

    Fatal, unlike everything else here: with no grammar the file cannot be parsed, so it
    is not a weaker gate but no gate at all, and the index would silently omit the code.
    """
    missing: list[str] = []
    for name in profile_names:
        p = profiles.by_name(name)
        if p is None or any(g not in GRAMMARS for g in p.grammars()):
            missing.append(name)
    return tuple(missing)
