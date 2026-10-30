"""`sandbox-exec` (Seatbelt) profile generation.

The profile is generated per wrap from filesystem and network flags rather than kept as a
static file, because the writable set differs per call and a profile that covered every
case would be the union of them — which is no boundary at all.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from chickadee.errors import InfraError
from chickadee.isolation.base import first_line, has, probe
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.paths import AnyOne, AnyRun, CharClass, segment_tokens
from chickadee.process import Completed

BINARY = "sandbox-exec"


NULL_DEVICE_WRITE = '(allow file-write-data (literal "/dev/null"))'
"""The one `/dev` write a wrap allows."""


def profile_for(isolation: Isolation) -> str:
    """A Seatbelt profile: deny by default, then allow what this wrap permits."""
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec process-fork signal)",
        "(allow sysctl-read mach-lookup)",
        _file_read_allow(isolation),
    ]
    if isolation.repo_writable:
        lines.append(f"(allow file-write* (subpath {_quoted(isolation.cwd)}))")
        lines.extend(
            f"(allow file-write* (subpath {_quoted(extra)}))" for extra in isolation.writable
        )
        # After the allows: Seatbelt takes the last matching rule, so these win.
        lines.extend(
            f"(deny file-write* (subpath {_quoted(kept)}))" for kept in isolation.read_only
        )
    # A scratch tmpdir stays writable even for read-only wraps: compilers and linters
    # write temporaries, and denying that turns a read-only *intent* into a broken toolchain.
    lines.append('(allow file-write* (subpath "/private/tmp") (subpath "/tmp"))')
    # The repository's tool cache and the session's temporary
    # directory, the harness's own scratch, writable whatever the wrap.
    lines.extend(
        f"(allow file-write* (subpath {_quoted(extra)}))"
        for extra in (*isolation.scratch, *((isolation.tmp,) if isolation.tmp else ()))
    )
    # git opens /dev/null read-write, so a sandboxed `git status` (gate A3's baseline) failed
    # with EPERM and every patch escalated on a Mac (the twin of G3 F1 on Linux). The
    # null device and nothing else under /dev: data written there goes nowhere.
    lines.append(NULL_DEVICE_WRITE)
    if isolation.network_off:
        lines.append("(deny network*)")
    else:
        lines.append("(allow network*)")
    return "\n".join(lines)


PATTERN_ROWS: tuple[str, ...] = (
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "service-account*.json",
    ".env",
    ".env.*",
    ".npmrc",
    ".netrc",
    ".pypirc",
)
"""The rows for a caller that resolved no list (`Isolation.deny_patterns is None`): the
bare-name half of the deny list's built-ins (`secrets/denylist.py`), the entries a name
can match anywhere on disk. A literal here rather than an import, because Isolation does
not know the deny list (types.py); `test_sandbox_macos.py` pins the two equal.

A caller that resolved a list hands over that list's own half instead, so a removed
built-in has no row and a project's addition has one. These were the only rows
before, and on macOS a removed built-in stayed masked for spawned processes while harness
reads, the pre-filter and bubblewrap honoured the removal."""

_NEGATED_SUFFIXES = (".example", ".template")


def deny_regexes(isolation: Isolation) -> tuple[str, ...]:
    """Basename regexes for the name-pattern rows: one per pattern in `deny_patterns`.

    `None` is a caller that resolved no list, and keeps the old reading: the built-in rows
    while `deny_read` is non-empty, because `deny_read=()` meant `allow_secret_read`. That
    reading is only safe without a list: a removal can empty `deny_read` while
    `*.pem` and the rest are still in force.
    """
    patterns = isolation.deny_patterns
    if patterns is None:
        patterns = PATTERN_ROWS if isolation.deny_read else ()
    return tuple(rx for glob in patterns if (rx := _basename_regex(glob)) is not None)


def _basename_regex(glob: str) -> str | None:
    """`id_rsa*` becomes `/id_rsa[^/]*$`: anchored at a separator on the left and the end
    of the path on the right, so it is the basename that is matched. None for a glob that
    no name can match, which needs no row.

    The glob comes from config and is written into the profile, so it is held to
    `_quoted`'s rule. It must also be a bare name: a pattern with a `/` is about where a
    file sits, which a basename regex cannot say, and it reaches the wrap as resolved
    paths in `deny_read` instead.
    """
    if _UNQUOTABLE.search(glob):
        raise InfraError(
            f"refusing to build a Seatbelt profile around the pattern {glob!r}: a quote, "
            "backslash or control character would end the profile's string early and be "
            "read as profile"
        )
    if "/" in glob:
        raise InfraError(
            f"refusing to build a Seatbelt profile around the pattern {glob!r}: a pattern "
            "with a `/` names a directory, which reaches the wrap as resolved paths"
        )
    body = _glob_regex(glob)
    return None if body is None else f"/{body}$"


_ANY_RUN = "[^/]*"
_ANY_ONE = "[^/]"


def _glob_regex(glob: str) -> str | None:
    """A regex for exactly the names the deny list matches with `glob`, or None if none can:
    the glob as `paths.segment_tokens` reads it, the one reading, written in
    the profile's dialect.

    `*` and `?` stop at a separator, so they match within the basename. A class is written
    back member by member (`_bracket`). Everything else is escaped. Before, only `*` was
    translated, so `secret?.txt` and `*.[pk]em` rendered as literals and masked nothing.

    **Case is ignored, as the list ignores it**. The list denies a
    name when its case-folded spelling matches the case-folded glob, and Seatbelt's regex has
    no flag for that, so the row says it per character: a letter is both its cases (`[eE]`),
    and a class takes a character exactly when its lowercase does. Exact for ASCII; a letter
    beyond it is its one-character upper and lower forms, which misses a fold that changes
    length (`ß` folds to `ss`).
    """
    out: list[str] = []
    for token in segment_tokens(glob.casefold()):
        if isinstance(token, AnyRun):
            out.append(_ANY_RUN)
        elif isinstance(token, AnyOne):
            out.append(_ANY_ONE)
        elif isinstance(token, CharClass):
            spans = _folded(token.spans)
            if not spans and not token.negated:
                return None  # a class with no member matches no character
            out.append(_bracket(spans, negated=token.negated))
        else:
            cases = _cases(token.char)
            out.append(
                _bracket([(c, c) for c in cases], negated=False)
                if len(cases) > 1
                else re.escape(token.char)
            )
    return "".join(out)


def _cases(char: str) -> list[str]:
    """`char` (already folded) and every one-character spelling that folds to it."""
    found = {char} | {
        c for c in (char.upper(), char.lower()) if len(c) == 1 and c.casefold() == char
    }
    return sorted(found)


def _folded(spans: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    """The characters whose lowercase is in `spans`: the class as the list reads a folded
    name. An uppercase ASCII letter is in only when its lowercase is, and every other
    character is in when it is itself; a lowercase letter beyond ASCII brings its one-character
    uppercase with it."""
    upper_a, upper_z = ord("A"), ord("Z")
    out: list[tuple[str, str]] = []
    for low, high in spans:
        lo, hi = ord(low), ord(high)
        # Itself, less the uppercase ASCII letters: their folded form is what the list reads.
        if lo < upper_a:
            out.append((low, chr(min(hi, upper_a - 1))))
        if hi > upper_z:
            out.append((chr(max(lo, upper_z + 1)), high))
        # The uppercase of every lowercase ASCII letter in the span.
        a, z = max(lo, ord("a")), min(hi, ord("z"))
        if a <= z:
            out.append((chr(a - 32), chr(z - 32)))
        # Beyond ASCII, a one-character uppercase that folds back.
        if lo == hi and lo > 127:
            out.extend((c, c) for c in _cases(low) if c != low)
    return sorted(set(out))


def _bracket(members: list[tuple[str, str]], *, negated: bool) -> str:
    """A bracket expression holding `members`, with each character where it means itself.

    Four characters are read by position: `]` closes the class unless it comes first, `^`
    negates it when it comes first, `-` makes a range unless it comes first or last, and
    `[` opens `[:`, `[.` or `[=`. So they are split off the ends of any range and written in
    fixed places: `]` first, `-` first or last, and `[` then `^` after every other member,
    where only `^`, `-` or the close can follow the `[`. That keeps backslashes out of the
    class, where regex dialects disagree about them.

    A class never matches `/`: the row is matched against a whole path, where `a[--z]b` would
    match the file `b` in a directory `a`. `segment_tokens` already takes `/` out of every
    span, and a negated class leaves it out here, as `?` does.
    """
    special: set[str] = set()
    plain: list[str] = []
    for lo, hi in members:
        while lo <= hi and lo in "]^-":
            special.add(lo)
            lo = chr(ord(lo) + 1)
        while lo <= hi and hi in "-[":
            special.add(hi)
            hi = chr(ord(hi) - 1)
        if lo < hi:
            plain.append(f"{lo}-{hi}")
        elif lo == hi:
            plain.append(lo)
    if not negated and len(special) + len(plain) == 1:
        only = next(iter(special)) if special else plain[0]
        if len(only) == 1:
            return re.escape(only)  # one character is itself, and `[^]` would negate
    head = "]" if "]" in special else ""
    dash = "-" if "-" in special else ""
    return (
        "["
        + ("^" if negated else "")
        + head
        + ("" if head else dash)
        + "".join(plain)
        + ("/" if negated else "")
        + ("[" if "[" in special else "")
        + ("^" if "^" in special else "")
        + (dash if head else "")
        + "]"
    )


def _file_read_allow(isolation: Isolation) -> str:
    """Allow reads except the concrete and pattern-shaped masks.

    Seatbelt allow rules are additive: a broad `(allow file-read*)` cannot be narrowed by
    later deny rows. The exclusions therefore belong inside the broad allow as boolean
    filters. `require-not` is a native Seatbelt filter and compiles to its DFA; unlike a
    regex lookahead, it is supported by the profile language itself.
    """
    excluded: list[str] = []
    for denied in isolation.deny_read:
        kind = "subpath" if denied.is_dir() else "literal"
        excluded.append(f"({kind} {_quoted(denied)})")
    certificate = {
        rx for glob in isolation.spared_patterns if (rx := _basename_regex(glob)) is not None
    }
    excluded.extend(
        _pattern_filter(rx, isolation if rx in certificate else None)
        for rx in deny_regexes(isolation)
    )
    if not excluded:
        return "(allow file-read*)"
    filters = " ".join(f"(require-not {item})" for item in excluded)
    return f"(allow file-read* (require-all {filters}))"


_UNQUOTABLE = re.compile(r'["\\\x00-\x1f\x7f]')


def _quoted(path: Path) -> str:
    """`path` as a profile string literal, or a refusal to build the profile.

    A path is written between double quotes, so a `"` in it ends the string early and the
    rest is read as profile. This is a legal file name, and it matches `id_rsa*`:

        id_rsa")))) (allow file-read*) (allow file-read* (require-all (require-not (literal "x

    Resolved into `deny_read`, it closed the read rule early, added an unconditional
    `(allow file-read*)` as a rule of its own, and opened one more rule to absorb the rest,
    so the profile stayed well-formed and no mask held. Refused rather than escaped: an
    escape would be written blind, because this suite cannot run Seatbelt off macOS, and a
    wrong escape here is a sandbox escape. Backslash and control characters are refused for
    the same reason. A refused wrap runs nothing, which is the fail-closed direction.
    """
    text = str(path)
    if _UNQUOTABLE.search(text):
        raise InfraError(
            f"refusing to build a Seatbelt profile around {text!r}: a quote, backslash or "
            "control character in a path would end the profile's string early and be read "
            "as profile; rename the path"
        )
    return f'"{text}"'


def _pattern_filter(regex: str, spared: Isolation | None = None) -> str:
    """The positive deny pattern minus the deny list's template negations; for a certificate
    pattern, minus the trust stores `spared` names too: its directories, its
    paths and the CA bundles. Filesystem-wide otherwise, as it always was."""
    exceptions = [
        f'(require-not (regex #"{re.escape(suffix)}$"))' for suffix in _NEGATED_SUFFIXES
    ]
    if spared is not None:
        exceptions += [f'(require-not (regex #"/{_segments(d)}/"))' for d in spared.spared_dirs]
        exceptions += [f"(require-not (subpath {_quoted(p)}))" for p in spared.spared_paths]
    return f'(require-all (regex #"{regex}") {" ".join(exceptions)})'


def _segments(tail: str) -> str:
    """A directory tail (`site-packages/certifi`) as profile regex: each segment's characters
    escaped as a glob's literals are (`_glob_regex`), so a `.` or `-` means itself."""
    return "/".join(_glob_regex(segment) or "" for segment in tail.split("/"))


class Seatbelt:
    def available(self) -> bool:
        """It runs, not that it is on the path."""
        return probe(self).ran

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        return (BINARY, "-p", profile_for(isolation), *isolation.argv)

    # -- what the probe asks -------------------------------------------------------

    def probe_key(self) -> tuple[str, ...]:
        return (shutil.which(BINARY) or "",)

    def not_ready(self) -> tuple[str, str] | None:
        if has(BINARY):
            return None
        return (
            f"{BINARY} is not on the PATH",
            f"{BINARY} ships with macOS at /usr/bin/{BINARY}; put /usr/bin on the PATH",
        )

    def diagnose(self, done: Completed) -> tuple[str, str]:
        return (
            f"{BINARY} exited {done.exit_code}: {first_line(done)}",
            f"run `{BINARY} -p '(version 1)(allow default)' true` in a terminal to see the "
            "error; a sandbox inside another sandbox (an IDE's, a CI runner's) is refused",
        )
