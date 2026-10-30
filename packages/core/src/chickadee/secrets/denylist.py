"""Layer 1 — never see.

A path on this list is unreadable by **every** rung and surface: scout, packer, planner,
reader, graph. Prevention rather than detection, which is why it is layer 1 and redaction is
only a backstop — a value the model never reads is a value that cannot leak from a prompt.

A denial is a typed refusal carrying the reason, never silence and never empty content. The
distinction matters: a scout told "denied: secret-bearing path" learns the file exists and
stops looking, whereas one handed an empty string concludes the file is empty and may go
looking for the value somewhere less guarded.
"""

from __future__ import annotations

import functools
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from chickadee.paths import GlobSet, glob_match, name_under, spelled_out, within
from chickadee.secrets.walk_memo import RACY_NS, Seen, WalkMemo, memo_key

DEFAULT_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    ".ssh/**",
    ".aws/**",
    ".gnupg/**",
    ".npmrc",
    ".netrc",
    ".pypirc",
    ".docker/config.json",
    "service-account*.json",
)
"""The built-ins. Every entry names a file *format* (`*.pem`, `*.p12`) or a
well-known credential *path* (`.env`, `.ssh/**`, `.docker/config.json`), and is right by
construction. `*secret*` and `*credential*` were on this list and are not: they were
substring guesses about intent, they denied this repository its own `22-secrets.md` and
five other files, and SE-11 had already rejected deciding secrecy by looking for the word —
for content; D118 applies the same argument to names. A project that wants them writes them
into its own `[secrets].deny_add` and owns the false positives they bring."""

DEFAULT_NEGATIONS: tuple[str, ...] = (
    ".env.example",
    ".env.template",
    "*.example",
    "*.template",
)
"""Templates carry names, not values. Excluding them is what keeps the list usable — a
denylist that hides `.env.example` teaches people to turn it off."""

TRUST_STORE_DIRS: tuple[str, ...] = (
    "site-packages/certifi",
    "dist-packages/certifi",
    "_vendor/certifi",
    "rubygems/ssl_certs",
)
"""The installed packages that ship public CA certificates, as the directories
they install to: Python's `certifi` and pip's vendored copy, and RubyGems' `ssl_certs`.
`certifi/cacert.pem` is what pip, requests and uv check TLS against, and masking it failed
every install on 2026-10-18 before a byte was downloaded. Named by where a package manager
puts them, never by a bare name: a project's own `ca-certificates/` or `certs/` can hold the
keys `cfssl` or `openssl` wrote beside a CA, and stays masked."""

TRUST_STORE_PATHS: tuple[str, ...] = (
    "/etc/ssl/certs",
    "/etc/ssl/cert.pem",
    "/etc/pki/ca-trust",
    "/etc/pki/tls/certs",
    "/etc/pki/tls/cert.pem",
    "/etc/ca-certificates",
    "/usr/lib/ssl/certs",
    "/usr/share/ca-certificates",
    "/usr/local/share/ca-certificates",
    "/private/etc/ssl/certs",
    "/private/etc/ssl/cert.pem",
    "/System/Library/OpenSSL/certs",
)
"""The system's trust stores, by absolute path: the certificate directories and
bundles, never a `private/` directory beside them, where a server's keys live
(`/etc/ssl/private`, `/etc/pki/tls/private`). Debian keeps its CAs as `/etc/ssl/certs/*.pem`."""

CERT_PATTERNS: tuple[str, ...] = ("*.pem", "*.crt", "*.cer")
"""The bare-name patterns a trust store is spared from: a certificate's formats.
Every other entry (`.env`, `id_rsa*`, `*.key`) still denies a file wherever it sits, a trust
store's directory included."""

CA_FILE_ENV: tuple[str, ...] = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PIP_CERT",
    "GIT_SSL_CAINFO",
    "NODE_EXTRA_CA_CERTS",
)
"""The variables that name a CA bundle: the file each names is a trust store,
whatever it is called and wherever it sits, so a bare-name pattern never denies it."""

_UNWRITABLE = re.compile(r'["\\\x00-\x1f\x7f]')
"""What a pattern may not contain: Seatbelt writes each pattern into its profile between
quotes (`isolation/macos.py`, which refuses the same set when it renders)."""


@dataclass(frozen=True)
class Denial:
    path: str
    pattern: str

    @property
    def reason(self) -> str:
        return f"denied: secret-bearing path ({self.path}) matched {self.pattern!r}"


@dataclass(frozen=True)
class TrustStores:
    """What a certificate pattern never masks: files below `dirs` (as directory
    paths, `site-packages/certifi`), below or at `paths` (the system's stores and the CA bundles
    the environment names), for the `patterns` the list holds of `CERT_PATTERNS`."""

    dirs: tuple[str, ...] = ()
    paths: tuple[Path, ...] = ()
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Denylist:
    patterns: tuple[str, ...] = DEFAULT_PATTERNS
    negations: tuple[str, ...] = DEFAULT_NEGATIONS
    allow_secret_read: bool = False
    """SE-8's all-or-nothing switch, off by default: Layer 1 off entirely, in the harness's
    reads and in the sandbox (`resolve` → `()`). Loud in the run header and the handoff.
    Since D118 it is no longer the only way to correct one wrong entry — `removed` is — so
    the loud switch is kept for the loud decision."""
    removed: tuple[str, ...] = ()
    """The built-in patterns this list removes, **by name**. Per entry:
    removing `.env` leaves `.ssh/**`, `*.pem` and the rest in force. The run header names
    these — a list that can shrink says that it has — and a removal a
    repository *ships* is executable config, inert on a clone until its owner accepts the
    trust digest (`state/trust.py`)."""

    def __post_init__(self) -> None:
        """A removal must **name** the entry it removes, enforced rather than described.

        Three ways to shrink the list, each with its own rule. **Dropping** a built-in from
        `patterns` is refused unless `removed` names it — `Denylist(patterns=())` made `.env`
        readable with nothing saying so, which is a config silently turning the protection
        off. **Naming** something that is not a built-in is refused whatever the flag says:
        a typo in a removal must not read as a removal that worked. **Adding a negation** is
        refused without `allow_secret_read`, as before. `allow_secret_read` still waives the
        first rule, because it turns the list off whole.

        Two more refuse a list that reads as protection it does not give: an addition that
        can never deny anything, and a list left with no patterns while the flag is off.
        The flag is the one switch that turns everything off, and it is the loud one.
        """
        not_builtin = [r for r in self.removed if r not in DEFAULT_PATTERNS]
        if not_builtin:
            what = (
                "is not a built-in deny pattern"
                if len(not_builtin) == 1
                else "are not built-in deny patterns"
            )
            raise ValueError(
                f"removed {not_builtin}, which {what}; "
                f"the built-ins are {list(DEFAULT_PATTERNS)} — "
                "a removal names the entry it removes, and a typo must not read as one"
            )
        still_listed = [r for r in self.removed if r in self.patterns]
        if still_listed:
            raise ValueError(
                f"{still_listed} {'is' if len(still_listed) == 1 else 'are'} both removed "
                "and listed; a pattern is on the list or off it"
            )
        self._refuse_additions_that_protect_nothing()
        self._refuse_additions_a_profile_cannot_hold()
        if self.allow_secret_read:
            return
        unnamed = sorted(set(DEFAULT_PATTERNS) - set(self.patterns) - set(self.removed))
        if unnamed:
            raise ValueError(
                f"denylist drops built-in patterns {unnamed} without naming them in "
                "`removed` — a removal is per entry and names what it removes (D118); "
                "turning the list off whole is allow_secret_read=True"
            )
        if not self.patterns:
            # Every built-in removed and nothing added reads the same as the flag — everything
            # readable — but says nothing. SE-8 makes turning Layer 1 off loud (run header,
            # handoff), and anything keyed on the flag would report the list as on.
            raise ValueError(
                "the deny list resolves to no patterns, which turns Layer 1 off without the "
                "switch that says so; turning it off whole is allow_secret_read=True (SE-8), "
                "and a removal must leave at least one pattern in force"
            )
        custom = [n for n in self.negations if n not in DEFAULT_NEGATIONS]
        if custom:
            # Any custom negation, not just one that *looks* like it cancels a built-in.
            # The first attempt compared pattern **text** — asking whether the negation
            # string matched a pattern — which decides nothing: `".*"` shares no text with
            # `".env"` and makes it readable anyway. Glob intersection is not answerable by
            # comparison, so the honest rule is that adding a negation is itself the
            # deliberate act the flag exists to gate. Extending `patterns` stays free,
            # because that can only tighten.
            raise ValueError(
                f"custom negations {sorted(custom)} can only widen what is readable and "
                "need allow_secret_read=True; extending `patterns` needs no flag"
            )

    def _refuse_additions_that_protect_nothing(self) -> None:
        """An addition that can never deny anything reads as protection that is not there.

        Three shapes, all refused whatever the flag says, because each is a mistake in what
        the person wrote rather than a choice about the list:

        - **blank**: matches nothing;
        - **padded** (`" *.kdbx"`): patterns match names exactly, so the space is part of
          the name and nothing on disk has it;
        - **always exempt** (`.env.example`, `*.template`): `match` checks the negations
          first, so a name a negation allows is readable whatever else the list says, and
          config has no way to drop a negation. The test is whether a negation matches the
          addition's own text, which holds exactly when the addition ends in an exempt
          literal. That can miss an exotic spelling, but it never refuses a pattern that
          could deny something.
        """
        added = [p for p in self.patterns if p not in DEFAULT_PATTERNS]
        if any(not p.strip() for p in added):
            raise ValueError(
                "deny_add entries must be non-empty glob patterns: a blank pattern matches "
                "nothing and would read as protection that is not there"
            )
        padded = [p for p in added if p != p.strip()]
        if padded:
            raise ValueError(
                f"deny_add entries {padded} have leading or trailing whitespace; a pattern "
                "matches names exactly, so as written it would protect nothing (did you "
                f"mean {[p.strip() for p in padded]}?)"
            )
        # Folded, as `match` folds both sides: `CONFIG.EXAMPLE` is as exempt as
        # `config.example`, so it protects nothing either.
        exempt = [
            p
            for p in added
            if any(
                _matches(p.casefold(), PurePosixPath(p.casefold()).name, n.casefold())
                for n in self.negations
            )
        ]
        if exempt:
            raise ValueError(
                f"deny_add entries {exempt} only match names the exceptions "
                f"{list(self.negations)} always allow (templates carry names, not values), "
                "so they would protect nothing; keep real values out of template files"
            )

    def _refuse_additions_a_profile_cannot_hold(self) -> None:
        """An addition with a quote, backslash or control character in it is refused.

        Seatbelt writes each pattern into the sandbox profile between quotes, and a `"` there
        ends the string early, so the rest of the pattern is read as profile. D125 applies
        `deny_add` without the trust digest, so a repository's addition reaches every clone's
        profile unasked. Refused here so the config error names the entry: whatever the flag
        says, and on every platform, so a config means the same thing wherever it loads. The
        renderer refuses the same set, for a list built outside config.
        """
        added = [p for p in self.patterns if p not in DEFAULT_PATTERNS]
        unwritable = [p for p in added if _UNWRITABLE.search(p)]
        if unwritable:
            raise ValueError(
                f"deny_add entries {unwritable} contain a quote, backslash or control "
                "character; the macOS sandbox writes each pattern between quotes, where one "
                "of these would end the string early and be read as profile"
            )

    @staticmethod
    def configure(
        add: Iterable[str] = (),
        remove: Iterable[str] = (),
        *,
        allow_secret_read: bool = False,
    ) -> Denylist:
        """The list a config resolves to (`config.SecretsPolicy`): the built-ins, less
        the ones `remove` names, plus `add`. Every refusal is `__post_init__`'s: an unknown
        removal, a pattern both added and removed, an addition that protects nothing, and a
        list left with no patterns. So there is one rule and this is a call to it. A
        duplicated entry collapses to one, because listing a pattern twice changes nothing."""
        added = tuple(add)
        removed = tuple(dict.fromkeys(remove))
        kept = tuple(p for p in DEFAULT_PATTERNS if p not in removed)
        extra = tuple(a for a in dict.fromkeys(added) if a not in kept)
        return Denylist(
            patterns=kept + extra,
            removed=removed,
            allow_secret_read=allow_secret_read,
        )

    def match(self, path: str, *, is_dir: bool = False) -> Denial | None:
        """The pattern that denies this path, or None.

        Negations win over patterns. Every pattern is read as `.gitignore` reads it, by
        `paths.glob_match`. A pattern that names a **directory** (`.ssh/**`,
        `.docker/config.json`) is matched against every tail of the path, so it is that entry
        wherever it sits (`sub/.docker/config.json`), and `dir/**` denies `dir` itself;
        one that names a **file** (`.env`, `*.pem`, `id_rsa*`) is matched against the bare
        name, so `secrets/prod.pem` and `prod.pem` are still the same risk without the
        directory `secrets/` becoming one. The same verdict as `name_denied`, the read rule,
        so a write is refused wherever a read is.

        `is_dir` says the path is a directory, and then only a directory-shaped pattern can
        deny it. The list spells a secret directory as `dir/**`; without this, a bare-name
        pattern like `*secret*`
        swallowed every ordinary source tree named `secrets/` — masking it whole under D91's
        resolver and refusing it in the executor's pre-filter.

        **Case is ignored, on both sides**. A case-insensitive filesystem (macOS by
        default, Windows) opens `.env`'s bytes for `.ENV`, so a case-sensitive match let any
        other spelling of a secret through. On a case-sensitive filesystem the other spelling
        is another file, and refusing it fails closed. Negations ignore case too, so a
        template stays readable however it is spelled. The `Denial` names the pattern as
        listed and the tail of the path as asked that it matched: the whole path for a bare
        name, `.ssh/config` for `.ssh/**`, and never the directories above it.
        """
        return name_denied(path, self, is_dir=is_dir)

    def readable(self, path: str, *, is_dir: bool = False) -> bool:
        return self.match(path, is_dir=is_dir) is None

    def trust_stores(self) -> TrustStores:
        """What a certificate pattern spares, for a sandbox that matches names
        itself (Seatbelt): the one statement of it, so the sandbox and `name_denied` agree.
        Empty under `allow_secret_read`, where nothing is masked."""
        if self.allow_secret_read:
            return TrustStores()
        return TrustStores(
            dirs=TRUST_STORE_DIRS,
            paths=(*(Path(p) for p in TRUST_STORE_PATHS), *_ca_files()),
            patterns=tuple(p for p in CERT_PATTERNS if p in self.patterns),
        )

    def name_patterns(self) -> tuple[str, ...]:
        """The bare-name half of the list, in order: the patterns with no `/`, each of which
        denies a file by its name wherever it sits (`*.pem`, `id_rsa*`, a project's `*.kdbx`).

        A sandbox that can match names anywhere on disk takes these (`Isolation.deny_patterns`,
        Seatbelt's rows), so a key written mid-session, or kept outside the roots `resolve`
        walks, is still masked. The directory-shaped half reaches the sandbox as the paths
        `resolve` finds. `()` under `allow_secret_read`, the one switch that turns both off.
        """
        if self.allow_secret_read:
            return ()
        return tuple(p for p in self.patterns if not _names_a_directory(p))


def _names_a_directory(pattern: str) -> bool:
    """A pattern with a separator is about where the file sits; one without is about what
    it is called. The deny list already writes its secret directories as `dir/**`."""
    return "/" in pattern


def _ca_files() -> tuple[Path, ...]:
    found: list[Path] = []
    for name in CA_FILE_ENV:
        value = os.environ.get(name)
        if value:
            path = Path(os.path.realpath(value))
            if path not in found:
                found.append(path)
    return tuple(found)


def _trust_store(path: str) -> bool:
    """`path` lies in a trust store: below a `TRUST_STORE_DIRS` directory or a
    `TRUST_STORE_PATHS` entry, or it is a CA bundle the environment names."""
    pure = PurePosixPath(path)
    dirs = "/" + "/".join(p.casefold() for p in pure.parts[:-1] if p != "/") + "/"
    if any(f"/{d}/" in dirs for d in TRUST_STORE_DIRS):
        return True
    if not pure.is_absolute():
        return False
    if any(within(Path(root), path) is not None for root in TRUST_STORE_PATHS):
        return True
    return Path(os.path.realpath(path)) in _ca_files()


def _spared(denial: Denial, path: str) -> bool:
    """A certificate pattern's denial of a file in a trust store: never a key's
    pattern, and never a directory-shaped entry, which still denies there."""
    return denial.pattern in CERT_PATTERNS and _trust_store(path)


def _first_denial(
    denylist: Denylist, tails: Sequence[str], *, is_dir: bool = False
) -> Denial | None:
    """The list's verdict on `tails` (every tail of one path, longest first): `_exact_denial`,
    asked only when the path could be denied at all.

    Almost no path is, and `resolve` asks of every entry under the workspace and `$HOME`:
    tens of thousands, each matched per entry and, for a directory-shaped entry, per tail.
    That made a session's first turn take seconds in a home with a package cache. The screen
    is one expression per kind (`_screens`): no bare-name entry matches the name, and no
    directory-shaped entry matches any tail, so no entry can deny it, whatever the negations
    say, since a negation only clears. Anything the screen passes gets the exact answer, so
    the verdict, and the tail and entry a denial names, are unchanged."""
    if denylist.allow_secret_read or not tails or not _could_deny(denylist, tails[0], is_dir):
        return None
    return _exact_denial(denylist, tails, is_dir=is_dir)


def _could_deny(denylist: Denylist, path: str, is_dir: bool) -> bool:
    """The screen, on a path as spelled: False when no entry of the list can deny it. A
    spelling and the tails built from it normalise to one path (`GlobSet` drops empty and
    `.` segments as `PurePosixPath` does), so asking before the tails are built is the same
    question, without building them for the many paths it clears."""
    folded = path.casefold()
    segments = [seg for seg in folded.split("/") if seg and seg != "."]
    if not segments:
        return False
    bare, shaped = _screens(denylist.patterns)
    return (not is_dir and bare.matches(segments[-1])) or shaped.matches(folded)


@functools.cache
def _screens(patterns: tuple[str, ...]) -> tuple[GlobSet, GlobSet]:
    """The bare-name entries as one set over a name, and the directory-shaped ones over any
    tail of a path, each folded as `_matches` folds it. A directory-shaped entry that names a
    directory whole (`dir/**`) denies `dir` itself too (`_matches`), so both spellings join."""
    bare = [p.casefold() for p in patterns if not _names_a_directory(p)]
    shaped: list[str] = []
    for p in patterns:
        if _names_a_directory(p):
            folded = p.casefold()
            shaped.append(folded)
            whole = spelled_out(folded)
            if whole.endswith("/**"):
                shaped.append(whole[: -len("/**")])
    return GlobSet(bare), GlobSet(shaped, any_tail=True)


def _exact_denial(
    denylist: Denylist,
    tails: Sequence[str],
    *,
    is_dir: bool = False,
    skip: tuple[str, ...] = (),
) -> Denial | None:
    """`match` asked of each of `tails` in turn, the first denial: the one implementation of
    the list's verdict, for one path (`match`) and for every tail of one (`name_denied`).
    `_first_denial` asks it only of a path its screen cannot clear.

    Every tail of a path has the same last name, so a bare-name entry (`*.pem`, `.env`) gives
    every tail the same answer and is matched once; only the directory-shaped entries are
    matched per tail. Asking `match` once per tail matched every entry on every tail, which
    made `resolve`'s walk of a deep tree several times slower (review of #370). The verdict
    is the same: a negation that matches a tail clears that tail, and the first entry in
    list order that matches it denies it.

    `skip`: entries left out, as listed. A trust store's file is asked again without the
    certificate patterns (`CERT_PATTERNS`), so every other entry still decides it.
    """
    if denylist.allow_secret_read or not tails:
        return None
    name = PurePosixPath(tails[0].casefold()).name
    negations, patterns = _prepared(denylist.negations, denylist.patterns)
    if any(not shaped and _matches("", name, n, is_dir=is_dir) for n, shaped in negations):
        return None
    by_name = {
        p
        for p, folded, shaped in patterns
        if not shaped and p not in skip and _matches("", name, folded, is_dir=is_dir)
    }
    for tail in tails:
        folded_tail = tail.casefold()
        if any(
            shaped and _matches(folded_tail, name, n, is_dir=is_dir) for n, shaped in negations
        ):
            continue
        for pattern, folded, shaped in patterns:
            if pattern in skip:
                continue
            if shaped:
                hit = _matches(folded_tail, name, folded, is_dir=is_dir)
            else:
                hit = pattern in by_name
            if hit:
                return Denial(tail, pattern)
    return None


@functools.cache
def _prepared(
    negations: tuple[str, ...], patterns: tuple[str, ...]
) -> tuple[tuple[tuple[str, bool], ...], tuple[tuple[str, str, bool], ...]]:
    """A list's entries folded and classified once, not on every path a walk asks about:
    `(folded, directory-shaped)` per negation, `(as listed, folded, directory-shaped)` per
    pattern. Keyed by the entries, so two equal lists share it."""
    return (
        tuple((n.casefold(), _names_a_directory(n)) for n in negations),
        tuple((p, p.casefold(), _names_a_directory(p)) for p in patterns),
    )


def _matches(path: str, name: str, pattern: str, *, is_dir: bool = False) -> bool:
    """One entry against one tail of a path, both already case-folded, through the one
    path-glob matcher (`paths.glob_match`)."""
    if not _names_a_directory(pattern):
        # A bare-name pattern denies a FILE. A project's `*secret*` is `my-secrets.yaml`,
        # never the directory it happens to keep its secret *handling code* in.
        return not is_dir and glob_match(name, (pattern,)) is not None
    if glob_match(path, (pattern,)) is not None:
        return True
    # `dir/**`, and `dir/` which says the same, deny `dir` itself as well as
    # what is in it: the mask covers the directory whole (`resolve`), so a file written into
    # it later is masked too. A file named `dir` is denied as well, whatever `is_dir` says:
    # not every caller knows, and the list fails closed.
    whole = spelled_out(pattern)
    return whole.endswith("/**") and glob_match(path, (whole[: -len("/**")],)) is not None


def write_denied(path: str, denylist: Denylist | None = None) -> Denial | None:
    """SE-3 — is this write refused *because the path is secret-bearing*?

    Lives here rather than being restated inside the applier's gate, so there is **one**
    rule and the gate is a call to it. A rule expressed twice disagrees with itself the
    first time either copy is edited.

    **It deliberately does not ask whether the path is a declared target.** §2.1 phrases the
    exception as "allowed iff `target.path` is itself that file *and* the path matches an
    `.example`/`.template` negation", and the first half of that is gate 2's question, which
    the applier already answers. An earlier version took a `declared_target` argument and
    checked it here — the branch was unreachable, because a negated path returns before
    reaching it, and duplicating the scope gate would have been the same mistake this
    function exists to avoid.

    Composed, the two gates give the spec's "iff": SE-3 refuses `.env`, and gate 2 refuses
    an undeclared `.env.example`.
    """
    return (denylist or Denylist()).match(path)


def name_denied(path: str, denylist: Denylist, *, is_dir: bool = False) -> Denial | None:
    """The rule for a path as it is spelled. Every tail of it is matched, so
    a nested or an absolute spelling of a directory-shaped entry (`.docker/config.json`) is
    that entry wherever it sits: `/home/u/.docker/config.json` and `sub/.docker/config.json`
    are the file the entry names, as `resolve` spells it relative to a root. For a name a
    caller judges and will not open. None under `allow_secret_read`.

    A name holding a NUL byte is also judged as spelled up to it: no file is named with one,
    and anything that reads the name as a C string opens that prefix (`.env\x00x` is `.env`
    there), so it is refused as the prefix would be."""
    spellings = (path, path.split("\x00", 1)[0]) if "\x00" in path else (path,)
    for spelling in spellings:
        if denylist.allow_secret_read or not _could_deny(denylist, spelling, is_dir):
            continue
        parts = [p for p in PurePosixPath(spelling).parts if p != "/"]
        tails = ["/".join(parts[i:]) for i in range(len(parts))]
        denial = _first_denial(denylist, tails, is_dir=is_dir)
        if denial is not None and _spared(denial, spelling):
            # A trust store's file is no key by its name; a directory entry still denies it.
            denial = _exact_denial(denylist, tails, is_dir=is_dir, skip=CERT_PATTERNS)
        if denial is not None:
            return denial
    return None


def read_denied(
    asked: str,
    at: Path,
    denylist: Denylist,
    *,
    workspace: Path,
    real: Path | None = None,
) -> Denial | None:
    """The one read rule (SE-1 layer 1): the name asked for, then where `at`
    lands, so a link cannot launder a key. `at` is the path the caller will open; a relative
    one is taken under `workspace`, never under the process's working directory. `real` is the
    file a caller opened and knows it holds; without it, `realpath(at)`. A path that is its own
    landing was judged by its name.

    `workspace` is the workspace's **real** path, resolved once by the caller: a landing is
    resolved, so it is spelled from that path inside it and absolutely outside it, then judged
    by `name_denied`. Against an unresolved workspace reached through a link, no landing would
    be inside it, and the workspace's ancestors would be judged as part of every file's name.
    A caller that passes one anyway over-refuses; it never under-refuses.

    **It never raises.** When `realpath` cannot resolve `at` (a NUL byte is `ValueError`), no
    landing is known and the name's verdict stands, which judges a NUL name's prefix too
    (`name_denied`). Such a path cannot be opened either, but the open raises `ValueError`
    rather than `OSError`, so a caller that opens the path refuses a NUL byte before it asks,
    as the Reader does.

    Asked before an open and again after it with `real`: before, so a link to a secret is
    refused even when nothing is there to read; after, so a path that changed between the
    check and the open is judged as the file that was opened."""
    denial = name_denied(asked, denylist)
    if denial is not None:
        return denial
    if not at.is_absolute():
        at = workspace / at
    if real is not None:
        landed = real
    else:
        try:
            landed = Path(os.path.realpath(at))
        except (OSError, ValueError):
            return None
    if landed == at:
        return None
    spelled = name_under(workspace, landed)
    return name_denied(spelled if spelled is not None else landed.as_posix(), denylist)


def resolve(
    denylist: Denylist, roots: Iterable[Path], *, memo: WalkMemo | None = None
) -> tuple[Path, ...]:
    """The concrete paths a sandbox masks: every file and directory under each
    root that `name_denied` denies, as absolute paths, sorted. Each walked entry is judged by
    every tail of its name, so a nested `sub/.docker/config.json` is masked as `read` refuses
    it, and a nested `sub/.ssh/` is one mask, as a root-level `.ssh/` is.

    Pure apart from the walk. A denied *directory* resolves to the directory and the walk
    does not descend into it — one mask over `.ssh/`, not one per key. Only a
    directory-shaped pattern denies a directory (`is_dir=True`): a bare-name pattern
    (`*.pem`, or a project's own `*secret*`) masks the files it names and leaves an ordinary
    `secrets/` source tree alone, which is what keeps the mask from making a repository
    unbuildable under its own gates. Symlinks are not followed: a link to a secret is masked
    where it sits, and the target is masked where *it* sits if it is under a root; a link
    out of the roots is the coverage limit §2.1 states, not something to chase. `()` under
    `allow_secret_read`, so the flag turns the mask off the same way it turns the read check
    off — one switch, two consumers.

    Cost is a walk of each root, a `stat` per directory where `memo` (`secrets/walk_memo.py`)
    holds a verdict for it that its `mtime` says is still true. The caller resolves once per
    session (the workspace and `$HOME` do not change shape under a run), never per spawn.
    """
    if denylist.allow_secret_read:
        return ()
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        key = memo_key(
            [denylist.patterns, denylist.negations, str(root), [str(p) for p in _ca_files()]]
        )
        known: Mapping[str, Seen] = memo.of(key) if memo is not None else {}
        fresh: dict[str, Seen] = {}
        now = time.time_ns()
        stack = [root]
        while stack:
            here = stack.pop()
            try:
                # Followed: a root may be a link (a linked `$HOME`), and its own `mtime` would
                # never move with what is added inside. No other directory walked is a link.
                mtime = os.stat(here).st_mtime_ns
            except OSError:
                continue  # as `os.walk`: a directory that cannot be read is passed over
            seen = known.get(str(here))
            if seen is None or seen.mtime_ns != mtime:
                seen = _judged(denylist, root, here, mtime)
                if seen is None:
                    continue
            if seen.keepable and now - mtime > RACY_NS:
                fresh[str(here)] = seen
            found.update(here / name for name in seen.denied)
            stack.extend(here / name for name in seen.descend)
        if memo is not None:
            memo.keep(key, fresh)
    return tuple(sorted(found))


def _judged(denylist: Denylist, root: Path, here: Path, mtime: int) -> Seen | None:
    """One directory's entries judged, as `os.walk(followlinks=False)` lists them: a link to
    a directory is a directory to the list, and is not entered. None when it cannot be
    listed."""
    try:
        entries = list(os.scandir(here))
    except OSError:
        return None
    rel_dir = here.relative_to(root).as_posix()
    denied: list[str] = []
    descend: list[str] = []
    linked = False
    for entry in entries:
        rel = entry.name if rel_dir == "." else f"{rel_dir}/{entry.name}"
        try:
            is_dir = entry.is_dir()
            link = entry.is_symlink()
        except OSError:
            is_dir, link = False, False
        linked = linked or link
        if is_dir:
            if name_denied(rel, denylist, is_dir=True) is not None:
                denied.append(entry.name)  # the directory is the mask; do not descend
            elif not link:
                descend.append(entry.name)
            continue
        denial = name_denied(rel, denylist)
        # The absolute spelling too: a CA bundle the environment names is a trust store
        # wherever it sits.
        if denial is not None and not _spared(denial, entry.path):
            denied.append(entry.name)
    return Seen(mtime, tuple(sorted(denied)), tuple(sorted(descend)), not linked)
