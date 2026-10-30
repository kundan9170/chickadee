"""Seatbelt profile generation, and a real denial where the platform allows it."""

from __future__ import annotations

import itertools
import os
import platform
import re
import subprocess
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.errors import InfraError
from chickadee.isolation.base import Probe, forget_probes, probe
from chickadee.isolation.macos import (
    BINARY,
    NULL_DEVICE_WRITE,
    PATTERN_ROWS,
    Seatbelt,
    deny_regexes,
    profile_for,
)
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.paths import glob_match
from chickadee.secrets.denylist import DEFAULT_PATTERNS, Denylist, resolve

from .isolation_helpers import fake_binary, iso_for

darwin_only = pytest.mark.skipif(platform.system() != "Darwin", reason="macOS backend")


def test_read_only_wrap_gets_no_repo_write(work: Path) -> None:
    profile = profile_for(iso_for(work, ("echo", "x"), repo_writable=False))
    assert "(deny default)" in profile
    assert f'(allow file-write* (subpath "{work}"))' not in profile


def test_writable_wrap_gets_repo_write(work: Path) -> None:
    profile = profile_for(iso_for(work, ("echo", "x"), repo_writable=True))
    assert f'(allow file-write* (subpath "{work}"))' in profile


def test_network_is_denied_unless_left_on(work: Path) -> None:
    assert "(deny network*)" in profile_for(iso_for(work, ("echo", "x")))
    allowed = profile_for(iso_for(work, ("echo", "x"), network_off=False))
    assert "(deny network*)" not in allowed
    assert "(allow network*)" in allowed


def test_a_scratch_tmpdir_stays_writable(work: Path) -> None:
    """Denying it turns a read-only wrap into a broken toolchain: compilers and linters
    write temporaries."""
    profile = profile_for(iso_for(work, ("echo", "x"), repo_writable=False))
    assert '(subpath "/tmp")' in profile


@pytest.mark.parametrize("writable", [False, True])
def test_the_null_device_is_writable_and_nothing_else_under_dev(
    work: Path, writable: bool
) -> None:
    """#540: git opens /dev/null read-write, so a wrap without this fails gate A3's baseline."""
    profile = profile_for(iso_for(work, ("git", "status"), repo_writable=writable))
    assert NULL_DEVICE_WRITE in profile.split("\n")
    dev_writes = [ln for ln in profile.split("\n") if "write" in ln and '"/dev' in ln]
    assert dev_writes == [NULL_DEVICE_WRITE]


def test_the_level_is_native() -> None:
    assert Seatbelt().level() is SandboxLevel.NATIVE


@darwin_only
def test_the_profile_denies_a_real_repo_write(work: Path) -> None:
    """The assertion that matters: not that the profile *says* deny, but that the kernel
    does. A generated profile with a syntax error would allow everything and still read
    correctly above."""
    if not Seatbelt().available():  # pragma: no cover
        pytest.skip(f"{BINARY} unavailable")
    isolation = iso_for(work, ("sh", "-c", f"echo x > {work}/denied.txt"), repo_writable=False)
    out = subprocess.run(
        Seatbelt().wrap(isolation), capture_output=True, text=True, check=False
    )
    assert out.returncode != 0
    assert not (work / "denied.txt").exists()


@darwin_only
def test_the_profile_permits_a_declared_write(work: Path) -> None:
    if not Seatbelt().available():  # pragma: no cover
        pytest.skip(f"{BINARY} unavailable")
    isolation = iso_for(work, ("sh", "-c", f"echo x > {work}/allowed.txt"), repo_writable=True)
    subprocess.run(Seatbelt().wrap(isolation), capture_output=True, text=True, check=False)
    assert (work / "allowed.txt").exists()


@darwin_only
def test_a_read_only_wrap_runs_git_status_and_still_refuses_writes(work: Path) -> None:
    """#540 on the kernel: `git status` opens /dev/null read-write and exits 0 under a
    read-only wrap, and a write to tracked source is still refused."""
    if not Seatbelt().available():  # pragma: no cover
        pytest.skip(f"{BINARY} unavailable")
    tracked = work / "calc.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    init_repo(work, commit="one")
    status = iso_for(
        work, ("sh", "-c", "git status --porcelain >/dev/null"), repo_writable=False
    )
    out = subprocess.run(Seatbelt().wrap(status), capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    before = tracked.read_bytes()
    write = iso_for(work, ("sh", "-c", f"echo x > {tracked}"), repo_writable=False)
    refused = subprocess.run(
        Seatbelt().wrap(write), capture_output=True, text=True, check=False
    )
    assert refused.returncode != 0
    assert tracked.read_bytes() == before


@darwin_only
def test_a_deny_listed_read_is_still_refused_beside_the_null_device(work: Path) -> None:
    if not Seatbelt().available():  # pragma: no cover
        pytest.skip(f"{BINARY} unavailable")
    secret = work / ".env"
    secret.write_text("TOKEN=x\n")
    isolation = Isolation(argv=("cat", str(secret)), cwd=work, deny_read=(secret,))
    out = subprocess.run(
        Seatbelt().wrap(isolation), capture_output=True, text=True, check=False
    )
    assert out.returncode != 0
    assert "TOKEN" not in out.stdout


# --- D91: the read mask -------------------------------------------------------------


def test_read_allow_excludes_each_mask_by_shape(work: Path) -> None:
    secret = work / ".env"
    secret.write_text("x")
    keys = work / ".ssh"
    keys.mkdir()
    profile = profile_for(Isolation(argv=("cat",), cwd=work, deny_read=(secret, keys)))
    assert f'(require-not (literal "{secret}"))' in profile
    assert f'(require-not (subpath "{keys}"))' in profile


def test_regex_rows_render_every_pattern_shaped_entry(work: Path) -> None:
    profile = profile_for(Isolation(argv=("cat",), cwd=work, deny_read=(work / ".env",)))
    rows = [ln for ln in profile.split("\n") if "(regex #" in ln]
    assert len(rows) == 1
    assert rows[0].startswith("(allow file-read* (require-all ")
    assert rows[0].count("(require-all (regex #") == len(PATTERN_ROWS)
    assert "(?!" not in rows[0], "Seatbelt regexes are DFA-compatible and use no lookahead"
    assert '(require-not (regex #"\\.example$"))' in rows[0]
    assert '(require-not (regex #"\\.template$"))' in rows[0]


def test_the_two_name_guesses_left_the_seatbelt_rows_too() -> None:
    """D118 removed `*secret*` and `*credential*` from the defaults; the regex rows are a
    copy, so they go here in the same change or macOS alone would keep denying them."""
    assert "*secret*" not in PATTERN_ROWS and "*credential*" not in PATTERN_ROWS


def test_pattern_rows_are_the_deny_lists_pattern_shaped_defaults() -> None:
    """Isolation cannot import the deny list, so the two literals are pinned equal here:
    every default that is NOT a directory glob (`.ssh/**`) and not a fixed relative path
    (`.docker/config.json`) is a row."""
    expected = {p for p in DEFAULT_PATTERNS if "**" not in p and "/" not in p}
    assert set(PATTERN_ROWS) == expected


def test_no_mask_no_regex_rows(work: Path) -> None:
    assert deny_regexes(Isolation(argv=("cat",), cwd=work)) == ()
    assert "(regex #" not in profile_for(Isolation(argv=("cat",), cwd=work))


def test_a_removal_that_leaves_nothing_to_mask_keeps_the_pattern_rows(work: Path) -> None:
    """D118 made an empty `deny_read` ambiguous. `deny_remove = [".env"]` over a workspace
    whose only secret is `.env` resolves to `()` while `*.pem`, `id_rsa*` and the rest stay
    in force. Read as "the list is off", that emptiness dropped every row, and a key written
    mid-session, or kept outside the scanned roots, was readable by a spawned process."""
    (work / ".env").write_text("x")
    listed = Denylist.configure(remove=(".env",))
    in_force = Isolation(
        argv=("cat",),
        cwd=work,
        deny_read=resolve(listed, (work,)),
        deny_patterns=listed.name_patterns(),
    )
    assert in_force.deny_read == ()
    assert len(deny_regexes(in_force)) == len(PATTERN_ROWS) - 1, "every row but `.env`'s"
    assert "(regex #" in profile_for(in_force)


def test_the_list_turned_off_emits_no_rows(work: Path) -> None:
    """`allow_secret_read` is the one switch that turns the list off. The caller
    says so, and the rows go with the paths."""
    (work / ".env").write_text("x")
    listed = Denylist.configure(allow_secret_read=True)
    off = Isolation(
        argv=("cat",),
        cwd=work,
        deny_read=resolve(listed, (work,)),
        deny_patterns=listed.name_patterns(),
    )
    assert deny_regexes(off) == ()
    assert "(regex #" not in profile_for(off)


# --- #161: the rows are the configured list's bare-name patterns ----------------------


def _rows(*patterns: str) -> tuple[str, ...]:
    return deny_regexes(Isolation(argv=("cat",), cwd=Path("/w"), deny_patterns=patterns))


def test_the_rows_are_the_configured_name_patterns(work: Path) -> None:
    """A removed built-in has no row, an addition has one, and a directory-shaped addition
    has none, because it reaches the wrap as resolved paths. Before, the rows were the
    built-ins whatever the config said: a removed `.npmrc` stayed masked for a spawned
    process on macOS alone, and `*.kdbx` reached Seatbelt only where it resolved to a path
    under the scanned roots."""
    listed = Denylist.configure(add=("*.kdbx", "vault/**"), remove=(".npmrc",))
    iso = Isolation(argv=("cat",), cwd=work, deny_patterns=listed.name_patterns())
    rows = deny_regexes(iso)
    assert "/\\.[Nn][Pp][Mm][Rr][Cc]$" not in rows
    assert "/[^/]*\\.[Kk][Dd][Bb][Xx]$" in rows
    assert not any("vault" in row for row in rows)
    assert len(rows) == len(PATTERN_ROWS), "one removed, one added"
    assert '(regex #"/[^/]*\\.[Kk][Dd][Bb][Xx]$")' in profile_for(iso)


def test_none_keeps_the_built_in_rows_and_an_empty_set_has_none(work: Path) -> None:
    """`None` is a caller that resolved no list, and keeps the built-in rows while anything
    is masked. `()` is a list that keeps no bare-name pattern: no rows, while the concrete
    masks stay."""
    secret = work / ".env"
    legacy = Isolation(argv=("cat",), cwd=work, deny_read=(secret,))
    assert deny_regexes(legacy) == _rows(*PATTERN_ROWS)
    none_left = Isolation(argv=("cat",), cwd=work, deny_read=(secret,), deny_patterns=())
    profile = profile_for(none_left)
    assert "(regex #" not in profile
    assert f'(require-not (literal "{secret}"))' in profile


@pytest.mark.parametrize(
    ("glob", "row"),
    [
        ("secret?.txt", "/[Ss][Ee][Cc][Rr][Ee][Tt][^/]\\.[Tt][Xx][Tt]$"),
        ("*.[pk]em", "/[^/]*\\.[KPkp][Ee][Mm]$"),
        ("id_[!x]*", "/[Ii][Dd]_[^Xx/][^/]*$"),
        ("backup[0-9].key", "/[Bb][Aa][Cc][Kk][Uu][Pp][0-9]\\.[Kk][Ee][Yy]$"),
        ("a+b.c", "/[Aa]\\+[Bb]\\.[Cc]$"),
    ],
)
def test_a_glob_renders_as_the_regex_it_means(glob: str, row: str) -> None:
    """`?` and a class are one character of the name, and a negated class is one character
    that is none of its members and not a separator. A literal `.` or `+` stays escaped.
    Only `*` was translated before, so these rendered as literals and masked nothing."""
    assert _rows(glob) == (row,)


@pytest.mark.parametrize(
    ("glob", "row"),
    [
        ("[]x]", "/[]Xx]$"),  # a `]` straight after the `[` is a member
        ("[!]x]", "/[^]Xx/]$"),
        ("[^x]", "/[^Xx/]$"),  # a leading `^` negates, as `!` does
        ("[^]", "/\\[\\^\\]$"),  # a `[^` with nothing after it never closes: a literal
        ("[a-]", "/[-Aa]$"),  # a `-` goes first or last, where it is itself
        ("[+--]", "/[-+-,]$"),  # a range ending in `-` gives the `-` its own place
        ("[--z]", "/[-.0-@A-Z[-z]$"),  # a range over `/` is split around it
        ("x[z-a]", None),  # a descending range matches nothing, so neither can the pattern
        ("[!z-a]", "/[^/]$"),
        ("[[:alpha:]]", "/[:AHLPahlp[]\\]$"),  # globs have no `[:alpha:]`; its `[` goes last
        ("x[", "/[Xx]\\[$"),  # a `[` that never closes is a literal
    ],
)
def test_an_awkward_class_is_written_so_each_character_means_itself(
    glob: str, row: str | None
) -> None:
    """Written out member by member, never copied: `]`, `^`, `-` and `[` each mean something
    by position in a class, and no backslash is written inside one, where regex dialects
    disagree about it."""
    assert _rows(glob) == (() if row is None else (row,))


NAMES = tuple(
    "".join(chars) for n in range(4) for chars in itertools.product("ab.-]^[!_0A", repeat=n)
)
AWKWARD = (
    "secret?.txt",
    "*.[pk]em",
    "[!.]*",
    "[]-a]*",
    "[!]a-]?",
    "[^b]",
    "[a-c-e]",
    "[+-.]",
    "[--0]",
    "[[]",
    "[!a-c]b",
    "*[!-]",
    "[!z-a]",
)


@pytest.mark.parametrize("glob", AWKWARD)
def test_a_row_matches_exactly_the_names_the_deny_list_matches(glob: str) -> None:
    """The row matches a path whose basename the deny list matches (`glob_match`), and
    no other, from any directory. Read here by Python's `re`, which reads these classes as
    POSIX does because none holds a backslash or a `[:`; that Seatbelt compiles them is the
    darwin case's. `.` names no file, and `glob_match` reads it as the path it stands in."""
    (row,) = _rows(glob)
    with warnings.catch_warnings():
        # A class opening with `[` is valid POSIX; Python alone warns about it.
        warnings.simplefilter("ignore", FutureWarning)
        rx = re.compile(row)
    for name in (n for n in NAMES if n != "."):
        # The last path puts `name` in a directory: a class that matched `/` would let the
        # row reach across the separator.
        for path in (f"/w/{name}", f"/w/a/{name}", f"/w/{name}/b"):
            basename = path.rsplit("/", 1)[1]
            got = rx.search(path) is not None
            # The list folds case on both sides, and the row says the same.
            want = glob_match(basename.casefold(), (glob.casefold(),)) is not None
            assert got == want, (glob, row, path)


# --- paths are written into the profile between double quotes ------------------------

INJECTED = (
    'id_rsa")))) (allow file-read*) (allow file-read* (require-all (require-not (literal "x'
)
"""A legal file name that matches `id_rsa*`. Written into the profile as it was, it closed
the read rule early, added an unconditional `(allow file-read*)`, and opened one more rule
to absorb the rest, so the profile stayed well-formed and no mask held."""


def test_a_masked_name_that_would_end_its_string_early_is_refused(work: Path) -> None:
    key = work / INJECTED
    key.write_text("must-not-print\n")
    listed = Denylist()
    denied = resolve(listed, (work,))
    assert denied == (key,), "the name is on the list, so the resolver hands it to the wrap"
    with pytest.raises(InfraError, match="refusing to build a Seatbelt profile"):
        profile_for(
            Isolation(
                argv=("cat",),
                cwd=work,
                deny_read=denied,
                deny_patterns=listed.name_patterns(),
            )
        )


@pytest.mark.parametrize("name", ['a"b.pem', "a\\b.pem", "a\nb.pem", "a\x7fb.pem"])
def test_every_character_that_could_end_the_string_is_refused(work: Path, name: str) -> None:
    with pytest.raises(InfraError, match="refusing to build a Seatbelt profile"):
        profile_for(Isolation(argv=("cat",), cwd=work, deny_read=(work / name,)))


def test_a_writable_cwd_is_held_to_the_same_rule(tmp_path: Path) -> None:
    cwd = tmp_path / 'repo"name'
    cwd.mkdir()
    with pytest.raises(InfraError, match="refusing to build a Seatbelt profile"):
        profile_for(Isolation(argv=("true",), cwd=cwd, repo_writable=True))


def test_an_awkward_but_safe_name_is_written_as_it_is(work: Path) -> None:
    """None of these can end a double-quoted string, so refusing them would only make real
    repositories unrunnable."""
    key = work / "it's (prod); clé.pem"
    profile = profile_for(Isolation(argv=("cat",), cwd=work, deny_read=(key,)))
    assert f'(require-not (literal "{key}"))' in profile


@pytest.mark.parametrize("pattern", ['*".pem', "*\\.pem", "*.pe\nm", "*.pem\x7f", INJECTED])
def test_a_pattern_that_would_end_its_string_early_is_refused(work: Path, pattern: str) -> None:
    """Patterns are config text written into the profile between quotes, the same as paths.
    D125 applies `deny_add` without the trust digest, so a repository's addition reaches
    every clone's profile unasked; `Denylist` refuses these at load, and this refuses a list
    built outside config."""
    with pytest.raises(InfraError, match="refusing to build a Seatbelt profile around the"):
        profile_for(Isolation(argv=("cat",), cwd=work, deny_patterns=(pattern,)))


def test_a_pattern_with_a_separator_is_refused(work: Path) -> None:
    """`vault/**` is about where a file sits, which a basename row cannot say; it reaches
    the wrap as resolved paths. A caller that hands it over has skipped `name_patterns()`."""
    with pytest.raises(InfraError, match="names a directory"):
        profile_for(Isolation(argv=("cat",), cwd=work, deny_patterns=("vault/**",)))


@darwin_only
def test_a_nonempty_read_mask_compiles_and_is_enforced(work: Path) -> None:
    """Finding #6 on PR 146: compile the generated regex filters with Seatbelt itself,
    then prove that the mask denies a secret while preserving a negated template."""
    if not Seatbelt().available():  # pragma: no cover
        pytest.skip(f"{BINARY} unavailable")
    secret = work / ".env"
    secret.write_text("must-not-print\n")
    # Pattern-only: nothing in `deny_read` names it, so only the regex row can deny it.
    # `service-account*.json` survives D118 (the old `service-secret.txt` was `*secret*`),
    # and its hyphen is the `\-` `re.escape` emits — a case no other row measured.
    patterned = work / "service-account-prod.json"
    patterned.write_text("must-not-print-either\n")
    template = work / ".env.example"
    template.write_text("NAME=\n")

    masked = Isolation(argv=("/bin/cat", str(secret)), cwd=work, deny_read=(secret,))
    denied = subprocess.run(
        Seatbelt().wrap(masked), capture_output=True, text=True, check=False
    )
    assert denied.returncode != 0
    assert denied.stdout == ""
    assert "must-not-print" not in denied.stdout

    by_pattern = Isolation(argv=("/bin/cat", str(patterned)), cwd=work, deny_read=(secret,))
    pattern_denied = subprocess.run(
        Seatbelt().wrap(by_pattern), capture_output=True, text=True, check=False
    )
    assert pattern_denied.returncode != 0
    assert pattern_denied.stdout == ""

    allowed = Isolation(argv=("/bin/cat", str(template)), cwd=work, deny_read=(secret,))
    read = subprocess.run(Seatbelt().wrap(allowed), capture_output=True, text=True, check=False)
    assert read.returncode == 0
    assert read.stdout == "NAME=\n"

    # #161, under the kernel: the rows are the configured list. `vault1.kdbx` is matched by
    # an added pattern alone, whose `?` and class Seatbelt compiles here rather than Python's
    # `re`; `.npmrc` is a removed built-in, and reading it proves the same profile compiled.
    listed = Denylist.configure(add=("vault?.kdb[xy]",), remove=(".npmrc",))
    added = work / "vault1.kdbx"
    added.write_text("must-not-print-either\n")
    removed = work / ".npmrc"
    removed.write_text("registry=\n")

    def under_list(path: Path) -> subprocess.CompletedProcess[str]:
        iso = Isolation(
            argv=("/bin/cat", str(path)),
            cwd=work,
            deny_read=(secret,),
            deny_patterns=listed.name_patterns(),
        )
        return subprocess.run(Seatbelt().wrap(iso), capture_output=True, text=True, check=False)

    by_addition = under_list(added)
    assert by_addition.returncode != 0
    assert by_addition.stdout == ""
    unmasked = under_list(removed)
    assert unmasked.returncode == 0, unmasked.stderr
    assert unmasked.stdout == "registry=\n"


# -- the probe -------------------------------------------------------------

SEATBELT_RUNS = 'shift 2\nexec "$@"\n'
"""`sandbox-exec -p PROFILE cmd...`: drop the two profile words and run the rest."""


@pytest.fixture
def fresh_probes() -> Iterator[None]:
    forget_probes()
    yield
    forget_probes()


def test_seatbelt_probes_available_where_sandbox_exec_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    bin_dir = tmp_path / "bin"
    fake_binary(bin_dir, BINARY, SEATBELT_RUNS)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    assert probe(Seatbelt()) == Probe(SandboxLevel.NATIVE, ran=True)
    assert Seatbelt().available()


def test_a_sandbox_exec_that_fails_is_not_available_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_probes: None
) -> None:
    bin_dir = tmp_path / "bin"
    refused = 'echo "sandbox-exec: sandbox_apply: Operation not permitted" >&2\nexit 71\n'
    fake_binary(bin_dir, BINARY, refused)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    found = probe(Seatbelt())
    assert not found.ran
    assert "sandbox_apply: Operation not permitted" in found.reason and found.fix
    assert not Seatbelt().available()


@darwin_only
def test_the_real_seatbelt_probes_available(fresh_probes: None) -> None:  # pragma: no cover
    assert probe(Seatbelt()).ran
