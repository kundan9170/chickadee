"""Layer 1 — every read surface honours the deny list."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from chickadee.secrets.denylist import (
    DEFAULT_NEGATIONS,
    DEFAULT_PATTERNS,
    Denial,
    Denylist,
    _matches,  # pyright: ignore[reportPrivateUsage]
    name_denied,
    resolve,
)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        "certs/server.pem",
        "keys/private.key",
        "bundle.p12",
        "bundle.pfx",
        "id_rsa",
        "id_ed25519.pub",
        ".ssh/config",
        ".ssh/keys/deploy",
        ".aws/credentials",
        ".gnupg/secring.gpg",
        ".npmrc",
        ".netrc",
        ".pypirc",
        ".docker/config.json",
        "service-account-prod.json",
    ],
)
def test_every_default_pattern_denies(path: str) -> None:
    denial = Denylist().match(path)
    assert denial is not None, path
    assert path in denial.reason


@pytest.mark.parametrize("path", ["aws-credentials.txt", "my-secrets.yaml"])
def test_the_two_name_guesses_are_readable_now(path: str) -> None:
    """D118 item 4, and what it costs, stated as a test rather than buried in a diff.

    These two were `test_every_default_pattern_denies` cases, denied by `*credential*` and
    `*secret*`. Those patterns guessed at intent from a name — SE-11's rejected argument,
    one level up — and denied this repository six of its own files. A file that really is
    a credential is named by its path in `[secrets].deny_add`, and a project that wants the
    old guesses back writes them there and owns their false positives."""
    assert "*secret*" not in DEFAULT_PATTERNS and "*credential*" not in DEFAULT_PATTERNS
    assert Denylist().readable(path)
    assert not Denylist.configure(add=("*secret*", "*credential*")).readable(path)


@pytest.mark.parametrize(
    "path",
    [
        "design/22-secrets.md",
        "decisions/2026-08-21-d63-secret-path-reason.md",
        "tasks/T-2.4-secrets.md",
        "packages/core/tests/applier/test_secret_path_gate.py",
    ],
)
def test_the_document_that_defines_the_list_is_readable(path: str) -> None:
    """D118's first exhibit: the list denied the spec that defines it."""
    assert Denylist().readable(path)


@pytest.mark.parametrize("path", [".env.example", ".env.template", "config.example"])
def test_negations_win(path: str) -> None:
    """Templates carry names, not values. A list that hid them would teach people to
    switch it off."""
    assert Denylist().readable(path)


@pytest.mark.parametrize(
    "path", ["app.py", "README.md", "src/keychain_ui.tsx", "tests/test_a.py"]
)
def test_ordinary_files_are_readable(path: str) -> None:
    assert Denylist().readable(path)


def test_a_denial_names_its_reason_rather_than_returning_nothing() -> None:
    """A typed refusal, never a silent empty.

    A scout told 'denied' learns the file exists and stops; one handed an empty string
    concludes it is empty and may go looking for the value somewhere less guarded.
    """
    denial = Denylist().match(".env")
    assert denial is not None
    assert "denied: secret-bearing path" in denial.reason
    assert denial.pattern in denial.reason


def test_config_can_extend_but_the_builtins_remain() -> None:
    extended = Denylist.configure(add=("vault/**",))
    assert not extended.readable("vault/prod.json")
    for pattern in DEFAULT_PATTERNS:
        assert pattern in extended.patterns


def test_a_named_removal_drops_that_entry_and_leaves_every_other_in_force() -> None:
    """Successor to `test_shrinking_below_the_builtins_needs_the_explicit_flag`. The flag was
    the only way to correct one entry and it turned Layer 1 off whole; a
    removal is per entry, so `.env` going leaves `.ssh/**`, `*.pem` and the rest standing."""
    edited = Denylist.configure(remove=(".env",))
    assert edited.readable(".env")
    assert edited.removed == (".env",)
    for still in (
        ".env.production",
        "deploy/certs/prod.pem",
        ".ssh/keys/deploy",
        ".aws/credentials",
        ".gnupg/secring.gpg",
        "service-account-prod.json",
    ):
        assert not edited.readable(still), still
    assert not edited.allow_secret_read, "a removal is not the flag"


def test_allow_secret_read_still_turns_the_whole_list_off() -> None:
    """SE-8 keeps its meaning: the one switch that makes everything readable, and the only
    thing that resolves the sandbox mask to `()`."""
    assert not Denylist().readable(".env")
    off = Denylist.configure(allow_secret_read=True)
    assert off.readable(".env") and off.readable(".ssh/id_rsa")
    assert not Denylist.configure(remove=(".env",)).readable(".ssh/id_rsa")


def test_a_nested_path_is_denied_like_a_bare_one() -> None:
    """`secrets/prod.pem` and `prod.pem` are the same risk; a list that matched only one
    would depend on where the caller happened to be standing."""
    assert not Denylist().readable("deploy/certs/prod.pem")
    assert not Denylist().readable("prod.pem")


def test_an_unnamed_drop_is_refused_and_a_named_one_is_not() -> None:
    """Successor to `test_dropping_a_builtin_pattern_is_refused`. The old rule
    refused any drop; the new one refuses a drop that does not **name** what it drops.
    `Denylist(patterns=())` made `.env` readable with nothing saying so, and it still fails;
    the same list with `.env` named in `removed` is the edit a person meant."""
    with pytest.raises(ValueError, match="without naming them"):
        Denylist(patterns=())
    without_env = tuple(p for p in DEFAULT_PATTERNS if p != ".env")
    with pytest.raises(ValueError, match=r"\['\.env'\]"):
        Denylist(patterns=without_env)
    assert Denylist(patterns=without_env, removed=(".env",)).readable(".env")


@pytest.mark.parametrize("name", [".nmprc", "*secret*", ".ssh/*", ".env.example"])
def test_a_removal_naming_no_built_in_refuses_the_load(name: str) -> None:
    """A typo in a removal must not read as a removal that worked — `.nmprc` removes
    nothing, and a list that accepted it would leave the person believing `.npmrc` was
    readable. `*secret*` is refused too: it is no longer a built-in to remove. A negation is
    not a pattern and cannot be removed this way either."""
    with pytest.raises(ValueError, match="built-in deny pattern"):
        Denylist.configure(remove=(name,))


def test_a_pattern_is_on_the_list_or_off_it() -> None:
    with pytest.raises(ValueError, match="both removed and listed"):
        Denylist.configure(add=(".env",), remove=(".env",))
    with pytest.raises(ValueError, match="non-empty"):
        Denylist.configure(add=("",))


def test_removing_every_built_in_needs_the_flag() -> None:
    """Every built-in removed and nothing added reads like `allow_secret_read` (everything
    readable) while the flag says the list is on. SE-8 makes turning Layer 1 off loud, and
    the run header keys on the flag, so the quiet spelling of the same act is refused."""
    with pytest.raises(ValueError, match="no patterns"):
        Denylist.configure(remove=DEFAULT_PATTERNS)
    own = Denylist.configure(remove=DEFAULT_PATTERNS, add=("*.kdbx",))
    assert own.readable(".env") and not own.readable("vault.kdbx"), "a list of one's own"
    off = Denylist.configure(remove=DEFAULT_PATTERNS, allow_secret_read=True)
    assert off.readable(".env") and off.allow_secret_read, "the loud switch still works"


@pytest.mark.parametrize(
    "addition",
    [
        ".env.example",
        ".env.template",
        "*.example",
        "db.example",
        "config/*.template",
        # Another case is as exempt: `match` folds both sides, so the negation
        # allows `DB.EXAMPLE` too. Accepted until the review of #370.
        "DB.EXAMPLE",
        "*.Template",
    ],
)
def test_an_addition_a_built_in_exception_always_allows_is_refused(addition: str) -> None:
    """Templates are exempt, negations win over patterns, and config cannot drop a negation.
    So these would load, and the file would stay readable: protection that is not there."""
    with pytest.raises(ValueError, match="always allow"):
        Denylist.configure(add=(addition,))


def test_an_addition_only_partly_covered_by_an_exception_is_kept() -> None:
    """`x.example.bak` is not a template, so this addition denies something."""
    listed = Denylist.configure(add=("*.example.bak",))
    assert not listed.readable("prod.example.bak")


@pytest.mark.parametrize("addition", [" *.kdbx", "*.kdbx ", "\t*.kdbx"])
def test_a_padded_addition_is_refused(addition: str) -> None:
    """Patterns match names exactly, so a stray space makes `*.kdbx` match nothing."""
    with pytest.raises(ValueError, match="whitespace"):
        Denylist.configure(add=(addition,))


@pytest.mark.parametrize(
    "addition",
    ['*".kdbx', "*\\.kdbx", "*.kd\nbx", "*.kd\x00bx", "*.kdbx\x7f", 'x")) (allow file-read*'],
)
def test_an_addition_a_sandbox_profile_cannot_hold_is_refused_at_load(addition: str) -> None:
    """Seatbelt writes each pattern into its profile between quotes, where a `"` ends the
    string early and the rest is read as profile. D125 applies `deny_add` without the trust
    digest, so a repository's addition reaches every clone's profile unasked. Refused at
    load, naming the entry, and whatever the flag says: the text is the mistake."""
    for allow_secret_read in (False, True):
        with pytest.raises(ValueError, match="quote, backslash or control") as refused:
            Denylist.configure(add=(addition,), allow_secret_read=allow_secret_read)
        assert repr(addition) in str(refused.value), "the error names the entry"


def test_a_duplicated_addition_collapses_to_one() -> None:
    """Listing a pattern twice changes nothing, so it is not an error."""
    listed = Denylist.configure(add=("*.kdbx", "*.kdbx"))
    assert listed.patterns.count("*.kdbx") == 1


@pytest.mark.parametrize("negation", [".env", ".*", "*", "*.pem"])
def test_any_custom_negation_is_refused(negation: str) -> None:
    """The second way to shrink it, and the less obvious one.

    The first fix compared pattern **text**, asking whether the negation string matched a
    built-in — which decides nothing: `".*"` shares no text with `".env"` and made it
    readable anyway. Glob intersection is not answerable by comparison, so adding a negation
    is itself the deliberate act the flag gates.
    """
    with pytest.raises(ValueError, match="custom negations"):
        Denylist(negations=(*DEFAULT_NEGATIONS, negation))


def test_extending_patterns_still_needs_no_flag() -> None:
    """Adding a pattern can only tighten, so it stays free."""
    assert not Denylist.configure(add=("vault/**",)).readable("vault/prod.json")


def test_the_flag_permits_both() -> None:
    """Shrinking is a deliberate act, so it stays possible — just never by accident."""
    assert Denylist(patterns=(), allow_secret_read=True).readable(".env")


# --- the bare-name half: what a sandbox can match by name anywhere on disk ------


def test_the_name_half_keeps_order_and_leaves_directory_patterns_out() -> None:
    """The half Seatbelt renders as rows: a removed built-in is gone, an addition is in,
    and a directory-shaped entry is not, because it reaches the sandbox as resolved paths."""
    listed = Denylist.configure(add=("vault/**", "*.kdbx"), remove=(".npmrc",))
    kept = tuple(p for p in DEFAULT_PATTERNS if "/" not in p and p != ".npmrc")
    assert listed.name_patterns() == (*kept, "*.kdbx")
    assert "vault/**" in listed.patterns, "still on the list, as a path pattern"


def test_the_name_half_is_empty_when_the_list_is_off() -> None:
    """`allow_secret_read` turns off the rows with the paths."""
    assert Denylist.configure(allow_secret_read=True).name_patterns() == ()
    assert Denylist.configure(add=("*.kdbx",), allow_secret_read=True).name_patterns() == ()


def test_removing_every_name_pattern_leaves_the_directory_half_in_force() -> None:
    """`()` is also a list that is on but keeps no bare-name pattern: no rows, while
    `.ssh/**` and the other directory entries still mask what they resolve to."""
    names = tuple(p for p in DEFAULT_PATTERNS if "/" not in p)
    listed = Denylist.configure(remove=names)
    assert listed.name_patterns() == ()
    assert not listed.allow_secret_read
    assert not listed.readable(".ssh/id_rsa") and listed.readable("prod.pem")


# --- D91: resolve() — the concrete paths a sandbox masks --------------------------------


def _tree(root: Path, files: tuple[str, ...]) -> None:
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def test_resolve_names_every_default_pattern_as_an_absolute_path(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _tree(
        work,
        (
            ".env",
            ".env.prod",
            "certs/server.pem",
            "certs/server.key",
            "bundle.p12",
            "bundle.pfx",
            ".npmrc",
            ".netrc",
            ".pypirc",
            ".docker/config.json",
            "cfg/db-credentials.txt",
            "cfg/app-secret.yaml",
            "gcp/service-account-prod.json",
            "src/main.py",
            "README.md",
        ),
    )
    found = resolve(Denylist(), [work])
    rel = {p.relative_to(work).as_posix() for p in found}
    # `cfg/db-credentials.txt` and `cfg/app-secret.yaml` stay in the tree and out of the
    # mask: they were `*credential*` and `*secret*`, which D118 removed.
    assert rel == {
        ".env",
        ".env.prod",
        "certs/server.pem",
        "certs/server.key",
        "bundle.p12",
        "bundle.pfx",
        ".npmrc",
        ".netrc",
        ".pypirc",
        ".docker/config.json",
        "gcp/service-account-prod.json",
    }
    assert all(p.is_absolute() for p in found)
    assert list(found) == sorted(found)


def test_resolve_masks_a_directory_as_a_directory_not_its_contents(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _tree(
        home,
        (
            ".ssh/id_rsa",
            ".ssh/id_rsa.pub",
            ".ssh/known_hosts",
            ".aws/credentials",
            ".aws/config",
            ".gnupg/pubring.kbx",
            "notes.txt",
        ),
    )
    found = resolve(Denylist(), [home])
    assert {p.relative_to(home).as_posix() for p in found} == {".ssh", ".aws", ".gnupg"}


def test_resolve_walks_every_root_and_skips_a_missing_one(tmp_path: Path) -> None:
    work, home = tmp_path / "work", tmp_path / "home"
    _tree(work, (".env",))
    _tree(home, (".netrc",))
    found = resolve(Denylist(), [work, home, tmp_path / "absent"])
    assert {p.name for p in found} == {".env", ".netrc"}


def test_resolve_honours_negations(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _tree(work, (".env.example", ".env.template", "cfg.template", ".env"))
    assert {p.name for p in resolve(Denylist(), [work])} == {".env"}


def test_resolve_is_empty_under_allow_secret_read(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _tree(work, (".env", ".ssh/id_rsa"))
    assert resolve(Denylist(allow_secret_read=True), [work]) == ()


def test_resolve_follows_a_configured_list(tmp_path: Path) -> None:
    """The mask is the list the config resolved: a removal unmasks exactly its entry, an
    addition masks what it names, and everything else is as the built-ins left it."""
    work = tmp_path / "work"
    _tree(work, (".env", ".npmrc", "vault.kdbx", "certs/server.pem", "src/app.py"))
    edited = Denylist.configure(add=("*.kdbx",), remove=(".npmrc",))
    assert {p.name for p in resolve(edited, [work])} == {".env", "vault.kdbx", "server.pem"}


# --- a bare-name pattern names a file, never a source directory -------------------------


def test_a_source_directory_named_secrets_is_not_a_secret(tmp_path: Path) -> None:
    """`*secret*` is `my-secrets.yaml`, not the package a project keeps its secret-handling
    *code* in. Matching it against the whole path masked `chickadee/secrets/` wholesale, so
    a gate could not import the very module that enforces the deny list."""
    work = tmp_path / "work"
    _tree(work, ("src/chickadee/secrets/denylist.py", "src/credentials/loader.py"))
    # Against a list a project extended with the two name guesses D118 took off the
    # built-ins: nothing else can reach a directory called `secrets`, so without them this
    # would pass whether or not the directory rule works.
    listed = Denylist.configure(add=("*secret*", "*credential*"))
    assert resolve(listed, [work]) == ()
    assert listed.readable("src/chickadee/secrets/denylist.py")
    assert listed.readable("src/chickadee/secrets", is_dir=True)
    assert listed.readable("src/credentials", is_dir=True)


def test_a_secret_named_file_inside_such_a_directory_is_still_denied(tmp_path: Path) -> None:
    """The directory is not the secret; a file the list names still is — by a built-in,
    and by a bare-name pattern a project adds, which masks its files and not the tree."""
    work = tmp_path / "work"
    _tree(
        work,
        (
            "secrets/loader.py",
            "secrets/my-secrets.yaml",
            "secrets/prod.pem",
            "secrets/service-account-prod.json",
        ),
    )
    names = {p.name for p in resolve(Denylist(), [work])}
    assert names == {"prod.pem", "service-account-prod.json"}
    project = Denylist.configure(add=("*secret*",))
    assert {p.name for p in resolve(project, [work])} == names | {"my-secrets.yaml"}
    assert project.readable("secrets", is_dir=True)


def test_a_directory_shaped_pattern_still_masks_its_directory(tmp_path: Path) -> None:
    """`.ssh/**` names a directory, so it masks one — one mask, not one per key."""
    work = tmp_path / "work"
    _tree(work, (".ssh/id_rsa", ".ssh/keys/deploy", ".aws/credentials", ".gnupg/secring.gpg"))
    assert {p.name for p in resolve(Denylist(), [work])} == {".ssh", ".aws", ".gnupg"}


@pytest.mark.parametrize(
    "path",
    [
        "deploy/certs/prod.pem",
        "prod.pem",
        "gcp/service-account-prod.json",
        ".gnupg/secring.gpg",
    ],
)
def test_the_nested_and_bare_cases_are_unchanged(path: str) -> None:
    """The narrowing is about directories only: every path the list denied, it still denies."""
    assert not Denylist().readable(path)


# -- case (owner-directed) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        (".ENV", ".env"),
        ("config/.Env", ".env"),
        ("certs/PROD.PEM", "*.pem"),
        ("ID_RSA", "id_rsa*"),
        (".SSH/config", ".ssh/**"),
        (".Docker/Config.json", ".docker/config.json"),
    ],
)
def test_a_secret_spelled_in_another_case_is_denied(path: str, pattern: str) -> None:
    """A case-insensitive filesystem (macOS by default, Windows) opens `.env`'s bytes for
    `.ENV`, so the list matches without regard to case. On a case-sensitive one the other
    spelling is another file, and refusing it fails closed. The denial keeps the path as
    asked and the pattern as listed."""
    denial = Denylist().match(path)
    assert denial is not None and denial.pattern == pattern and denial.path == path


@pytest.mark.parametrize("path", [".ENV.EXAMPLE", "config/.Env.Template", "KEY.PEM.EXAMPLE"])
def test_a_template_spelled_in_another_case_stays_readable(path: str) -> None:
    """Negations match without regard to case too, so they still win."""
    assert Denylist().readable(path)


# -- one verdict over every tail (review of #370) -----------------------------------------


def _per_tail(denylist: Denylist, path: str, *, is_dir: bool) -> Denial | None:
    """The rule before the review of #370: the whole list matched against each tail of the
    path in turn, the first denial winning. Kept here as the reference the faster matcher
    must agree with."""
    parts = [p for p in PurePosixPath(path).parts if p != "/"]
    for i in range(len(parts)):
        tail = "/".join(parts[i:])
        folded = tail.casefold()
        name = PurePosixPath(folded).name
        if any(_matches(folded, name, n.casefold(), is_dir=is_dir) for n in denylist.negations):
            continue
        for pattern in denylist.patterns:
            if _matches(folded, name, pattern.casefold(), is_dir=is_dir):
                return Denial(tail, pattern)
    return None


def _with_a_directory_negation() -> Denylist:
    """A directory-shaped negation clears only the tails it matches, while a bare-name
    pattern matches every tail: the case a per-tail shortcut gets wrong. No valid list holds
    one today (the built-ins are bare names, and a custom negation is refused), so this is
    set past the validator, to keep the matcher right if a built-in ever is one."""
    listed = Denylist()
    object.__setattr__(listed, "negations", (*DEFAULT_NEGATIONS, "keep/.env"))
    return listed


_LISTS = (
    Denylist(),
    Denylist.configure(add=("settings_prod.py", "vault/**", "deploy/*.key")),
    _with_a_directory_negation(),
)
_HEADS = ("", "/home/u/", "sub/", "a/b/", "keep/", "x/keep/", ".ssh/", "vault/", "deploy/")
_TAILS = (
    ".env",
    ".ENV",
    ".env.example",
    ".env.local",
    "app.py",
    "prod.pem",
    ".docker/config.json",
    ".docker/other.json",
    ".ssh/id_rsa",
    ".ssh",
    ".aws/credentials",
    "settings_prod.py",
    "x.key",
    "secrets/",
)


@pytest.mark.parametrize("listed", range(len(_LISTS)))
def test_name_denied_gives_the_per_tail_verdict(listed: int) -> None:
    denylist = _LISTS[listed]
    for head in _HEADS:
        for tail in _TAILS:
            path = head + tail
            for is_dir in (False, True):
                want = _per_tail(denylist, path, is_dir=is_dir)
                assert name_denied(path, denylist, is_dir=is_dir) == want, (path, is_dir)
                if "/" not in path.strip("/"):
                    assert denylist.match(path, is_dir=is_dir) == want, (path, is_dir)
