"""Per-class rules: filesystem, network, effect enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
from executor_helpers import request_for

from chickadee.executor.executor import Executor
from chickadee.executor.policy import (
    SECRET_PATH,
    isolation_for,
    prefilter,
    rules_for,
    secret_path_denial,
)
from chickadee.executor.types import NetworkPolicy, PolicyClass
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.secrets.denylist import Denylist, resolve


@pytest.mark.parametrize(
    ("policy", "writable", "enforced"),
    [
        (PolicyClass.DISCOVERY, False, True),
        (PolicyClass.ACTION, True, True),
        (PolicyClass.WORKTREE, True, False),
        (PolicyClass.HUMAN, True, False),
    ],
)
def test_the_rule_table_matches_the_spec(
    policy: PolicyClass, writable: bool, enforced: bool
) -> None:
    rules = rules_for(policy)
    assert rules.repo_writable is writable
    assert rules.effects_enforced is enforced


def test_verification_effects_are_advisory(unsandboxed: Executor, work: Path) -> None:
    """A test run touching `__pycache__` is not a policy breach, and failing on it would
    produce constant false failures."""
    result = unsandboxed.run(
        request_for(work, ("sh", "-c", "echo x > undeclared.txt"), PolicyClass.WORKTREE),
        operation_id="op-1",
    )
    assert result.violations == ()


def test_action_effects_are_enforced(unsandboxed: Executor, work: Path) -> None:
    """The applier's scope gate generalised to processes."""
    result = unsandboxed.run(
        request_for(work, ("sh", "-c", "echo x > undeclared.txt"), PolicyClass.ACTION),
        operation_id="op-1",
    )
    assert [v.path for v in result.violations] == ["undeclared.txt"]
    assert not result.ok


def test_a_declared_effect_is_not_a_violation(unsandboxed: Executor, work: Path) -> None:
    result = unsandboxed.run(
        request_for(
            work,
            ("sh", "-c", "echo x > declared.txt"),
            PolicyClass.ACTION,
            effects=("declared.txt",),
        ),
        operation_id="op-1",
    )
    assert result.violations == ()
    assert result.ok


def test_the_discovery_prefilter_rejects_obvious_mutation(work: Path) -> None:
    denial = prefilter(request_for(work, "rm -rf /tmp/x", PolicyClass.DISCOVERY))
    assert denial is not None and "read-only" in denial


def test_the_prefilter_is_only_a_prefilter(work: Path) -> None:
    """`python -c` can do anything and only the kernel stops it. A pass here proves
    nothing, and the docstring says so — this asserts the limitation is real rather than
    letting a reader assume the filter is the boundary."""
    assert (
        prefilter(request_for(work, ("python", "-c", "open('x','w')"), PolicyClass.DISCOVERY))
        is None
    )


def test_the_prefilter_ignores_other_classes(work: Path) -> None:
    assert prefilter(request_for(work, "rm -rf x", PolicyClass.ACTION)) is None


def test_isolation_for_maps_discovery_read_only(work: Path) -> None:
    iso = isolation_for(request_for(work, ("echo", "x"), PolicyClass.DISCOVERY))
    assert iso.repo_writable is False
    assert iso.network_off is True
    assert iso.argv == ("echo", "x")


def test_isolation_for_maps_action_writable(work: Path) -> None:
    """Writable because it declared what it writes. Before D93 an ACTION was writable
    whatever it declared; the undeclared case is now the sealed one below."""
    iso = isolation_for(request_for(work, ("echo", "x"), PolicyClass.ACTION, effects=("x",)))
    assert iso.repo_writable is True


# --- D93: undeclared runs sealed ------------------------------------------------------


@pytest.mark.parametrize("policy", [PolicyClass.DISCOVERY, PolicyClass.ACTION])
def test_an_undeclared_request_maps_to_the_sealed_profile(
    work: Path, policy: PolicyClass
) -> None:
    """Network off, the repository read-only, the deny list masked: under this profile
    there is nothing to undo, so the call is T0 without a classifier being consulted. In a
    class whose writes are held to its declaration, declaring none is declaring read-only,
    and the sandbox enforces it rather than the effect check reporting it afterwards."""
    (work / ".env").write_text("KEY=1\n")
    listed = Denylist()
    masks = resolve(listed, (work,))
    iso = isolation_for(
        request_for(work, ("python", "-c", "print(1)"), policy),
        deny_read=masks,
        denylist=listed,
    )
    assert iso.network_off is True
    assert iso.repo_writable is False
    assert iso.deny_read == masks and masks == (work / ".env",)
    assert iso.deny_patterns == listed.name_patterns()


@pytest.mark.parametrize("policy", [PolicyClass.WORKTREE, PolicyClass.HUMAN])
def test_verification_and_human_keep_their_writes_and_are_masked_too(
    work: Path, policy: PolicyClass
) -> None:
    """A test run writes its caches whatever it declared, and a person's own command is
    theirs. Neither is exempt from the mask: every spawn is masked."""
    masks = (work / ".env",)
    iso = isolation_for(request_for(work, ("pytest", "-q"), policy), deny_read=masks)
    assert iso.repo_writable is True
    assert iso.deny_read == masks
    assert iso.deny_patterns == Denylist().name_patterns(), "None is the built-in list"


def test_a_declaration_opens_only_what_it_names(work: Path) -> None:
    """Declaring a network does not open the repository, and declaring an effect does not
    open the network: each declaration raises its own dimension."""
    networked = isolation_for(
        request_for(work, ("uv", "sync"), PolicyClass.ACTION, network=NetworkPolicy(open=True))
    )
    assert networked.network_off is False and networked.repo_writable is False
    writing = isolation_for(request_for(work, ("make",), PolicyClass.ACTION, effects=("out/",)))
    assert writing.network_off is True and writing.repo_writable is True


def test_the_configured_list_reaches_the_mask(work: Path) -> None:
    """D118 through to the wrap: a removed built-in has no name row, an addition has one, and
    `allow_secret_read` empties both halves however the caller resolved the paths."""
    edited = Denylist.configure(add=("*.kdbx",), remove=(".npmrc",))
    iso = isolation_for(request_for(work, ("ls",), PolicyClass.DISCOVERY), denylist=edited)
    assert iso.deny_patterns is not None
    assert "*.kdbx" in iso.deny_patterns and ".npmrc" not in iso.deny_patterns
    off = Denylist(allow_secret_read=True)
    stale = (work / ".env",)
    iso = isolation_for(
        request_for(work, ("ls",), PolicyClass.DISCOVERY), deny_read=stale, denylist=off
    )
    assert iso.deny_read == () and iso.deny_patterns == ()


class _WrapRecorder:
    """A backend that keeps the `Isolation` it was asked to wrap and runs nothing wrapped."""

    def __init__(self) -> None:
        self.wrapped: list[Isolation] = []

    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        self.wrapped.append(isolation)
        return isolation.argv


def test_the_executor_wraps_every_run_in_its_configured_mask(work: Path) -> None:
    listed = Denylist.configure(add=("*.kdbx",))
    masks = (work / "vault.kdbx",)
    backend = _WrapRecorder()
    executor = Executor(backend=backend, denylist=listed, deny_read=masks)
    executor.run(request_for(work, ("true",), PolicyClass.ACTION), operation_id="op-1")
    (iso,) = backend.wrapped
    assert iso.deny_read == masks
    assert iso.deny_patterns == listed.name_patterns()
    assert iso.repo_writable is False, "an undeclared ACTION runs sealed"


def test_the_executor_prefilters_against_its_configured_list(work: Path) -> None:
    """#161's last row: the executor's pre-filter read the built-in list whatever the run
    had resolved, so it refused what a config removed and spawned what it added."""
    added = Executor(backend=Passthrough(), denylist=Denylist.configure(add=("*.kdbx",)))
    refused = added.run(request_for(work, ("cat", "vault.kdbx")), operation_id="op-1")
    assert refused.exit_code == 126 and refused.stderr.startswith(SECRET_PATH + ":")

    (work / ".npmrc").write_text("registry=\n")
    removed = Executor(backend=Passthrough(), denylist=Denylist.configure(remove=(".npmrc",)))
    ran = removed.run(request_for(work, ("cat", ".npmrc")), operation_id="op-2")
    assert ran.exit_code == 0 and ran.stdout == "registry=\n"


def test_request_for_rejects_a_mistyped_keyword(work: Path) -> None:
    """A discarded keyword built the request with the default, so a test asserted against a
    configuration it never asked for — and read as though it had."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        request_for(work, ("echo", "hi"), timeot_s=1)


# --- D91: a deny-listed path NAMED in an argv is refused in every class --------------


@pytest.mark.parametrize("policy", list(PolicyClass))
@pytest.mark.parametrize(
    "command",
    [
        ("cat", ".env"),
        ("python", "x.py", "~/.aws/credentials"),
        'sh -c "cat .env"',
        ("cat", "secrets/prod.pem"),
    ],
    ids=["cat-env", "home-aws", "sh-c", "nested-pem"],
)
def test_a_named_secret_path_is_refused_in_every_class(
    work: Path, policy: PolicyClass, command: str | tuple[str, ...]
) -> None:
    request = request_for(work, command, policy)
    typed = secret_path_denial(request)
    assert typed is not None, "typed, not a string the caller has to parse"
    assert typed.denial.pattern in {".env", ".aws/**", "*.pem"}
    reason = prefilter(request)
    assert reason is not None and reason.startswith(SECRET_PATH + ":")


def test_a_template_is_not_a_secret(work: Path) -> None:
    assert (
        secret_path_denial(request_for(work, ("cat", ".env.example"), PolicyClass.DISCOVERY))
        is None
    )
    assert prefilter(request_for(work, ("cat", ".env.example"), PolicyClass.ACTION)) is None


def test_a_word_is_not_a_path(work: Path) -> None:
    """`grep -r id_rsa src/` names a word; `id_rsa*` must not refuse a search. (The example
    was `secret_key` against `*secret*` until D118 removed that pattern; the rule is the same
    for any bare-name pattern.)"""
    assert (
        secret_path_denial(
            request_for(work, ("grep", "-r", "id_rsa", "src/"), PolicyClass.DISCOVERY)
        )
        is None
    )


def test_a_word_that_is_also_a_file_is_a_path(work: Path) -> None:
    (work / "id_rsa").write_text("x")
    assert (
        secret_path_denial(request_for(work, ("cat", "id_rsa"), PolicyClass.DISCOVERY))
        is not None
    )


def test_the_secret_check_precedes_the_mutation_check(work: Path) -> None:
    reason = prefilter(request_for(work, "rm .env", PolicyClass.DISCOVERY))
    assert reason is not None and reason.startswith(SECRET_PATH + ":")


def test_allow_secret_read_turns_the_check_off(work: Path) -> None:
    request = request_for(work, ("cat", ".env"), PolicyClass.DISCOVERY)
    assert secret_path_denial(request, Denylist(allow_secret_read=True)) is None


def test_the_prefilter_refuses_against_the_list_it_is_handed(work: Path) -> None:
    """`prefilter` is the pre-spawn check `Executor.run` makes; the list a config resolves
    has to reach it, or the
    executor refuses what the repository allowed and spawns what it added to the list."""
    extended = Denylist.configure(add=("*.kdbx",))
    vault = request_for(work, ("cat", "passwords.kdbx"), PolicyClass.DISCOVERY)
    assert prefilter(vault) is None, "the built-in list does not know the extension"
    reason = prefilter(vault, extended)
    assert reason is not None and reason.startswith(SECRET_PATH + ":")
    assert "*.kdbx" in reason

    dotenv = request_for(work, ("cat", ".env"), PolicyClass.DISCOVERY)
    assert prefilter(dotenv) is not None
    assert prefilter(dotenv, Denylist(allow_secret_read=True)) is None
    assert prefilter(dotenv, None) is not None, "None is the built-in list, not 'no list'"


def test_a_directory_named_secrets_is_not_a_credential(work: Path) -> None:
    """A repository's own lint and test commands must survive the pre-filter. Matching
    `*secret*` against the whole path refused `ruff check src/secrets/denylist.py` and
    `pytest tests/secrets` in every policy class, with exit 126 and a SECRET_PATH message
    pointing at the wrong thing.

    Against a list with `*secret*` added: D118 took it off the built-ins, and with the
    built-ins alone nothing can reach a directory called `secrets`, so this would pass
    whether or not the directory check works."""
    listed = Denylist.configure(add=("*secret*",))
    (work / "src" / "secrets").mkdir(parents=True)
    (work / "src" / "secrets" / "denylist.py").write_text("code")
    (work / "tests" / "secrets").mkdir(parents=True)
    for argv in (
        ("ruff", "check", "src/secrets/denylist.py"),
        ("pytest", "-q", "tests/secrets"),
        ("pytest", "-q", "tests/secrets/"),
    ):
        request = request_for(work, argv, PolicyClass.WORKTREE)
        assert prefilter(request, listed) is None, argv


@pytest.mark.parametrize(
    "argv",
    [
        ("cat", ".env"),
        ("cat", "src/secrets/prod.pem"),
        ("cat", "src/secrets/service-account-prod.json"),
        ("cat", "~/.aws/credentials"),
    ],
    ids=["env", "nested-pem", "named-secret", "home-aws"],
)
def test_the_narrowing_does_not_reopen_a_named_secret(
    work: Path, argv: tuple[str, ...]
) -> None:
    """Only the directory question moved. A file the list names is refused as it always was."""
    reason = prefilter(request_for(work, argv, PolicyClass.ACTION))
    assert reason is not None and reason.startswith(SECRET_PATH + ":")


# -- a heredoc's text ------------------------------------------------------


def test_a_heredoc_copied_into_a_file_is_data_not_a_secret_path(work: Path) -> None:
    """The owner's run: writing `config.py` was refused twice because its heredoc held
    `env_file = ".env"`. `cat` and `tee` copy those bytes; they open nothing the body names."""
    for line in (
        "cat > app/config.py << 'EOF'\nclass Config:\n    env_file = \".env\"\nEOF",
        "tee app.py <<'EOF'\nx = '.env'\nEOF",
        "mkdir -p a && cat > a/b.py <<'EOF'\nx = '.env'\nEOF",
    ):
        assert secret_path_denial(request_for(work, line, PolicyClass.ACTION)) is None, line


@pytest.mark.parametrize(
    "line",
    [
        "sh <<'EOF'\ncat .env\nEOF",
        "cat <<'EOF' | sh\ncat .env\nEOF",
        "(cat <<'EOF'\ncat .env\nEOF\n) | bash",
        "sh -c \"$(cat <<'EOF'\ncat .env\nEOF\n)\"",
        "cat > a.py <<EOF\nx = $(cat .env)\nEOF",
        "cat .env - <<'EOF'\nhi\nEOF",
    ],
)
def test_a_heredoc_anything_runs_is_still_read(work: Path, line: str) -> None:
    """A body fed to a shell, piped or substituted into one, a substitution inside an unquoted
    body, and a secret `cat` is itself given: each still a secret path."""
    assert secret_path_denial(request_for(work, line, PolicyClass.ACTION)) is not None


def test_a_proposed_command_runs_without_the_harnesss_venv(
    monkeypatch: pytest.MonkeyPatch, work: Path, tmp_path: Path
) -> None:
    """D202 decision 2, end to end: `Executor.run` names the command's directory, so the
    harness's own venv leaves the `PATH` the command sees."""
    import os
    import sys

    venv = tmp_path / "harness" / ".venv"
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setenv("PATH", os.pathsep.join((str(venv / "bin"), os.environ["PATH"])))
    done = Executor(Passthrough()).run(
        request_for(work, "echo $PATH", PolicyClass.ACTION), operation_id="S:path"
    )
    assert str(venv / "bin") not in done.stdout
