"""What the sandbox masks and passes to a command.

Pinned in older versions: a trust store's `cacert.pem` was masked by `*.pem` like a key, and a
networked command lost the proxy and CA variables. Moved with T-3.178, listed in its Log: a
trust store is never masked by a name, a networked command keeps `NETWORK_ENV`, the
repository's tool cache is writable and named to the package managers, and the session's `/tmp`
lasts it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sandboxes import Enforcing, Scripted

from chickadee.executor.executor import Executor, cache_env
from chickadee.executor.policy import isolation_for
from chickadee.executor.types import ExecRequest, NetworkPolicy, PolicyClass
from chickadee.isolation.linux import bwrap_argv
from chickadee.isolation.macos import profile_for
from chickadee.isolation.types import Isolation
from chickadee.process.supervisor import NETWORK_ENV, bind_session, scrub_env
from chickadee.secrets.denylist import (
    CA_FILE_ENV,
    TRUST_STORE_DIRS,
    Denylist,
    name_denied,
    resolve,
)

CERTIFI = ".venv/lib/python3.12/site-packages/certifi/cacert.pem"
PIP_CERTIFI = "lib/python3.12/site-packages/pip/_vendor/certifi/cacert.pem"


def test_a_trust_store_is_never_masked_by_a_name() -> None:
    for path in (
        CERTIFI,
        PIP_CERTIFI,
        "/etc/ssl/certs/ca.pem",
        "/System/Library/OpenSSL/certs/x.pem",
    ):
        assert name_denied(path, Denylist()) is None, path


def test_a_key_is_masked_and_a_key_dir_inside_a_trust_store_still_is() -> None:
    assert name_denied("deploy/prod.pem", Denylist()) is not None
    assert name_denied("/home/u/keys/a.pem", Denylist()) is not None
    # The review of T-3.178: a trust store spares certificates, never keys beside them.
    for key in (
        "certifi/.ssh/id_rsa",
        "/etc/ssl/private/server.key",
        "/etc/ssl/private/server.pem",
        "/etc/pki/tls/private/host.key",
        "vendor/certifi/.env",
        "ca-certificates/id_rsa",
        # The second review: a project's own `ca-certificates/` or `ssl_certs/` is no trust
        # store (cfssl writes `ca-key.pem` beside its CA), and in a real trust store every
        # other pattern still decides.
        "ca-certificates/server.pem",
        "certs/ssl_certs/client-key.pem",
        "lib/site-packages/certifi/id_rsa.pem",
    ):
        assert name_denied(key, Denylist()) is not None, key
    added = Denylist.configure(add=["*-key.pem"])
    denial = name_denied("lib/site-packages/certifi/ca-key.pem", added)
    assert denial is not None and denial.pattern == "*-key.pem"


def test_a_ca_bundle_the_environment_names_is_never_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "corp-root.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    (tmp_path / "id.pem").write_text("key\n")
    for name in CA_FILE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    assert name_denied(str(bundle), Denylist()) is None
    assert resolve(Denylist(), [tmp_path]) == (tmp_path / "id.pem",)
    stores = Denylist().trust_stores()
    assert stores.paths[-1] == bundle.resolve() and stores.dirs == TRUST_STORE_DIRS
    assert stores.patterns == ("*.pem",), "the certificate patterns the list holds"
    assert Denylist(allow_secret_read=True).trust_stores().paths == ()


def test_resolve_leaves_a_venvs_trust_store_unmasked(tmp_path: Path) -> None:
    store = tmp_path / ".venv" / "lib" / "site-packages" / "certifi"
    store.mkdir(parents=True)
    (store / "cacert.pem").write_text("ca\n")
    assert resolve(Denylist(), [tmp_path]) == ()


def test_seatbelt_spares_the_trust_stores_from_the_certificate_rows_alone(
    tmp_path: Path,
) -> None:
    """The deny list states the trust stores once; the profile renders what `isolation_for`
    hands it."""
    request = ExecRequest(
        command="true", cwd=tmp_path, policy=PolicyClass.WORKTREE, timeout_s=5
    )
    listed = Denylist.configure(add=["*.key"])
    profile = profile_for(isolation_for(request, denylist=listed))
    rows = {row.split('"', 1)[0]: row for row in profile.split('(require-all (regex #"')[1:]}
    pem = next(row for rx, row in rows.items() if "[Pp][Ee][Mm]" in rx)
    key = next(row for rx, row in rows.items() if "[Kk][Ee][Yy]" in rx)
    certifi = "[Cc][Ee][Rr][Tt][Ii][Ff][Ii]"
    assert certifi in pem and '(require-not (subpath "/etc/ssl/certs"))' in pem
    assert certifi not in key and "/etc/ssl" not in key, "a key is masked everywhere"


def test_a_networked_command_keeps_how_the_network_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    monkeypatch.setenv("UV_TOKEN", "never")
    sealed = scrub_env()
    assert "HTTPS_PROXY" not in sealed and "SSL_CERT_FILE" not in sealed
    opened = scrub_env(network=True)
    assert opened["HTTPS_PROXY"] == "http://proxy.local:3128"
    assert opened["SSL_CERT_FILE"] == "/etc/ssl/cert.pem"
    assert "UV_TOKEN" not in opened, "the credential scrub still judges every name"
    assert {"PIP_CERT", "GIT_SSL_CAINFO", "NODE_EXTRA_CA_CERTS"} <= set(NETWORK_ENV)


def test_the_tool_cache_and_the_session_tmp_reach_every_wrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_CACHE_DIR", "/home/u/.cache/uv")
    cache = tmp_path / "tool-cache"
    cache.mkdir()
    backend, supervisor = Enforcing(), Scripted()
    executor = Executor(backend, supervisor=supervisor, tool_cache=cache)
    bind_session("s-1")
    try:
        executor.run(
            ExecRequest(
                command="uv sync",
                cwd=tmp_path,
                policy=PolicyClass.WORKTREE,
                timeout_s=30,
                network=NetworkPolicy(True),
            ),
            operation_id="op-1",
        )
    finally:
        bind_session("")
    [wrap] = backend.seen
    tmp = tmp_path / "cache" / "chickadee" / "tmp" / "s-1"
    assert wrap.scratch == (cache,) and wrap.tmp == tmp and tmp.is_dir()
    assert supervisor.env["UV_CACHE_DIR"] == str(cache / "uv"), "the person's own is read-only"
    assert supervisor.env["TMPDIR"] == str(tmp)
    assert set(cache_env(cache)) <= set(supervisor.env)


def test_bubblewrap_binds_the_session_tmp_over_tmp_and_the_cache_writable(
    tmp_path: Path,
) -> None:
    cache, tmp = tmp_path / "cache", tmp_path / "tmp"
    argv = bwrap_argv(Isolation(argv=("true",), cwd=tmp_path, scratch=(cache,), tmp=tmp))
    text = " ".join(argv)
    assert f"--bind {tmp} /tmp" in text and "--tmpfs /tmp" not in text
    assert f"--bind {cache} {cache}" in text
    assert "--tmpfs /tmp" in " ".join(bwrap_argv(Isolation(argv=("true",), cwd=tmp_path)))


def test_a_crashed_sessions_tmp_is_swept_and_a_read_only_tree_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from chickadee.isolation.types import remove_session_tmp, session_tmp, sweep_session_tmp

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    old, live = session_tmp("crashed"), session_tmp("open")
    stuck = old / "ro" / "deep"
    stuck.mkdir(parents=True)
    (stuck / "f").write_text("x")
    for d in (stuck, stuck.parent):
        d.chmod(0o500)
    os.utime(old, (0, 0))
    sweep_session_tmp()
    assert not old.exists() and live.is_dir()
    remove_session_tmp("open")
    assert not live.exists()
