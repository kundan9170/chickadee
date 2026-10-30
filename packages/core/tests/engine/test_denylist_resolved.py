"""The deny list is resolved once, through the trust digest.

A removal a repository ships is inert on a clone until its owner accepts the digest of the
repository's executable subset. The loader answers that question over the
project layer (`Resolved.trust`), the engine resolves its list from the answer once, at
construction, and the run header reads that one value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repos import init_repo

from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError, HarnessBug
from chickadee.secrets.denylist import Denylist
from chickadee.state.layers import Layer
from chickadee.state.resolve import Resolved, TrustDecision, resolve
from chickadee.state.roots import roots as resolve_roots
from chickadee.state.trust import TRUST_FILE, acceptance_record

SID = "S-deny"


def resolved(
    raw: dict[str, Any], secrets: dict[str, Any], *, mine: dict[str, Any] | None = None
) -> Resolved:
    """The person's engine config (with `mine` as its own `[secrets]`) plus a repository's
    `[secrets]`, merged as a load does."""
    user = {**raw, "secrets": mine} if mine else raw
    project = {"secrets": secrets} if secrets else {}
    return resolve(
        [
            Layer("user", None, user),
            Layer("project", Path("/repo/.chickadee/config.toml"), project),
        ]
    )


def accept(config: Resolved, repo_local: Path, overrides: dict[str, Any] | None = None) -> None:
    """What the acceptance surface records: the digest of the repository's layer, with the
    project profile's executable overrides merged in."""
    _, body = acceptance_record({**config.project, **(overrides or {})}, repo_local)
    repo_local.mkdir(parents=True, exist_ok=True)
    (repo_local / TRUST_FILE).write_text(body, encoding="utf-8")


def listed(engine: PacketEngine) -> Denylist:
    return engine._denylist  # pyright: ignore[reportPrivateUsage]


def engine_for(tmp_path: Path, config: Resolved, repo_local: Path) -> PacketEngine:
    return PacketEngine(tmp_path, config.engine, trust=config.trust(repo_local))


# -- the rule --------------------------------------------------------------------------------


def test_a_repository_removal_is_inert_until_accepted_then_in_force(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo_local = tmp_path / "repo_local"
    config = resolved(raw, {"deny_remove": [".npmrc"]})

    assert config.trust(repo_local).accepted is False
    assert not listed(engine_for(tmp_path, config, repo_local)).readable(".npmrc")

    accept(config, repo_local)
    assert config.trust(repo_local).accepted is True
    assert listed(engine_for(tmp_path, config, repo_local)).readable(".npmrc")


def test_allow_secret_read_is_inert_until_accepted_then_in_force(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo_local = tmp_path / "repo_local"
    config = resolved(raw, {"allow_secret_read": True})

    before = listed(engine_for(tmp_path, config, repo_local))
    assert not before.allow_secret_read
    assert not before.readable(".env")

    accept(config, repo_local)
    after = listed(engine_for(tmp_path, config, repo_local))
    assert after.allow_secret_read
    assert after.readable(".env")


def test_an_addition_applies_whether_or_not_the_digest_is_accepted(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    """It can only tighten, so an inert addition would leave a clone looser than its
    project — the wrong direction for inert."""
    repo_local = tmp_path / "repo_local"
    config = resolved(raw, {"deny_add": ["*.kdbx"], "deny_remove": [".npmrc"]})

    unaccepted = listed(engine_for(tmp_path, config, repo_local))
    assert not unaccepted.readable("vault.kdbx")
    assert not unaccepted.readable(".npmrc")

    accept(config, repo_local)
    accepted = listed(engine_for(tmp_path, config, repo_local))
    assert not accepted.readable("vault.kdbx")
    assert accepted.readable(".npmrc")


def test_accepting_one_removal_never_accepts_a_wider_one(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo_local = tmp_path / "repo_local"
    accept(resolved(raw, {"deny_remove": [".npmrc"]}), repo_local)

    wider = resolved(raw, {"deny_remove": [".npmrc", ".env"]})
    assert wider.trust(repo_local).accepted is False
    assert not listed(engine_for(tmp_path, wider, repo_local)).readable(".npmrc")


def test_the_persons_own_setting_needs_no_acceptance(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    """§6 digests the project config: what a repository ships. `allow_secret_read` in the
    person's own engine config is their decision, not one a clone makes for them."""
    repo_local = tmp_path / "repo_local"
    config = resolved(raw, {}, mine={"allow_secret_read": True})
    assert config.trust(repo_local).accepted is True
    assert listed(engine_for(tmp_path, config, repo_local)).allow_secret_read


def test_a_project_bootstrap_override_is_part_of_the_digest(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    """The acceptance surface digests the repository's layer with its profile overrides
    merged in, so the same overrides must reach `trust()` or an accepted config reads as
    unaccepted."""
    repo_local = tmp_path / "repo_local"
    overrides = {"toolchain": {"bootstrap": {"python": ["uv", "sync"]}}}
    config = resolved(raw, {"deny_remove": [".npmrc"]})
    accept(config, repo_local, overrides)

    assert config.trust(repo_local, overrides=overrides).accepted is True
    assert config.trust(repo_local).accepted is False


def test_the_engine_defaults_to_the_whole_list(tmp_path: Path, raw: dict[str, Any]) -> None:
    """A host that does not pass the loader's answer gets the safe reading, not a guess."""
    engine = PacketEngine(tmp_path, resolved(raw, {"deny_remove": [".npmrc"]}).engine)
    assert not listed(engine).readable(".npmrc")


def test_a_decision_taken_for_another_config_is_refused(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo_local = tmp_path / "repo_local"
    narrow = resolved(raw, {"deny_remove": [".npmrc"]})
    accept(narrow, repo_local)
    wide = resolved(raw, {"allow_secret_read": True})
    with pytest.raises(HarnessBug, match="different config"):
        PacketEngine(tmp_path, wide.engine, trust=narrow.trust(repo_local))
    # A decision names the load it was asked of.
    decision = narrow.trust(repo_local)
    assert isinstance(decision, TrustDecision)
    assert decision.engine is narrow.engine


@pytest.mark.parametrize("body", ["accepted = [", "accepted = 5"])
def test_an_unreadable_trust_file_is_a_config_error(
    tmp_path: Path, raw: dict[str, Any], body: str
) -> None:
    # Only a config that shrinks the list has anything to accept, so only it reads the file.
    repo_local = tmp_path / "repo_local"
    repo_local.mkdir()
    (repo_local / TRUST_FILE).write_text(body, encoding="utf-8")
    config = resolved(raw, {"deny_remove": [".npmrc"]})
    with pytest.raises(ConfigError, match=TRUST_FILE):
        config.trust(repo_local)


# -- one value, read by the header -----------------------------------------------------------


def repository(tmp_path: Path) -> Path:
    repo = init_repo(tmp_path / "repo", commit=None)
    Session.create(sessions_root(resolve_roots(repo)), SID).close()
    return repo


def deny_line(engine: PacketEngine) -> str:
    lines = [line.text for line in engine.header(SID).lines if line.kind == "deny_list"]
    assert len(lines) == 1
    return lines[0]


def test_the_list_is_resolved_once_at_construction_and_the_header_reads_it(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo = repository(tmp_path)
    repo_local = resolve_roots(repo).repo_local
    config = resolved(raw, {"deny_remove": [".npmrc"]})
    accept(config, repo_local)

    engine = engine_for(repo, config, repo_local)
    assert listed(engine).readable(".npmrc")
    assert "inert" not in deny_line(engine)

    # Withdrawing the acceptance mid-session changes nothing this engine holds: the list
    # was resolved once, and the header reads that list rather than asking again.
    (repo_local / TRUST_FILE).unlink()
    assert listed(engine).readable(".npmrc")
    assert "inert" not in deny_line(engine)

    # The next engine resolves again, and both the list and its header say inert.
    fresh = engine_for(repo, config, repo_local)
    assert not listed(fresh).readable(".npmrc")
    assert "inert until" in deny_line(fresh)
