"""D22 sealing: `require_sandbox` aborts, and the environment an arm runs in is scrubbed.

Two properties, and they are not the same property. One is that a run refuses to start
when it cannot isolate. The other is that what an arm's subprocess actually
inherits carries no proxy and no credential — which is deterministic and therefore worth
asserting exactly, unlike a live network probe whose answer depends on the machine.

The probe's *honesty* is tested rather than its verdict: `Seal` must report three states —
proven unreachable, proven reachable, or not established — because a report that reads the
same whether or not sealing worked is the failure D22 exists to prevent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from chickadee.schema import SandboxLevel
from chickadee_eval import runner


def test_require_sandbox_aborts_before_problem_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sandbox unavailable and `require_sandbox=true` → abort before problem 1.

    Before, not during: a run that stops halfway leaves a partial set whose absences
    correlate with whatever broke, which is worse than no set at all.
    """
    monkeypatch.setattr(runner, "available_backend", lambda: SandboxLevel.NONE)

    with pytest.raises(runner.SandboxUnavailable, match="require_sandbox"):
        runner.seal(require_sandbox=True, probe=False)


def test_unsealed_is_possible_only_when_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller may opt out, and the resulting Seal says so rather than looking sealed."""
    monkeypatch.setattr(runner, "available_backend", lambda: SandboxLevel.NONE)

    sealed = runner.seal(require_sandbox=False, probe=False)
    assert sealed.applied is SandboxLevel.NONE
    assert not sealed.enforced


def test_backend_is_detected_not_assumed() -> None:
    sealed = runner.seal(require_sandbox=False, probe=False)
    assert sealed.available in set(SandboxLevel)
    assert sealed.applied in set(SandboxLevel)


def test_sealed_env_drops_proxies_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deterministic half of D22. A sealed child inherits neither a route out nor a
    way to pay for one."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("http_proxy", "http://proxy.internal:8080")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-inherited")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should-never-be-inherited")
    monkeypatch.setenv("MY_DB_PASSWORD", "hunter2")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    sealed = runner.seal(require_sandbox=False, probe=False)

    for gone in (
        "HTTPS_PROXY",
        "http_proxy",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "MY_DB_PASSWORD",
    ):
        assert gone not in sealed.env, f"{gone} reached the sealed environment"
    assert sealed.env.get("PATH") == "/usr/bin:/bin", "a sealed child still needs to start"


def test_scrub_is_removal_not_blanking(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HTTPS_PROXY=""` is read as "no proxy" by some tools and as a malformed one by
    others, and the two behave differently. Absence has one meaning."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    sealed = runner.seal(require_sandbox=False, probe=False)
    assert "HTTPS_PROXY" not in sealed.env


def test_sealed_env_actually_reaches_the_child(tmp_path: Path) -> None:
    """The scrub is worth nothing if the subprocess inherits the parent anyway."""
    monkey = "SEALING_CANARY_API_KEY"
    import os

    os.environ[monkey] = "leaked"
    try:
        sealed = runner.seal(require_sandbox=False, probe=False)
        proc = subprocess.run(
            ["python3", "-c", f"import os,sys; sys.exit(1 if {monkey!r} in os.environ else 0)"],
            env=dict(sealed.env),
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, "the canary survived into the sealed child"
    finally:
        del os.environ[monkey]


def _fake_proc(returncode: int) -> Any:
    class P:
        pass

    p = P()
    p.returncode = returncode  # type: ignore[attr-defined]
    return p


def test_probe_reports_unreachable_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_proc(0))
    sealed = runner.seal(require_sandbox=False, probe=True)
    assert sealed.network_verified is True
    assert "failed" in sealed.verification


def test_probe_reports_a_reachable_network_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dangerous case: sealing did not work. It must not read as "unverified"."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_proc(7))
    sealed = runner.seal(require_sandbox=False, probe=True)
    assert sealed.network_verified is False
    assert "SUCCEEDED" in sealed.verification


def test_probe_distinguishes_inconclusive_from_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three states, never two. An unestablished result is not a passing one."""

    def boom(*_a: object, **_k: object) -> Any:
        raise subprocess.TimeoutExpired(cmd="probe", timeout=5)

    monkeypatch.setattr(subprocess, "run", boom)
    sealed = runner.seal(require_sandbox=False, probe=True)
    assert sealed.network_verified is False
    assert "did not complete" in sealed.verification


def test_disabled_probe_never_claims_verification() -> None:
    sealed = runner.seal(require_sandbox=False, probe=False)
    assert sealed.network_verified is False
    assert sealed.verification


def test_seal_is_recorded_on_every_row(tmp_path: Path) -> None:
    """The record carries the sandbox level and the probe verdict, so a reader can tell
    an unsealed run from a sealed one without leaving the results file."""
    d = tmp_path / "p"
    d.mkdir()
    (d / "instructions.md").write_text("x\n", encoding="utf-8")
    (d / "lib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (d / "lib_test.py").write_text(
        "from lib import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8"
    )

    from chickadee_eval.manifest import Language
    from chickadee_eval.problems import polyglot

    problem = polyglot.load(tmp_path, "p", Language.PYTHON)
    sealed = runner.seal(require_sandbox=False, probe=False)

    class Noop:
        name = "noop"

        def attempt(self, problem: object, workspace: Path) -> runner.Attempt:
            return runner.Attempt(solved_claim=False, cost_usd=0.0)

    record = runner.run_problem(problem, Noop(), sealed)
    assert record.sandbox == sealed.applied
    assert record.network_verified == sealed.network_verified


# --- the sandbox is applied, not merely detected -----------------------------
# Greptile P1 on #56: `detect_backend` reported `native` from executable discovery while
# nothing was ever wrapped, so rows claimed isolation that had not happened. These are the
# tests that would have caught it.


def test_wrap_prefixes_argv_for_each_backend() -> None:
    assert runner.wrap(["x"], SandboxLevel.NATIVE)[:2] == ["unshare", "-rn"]
    assert runner.wrap(["x"], SandboxLevel.CONTAINER)[0] == "bwrap"
    assert runner.wrap(["x"], SandboxLevel.NONE) == ["x"]


def test_available_is_not_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend on PATH that cannot start a child here is reported as available and NOT
    applied. Conflating the two is what let a row claim a sandbox it never entered."""
    monkeypatch.setattr(runner, "available_backend", lambda: SandboxLevel.NATIVE)
    monkeypatch.setattr(runner, "_wrap_works", lambda _l, _e: False)

    sealed = runner.seal(require_sandbox=False, probe=False)
    assert sealed.available is SandboxLevel.NATIVE
    assert sealed.applied is SandboxLevel.NONE
    assert not sealed.enforced


def test_require_sandbox_aborts_when_backend_present_but_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed-but-broken must abort exactly like absent."""
    monkeypatch.setattr(runner, "available_backend", lambda: SandboxLevel.NATIVE)
    monkeypatch.setattr(runner, "_wrap_works", lambda _l, _e: False)

    with pytest.raises(runner.SandboxUnavailable, match="available"):
        runner.seal(require_sandbox=True, probe=False)


def test_arm_isolation_is_stated_and_says_none() -> None:
    """The arm is in-process and is not wrapped. Saying so is the honest report; a
    `SandboxLevel` field would have implied otherwise."""
    sealed = runner.seal(require_sandbox=False, probe=False)
    assert "none" in sealed.arm_isolation
    assert "in-process" in sealed.arm_isolation


def _sandbox_applies() -> bool:
    """Whether this machine can actually *apply* a sandbox, not merely offer one.

    Guarding on `available_backend()` was wrong in exactly the way this PR exists to fix:
    GitHub's runners have `unshare` on PATH with unprivileged user namespaces disabled, so
    availability is `native` and application is `none`. The test then demanded a seal the
    machine could not give and failed — while the code under test behaved correctly by
    refusing to claim one.
    """
    try:
        return runner.seal(require_sandbox=True, probe=False).enforced
    except runner.SandboxUnavailable:
        return False


@pytest.mark.skipif(not _sandbox_applies(), reason="no sandbox can be applied on this machine")
def test_applied_sandbox_really_blocks_the_network() -> None:
    """The end of the chain: where a backend can be applied, a wrapped child genuinely
    cannot connect. This is the claim `network_verified` makes, measured rather than
    asserted."""
    sealed = runner.seal(require_sandbox=True, probe=True)
    assert sealed.enforced
    assert sealed.network_verified, sealed.verification


def test_an_unappliable_backend_never_reports_as_enforced() -> None:
    """The property that holds on every machine, including the runners that skip the test
    above: `enforced` follows `applied`, so a box with `unshare` present but namespaces
    disabled reports honestly rather than optimistically."""
    try:
        sealed = runner.seal(require_sandbox=False, probe=False)
    except runner.SandboxUnavailable:  # pragma: no cover - require_sandbox is False
        pytest.fail("require_sandbox=False must never raise")
    assert sealed.enforced == (sealed.applied is not SandboxLevel.NONE)
    if sealed.applied is SandboxLevel.NONE:
        assert not sealed.enforced
