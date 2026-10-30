"""Every refusal a model reads ends with what to do next.

Was (2026-10-22): most failures stated a fact and stopped. Athena's `note` was refused 25 times
with `a note is at most 200 tokens (athena.note_max)`, in a unit she could not measure and with
no size to cut to, and from the fourth identical call the answer dropped even that.

What these pin: every failure type renders an instruction, and a new type is held to it (the
types are found, not listed); a bound says what to do at each resource; the repetition note says
the next identical calls are refused; a note over its cap is measured in characters with how
much to cut.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil

import pytest

import chickadee
from chickadee.boxctx.athena import write_note
from chickadee.context.request import est_text
from chickadee.context.results import (
    CallCancelled,
    FetchFailed,
    HandleExpired,
    IndexUnavailable,
    Interrupted,
    InvalidArgument,
    JobsFull,
    NotSupported,
    NotText,
    SearchFailed,
    SecretPath,
    UnknownJob,
    UnresolvedSearch,
    UrlRefused,
)
from chickadee.edits.draft import DraftRefused
from chickadee.edits.match import MatchFailure, find
from chickadee.hooks.failures import HookDenied, HookHalted
from chickadee.limits import chars_for_tokens
from chickadee.schema.authorization import Tier
from chickadee.schema.execution import SandboxLevel
from chickadee.shell import (
    NetworkDenied,
    PreferenceDenied,
    RefusedAboveT0,
    SandboxRequired,
    TimedOut,
    WriteDenied,
)
from chickadee.tools.catalogue import InvalidCall
from chickadee.tools.rules import _DO, BoundReached, Declined, NotOffered, repeated
from chickadee.tools.tool import CallTimedOut
from chickadee.tools.validate import InvalidArguments

CASES: dict[type, object] = {
    InvalidArgument: InvalidArgument("text", "it is empty", do="send text"),
    HandleExpired: HandleExpired("h_1", "absent"),
    SecretPath: SecretPath(".env", ".env*"),
    NotText: NotText("binary", 10),
    NotSupported: NotSupported("images", "low-param"),
    IndexUnavailable: IndexUnavailable("a.py", "none"),
    UnresolvedSearch: UnresolvedSearch("x", ("a",), ("b",)),
    UrlRefused: UrlRefused("https://a.com/?q=x", "not_seen"),
    SearchFailed: SearchFailed("github", "rate_limited", 403),
    FetchFailed: FetchFailed("https://a.com/", "http_status", 404),
    UnknownJob: UnknownJob("J9"),
    JobsFull: JobsFull(("J1",)),
    CallCancelled: CallCancelled("shell"),
    Interrupted: Interrupted("shell", "the session ended"),
    TimedOut: TimedOut("pytest", 60),
    RefusedAboveT0: RefusedAboveT0(Tier.T1, "install", "approval"),
    NetworkDenied: NetworkDenied("could not resolve host"),
    WriteDenied: WriteDenied("/etc/x", "read-only file system"),
    SandboxRequired: SandboxRequired(SandboxLevel.NONE, "no backend"),
    PreferenceDenied: PreferenceDenied("rm -rf x", "deny rm"),
    BoundReached: BoundReached("tool_calls", 100, 100, "request"),
    Declined: Declined("shell", Tier.T1),
    HookDenied: HookDenied("lint", "the file has new errors"),
    HookHalted: HookHalted("budget", "the episode's time is up"),
    NotOffered: NotOffered("str_replace_editor", "read", ("read",)),
    CallTimedOut: CallTimedOut("grep", 30),
    InvalidCall: InvalidCall(InvalidArguments("read", (), {})),
    DraftRefused: DraftRefused("NO_MATCH", "old matched nowhere", do="copy old exactly"),
}


def _failure_types() -> set[type]:
    """Every class in the package with a `failure` field and a `render`: what a model reads."""
    found: set[type] = set()
    for info in pkgutil.walk_packages(chickadee.__path__, "chickadee."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # an optional dependency's module; not a failure's home
            continue
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and value.__module__ == module.__name__
                and dataclasses.is_dataclass(value)
                and "failure" in {f.name for f in dataclasses.fields(value)}
                and callable(getattr(value, "render", None))
            ):
                found.add(value)
    return found


def test_every_failure_type_is_held_here() -> None:
    """A new failure type is a new row in `CASES`, so it is held to an instruction too."""
    missing = sorted(t.__name__ for t in _failure_types() - set(CASES))
    assert missing == []


@pytest.mark.parametrize("kind", list(CASES), ids=lambda t: t.__name__)
def test_every_failure_renders_what_to_do(kind: type) -> None:
    rendered = CASES[kind].render()  # type: ignore[attr-defined]
    assert "Do: " in rendered or "Do not call it again" in rendered, rendered


@pytest.mark.parametrize("resource", sorted(_DO))
def test_every_bound_says_what_to_do_at_it(resource: str) -> None:
    assert f"Do: {_DO[resource]}" in BoundReached(resource, 1, 1, "request").render()


def test_a_draft_refusal_cannot_be_built_without_its_instruction() -> None:
    """Its code names a kind of refusal, never the cause (`DENIED` is a path out of scope, a
    file that exists, a frozen design, a run holding the repository...), so no instruction can
    be chosen by code: one so chosen told Athena never to write her plan's README. The site that
    knows the cause states it, and pyright checks every
    site for it."""
    with pytest.raises(TypeError, match="do"):
        DraftRefused("DENIED", "app.py is not a TARGETS path")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("old", "scopes", "said"),
    [
        ("", (), "to add text, set old to the line it goes after"),
        ("x = 1", (), "add the lines around old"),
        ("nowhere", (), "copy old exactly"),
        ("x = 1", ("def go():",), "name a scope line that occurs once"),
        ("x = 1", ("def stop():",), "copy the scope line exactly"),
    ],
)
def test_each_match_cause_says_its_own_next_step(
    old: str, scopes: tuple[str, ...], said: str
) -> None:
    """Five causes, two codes: an empty `old`, a repeated one and a repeated scope are all
    `AMBIGUOUS`, and each is fixed another way. Each instruction is set where its cause is."""
    text = "def go():\n    x = 1\n\ndef go():\n    x = 1\n"
    found = find(text, old, "y", lines=False, scopes=scopes)
    assert isinstance(found, MatchFailure)
    assert said in found.do


def test_the_repetition_note_says_the_next_ones_are_refused() -> None:
    note = repeated("repetition{count: 3}", 3)
    assert note.startswith("repetition{count: 3}. Do: you sent this exact call 3 times")
    assert "the next identical ones are refused" in note


class _RefusalLog:
    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()
        self.next_seq = 1
        self.lines: list[object] = []

    def append(self, line: object, *, run_id: str) -> None:
        self.lines.append(line)


def test_a_note_over_its_cap_is_measured_in_characters_with_the_cut() -> None:
    text = "take_hit: knockdown sets hitstun_timer=0 so pose section 6 does not override. " * 12
    log = _RefusalLog()
    refused = write_note(log, "a" * 64, "r", 200, text)  # type: ignore[arg-type]
    assert isinstance(refused, InvalidArgument) and log.lines == []
    most = chars_for_tokens(200)
    shown = refused.render()
    assert f"this note is about {est_text(text)} tokens ({len(text)} characters)" in shown
    assert f"about {most} characters" in shown
    assert f"Do: send note again with text of at most {most} characters" in shown
    assert f"cut at least {len(text) - most} characters" in shown
    kept = write_note(log, "a" * 64, "r", 200, text[:most])  # type: ignore[arg-type]
    assert not isinstance(kept, InvalidArgument), "a note cut to the limit is kept"
