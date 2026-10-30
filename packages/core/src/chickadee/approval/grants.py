"""Grants: a person's *always*, and the one check that honours it.

**A grant is not a preference.** It changes no tier and lowers no floor. It supplies the
dispatch-time confirmation for an action that equals what it binds, which is the ticket: the
command's class, prefix and destination and every declaration the
approved call made (`CommandBinding`). A call that differs in any of them is asked again.

**Two scopes, one check.** *Always for this session* is a `command_grant` line on the session
log, so the same session resumed keeps it and a fork or a new session does not (a fork's first
line is its own `fork`, and the fold starts over there). *Always for this repo* is a line under
`[[approval.grants]]` in the repository's `approval.toml`: committed, so executable config,
inert on a clone until its owner accepts the list's digest.
`GrantingPort` is the only reader and the only writer of both: every asker a session has
reaches the person through it, and none checks a grant itself.

*Once* is not a grant. It stays what it was: the answer to this question, or a host's
`spec_sha` authorization.
"""

from __future__ import annotations

import json
import threading
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast

from pydantic import TypeAdapter, with_config

from chickadee.approval.commands import (
    NEVER_GRANTED,
    Classification,
    CommandClass,
    binding_of,
    exact_binding_of,
    repo_any_arguments,
    repo_offered,
)
from chickadee.approval.prefs import (
    PreferenceError,
    parse_level,
    paths_for,
    read_approval,
    repo_approval,
)
from chickadee.context.lines import LINE, SessionKind
from chickadee.context.questions import for_run
from chickadee.durable.files import write_atomic
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    ApprovalDecision,
    ApprovalStop,
    CommandBinding,
    GrantScope,
    InteractionPort,
    Question,
    StopDecision,
    network_of_hosts,
)
from chickadee.state.roots import Roots
from chickadee.state.trust import TRUST_FILE, acceptance_body

if TYPE_CHECKING:
    from chickadee.context.session import Session

__all__ = [
    "ACTION_GRANTED",
    "COMMAND_GRANT",
    "NEVER_ASKED",
    "ActionGrantedLine",
    "CommandGrantLine",
    "GrantingPort",
    "Offer",
    "RepoGrants",
    "accept_for_clone",
    "grantable",
    "offer",
    "record_repo_grant",
    "repo_grants",
    "session_grants",
]

COMMAND_GRANT: Final[SessionKind] = "command_grant"
ACTION_GRANTED: Final[SessionKind] = "action_granted"

COMMAND_EFFECT: Final = "command"
"""`[[approval.grants]]`'s `effect` for a command grant. Other effects (`expose_path`) are
their owner's, read by it and left alone here."""

_KEYS: Final = frozenset(
    {"effect", "class", "prefix", "destination", "network", "hosts", "effects", "credentials"}
)
"""`hosts` is a row written by an older version, read as `network` (`network_of_hosts`)."""


def grantable(
    found: Classification | None,
    *,
    network: bool = False,
    effects: Sequence[str] = (),
    credentials: Sequence[str] = (),
) -> CommandBinding | None:
    """What *always* would bind for this call, or `None` for *once* only: an unrecognised
    command, a line no single grant could cover (`binding_of`), or a class no grant
    covers (`NEVER_GRANTED`). The call's
    declarations are bound beside the table's three fields, sorted, so a grant covers a later
    call only when that call reaches no further."""
    if found is None or found.classes & NEVER_GRANTED:
        return None
    return _declared(binding_of(found), network, effects, credentials)


@dataclass(frozen=True)
class Offer:
    """What a prompt offers for a call: `grant`, what *always for this repo*
    binds (the shape, or any arguments); `exact`, what *this session, exact* binds; whether the
    prompt offers the repository answer at all (`offer_repo`, E.3); and whether that answer
    covers any arguments, which the prompt then says in plain words."""

    grant: CommandBinding | None
    exact: CommandBinding | None
    offer_repo: bool
    repo_any_arguments: bool


def offer(
    found: Classification | None,
    words: Sequence[str],
    *,
    network: bool = False,
    effects: Sequence[str] = (),
    credentials: Sequence[str] = (),
) -> Offer:
    """The `Offer` for a call whose line reads as `found` and whose exact words are `words`
    (`shell_line.exact_words`). A class no grant covers offers *once* only, as `grantable`."""
    grant = grantable(found, network=network, effects=effects, credentials=credentials)
    if found is None or found.classes & NEVER_GRANTED:
        return Offer(grant, None, offer_repo=False, repo_any_arguments=False)
    exact = _declared(exact_binding_of(found, words), network, effects, credentials)
    return Offer(
        grant,
        exact,
        offer_repo=grant is not None and repo_offered(found),
        repo_any_arguments=repo_any_arguments(found),
    )


def _declared(
    binding: CommandBinding | None,
    network: bool,
    effects: Sequence[str],
    credentials: Sequence[str],
) -> CommandBinding | None:
    """`binding` with the call's declarations bound beside it, sorted, so a grant covers a later
    call only when that call reaches no further."""
    if binding is None:
        return None
    return binding.model_copy(
        update={
            "network": network,
            "effects": tuple(sorted(set(effects))),
            "credentials": tuple(sorted(set(credentials))),
        }
    )


# -- the session's grants ------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class CommandGrantLine:
    """*Always for this session*: one line per grant, written when the person gives it."""

    KIND: ClassVar[SessionKind] = COMMAND_GRANT
    cls: str
    prefix: tuple[str, ...]
    destination: str | None
    network: bool
    """Whether the granted call opened the network; a line written before reads its
    `hosts` as this (`network_of_hosts`)."""
    effects: tuple[str, ...]
    credentials: tuple[str, ...]
    action_id: str
    """The confirmation that issued it."""
    exact: bool = False
    """*This session, exact*: `prefix` is the line's every word, and the grant
    covers only a question whose `exact` binding it is. Without it, a line whose words are its
    row's own (`make`, `git push`) would bind as the row, and cover `make deploy`. A line from
    before then has none, and is the row grant it always was."""

    @classmethod
    def of(cls, binding: CommandBinding, action_id: str, *, exact: bool) -> CommandGrantLine:
        return cls(
            binding.cls,
            binding.prefix,
            binding.destination,
            binding.network,
            binding.effects,
            binding.credentials,
            action_id,
            exact,
        )

    def binding(self) -> CommandBinding:
        return CommandBinding(
            cls=self.cls,
            prefix=self.prefix,
            destination=self.destination,
            network=self.network,
            effects=self.effects,
            credentials=self.credentials,
        )


_LINE = TypeAdapter(CommandGrantLine)


def _line(raw: bytes) -> CommandGrantLine:
    """A `command_grant` line, one written by an older version with its `hosts` read as
    `network`. The
    line is strict JSON, so only an old one is parsed twice, to rename the key in the JSON it is
    read from; every other is parsed once."""
    if b'"hosts"' not in raw:
        return _LINE.validate_json(raw)
    return _LINE.validate_json(json.dumps(network_of_hosts(json.loads(raw))))


def session_grants(session: Session) -> tuple[CommandGrantLine, ...]:
    """The session's grants, folded from its log. A `fork` line starts the fold over: a fork
    reads its parent's lines through, and a grant dies with the session that was given it.
    A rewind keeps them (`rewind.REWIND`): a grant is the person's decision about the
    session, not part of the work an undo takes back. Each line says whether it is exact, which
    a bare binding would lose (`CommandGrantLine.exact`)."""
    held: list[CommandGrantLine] = []
    for event, _record in session.state.lines():
        if event.kind == "fork":
            held.clear()
        elif event.kind == COMMAND_GRANT:
            held.append(_line(session.blob(event.payload_sha)))
    return tuple(held)


# -- the repository's grants ---------------------------------------------------------------


@dataclass(frozen=True)
class RepoGrants:
    """The project level's command grants, and whether this clone has accepted them."""

    bindings: tuple[CommandBinding, ...]
    digest: str
    """Of the whole `approval.grants` list, every effect, and the file's `allow` rules
    (`prefs.RepoApproval`): what `trusted.toml` accepts."""
    trusted: bool
    """A file with neither is trusted by construction: there is nothing to accept."""
    empty: bool
    declared: tuple[Mapping[str, Any], ...] = ()
    """The list's tables as the file holds them, every effect: what `chickadee trust` shows
    verbatim beside `digest`."""
    lowerings: tuple[str, ...] = ()
    """The command classes the file sets to `allow`, inert until accepted with the grants."""

    @property
    def honoured(self) -> tuple[CommandBinding, ...]:
        return self.bindings if self.trusted else ()


def repo_grants(roots: Roots) -> RepoGrants:
    """`[[approval.grants]]` from `<repo>/.chickadee/approval.toml`. Strict: an unknown key or
    class refuses, as the rest of the file does (`prefs.py`). Never the user level: a grant
    a person carried into every repository would be a preference, which it is not."""
    project, _user = paths_for(roots)
    return _judged(read_approval(project), roots, str(project))


def _judged(data: Mapping[str, Any], roots: Roots, source: str) -> RepoGrants:
    parts = repo_approval(data, source)
    bindings = tuple(
        _binding(g, source, n)
        for n, g in enumerate(parts.grants)
        if g.get("effect") == COMMAND_EFFECT
    )
    return RepoGrants(
        bindings,
        parts.digest,
        parts.accepted(roots),
        empty=parts.empty,
        declared=parts.grants,
        lowerings=parts.lowerings,
    )


def _binding(raw: Mapping[str, Any], source: str, n: int) -> CommandBinding:
    where = f"{source}: approval.grants[{n}]"
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise PreferenceError(f"{where}: unknown key(s) {unknown}")
    cls = raw.get("class")
    if cls not in {c.value for c in CommandClass}:
        raise PreferenceError(f"{where}: class {cls!r} is not a command class")
    if CommandClass(cls) in NEVER_GRANTED:
        raise PreferenceError(f"{where}: class {cls!r} is never granted (D106 §3)")
    prefix = raw.get("prefix")
    if (
        not isinstance(prefix, list)
        or not prefix
        or not all(isinstance(t, str) for t in cast("list[object]", prefix))
    ):
        raise PreferenceError(f"{where}: prefix must be a non-empty list of tokens")
    lists: dict[str, tuple[str, ...]] = {}
    for key in ("hosts", "effects", "credentials"):
        value = raw.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(v, str) for v in cast("list[object]", value)
        ):
            raise PreferenceError(f"{where}: {key} must be a list of strings")
        lists[key] = tuple(sorted(set(cast("list[str]", value))))
    destination = raw.get("destination")
    if destination is not None and not isinstance(destination, str):
        raise PreferenceError(f"{where}: destination must be a string")
    del lists["hosts"]
    try:
        # A row from an older version names hosts, read as every grant's are.
        read = cast("dict[str, object]", network_of_hosts(dict(raw)))
    except ValueError as exc:
        raise PreferenceError(f"{where}: {exc}") from exc
    network = read.get("network", False)
    if not isinstance(network, bool):
        raise PreferenceError(f"{where}: network must be true or false")
    return CommandBinding(
        cls=cast("str", cls),
        prefix=tuple(cast("list[str]", prefix)),
        destination=destination,
        network=network,
        **lists,
    )


def accept_for_clone(roots: Roots, digests: Sequence[str]) -> Path:
    """Record `digests` as accepted in this clone's `trusted.toml`, beside every digest there:
    `chickadee trust`'s one write, on the person's yes, after it showed what each names.
    Returns the file. The one writer of the file, as
    `record_repo_grant` is for a grant's own acceptance."""
    path = roots.repo_local / TRUST_FILE
    write_atomic(path, acceptance_body(roots.repo_local, *digests).encode("utf-8"))
    return path


def record_repo_grant(roots: Roots, binding: CommandBinding) -> str:
    """Append *always for this repo*'s line, and accept the new list for this clone only when
    the list before it was already accepted (or empty): the person's answer accepts their own
    line, never a teammate's they have not read. Returns what the answer's note says.

    The line is written first and accepted second, so a crash between them leaves the grant
    inert, the safe side. The new file is parsed back, as the loader will read it, before it
    replaces the old one: a file this would break is left as it was, and the grant refused."""
    project, _user = paths_for(roots)
    before = read_approval(project)
    prior = _judged(before, roots, str(project))
    text = project.read_text(encoding="utf-8") if project.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    text += _rendered(binding)
    try:
        after = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PreferenceError(
            f"{project}: the grant could not be added to this file: {exc}"
        ) from exc
    parse_level(after, "repo", source=str(project))
    now = _judged(after, roots, str(project))
    write_atomic(project, text.encode("utf-8"))
    if not prior.trusted:
        return (
            f"written to {project}, and inert on this clone: the file holds grants this clone "
            "has not accepted, and answering for one line never accepts the others"
        )
    write_atomic(
        roots.repo_local / TRUST_FILE,
        acceptance_body(roots.repo_local, now.digest).encode("utf-8"),
    )
    return f"granted for this repository in {project}"


def _rendered(binding: CommandBinding) -> str:
    def string(value: str) -> str:
        # A JSON string is a TOML basic string for every character JSON escapes.
        return json.dumps(value, ensure_ascii=False)

    def strings(values: Sequence[str]) -> str:
        return "[" + ", ".join(string(v) for v in values) + "]"

    lines = [
        "",
        "[[approval.grants]]",
        f"effect = {string(COMMAND_EFFECT)}",
        f"class = {string(binding.cls)}",
        f"prefix = {strings(binding.prefix)}",
    ]
    if binding.destination is not None:
        lines.append(f"destination = {string(binding.destination)}")
    if binding.network:
        lines.append("network = true")
    for key in ("effects", "credentials"):
        values: tuple[str, ...] = getattr(binding, key)
        if values:
            lines.append(f"{key} = {strings(values)}")
    return "\n".join(lines) + "\n"


# -- the check -----------------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class ActionGrantedLine:
    """An action a standing grant approved, so no one was asked: one line, never a
    question. What ran, under which grant, and what that grant covers."""

    KIND: ClassVar[SessionKind] = ACTION_GRANTED
    action_id: str
    operation_id: str
    display: str
    """The action as the person reads it (`ActionApproval.display`)."""
    scope: GrantScope | Literal["never_ask"]
    """`never_ask`: no grant, the person's `sandbox.never_ask`."""
    cls: str
    prefix: tuple[str, ...]
    """What the grant covers; under `never_ask`, the command's own binding, or `NEVER_ASKED`
    and `()` for a line no binding reads."""


NEVER_ASKED: Final = "unbound"
"""`ActionGrantedLine.cls` for a command `never_ask` ran that no binding reads."""


class GrantingPort:
    """The session's `InteractionPort`, over its `RecordingPort`: the one dispatch check for
    both scopes, and where the person's *always* is written down.

    It sits outside the recorder, so a grant is checked once and before anything is
    recorded: an action a standing grant covers is one `action_granted` line, tagged with the
    run that asked, and never a question, so no one is shown a prompt that is already
    answered. Anything else goes to the recorder, which records it and asks the host's
    port; that port is told nothing about grants."""

    def __init__(
        self,
        inner: InteractionPort,
        session: Session,
        roots: Roots,
        run_id: str | None = None,
        *,
        lock: threading.Lock | None = None,
        never_ask: bool = False,
    ) -> None:
        """`never_ask` is the person's `sandbox.never_ask`: a command no grant
        covers is approved here, before anyone is asked, attended or not."""
        self.inner = inner
        self._session = session
        self._roots = roots
        self._run_id = run_id
        self._never_ask = never_ask
        self._lock = lock if lock is not None else threading.Lock()
        """One grant check and issue at a time, across the session's run-scoped copies: two
        calls confirmed at once must not both write the same repository line."""

    def for_run(self, run_id: str) -> GrantingPort:
        """The same check for run `run_id`'s confirmations (`context.questions.for_run`): its
        lines, and the recorder's under it, carry the run."""
        return GrantingPort(
            for_run(self.inner, run_id),
            self._session,
            self._roots,
            run_id,
            lock=self._lock,
            never_ask=self._never_ask,
        )

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        return self.inner.ask(q, cancel=cancel)

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        """No grant covers a stop: every stop is asked."""
        return self.inner.confirm_stop(stop, cancel=cancel)

    def _settle(self, approval: ActionApproval) -> ApprovalDecision | None:
        """The approval a standing grant gives `approval`, its `action_granted` line written;
        then, for a command, the one `never_ask` gives; None when someone must be asked. The
        shell's `deny` and secret-path refusals are made before any approval reaches here, so
        `never_ask` never runs what they refuse."""
        held, binding = self._covering(approval)
        if held is None or binding is None:
            return self._never_asked(approval)
        self._session.append(
            ActionGrantedLine(
                approval.action_id,
                approval.operation_id,
                approval.display,
                held,
                binding.cls,
                binding.prefix,
            ),
            run_id=self._run_id,
        )
        return ApprovalDecision(
            action_id=approval.action_id,
            verdict="approve",
            source="grant",
            scope=held,
            note=_SCOPE_WORDS[held],
        )

    def _never_asked(self, approval: ActionApproval) -> ApprovalDecision | None:
        """`sandbox.never_ask`'s approval of a command, one `action_granted` line naming it;
        None for anything else, which is asked as before (a mutating MCP call, a git action)."""
        if not self._never_ask or approval.kind != "command":
            return None
        bound = approval.exact or approval.grant
        self._session.append(
            ActionGrantedLine(
                approval.action_id,
                approval.operation_id,
                approval.display,
                "never_ask",
                NEVER_ASKED if bound is None else bound.cls,
                () if bound is None else bound.prefix,
            ),
            run_id=self._run_id,
        )
        return ApprovalDecision(
            action_id=approval.action_id,
            verdict="approve",
            source="preference",
            note="run unasked: sandbox.never_ask",
        )

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        settled = self._settle(approval)
        if settled is not None:
            return settled
        decision = self.inner.confirm_action(approval, cancel=cancel)
        if decision.verdict != "approve" or decision.scope == "once":
            return decision
        # *This session* binds the exact line; a question that carries no exact
        # binding (the packet loop's) binds its `grant`, as before. *This repo* binds `grant`,
        # and only where the question offered it (E.3).
        binding = (
            (approval.exact or approval.grant)
            if decision.scope == "session"
            else (approval.grant if approval.offer_repo else None)
        )
        if binding is None:
            # An *always* the question did not offer is the *once* it can be.
            return decision.model_copy(update={"scope": "once"})
        with self._lock:
            if decision.scope == "session":
                self._session.append(
                    CommandGrantLine.of(
                        binding, approval.action_id, exact=approval.exact is not None
                    )
                )
                return decision.model_copy(
                    update={"note": _joined(decision.note, _SCOPE_WORDS["session"])}
                )
            try:
                note = record_repo_grant(self._roots, binding)
            except (PreferenceError, OSError) as exc:
                # The person said yes to this call: it runs, once. The grant they also asked
                # for could not be written; the answer says why instead of failing the call.
                return decision.model_copy(
                    update={
                        "scope": "once",
                        "note": _joined(decision.note, f"approved once; not granted: {exc}"),
                    }
                )
        return decision.model_copy(update={"note": _joined(decision.note, note)})

    def _covering(
        self, approval: ActionApproval
    ) -> tuple[GrantScope | None, CommandBinding | None]:
        """The held grant that covers `approval`, and its scope: a session grant of its exact
        line or of its binding, then a repository grant of its binding. An exact grant covers
        only an exact binding and a row grant only a row binding, so the two never stand in
        for each other however alike their words. A repository grant written by hand covers a
        binding the prompt would not have offered (E.3)."""
        held = session_grants(self._session)
        exact = [line.binding() for line in held if line.exact]
        rows = [line.binding() for line in held if not line.exact]
        if approval.exact is not None and approval.exact in exact:
            return "session", approval.exact
        if approval.grant is not None and approval.grant in rows:
            return "session", approval.grant
        if approval.grant is not None and approval.grant in repo_grants(self._roots).honoured:
            return "repo", approval.grant
        return None, None


_SCOPE_WORDS: Final[Mapping[GrantScope, str]] = {
    "once": "approved once",
    "session": "granted for this session",
    "repo": "granted for this repository",
}


def _joined(first: str | None, second: str) -> str:
    return second if not first else f"{first}; {second}"
