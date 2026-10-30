"""The worker's draft: one per candidate, where its incremental edits land.

Moved from `packer/draft.py`, off `Packet`: the draft is given `Targets`, the paths
and their modes, and the packet loop turns its packet's targets into them
(`packer/view.draft_targets`). It stays a leaf, as `edits/` is: it reads no file itself (its
`read` is given), runs no process (its `judge` is given) and imports nothing above `edits/`.

A draft is made over the targets as rollback left them: an `edit` target's text, as
the newline policy shows it (`edits/lines.decode`), a `create` target absent until written, a
`delete` target not editable. Every edit, from every tool, ends in `CandidateDraft.apply`, the
one write (guard row E7), inside one call (`CandidateDraft.call`): every change a call makes
stands, or none does. What the candidate hands in is `handed`, the targets it changed, whole,
in the file's own newlines; the driver turns that into the attempt's one gated `PatchSet`
(only the Applier writes).

When a line tool is offered the draft holds the run's anchor style: its answers show
their lines in it, a line anchor is resolved against the text as it now reads (`line`), and the
port's views of a target ask it for the same style (`packer/view.py`).

At a call's answer every file the call changed is judged by its `judge` (the packet
loop's is `box/syntax.SyntaxChecker.inproc`, never a process): under `reject` a call that broke
a file that parsed before it is undone whole; under `report` it stands and its answer says so;
`off` judges nothing.

**Concurrency, stated where the state is.** A draft belongs to one candidate's port view, which
serves one call at a time on the driver's thread; nothing else holds it.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from chickadee.context.results import instructed
from chickadee.context.spans import Span
from chickadee.edits.elision import find_elision
from chickadee.edits.formats.base import Create, FileEdit, FileEdits, Targets, Whole
from chickadee.edits.lines import (
    AnchorKind,
    AnchorStyle,
    Decoded,
    Stale,
    as_decoded,
    encode,
    split_lines,
)
from chickadee.edits.match import Region, splice
from chickadee.edits.symbols import SymbolMiss, SymbolSpan, find_symbol
from chickadee.errors import HarnessBug
from chickadee.paths import ROOT_RULE
from chickadee.profiles.resolve import FileLanguage
from chickadee.schema.journal import SyntaxDepth

EditCheck = Literal["reject", "report", "off"]
"""What a syntax error an edit call made does (`RolePolicy.edit_check`)."""

__all__ = [
    "CandidateDraft",
    "DraftFailure",
    "DraftMode",
    "DraftRefused",
    "Judge",
    "Meant",
    "Opens",
    "Removable",
    "SyntaxCheck",
    "emptied",
]


@dataclass(frozen=True)
class SyntaxCheck:
    """A file's syntax verdict (gate A1's, moved here with the draft that judges by it)."""

    ok: bool
    detail: str | None
    depth: SyntaxDepth | None = None
    """How far the check went (`SyntaxDepth`), which the facts record. A grammar's parse
    (`parsed`) is weaker than the language's compiler: its pass is said
    `parsed`, never `syntax ok`, so a reader is never told more was checked than was."""

    @property
    def parsed_only(self) -> bool:
        return self.depth == "parsed"


def emptied(content: str) -> bool:
    """Whether `content` is empty or whitespace only: a body that must never stand in for a
    file's text (the Applier's `empty_defense`, whose rule this is). A model that returns
    nothing is failing; writing that nothing over a file is data loss."""
    return not content.strip()


Judge = Callable[[str, bytes], SyntaxCheck | None]
"""A written file's syntax verdict, its path and its bytes as they would be written; None when
nothing judges that file (`syntax unchecked`)."""

Languages = Callable[[str], FileLanguage | None]
"""A path's language and grammar, the profile set's answer; None when no
profile reads it. What a symbol edit parses the file's text as."""


DraftFailure = Literal[
    "NO_MATCH",
    "AMBIGUOUS",
    "STALE_ANCHOR",
    "FILE_ABSENT",
    "ELIDED",
    "BAD_FORMAT",
    "DENIED",
    "UNSUPPORTED",
    "SYNTAX_FAIL",
]


@dataclass(frozen=True)
class DraftRefused:
    """An edit the worker's draft refused, answered to the worker at once: an anchor
    that matched nowhere or more than once (`NO_MATCH`, `AMBIGUOUS`, naming the nearest lines,
    or for a symbol's name the file's declarations), a line anchor naming no line of the draft
    as it now reads (`STALE_ANCHOR`, with the lines there now), an edit of a create target
    not yet written (`FILE_ABSENT`), a placeholder standing in for code (`ELIDED`), a patch its
    format's parser refused or a line range that ends before it starts (`BAD_FORMAT`, with the
    hint), a path it may not write (`DENIED`), or a symbol edit of a file no parser reads, or
    of a declaration sharing a line with other code (`UNSUPPORTED`), or a call that broke a
    file that parsed before it. Never evidence the
    packet was cut too narrow: no repack widens what may be written."""

    failure: DraftFailure
    reason: str
    did_you_mean: str | None = None
    """For a path that does not resolve, the file it most likely meant (`paths.suggest`)."""
    do: str = field(kw_only=True)
    """What to do next, written where the cause is known. Required: the code
    names a kind of refusal, never its cause (`DENIED` is a path out of scope, a file that
    exists, a frozen design, a run holding the repository...), so an instruction chosen by
    code was wrong for most of its sites; one told Athena never to write her plan's README."""

    @property
    def instruction(self) -> str:
        if self.did_you_mean is not None:
            return f"send it again with path {self.did_you_mean} if that is the file you meant"
        return self.do

    def render(self) -> str:
        text = f"{self.failure}: {self.reason}"
        if self.did_you_mean is not None:
            text += f"; did you mean {self.did_you_mean}? {ROOT_RULE}"
        return instructed(text, self.instruction)


STALE_DO: Final = (
    "the file changed since you read it: take the anchors from the lines shown above, or read "
    "it again, then send the edit"
)
"""A line anchor that names no line of the text as it now reads: its one cause."""


@dataclass(frozen=True)
class _Change:
    """What one `apply` wrote, for the call's answer: the path, and the span of the new text
    that now holds the content; None for a whole file, or a file removed (`deleted`)."""

    path: str
    span: Region | None
    deleted: bool = False


DraftMode = Literal["edit", "create"]
"""How the draft holds a path: an existing file's text, or a new file absent until written."""

Opens = Callable[[str], DraftMode | DraftRefused]
"""Whether a path beyond the packet's targets may be taken on, and as what: `create` for a new
file, `edit` for one on disk; a refusal says why and what to do, in the rule owner's words.
Athena's verification draft is all such paths; a worker's has none."""

Refusal = Callable[[str], DraftRefused | None]
"""Why a path may not be written, with what to do instead, or None: the edit port's rules."""

Removable = Callable[[str], DraftRefused | None]
"""Why a file may not be deleted or moved, with what to do instead, or None when it may: the
box's undo check. A draft given none refuses both, as the packet loop's must.
"""

Meant = Callable[[str], str | None]
"""The path a refused one most likely meant, or None (`paths.suggest` over what `glob` lists):
what a refusal of a path that is not there names. The box's edit port gives it."""

Answered = Callable[[Mapping[str, bytes | None]], None]
"""Told each file an answered call changed, as it would be written now, in the order the draft
holds them, None for a file the call deleted: what the live draft stages on the worktree
(`orchestrator/live.py`), and what the box's edit port writes (`box/edits.py`)."""


class CandidateDraft:
    """`read` gives a path's text decoded, or None when it is missing or not UTF-8 text;
    `refusal` gives why a path may not be written whatever the targets declare (a deny-listed
    path, a test file without `test_authorship`: the Applier's own path gates), or None.
    `style` is the run's anchor style when a line tool is offered (`worker_tools.line_style`),
    else None. `targets` gives the writable paths, in order, and the deletes; `opens` admits
    more as they are written; `judge` gives a written file's syntax; `on_answered`
    hears what each answered call changed. `refuse_emptied` refuses an edit that leaves a
    non-empty file `emptied` (the box's edit tool, which no Applier gate follows); a new file
    may be written empty. `removable` lets a call delete or move a file it allows; without it
    both are refused. `languages` says what a symbol edit parses a path
    as; without it every symbol edit is `UNSUPPORTED`. `meant` names the path a refused one most
    likely meant."""

    def __init__(
        self,
        targets: Targets | None,
        read: Callable[[str], Decoded | None],
        refusal: Refusal,
        *,
        style: AnchorStyle | None = None,
        judge: Judge | None = None,
        edit_check: EditCheck = "reject",
        opens: Opens | None = None,
        on_answered: Answered | None = None,
        refuse_emptied: bool = False,
        removable: Removable | None = None,
        languages: Languages | None = None,
        meant: Meant | None = None,
    ) -> None:
        held = targets if targets is not None else Targets(writable=())
        self._read = read
        self._refusal = refusal
        self._style = style
        self._judge_file = judge
        self._edit_check: EditCheck = edit_check
        self._opens = opens
        self._on_answered = on_answered
        self._refuse_emptied = refuse_emptied
        self._removable = removable
        self._languages = languages
        self._meant = meant
        self._moves: dict[str, str] = {}
        """Target → source, for the call in flight and its answer: set by `move`."""
        self._order = list(held.writable)
        self._declared = frozenset(held.writable)
        self._deletes = held.deletes
        self._creates = set(held.creates)
        self._decoded: dict[str, Decoded] = {}
        self._unreadable: set[str] = set()
        self._texts: dict[str, str | None] = {}
        for path in self._order:
            self._hold(path, "create" if path in self._creates else "edit")
        self._changes: list[_Change] = []

    def _hold(self, path: str, mode: DraftMode) -> None:
        """`path` held: a create, absent until written; an edit, as `read` reads it."""
        if mode == "create":
            self._creates.add(path)
            self._texts[path] = None
            return
        decoded = self._read(path)
        if decoded is None:
            self._unreadable.add(path)
            return
        self._decoded[path] = decoded
        self._texts[path] = decoded.text

    @property
    def targets(self) -> Targets:
        return Targets(writable=tuple(self._order), deletes=self._deletes)

    @property
    def style(self) -> AnchorStyle | None:
        """The anchor style every view of a target is shown in; None, and they are plain."""
        return self._style

    def text(self, path: str) -> str | None:
        """A target's text now; None when the draft does not hold it: a create target not yet
        written, or a path that is not one of its targets."""
        return self._texts.get(path)

    def crlf(self, path: str) -> bool:
        """Whether `path` is held decoded from a uniformly `\r\n` file: what is written into
        it, or matched in it, is read as `edits/lines.as_decoded` reads it."""
        decoded = self._decoded.get(path)
        return decoded is not None and decoded.crlf

    def absent(self, path: str) -> bool:
        """Whether `path` is a create target not written yet."""
        return path in self._creates and self._texts.get(path) is None

    def readable(self, path: str) -> str | DraftRefused:
        """A target's text for an edit to anchor in, or why there is none."""
        refused = self._scope(path)
        if refused is not None:
            return refused
        current = self._texts[path]
        if current is None:
            return self._missing(path)
        return current

    def _missing(self, path: str) -> DraftRefused:
        """Why a held path has no text: a create not written yet, or a file deleted."""
        if path in self._creates and path in self._declared:
            return _absent(path)
        if path in self._creates:
            return self._meant_for(
                DraftRefused(
                    "FILE_ABSENT",
                    f"{path} does not exist",
                    do="write it whole with create first; check the path with glob",
                ),
                path,
            )
        return DraftRefused(
            "FILE_ABSENT",
            f"{path} was deleted in this call",
            do="write it whole with create to bring it back",
        )

    def _meant_for(self, refused: DraftRefused, path: str) -> DraftRefused:
        """`refused`, naming the path `path` most likely meant when `meant` knows one."""
        meant = self._meant(path) if self._meant is not None else None
        return (
            refused if meant is None or meant == path else replace(refused, did_you_mean=meant)
        )

    @property
    def moves(self) -> Mapping[str, str]:
        """The answered call's moves, target to source: what a writer needs to keep a moved
        file's mode. Read in `on_answered`; reset when the next call opens."""
        return dict(self._moves)

    def line(self, path: str, anchor: int | str, kind: AnchorKind) -> int | DraftRefused:
        """The line of `path` a line anchor names in the text as it now reads, or why not: a
        moved or missing line is `STALE_ANCHOR`, with the lines there now."""
        text = self.readable(path)
        if isinstance(text, DraftRefused):
            return text
        if self._style is None:
            raise HarnessBug(f"a line anchor on {path} in a draft with no anchor style")
        found = self._style.resolve(anchor, split_lines(text), kind)
        if isinstance(found, Stale):
            return DraftRefused("STALE_ANCHOR", f"{path}: {found.reason}", do=STALE_DO)
        return found

    def symbol(self, path: str, name: str) -> SymbolSpan | DraftRefused:
        """The one declaration of `path` that `name` names in the text as it now reads, so a
        symbol edited earlier in the attempt is found where it now is."""
        text = self.readable(path)
        if isinstance(text, DraftRefused):
            return text
        language = self._languages(path) if self._languages is not None else None
        found = find_symbol(path, text, name, language)
        if isinstance(found, SymbolMiss):
            return DraftRefused(found.reason, f"{path}: {found.detail}", do=found.do)
        return found

    def apply(
        self, path: str, region: Region | None, content: str | None
    ) -> DraftRefused | None:
        """The one write: scope, existence, elision, then the replacement. `region`
        is a span of the text as `readable` gave it; None writes the whole file. `content`
        None removes the file, when `removable` allows it: a delete is a write
        too, so the draft's text still changes in this one place."""
        refused = self._scope(path)
        if refused is not None:
            return refused
        current = self._texts[path]
        if content is None:
            refused = self._removable_here(path, region, current)
            if refused is not None:
                return refused
            self._texts[path] = None
            self._changes = [c for c in self._changes if c.path != path]
            self._changes.append(_Change(path, None, deleted=True))
            return None
        if current is None and region is not None:
            return self._missing(path)
        content = as_decoded(content, self.crlf(path))
        if region is not None and not 0 <= region.start <= region.end <= len(current or ""):
            raise HarnessBug(f"an edit region {region} outside {path}'s {len(current or '')}")
        old = (
            ""
            if current is None
            else current
            if region is None
            else current[region.start : region.end]
        )
        elided = find_elision(content, old)
        if elided is not None:
            return DraftRefused(
                "ELIDED",
                f"{path}: line {elided.line} stands in for code it leaves out "
                f"({elided.text.strip()!r})",
                do="send the code in full: write out every line the placeholder stood for",
            )
        if self._refuse_emptied and current is not None and not emptied(current):
            after = content if region is None else splice(current, region, content)
            if emptied(after):
                return DraftRefused(
                    "DENIED",
                    f"{path}: this would leave the file empty or whitespace only",
                    do="to remove the file, delete it; otherwise send what it should keep",
                )
        if region is None or current is None:
            self._texts[path] = content
            self._record(path, None, len(content))
            return None
        self._texts[path] = splice(current, region, content)
        self._record(path, region, len(content))
        return None

    def _removable_here(
        self, path: str, region: Region | None, current: str | None
    ) -> DraftRefused | None:
        """Why `apply` may not remove `path`: only a whole file that exists, and that
        `removable` lets go."""
        if region is not None:
            raise HarnessBug(f"a removal of {path} names a region")
        if current is None:
            return self._meant_for(
                DraftRefused(
                    "FILE_ABSENT",
                    f"{path} does not exist",
                    do="there is nothing to delete: check the path with glob",
                ),
                path,
            )
        if self._removable is None:
            return DraftRefused(
                "DENIED",
                f"{path}: deletes and moves are not offered here",
                do="leave the file where it is, and change only its text",
            )
        return self._removable(path)

    def move(self, source: str, target: str) -> DraftRefused | None:
        """`source` to `target`, its text carried whole: the source removed (`removable` asked)
        and the target created, which must not exist yet. Both in the call in flight, so a
        later refusal in the call undoes both."""
        text = self.readable(source)
        if isinstance(text, DraftRefused):
            return text
        if target == source:
            return DraftRefused(
                "DENIED",
                f"{source} is already there",
                do="name another target, or leave the file where it is",
            )
        refused = self._scope(target)
        if refused is not None:
            return refused
        if self._texts.get(target) is not None:
            return DraftRefused(
                "DENIED",
                f"{target} exists, and a move never replaces a file",
                do=f"delete {target} first, or move to a name that is free",
            )
        removed = self.apply(source, None, None)
        if removed is not None:
            return removed
        wrote = self.apply(target, None, text)
        if wrote is not None:
            return wrote
        decoded = self._decoded.get(source)
        if decoded is not None:
            # The target is written in the source's own newlines, as the file it is.
            self._decoded[target] = Decoded(text, decoded.crlf)
        self._moves[target] = source
        return None

    def _record(self, path: str, region: Region | None, written: int) -> None:
        """The call's changes once `region` of `path` holds `written` characters, so each is
        answered where it now is: an earlier change after it moves by the difference, one it
        overlaps grows to cover both, and a whole write stands for the path's earlier ones."""
        if region is None:
            self._changes = [c for c in self._changes if c.path != path]
            self._changes.append(_Change(path, None))
            return
        delta = written - (region.end - region.start)
        now = Region(region.start, region.start + written)
        changes: list[_Change] = []
        covered = False
        for change in self._changes:
            span = change.span
            if change.path != path or span is None or span.end < region.start:
                changes.append(change)
            elif span.start > region.end or (
                span.start == region.end and region.start < span.start
            ):
                changes.append(_Change(path, Region(span.start + delta, span.end + delta)))
            else:
                end = span.end + delta if span.end > region.end else now.end
                changes.append(
                    _Change(path, Region(min(span.start, now.start), max(end, now.end)))
                )
                covered = True
        if not covered:
            changes.append(_Change(path, now))
        self._changes = changes

    @contextmanager
    def call(self) -> Generator[_Call]:
        """One edit call: its changes stand when it ends answered, and are undone otherwise
        (a refusal, or an error)."""
        kept = dict(self._texts)
        self._changes = []
        self._moves = {}
        scope = _Call(self, kept)
        try:
            yield scope
        finally:
            if not scope.answered:
                self._restore(kept)
            self._changes = []
        if scope.answered and self._on_answered is not None:
            changed = {
                p: self.written(p)
                for p in self._order
                if p not in self._unreadable and self._texts.get(p) != self._before(p, kept)
            }
            if changed:
                self._on_answered(changed)

    def _before(self, path: str, kept: Mapping[str, str | None]) -> str | None:
        """`path`'s text as the call found it: as kept, or, for a file the call took on, as it
        was taken (absent if a create, its text on disk if an edit)."""
        if path in kept:
            return kept[path]
        return None if path in self._creates else self._decoded[path].text

    def _restore(self, kept: dict[str, str | None]) -> None:
        """The texts as the call found them. A file the call took on (D190's open scope) stays
        taken on, as it was taken: absent if a create, its text on disk if an edit."""
        for path in self._order:
            if path not in kept and path not in self._unreadable:
                kept[path] = None if path in self._creates else self._decoded[path].text
        self._texts = kept

    def answer(self, kept: dict[str, str | None]) -> str | DraftRefused:
        """`ok`, and each change of the call re-rendered as the whole lines it now spans, in the
        anchor style when there is one; a whole file is named with its size, not shown again.
        Then each changed file's syntax, judged against `kept`, the text before the call; or
        `SYNTAX_FAIL`, when the call broke a file and `reject` undoes it."""
        judged = self._judge(kept)
        if isinstance(judged, DraftRefused):
            return judged
        parts: list[str] = []
        moved_from = set(self._moves.values())
        for change in self._changes:
            if change.deleted:
                if change.path not in moved_from:
                    parts.append(f"ok: {change.path} deleted")
                continue
            text = self._texts[change.path] or ""
            source = self._moves.get(change.path)
            if source is not None and change.span is None:
                parts.append(f"ok: {source} moved to {change.path}")
                continue
            if change.span is None:
                parts.append(f"ok: {change.path} written, {len(split_lines(text))} lines")
                continue
            parts.append(f"ok: {change.path}\n{self._shown(text, change.span)}")
        return "\n".join([*parts, *judged] if parts else ["ok: nothing changed"])

    def _judge(self, kept: dict[str, str | None]) -> list[str] | DraftRefused:
        """Each changed file's syntax line, or the refusal that undoes the call: under
        `reject`, a file that parsed before the call (an unwritten create target counts as
        parsing) and fails after. One that failed before is reported, never refused for it, and
        so is a failure of a grammar's parse alone (`parsed_only`): a parse can fail where the
        compiler accepts the file (C's `#ifdef` around a brace), so it is a gap to report, never
        a refusal."""
        if self._judge_file is None or self._edit_check == "off":
            return []
        said: list[str] = []
        for path in dict.fromkeys(c.path for c in self._changes if not c.deleted):
            verdict = self._parse(path, self._texts[path] or "")
            if verdict is None:
                said.append(f"syntax unchecked: {path}")
                continue
            if verdict.ok:
                said.append(f"parsed: {path}" if verdict.parsed_only else f"syntax ok: {path}")
                continue
            detail = verdict.detail or f"{path}: syntax error"
            # A moved file was its source before the call, not a create that parsed.
            before = self._before(self._moves.get(path, path), kept)
            prior = self._parse(path, before) if before is not None else None
            was = before is None or (prior is not None and prior.ok)
            if verdict.parsed_only:
                said.append(f"parse error: {detail}" + ("" if was else ", as before this call"))
                continue
            if was and self._edit_check == "reject":
                return DraftRefused(
                    "SYNTAX_FAIL",
                    f"{detail}; the call is undone, every file as it was",
                    do="fix the syntax error named above and send the edit again",
                )
            said.append(f"syntax error: {detail}" + ("" if was else ", as before this call"))
        return said

    def _parse(self, path: str, text: str) -> SyntaxCheck | None:
        """The judge's verdict on `text` as the file would be written, its own newlines."""
        if self._judge_file is None:  # pragma: no cover - `_judge` asks only with one
            return None
        decoded = self._decoded.get(path)
        written = encode(text, decoded) if decoded is not None else text
        return self._judge_file(path, written.encode("utf-8"))

    def _shown(self, text: str, span: Region) -> str:
        """The whole lines of `text` that `span` touches, numbered in the style when there is
        one: an empty span, the line it sits in."""
        start, end = _lines_around(text, span)
        shown = split_lines(text[start:end])
        if not shown:
            return "(nothing remains there)"
        if self._style is None:
            return "\n".join(shown)
        first = text.count("\n", 0, start) + 1
        rows = self._style.rows(split_lines(text), (Span(first, first + len(shown) - 1),))
        return "\n".join(rows)

    def handed(self) -> FileEdits:
        """What the candidate hands in: each target it changed, whole, in target
        order and in the file's own newlines; a create target it wrote as a create."""
        edits: list[tuple[str, FileEdit]] = []
        for path in self._order:
            text = self._texts.get(path)
            if text is None:
                continue
            if path in self._creates:
                edits.append((path, Create(text)))
            elif text != self._decoded[path].text:
                edits.append((path, Whole(encode(text, self._decoded[path]))))
        return FileEdits(tuple(edits))

    def written(self, path: str) -> bytes | None:
        """`path` as the file would be written now, in its own newlines; None when the draft
        holds no text for it. What the live draft stages."""
        text = self._texts.get(path)
        if text is None:
            return None
        decoded = self._decoded.get(path)
        return (encode(text, decoded) if decoded is not None else text).encode("utf-8")

    def _scope(self, path: str) -> DraftRefused | None:
        """D172 §4 step 1 and §10: an `edit` or `create` target the run may write."""
        if path in self._deletes:
            return DraftRefused(
                "DENIED",
                f"{path} is a delete target: the applier removes it",
                do="leave it: nothing edits a file that is being deleted",
            )
        if path not in self._order and self._opens is not None:
            mode = self._opens(path)
            if isinstance(mode, DraftRefused):
                return self._meant_for(mode, path)
            refused = self._refusal(path)
            if refused is not None:
                return self._meant_for(refused, path)
            self._order.append(path)
            self._hold(path, mode)
        if path not in self._order:
            return DraftRefused(
                "DENIED",
                f"{path} is not a TARGETS path",
                do=(
                    f"change only the files this task may write ({', '.join(self._order)}), "
                    "or say in your report that it needs this one"
                ),
            )
        refused = self._refusal(path)
        if refused is not None:
            return refused
        if path in self._unreadable:
            return DraftRefused(
                "DENIED",
                f"{path} is not UTF-8 text",
                do="leave it: only text is edited here; if it must change, say so in your "
                "report",
            )
        return None


class _Call:
    """`DraftCall` over one draft (`tools/context.py`)."""

    def __init__(self, draft: CandidateDraft, kept: dict[str, str | None]) -> None:
        self._draft = draft
        self._kept = kept
        self.answered = False

    @property
    def targets(self) -> Targets:
        return self._draft.targets

    def text(self, path: str) -> str | DraftRefused:
        return self._draft.readable(path)

    def crlf(self, path: str) -> bool:
        return self._draft.crlf(path)

    def line(self, path: str, anchor: int | str, kind: AnchorKind) -> int | DraftRefused:
        return self._draft.line(path, anchor, kind)

    def symbol(self, path: str, name: str) -> SymbolSpan | DraftRefused:
        return self._draft.symbol(path, name)

    def apply(self, path: str, region: Region | None, content: str) -> DraftRefused | None:
        return self._draft.apply(path, region, content)

    def delete(self, path: str) -> DraftRefused | None:
        return self._draft.apply(path, None, None)

    def move(self, source: str, target: str) -> DraftRefused | None:
        return self._draft.move(source, target)

    def answer(self) -> str | DraftRefused:
        said = self._draft.answer(self._kept)
        # Refused, the call is not answered: its scope restores every file it changed.
        self.answered = not isinstance(said, DraftRefused)
        return said


def _absent(path: str) -> DraftRefused:
    return DraftRefused(
        "FILE_ABSENT",
        f"{path} is a create target not written yet",
        do="write it whole with create first, then edit it",
    )


def _lines_around(text: str, span: Region) -> tuple[int, int]:
    """Where the whole lines of `text` that `span` touches start and end, their last newline
    excluded: an empty span, the line it sits in."""
    start = text.rfind("\n", 0, span.start) + 1
    # Content ending its line ends there: the line after it is not part of the change.
    last = span.end - 1 if span.end > span.start and text[span.end - 1] == "\n" else span.end
    end = text.find("\n", last)
    return start, len(text) if end == -1 else end
