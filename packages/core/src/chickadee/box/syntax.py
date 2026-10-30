"""A written file's syntax check: gate A1's check, moved and given every profile.

Each file is judged by the profile of its own language, so an episode that writes Python and
JavaScript judges each by its own rule . In process where the profile says so
(`compile` for Python, or its grammar's parse, `syntax.mode = "parse"`); else
the profile's command, through `spawn`. A file no profile owns, or
one whose check needs a process when no `spawn` was given, is not judged: None, which the draft
says as `syntax unchecked: <path>`.

The edit port judges each written file by it (`box/edits.WorktreeEdits`), putting the call's
bytes in place for a check that is a command (`on_disk`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from chickadee.box.digest import build_digest
from chickadee.edits.draft import SyntaxCheck
from chickadee.executor.types import ExecResult
from chickadee.graph.grammars import GRAMMARS, first_error
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile
from chickadee.profiles.resolve import FileLanguage

__all__ = ["Spawn", "SyntaxChecker"]

Spawn = Callable[[tuple[str, ...], Path, int], ExecResult]
"""A sandboxed run of the harness's own argv, in a directory, under a timeout in seconds:
`box/spawn.run_argv` over the engine's `HarnessSpawn`."""


class SyntaxChecker:
    """The syntax of one written file, by the profile of `profiles` that owns its path.
    `languages` is the set the index read the workspace by: a path's language is its answer,
    judged only when that language is one of `profiles`, so a file the index calls a gap (two
    profiles, no manifest to pick one) is not judged by whichever of the two was detected.
    Without it, `profiles` alone answer."""

    def __init__(
        self,
        root: Path,
        profiles: Sequence[LanguageProfile],
        spawn: Spawn | None = None,
        *,
        timeout_s: int = 120,
        languages: ProfileSet | None = None,
    ) -> None:
        self._root = root
        self._languages = (
            languages if languages is not None else ProfileSet.of(profiles, repo=root)
        )
        self._judges = frozenset(p.name for p in profiles)
        self._spawn = spawn
        self._timeout_s = timeout_s
        self._judged: dict[tuple[str, str], SyntaxCheck | None] = {}
        """Each verdict by path and the sha of its bytes: the facts judge again the
        files the edit tool judged at its write, and a `cmd` check is a process per file. One
        episode's checker serves one call at a time, on its thread."""

    def language_of(self, path: str) -> FileLanguage | None:
        """`path`'s language and grammar, or None: a README or a config file is not code. The
        profile set's answer, which the draft's symbol edits ask too."""
        language = self._languages.language_of(path)
        return language if language is not None and language.name in self._judges else None

    def profile_for(self, path: str) -> LanguageProfile | None:
        """The profile that owns `path`, or None."""
        language = self.language_of(path)
        return language.profile if language is not None else None

    def on_disk(self, path: str) -> LanguageProfile | None:
        """The profile whose check of `path` is a command, which reads the file on disk, when
        there is a spawn to run it; None when the check needs no file or nothing runs one."""
        profile = self.profile_for(path)
        if profile is None or _mode(profile) != "cmd" or self._spawn is None:
            return None
        return profile

    def check(self, path: str, content: bytes) -> SyntaxCheck | None:
        """`content`'s verdict as `path` would be written: in process where its profile says
        so, else its command through `spawn`, over the file on disk (`on_disk`), so `content`
        must be what is there: gate A1's case, after the Applier wrote. None when nothing
        judges it."""
        key = (path, hashlib.sha256(content).hexdigest())
        if key not in self._judged:
            self._judged[key] = self._check(path, content)
        return self._judged[key]

    def _check(self, path: str, content: bytes) -> SyntaxCheck | None:
        judged = self.inproc(path, content)
        if judged is not None:
            return judged
        profile = self.profile_for(path)
        if profile is None or _mode(profile) != "cmd" or self._spawn is None:
            return None
        return self.command(path, profile)

    def inproc(self, path: str, content: bytes) -> SyntaxCheck | None:
        """The verdict without a process, or None when it needs one (a command-mode profile)
        or no profile owns `path`. What the packet worker's draft asks after each edit call:
        it never spawns, so the packet edit tools stay `effects = "none"`."""
        language = self.language_of(path)
        if language is None:
            return None
        mode = _mode(language.profile)
        if mode == "parse":
            # `init` refuses a profile naming a grammar the harness lacks (`missing_grammars`);
            # a repository override can still name one, and that file is unchecked.
            return (
                _parse(path, content, language.grammar)
                if language.grammar in GRAMMARS
                else None
            )
        if mode != "inproc":
            return None
        return self._compile(path, content)

    def command(self, path: str, profile: LanguageProfile) -> SyntaxCheck:
        """The profile's command over `path` as it is on disk, through `spawn`. The caller
        has checked that the profile has a command and that there is a `spawn`."""
        cmd = profile.syntax.cmd if profile.syntax is not None else None
        if cmd is None or self._spawn is None:
            raise ValueError(f"profile {profile.name!r} has no syntax command to spawn")
        result = self._spawn(cmd.render(file=path), self._root, self._timeout_s)
        if result.timed_out:
            return SyntaxCheck(False, "timeout", "command")
        if result.exit_code != 0:
            return SyntaxCheck(
                False, build_digest(result.stderr or result.stdout).text, "command"
            )
        return SyntaxCheck(True, None, "command")

    def _compile(self, path: str, source: bytes) -> SyntaxCheck:
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return SyntaxCheck(False, f"{path}: {exc}", "compiled")
        try:
            compile(text, str(self._root / path), "exec")
        except SyntaxError as exc:
            msg = exc.msg or "syntax error"
            where = f"{path}:{exc.lineno}" if exc.lineno else path
            return SyntaxCheck(False, f"{where}: {msg}", "compiled")
        except ValueError as exc:
            # NUL bytes and similar are not SyntaxError; they must still fail, not crash: the
            # applier asks this before a write, and an uncaught exception skips its rollback.
            return SyntaxCheck(False, f"{path}: {exc}", "compiled")
        except (RecursionError, MemoryError) as exc:
            # Source nested or chained past the parser's stack (`-` * 200_000, a 300k-long
            # attribute chain): Python refuses to compile it, so it fails, never crashes. The
            # draft asks this after every edit call, the applier before every write.
            return SyntaxCheck(
                False, f"{path}: too complex to compile ({type(exc).__name__})", "compiled"
            )
        return SyntaxCheck(True, None, "compiled")


def _mode(profile: LanguageProfile) -> str | None:
    """The profile's check, or None when it states none (a profile built in code)."""
    return profile.syntax.mode if profile.syntax is not None else None


def _parse(path: str, source: bytes, grammar: str) -> SyntaxCheck:
    """`source` parsed by `grammar`: the first `ERROR` or `MISSING` node fails it, with its
    line and column. A pass is `parsed`, never a compile's."""
    error = first_error(grammar, source)
    if error is not None:
        return SyntaxCheck(False, error.said(path), "parsed")
    return SyntaxCheck(True, None, "parsed")
