"""LanguageProfile — everything language-specific, so no component hardcodes a language.

The test for whether this part is doing its job: adding Go should be a new file in
`profiles/`, not a code change anywhere else.

**Templates are argv token lists, never shell strings.** They are split with
`shell_line.argv` at load and substituted per token afterwards, so a filename containing
`; rm -rf /` becomes one literal argument rather than a second command. This is the whole
reason `Template` is a type instead of a `str`: a string field would let any consumer
reintroduce shell interpolation, and the one that did would look identical to the ones
that did not.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, model_validator

from chickadee import shell_line
from chickadee.paths import glob_match

WIRE = ConfigDict(extra="forbid", frozen=True)


class ProfileError(ValueError):
    """A profile that cannot be used. Raised at load, never at run."""


class Template(BaseModel):
    """A command as parsed argv, plus the source text for reporting.

    Splitting happens once, at load. A template that cannot be split raises then rather
    than at the moment a gate needs to run, which would turn an operator's typo into a
    mid-run infra failure.
    """

    model_config = WIRE

    source: str
    argv: tuple[str, ...]

    @staticmethod
    def parse(source: str) -> Template:
        try:
            argv = shell_line.argv(source)
        except ValueError as exc:
            raise ProfileError(f"template {source!r} cannot be split: {exc}") from exc
        if not argv:
            raise ProfileError(f"template {source!r} is empty")
        return Template(source=source, argv=argv)

    def render(self, **values: str) -> tuple[str, ...]:
        """Substitute per token. A value is never re-split, so it stays one argument.

        `{file}` and `{selector}` are resolved by the caller against the leased execution
        worktree, never the user's checkout in worktree mode.
        """
        out: list[str] = []
        for token in self.argv:
            rendered = token
            for key, value in values.items():
                rendered = rendered.replace("{" + key + "}", value)
            out.append(rendered)
        return tuple(out)

    def placeholders(self) -> frozenset[str]:
        import re

        return frozenset(re.findall(r"\{([a-z_]+)\}", self.source))


def _as_template(value: object) -> object:
    """Accept a plain string from TOML and parse it into a Template."""
    if isinstance(value, str):
        return Template.parse(value)
    return value


def _table(data: object) -> dict[str, Any] | None:
    """Narrow a TOML table once, at the validator boundary.

    TOML is untyped the way JSON is; asserting the shape here is honest, and lets the rest
    of the module stay typed instead of propagating Unknown through every spec.
    """
    return cast("dict[str, Any]", data) if isinstance(data, dict) else None


class SyntaxSpec(BaseModel):
    """Gate A1, the check after every edit. `inproc` is the language's own compiler in process
    (Python's `compile()`); `cmd` spawns the profile's command; `parse` parses the file with the
    profile's tree-sitter grammar in process and fails on the first `ERROR` or `MISSING` node:
    no toolchain and no spawn, and weaker than a compiler, so its pass says
    `parsed`, never more.

    `mode` has no default: Python's `compile()` as a default judged any profile that left the
    table out as Python. A profile's TOML states it, or the loader refuses the file."""

    model_config = WIRE

    mode: Literal["inproc", "cmd", "parse"]
    cmd: Template | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_cmd(cls, data: object) -> object:
        table = _table(data)
        if table is not None and "cmd" in table:
            return {**table, "cmd": _as_template(table["cmd"])}
        return data

    @model_validator(mode="after")
    def _cmd_matches_mode(self) -> SyntaxSpec:
        if self.mode == "cmd" and self.cmd is None:
            raise ProfileError('syntax.mode = "cmd" requires syntax.cmd')
        if self.mode != "cmd" and self.cmd is not None:
            raise ProfileError(f'syntax.mode = "{self.mode}" must not carry syntax.cmd')
        return self


class TestSpec(BaseModel):
    model_config = WIRE

    globs: tuple[str, ...] = ()
    """What a test file is in this language: the applier's D20 lockdown, the router's row 7,
    the plan linter and the scout's snapshot all ask `matches`, so none can disagree."""
    runner: Template | None = None
    timeout_s: int = 120

    def matches(self, path: str) -> bool:
        """Whether `path` is a test file: the one answer, through the one matcher."""
        return glob_match(path, self.globs) is not None

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is not None and "runner" in table:
            return {**table, "runner": _as_template(table["runner"])}
        return data


def _patterns(value: object) -> object:
    """A pattern field as a tuple: one pattern or a list of them, each compiled once here so a
    bad one refuses at load, naming itself."""
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return value
    found = tuple(cast("list[object] | tuple[object, ...]", value))
    for pattern in found:
        if isinstance(pattern, str):
            try:
                re.compile(pattern, re.M)
            except re.error as exc:
                raise ProfileError(f"pattern {pattern!r} does not compile: {exc}") from exc
    return found


class SummarySpec(BaseModel):
    """One kind of summary line a test runner prints: `line` finds each, and
    the counts are read within it. A count pattern with a group sums its numbers; one without
    counts its matches. With no `passed`, it is `total` less the failed, errors and skipped."""

    model_config = WIRE

    line: str
    passed: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    total: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is None:
            return data
        _patterns(table.get("line"))
        return {k: v if k == "line" else _patterns(v) for k, v in table.items()}


class RunnerNames(BaseModel):
    model_config = WIRE

    failing: tuple[str, ...] = ()
    passing: tuple[str, ...] = ()
    broken: tuple[str, ...] = ()
    """With no summary, a line saying the suite could not run (a compile error)."""
    file: str | None = None
    """A line naming the file the names after it belong to (jest's `FAIL t/a.test.js`): each
    such name is read as `file > name`."""

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is None:
            return data
        _patterns(table.get("file"))
        return {k: v if k == "file" else _patterns(v) for k, v in table.items()}


class RunnerCommand(BaseModel):
    """How a runner is run, for the cautious selection: the words that run
    it, the flags that select nothing (with no value, or one), and what its arguments select.
    Any other flag makes a run's selection unknown."""

    model_config = WIRE

    words: tuple[str, ...]
    selects: Literal["paths", "names"]
    """`paths`: files or `path::id`s (pytest), or substrings of a file's path (jest, vitest);
    `names`: substrings of a test's name (cargo)."""
    flags: tuple[str, ...] = ()
    values: tuple[str, ...] = ()


class TestRunner(BaseModel):
    """A test runner's output as data (`profiles/runners/`): how a run is read
    by what it printed, whatever ran it. One file a runner, not a language: jest serves
    JavaScript and TypeScript."""

    __test__ = False
    model_config = WIRE

    name: str
    unit: Literal["test", "package"] = "test"
    summary: tuple[SummarySpec, ...]
    names: RunnerNames = RunnerNames()
    command: RunnerCommand | None = None
    """None: no run of it has a selection the harness reads."""


class BuildSpec(BaseModel):
    model_config = WIRE

    compile_cmd: Template | None = None
    import_smoke: Template | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is None:
            return data
        # An empty string in TOML means "this language has no compile step", which is a
        # declaration rather than an omission; it must not become an unsplittable template.
        return {k: (None if v == "" else _as_template(v)) for k, v in table.items()}


class DepsSpec(BaseModel):
    model_config = WIRE

    manifests: tuple[str, ...] = ()
    """Gate A3(b) diff targets — where a new dependency would show up."""
    install_cmd: Template | None = None
    """A template an action packet may use. Nothing is installed by this part."""

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is not None and "install_cmd" in table:
            return {**table, "install_cmd": _as_template(table["install_cmd"])}
        return data


class ExamplesSpec(BaseModel):
    model_config = WIRE

    snippet: str = ""
    """Prose, not a command — never split, never executed here. Examples may specify
    intent; only independent checks prove behavior."""


class ConventionsSpec(BaseModel):
    model_config = WIRE

    file: str | None = None


class LspSpec(BaseModel):
    model_config = WIRE

    server: Template | None = None
    """LSP-backed edges are labelled `resolved`; without a server they are `fallback`
    or `unknown`, never silently promoted. The index asks it at index time, sandboxed
    (`graph/lsp.CheckerResolver`)."""
    language_id: tuple[tuple[str, str], ...] = ()
    """The LSP `languageId` of each extension, one `(extension, id)` pair per extension, loaded
    from one name or a table as `tree_sitter_grammar` is (`{".jsx" = "javascriptreact"}`).
    Empty is the profile's name for every extension. Ask `LanguageProfile.language_id_for`."""
    install: Template | None = None
    """One command that installs the server into `{dir}` (its executables in `{dir}/bin`),
    Chickadee's servers directory: `chickadee setup` runs it when the person says yes, never
    on its own (`init/servers.py`). None where no one command does it portably."""
    settings: str = "{}"
    """The client's settings for the server, one nested table (`[lsp.settings.basedpyright.
    analysis]`), held as its canonical JSON so the profile stays a hashable value: the client's
    defaults, which a repository's own config file for that server overrides. Ask
    `configuration`."""

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is None:
            return data
        out = dict(table)
        for field in ("server", "install"):
            if field in out:
                out[field] = _as_template(out[field])
        if isinstance(out.get("settings"), dict):
            out["settings"] = json.dumps(out["settings"], sort_keys=True)
        return out

    def configuration(self, section: str) -> object:
        """The answer to `workspace/configuration` for `section`, a dotted path into the
        settings (`basedpyright`, `basedpyright.analysis`); None, the server's own default, for
        one not configured."""
        found: object = json.loads(self.settings)
        for part in section.split(".") if section else ():
            if not isinstance(found, dict):
                return None
            found = cast("dict[str, object]", found).get(part)
        return found


class LintSpec(BaseModel):
    """`[lint]`: the language's linter, as a server kept beside its language server.
    It reads the repository's own configuration, or runs with its defaults where there is none;
    not installed, it is no source, and its findings are absent, never clean."""

    model_config = WIRE

    server: Template | None = None
    """A server that answers `textDocument/diagnostic` when asked (`ruff server`): the only
    form the client reads."""

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is not None and "server" in table:
            return {**table, "server": _as_template(table["server"])}
        return data


class ToolchainSpec(BaseModel):
    """`[toolchain]` — what this language's toolchain needs from the harness."""

    model_config = WIRE

    bootstrap: Template | None = None
    """The argv that installs the lockfile into a fresh worktree — `uv sync
    --frozen`, `npm ci`. Runs **once, at worktree creation**, as the episode's first
    operation, through Isolation + ProcessSupervisor with the network its class allows.
    Optional: absent, the worktree has the checkout and nothing else.

    **An argv list in TOML, never a shell string.** Every other template here is a string
    that `shell_line.argv` turns into argv at load; this one runs with the network up and is
    the one command a *project* profile can put in front of the harness, so its form is
    the list and a string refuses the load naming the list form (the D104 rule: a shape
    that cannot carry a shell is the shape the field has). `[]` declares "no bootstrap",
    the way `compile_cmd = ""` declares no compile step.

    A project override of this field is executable config: inert until
    a person accepts the trust digest. The shipped profiles' bootstraps are the harness's
    own and need no acceptance."""

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: object) -> object:
        table = _table(data)
        if table is None or "bootstrap" not in table:
            return data
        raw = table["bootstrap"]
        if raw is None or isinstance(raw, Template | dict):
            return data  # absent, or an already-parsed template round-tripping
        if isinstance(raw, str):
            raise ProfileError(
                f"[toolchain].bootstrap must be an argv list such as "
                f'["uv", "sync", "--frozen"], not the string {raw!r} (D109: a command '
                "that runs with the network up is never a shell string)"
            )
        if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):  # pyright: ignore[reportUnknownVariableType]
            raise ProfileError(f"[toolchain].bootstrap must be a list of strings, got {raw!r}")
        argv = tuple(cast("list[str]", raw))
        if not argv:
            return {**table, "bootstrap": None}
        if any(not token for token in argv):
            raise ProfileError(f"[toolchain].bootstrap has an empty token: {list(argv)}")
        return {**table, "bootstrap": Template(source=shell_line.join(argv), argv=argv)}


def _per_extension(raw: object, extensions: list[object] | tuple[object, ...]) -> object:
    """One value for every extension, or a table of extension to value, as one
    `(extension, value)` pair per extension in `extensions`' order, whatever order the table
    was written in. A key that is not an extension sorts last, and `_check_per_extension`
    names it. Anything else passes as it is, for the field's own validation."""
    if isinstance(raw, str):
        return tuple((e, raw) for e in extensions)
    if isinstance(raw, dict):
        order = {e: i for i, e in enumerate(extensions) if isinstance(e, str)}
        pairs = cast("dict[str, object]", raw).items()
        return tuple(sorted(pairs, key=lambda kv: order.get(kv[0], len(order))))
    return raw


def _check_per_extension(
    profile: LanguageProfile, field: str, pairs: tuple[tuple[str, str], ...]
) -> None:
    """A per-extension field names one non-empty value for each of the profile's extensions
    and no other: one naming an extension the profile does not own, or none for one it does,
    leaves a file nothing reads or a value no file reaches."""
    named = [ext for ext, _v in pairs]
    if sorted(named) != sorted(profile.extensions):
        missing = sorted(set(profile.extensions) - set(named))
        extra = sorted(set(named) - set(profile.extensions))
        raise ProfileError(
            f"profile {profile.name!r}: {field} must name one value for each of its "
            f"extensions {list(profile.extensions)}"
            + (f"; none for {missing}" if missing else "")
            + (f"; {extra} are not its extensions" if extra else "")
        )
    empty = [ext for ext, v in pairs if not v]
    if empty:
        raise ProfileError(f"profile {profile.name!r}: {field} is empty for {empty}")


class LanguageProfile(BaseModel):
    model_config = WIRE

    name: str
    extensions: tuple[str, ...]
    tree_sitter_grammar: tuple[tuple[str, str], ...]
    """The grammar that parses each extension, one `(extension, grammar)` pair per extension in
    `extensions`' order. The TOML names one grammar for every extension
    (`"python"`) or a table where one language has two (`{".ts" = "typescript", ".tsx" =
    "tsx"}`); both load into these pairs, which hash, unlike a dict, so a profile stays a value.
    Ask `grammar_for`, never this field."""

    syntax: SyntaxSpec | None = None
    """None only for a profile built in code with no check: nothing judges its files (`syntax
    unchecked`), never a default mode. A TOML profile must state `[syntax].mode`
    (`loader._validate`)."""
    tests: TestSpec = TestSpec()
    build: BuildSpec = BuildSpec()
    deps: DepsSpec = DepsSpec()
    examples: ExamplesSpec = ExamplesSpec()
    conventions: ConventionsSpec = ConventionsSpec()
    lsp: LspSpec = LspSpec()
    lint: LintSpec = LintSpec()
    toolchain: ToolchainSpec = ToolchainSpec()

    def owns(self, path: str) -> bool:
        """Whether `path` is a file of this profile's language, by its extension: what decides
        which parser, compiler or import may judge it. A README is not Python."""
        return PurePosixPath(path).suffix in self.extensions

    def grammar_for(self, path: str) -> str | None:
        """The tree-sitter grammar that parses `path`, or None when this profile does not own
        it. One language may need two: TypeScript's `.tsx` is not its `.ts`."""
        suffix = PurePosixPath(path).suffix
        return next((g for ext, g in self.tree_sitter_grammar if ext == suffix), None)

    def language_id_for(self, path: str) -> str | None:
        """The LSP `languageId` of `path`, or None when this profile does not own it: its
        `[lsp].language_id`, else the profile's name."""
        if not self.owns(path):
            return None
        suffix = PurePosixPath(path).suffix
        named = next((i for ext, i in self.lsp.language_id if ext == suffix), None)
        return named or self.name

    def grammars(self) -> tuple[str, ...]:
        """Every grammar this profile parses with, each once, in `extensions`' order."""
        return tuple(dict.fromkeys(g for _ext, g in self.tree_sitter_grammar))

    @model_validator(mode="before")
    @classmethod
    def _per_extension_fields(cls, data: object) -> object:
        """`tree_sitter_grammar` and `[lsp].language_id` as their pairs, from one name or a
        table, through one normaliser (`_per_extension`). Already pairs (a profile
        round-tripping through `model_dump`) passes as it is; either way
        `_extensions_are_usable` checks the pairs against `extensions`."""
        table = _table(data)
        if table is None:
            return data
        extensions = table.get("extensions")
        if not isinstance(extensions, list | tuple):
            return data  # the field's own validation names what is wrong
        named = cast("list[object] | tuple[object, ...]", extensions)
        out = dict(table)
        if "tree_sitter_grammar" in out:
            out["tree_sitter_grammar"] = _per_extension(out["tree_sitter_grammar"], named)
        lsp = _table(out.get("lsp"))
        if lsp is not None and "language_id" in lsp:
            out["lsp"] = {**lsp, "language_id": _per_extension(lsp["language_id"], named)}
        return out

    @model_validator(mode="before")
    @classmethod
    def _gates_retired(cls, data: object) -> object:
        """`[gates]` named the packet loop's checks A4 and A5, which retired with the verifier:
        refused with that reason, never loaded and ignored."""
        table = _table(data)
        if table is not None and "gates" in table:
            raise ProfileError(
                "[gates] is no longer read: the gates retired with the packet loop (D196). "
                "Remove it; the task worker runs a linter or a type checker through its shell"
            )
        return data

    @model_validator(mode="after")
    def _extensions_are_usable(self) -> LanguageProfile:
        if not self.extensions:
            raise ProfileError(f"profile {self.name!r} declares no extensions")
        bad = [e for e in self.extensions if not e.startswith(".")]
        if bad:
            raise ProfileError(f"extensions must start with a dot, got {bad}")
        _check_per_extension(self, "tree_sitter_grammar", self.tree_sitter_grammar)
        if self.lsp.language_id:
            _check_per_extension(self, "[lsp].language_id", self.lsp.language_id)
        return self
