"""What a shell line runs, read without running it: its words, the simple commands in it, and
whether it is plain.

One home for taking a command line apart. A line is parsed by tree-sitter-bash,
the grammar Codex and OpenCode read commands with, in process; nothing here evaluates anything:
no expansion, no globbing, no variable is read. A line is text a model wrote, and this
only says what it names.

- **The commands.** Every simple command anywhere in the tree: in a list or a pipeline, `{ }`,
  a subshell, an `if`/`while`/`for`/`case` body, a function body, `$( )`, backticks and process
  substitution. A command can also run inside another: a `sh -c` body, a heredoc or a
  here-string fed to a shell (behind a wrapper too: `sudo sh <<EOF`), an `eval` string, an
  `env -S` string, the command a wrapper starts (`sudo rm x`, `xargs rm`, `timeout 5 curl …`,
  `find. -exec rm {};`) and a `git -c alias.x=!…` alias.
  Each is a simple command of its own, so a line cannot hide one from the table by putting it
  in second place, in a block, or behind a wrapper.
- **Plain** (Codex's word-only rule): every node in the tree is one of `_PLAIN_KINDS`, every
  operator is `;`, `&&`, `||` or `|`, and every word is literal (`_literal`): the shell would
  run exactly the words the text shows. A redirect, a heredoc, an assignment, a compound
  command, a function, a substitution, an expansion, a comment, an `ERROR` node, `eval`,
  `source`, `.`, `xargs`, an `env` with anything before the command it starts, and a line over
  `PLAIN_MAX_CHARS` are not plain. Only a plain line can
  be covered by a grant (`approval/commands.py`); a line that is not plain is still read, for
  its tier.
- **The dialect.** The executor runs a line as `/bin/sh -c`, which is dash on Debian, while the
  grammar is bash's. Bash-only syntax parses, but none of it is plain; inside the plain subset
  the two shells agree, and `tests/approval/test_classifier_against_sh.py` checks that against
  the real `/bin/sh`.
"""

from __future__ import annotations

import functools
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import tree_sitter_bash
from tree_sitter import Language, Node, Parser

from chickadee.paths import glob_match

__all__ = [
    "GIT_GLOBAL_OPTIONS",
    "PLAIN_MAX_CHARS",
    "PRINTERS",
    "RUNS_ITS_ARGUMENT",
    "LineReading",
    "argv",
    "exact_words",
    "first_operand",
    "join",
    "may_be",
    "read_argv",
    "read_line",
    "runs_its_body",
    "split",
]

PLAIN_MAX_CHARS = 10_000
"""A longer line is never plain, so no grant covers it (Claude Code's bound); it is still
read."""

_PLAIN_KINDS = frozenset(
    {
        "program",
        "list",
        "pipeline",
        "command",
        "command_name",
        "word",
        "string",
        "string_content",
        "raw_string",
        "number",
        "concatenation",
    }
)
"""Codex's allowlist of named node types for a word-only command sequence (`bash.rs`)."""

_PLAIN_TOKENS = frozenset({"&&", "||", ";", "|", '"', "'", "\n"})

_NOT_LITERAL = frozenset("{}*?[]\\~^#$`")
"""A word holding one of these may be rewritten by the shell before it runs (a brace, a glob, a
tilde, an escape, an expansion), so its text is not proof of the argv (Codex's
`is_literal_word_or_number`). A backslash makes a word not literal rather than being undone:
the rule never has to reproduce the shell's escaping."""

_RUNS_A_STRING = frozenset({"eval"})
_RUNS_A_FILE = frozenset({"source", "."})
_STDIN_ARGUMENTS = frozenset({"xargs"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_FIND_RUNS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_SHELL_VALUE_OPTIONS = frozenset({"-o", "+o", "-O", "+O", "--rcfile", "--init-file"})
"""A shell's options that take a separate value, so the value is not read as its `-c` body."""
_GIT_SHELL_ALIAS = re.compile(r"^alias\.[^=]+=!(.*)$", re.DOTALL)
_GIT_COMMAND_KEY = re.compile(
    r"^(core\.(sshcommand|pager|editor|fsmonitor|askpass)|diff\.external"
    r"|diff\.[^.=]+\.(command|textconv)|merge\.[^.=]+\.driver|sequence\.editor"
    r"|gpg\.program|gpg\.[^.=]+\.program|filter\.[^.=]+\.(clean|smudge|process)"
    r"|credential\.helper|uploadpack\.packobjectshook)=(.*)$",
    re.DOTALL | re.IGNORECASE,
)
"""A `git -c` key whose value is a command git runs (`core.sshCommand`, `core.pager`, a diff or
merge driver, a filter), beside the `!` alias: the value is read as a line."""
_PROGRAM_OPTIONS: Mapping[str, frozenset[str]] = {
    "rg": frozenset({"--pre"}),
    "sort": frozenset({"--compress-program"}),
    "tar": frozenset({"--use-compress-program", "--to-command"}),
    "rsync": frozenset({"--rsh"}),
}
"""Options whose value is a program the command runs (`rg --pre=rm` runs `rm` on every file it
searches), read as a line. A long option's abbreviation counts, as GNU `getopt_long` takes
it."""
GIT_GLOBAL_OPTIONS: Final = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
)
"""git's options that may come before its subcommand and take a separate value: `git -C sub
push` is `git push`. Without them every flag written first would hide the subcommand."""
_GIT_PROGRAM_OPTIONS: Mapping[str, frozenset[str]] = {
    "rebase": frozenset({"-x", "--exec"}),
    "difftool": frozenset({"-x", "--extcmd"}),
    "filter-branch": frozenset(
        {
            "--env-filter",
            "--tree-filter",
            "--index-filter",
            "--parent-filter",
            "--msg-filter",
            "--commit-filter",
            "--tag-name-filter",
        }
    ),
}
"""git subcommands' options whose value is a line git runs:
`git rebase -x "echo x > a.py" main` runs the `echo` after every commit it replays."""
_GIT_RUNS_THE_REST: Final = frozenset({("bisect", "run"), ("submodule", "foreach")})
"""git subcommands whose remaining words are a command git runs: `git bisect run make test`."""
_AWKS = frozenset({"awk", "gawk", "mawk", "nawk"})
_AWK_RUNS = re.compile(
    r'system\s*\(\s*"((?:[^"\\]|\\.)*)"|\|&?\s*"((?:[^"\\]|\\.)*)"'
    r'|"((?:[^"\\]|\\.)*)"\s*\|&?\s*getline'
)
"""The commands an awk program runs: `system("…")`, `print | "…"` and `"…" | getline`."""
_BRACE = re.compile(r"(?<!\$)\{([^{}]*,[^{}]*)\}")
_GLOB = frozenset("*?[")
_SUBSTITUTION = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_MAX_DEPTH = 8
"""How deep a line may hide a line (`sh -c "sh -c …"`); deeper is not plain and not read."""

_WRAPPERS: Mapping[str, frozenset[str]] = {
    # The command each one starts follows its own options. The set is the options that take
    # a separate value, so the value is not read as the command.
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-U", "-r", "-t", "-T"}),
    "doas": frozenset({"-u", "-C"}),
    "env": frozenset({"-u", "-C", "-S", "--unset", "--chdir", "--split-string"}),
    "xargs": frozenset({"-I", "-i", "-n", "-P", "-L", "-l", "-d", "-E", "-e", "-s", "-a"}),
    "nice": frozenset({"-n"}),
    "timeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
    "nohup": frozenset(),
    "time": frozenset({"-f", "-o"}),
    "exec": frozenset({"-a"}),
    "command": frozenset(),
    "stdbuf": frozenset({"-i", "-o", "-e"}),
}
"""Heads that start another command. `sudo` and `doas` are rows of their own as well
(`scope_expanding`); `env` is one only when it starts nothing, because then it prints the
environment. An `env` with an assignment or an option before its command is not plain: the
command runs with an environment or in a directory the text does not show it in, which is why
`GIT_SSH_COMMAND=… git fetch` is not plain either."""

RUNS_ITS_ARGUMENT = frozenset({"nice", "timeout", "nohup", "time", "exec", "command", "stdbuf"})
"""Wrappers that do nothing themselves but run their argument, so their own part is no command
for the table to name: `timeout 30 git push origin` is the `git push` it starts."""

_QUERIES: Mapping[str, frozenset[str]] = {
    "command": frozenset({"-v", "-V"}),
    "sudo": frozenset({"-l", "-v", "-k", "-K", "-V"}),
}
"""A wrapper's options that make it ask about a command rather than start one:
`command -v rm` looks `rm` up, `sudo -l rm` lists whether it may run."""


@dataclass(frozen=True)
class LineReading:
    """A line's simple commands, in order, wrapped and nested ones included, and whether the
    line is plain."""

    commands: tuple[tuple[str, ...], ...]
    plain: bool
    writes_to: tuple[str, ...] = ()
    """Every file an output redirect names (`> f`, `>> f`, `&> f`), nested lines included, as
    the shell would pass it after quote removal; a descriptor duplication (`2>&1`) and an
    input redirect name none."""
    whole: bool = True
    """Every part of the line was read: no text the grammar could not parse, and no nesting
    deeper than the reading follows. Not `plain`: `git switch x 2>/dev/null` is read whole
    and is not plain. A line not read whole may run a command `commands` does not hold."""

    @property
    def programs(self) -> tuple[str, ...]:
        """The program each command runs, by its file's name (`/usr/bin/git` is `git`)."""
        return tuple(_head(c) for c in self.commands if c)


@functools.cache
def _language() -> Language:
    return Language(tree_sitter_bash.language())


def _parse(line: str) -> Node:
    # A parser per call: one is cheap to make, and a shared one would need a lock.
    return Parser(_language()).parse(line.encode("utf-8")).root_node


def read_line(line: str) -> LineReading:
    """Every simple command `line` runs, and whether it is plain. Never raises: a line the
    grammar cannot read is not plain, and the commands it could read still count."""
    reading = _Reading()
    reading.line(line, depth=0)
    return LineReading(
        tuple(reading.commands), reading.plain, tuple(reading.writes_to), reading.whole
    )


PRINTERS: Final = frozenset({"echo", "printf", "true", ":"})
"""Commands that change nothing and check nothing: they print, or succeed."""


def prints_only(line: str) -> bool:
    """Whether `line` only prints a message (`echo "task completed"`): plain, so no redirect
    or substitution writes or runs anything, and every command in it is a printer. A worker
    that ends its task this way loops instead of replying (the owner's REPL run)."""
    reading = read_line(line)
    return (
        reading.plain
        and bool(reading.commands)
        and all(c and c[0].rsplit("/", 1)[-1] in PRINTERS for c in reading.commands)
    )


def status_command(line: str) -> tuple[str, ...] | None:
    """The simple command whose exit status is `line`'s own, as its words; None when no one
    command's is. After `&&`, `;` or a newline it is the last command's; a
    pipeline's is its last command's (`/bin/sh` has no `pipefail`), so `pytest | tail` is
    `tail`'s; a redirect changes nothing. A `||`, a negation, a background `&`, a compound
    command or a line the grammar cannot read has none: its status may be any of several."""
    root = _parse(line)
    if root.has_error:
        return None
    return _status_of(root)


def _status_of(node: Node) -> tuple[str, ...] | None:
    kind = node.type
    if kind in ("program", "list"):
        parts = [c for c in node.children if c.type != "comment"]
        if not parts or any(c.type in ("||", "&") for c in parts):
            return None
        last = [c for c in parts if c.is_named]
        return _status_of(last[-1]) if last else None
    if kind == "pipeline":
        return _status_of(node.named_children[-1]) if node.named_children else None
    if kind == "redirected_statement":
        return _status_of(node.named_children[0]) if node.named_children else None
    if kind == "command":
        return _argv_of(node) or None
    return None


def read_argv(argv: Sequence[str]) -> LineReading:
    """`argv` as the executor starts it, with no shell between: itself and every command it
    starts, and whether that is plain. Its own words are literal, since no shell rewrites
    them; what it starts may not be (a `sh -c` body)."""
    reading = _Reading()
    reading.command(tuple(argv), depth=0, scan_words=True)
    return LineReading(tuple(reading.commands), reading.plain)


def argv(line: str) -> tuple[str, ...]:
    """A declared command line as the argv it runs: one simple command, its words as written
    after quote removal, never evaluated, no shell in between. Raises `ValueError` for
    a line the grammar cannot read or one that is more than a simple command (an operator, a
    redirect, an assignment), so a template, a check or an action that is not one argv is
    refused where it is declared, never run as something else. A.6's literal test does not
    apply: no shell expands a declared argv, so a template's `{files}` is its word."""
    if not line.strip():
        return ()
    root = _parse(line)
    if root.has_error:
        raise ValueError(f"cannot read {line!r} as a command")
    nodes = root.children
    if len(nodes) != 1 or nodes[0].type != "command":
        # `children`, not `named_children`: a trailing `&` or `;` is an unnamed token, and
        # `pytest &` is not the argv `pytest`.
        raise ValueError(f"{line!r} is not one simple command")
    words: list[str] = []
    for child in nodes[0].named_children:
        if child.type in ("variable_assignment", "file_redirect", "herestring_redirect"):
            raise ValueError(f"{line!r} is not one simple command")
        target = child.named_children[0] if child.type == "command_name" else child
        words.append(_value(target)[0])
    return tuple(words)


def split(line: str) -> list[str]:
    """The words of `line`, for a reader rather than a runner, such as the secret check: every
    word in the tree, each as written after quote removal (a word the shell would compute is
    its source text), with the text of comments and heredoc bodies; for a line the grammar
    cannot read, its whitespace-separated words as well, so a caller looking for a refusal
    still sees every one of them.

    Not the text of a heredoc `cat` or `tee` copies: that is a file's
    bytes, `env_file = ".env"` in a `config.py` being written, never a name anything opens. A
    command substitution inside such a body runs, and its words are read, as the tree holds
    it; a heredoc fed to anything else (a shell, `python3`) is read whole."""
    root = _parse(line)
    words: list[str] = []
    for node in _walk(root):
        if node.type in _WORDISH:
            value = _value(node)[0]
            if value:
                words.append(value)
        elif node.type == "comment" or (node.type == "heredoc_body" and not _copied(node)):
            words.extend(_text(node).split())
    if root.has_error:
        words.extend(w for w in line.split() if w not in words)
    return words


_COPIERS: Final = frozenset({"cat", "tee"})
"""Commands that copy their stdin out as bytes, opening nothing it names."""


_HEREDOC_PARTS: Final = frozenset({"heredoc_start", "heredoc_body", "heredoc_end"})
"""A heredoc redirect that holds only these is the heredoc and nothing after it."""

_READS_OUTPUT: Final = frozenset({"pipeline", "command_substitution", "process_substitution"})
"""Where a command's output becomes another's input or words: `cat <<EOF | sh` runs the body."""


def _copied(body: Node) -> bool:
    """Whether a heredoc body is only copied: the command it feeds is `cat` or `tee`, and no
    pipe or substitution above it hands what they print to anything else."""
    redirect = body.parent
    if redirect is None or any(c.type not in _HEREDOC_PARTS for c in redirect.named_children):
        # The rest of the line after the marker sits inside the redirect: `cat <<EOF | sh`.
        return False
    statement = redirect.parent
    if statement is None or statement.type != "redirected_statement":
        return False
    inner = statement.named_children[0] if statement.named_children else None
    if inner is not None and inner.type == "list" and inner.named_children:
        # The grammar wraps `mkdir -p a && cat > a/b.py <<EOF` whole; the shell feeds the
        # heredoc to the last command, after the last operator.
        inner = inner.named_children[-1]
    if inner is None or inner.type != "command":
        return False
    argv = _argv_of(inner)
    if not argv or _head(argv) not in _COPIERS:
        return False
    above = statement.parent
    while above is not None:
        if above.type in _READS_OUTPUT:
            return False
        above = above.parent
    return True


def exact_words(line: str) -> tuple[str, ...]:
    """A plain line as what *this session, exact* binds: its words after quote
    removal and its operators, in order, so `a && b` and `a | b` never bind alike and a quoted
    word is one word. A newline between commands is the `;` it means, so two commands on two
    lines never bind as one command; a word spelled as an operator, or starting with a quote,
    is quoted, so the word `'&&'` never binds as the operator. Two lines bind alike only when
    the shell runs them alike. Empty for a line that is not plain, which no grant covers."""
    reading = read_line(line)
    if not reading.plain:
        return ()
    out: list[str] = []
    for node in _walk(_parse(line)):
        if (
            node.type in _WORDISH
            and node.parent is not None
            and node.parent.type != "concatenation"
        ):
            out.append(_word(_value(node)[0]))
        elif not node.is_named and node.type in _OPERATORS:
            out.append(";" if node.type == "\n" else node.type)
        elif (
            node.is_named
            and node.parent is not None
            and node.parent.type == "program"
            and node.prev_sibling is not None
            and node.prev_sibling.is_named
        ):
            out.append(";")
    return tuple(out)


_OPERATORS = frozenset({"&&", "||", ";", "|", "\n"})


def _word(value: str) -> str:
    """`value` as one of `exact_words`'s words: quoted when it could be read as an operator or
    as a word already quoted (`shlex.quote` always quotes those, and is one-to-one)."""
    if value in _OPERATORS or value.startswith("'"):
        return shlex.quote(value)
    return value


def first_operand(argv: Sequence[str], takes_value: frozenset[str]) -> int:
    """Where `argv`'s operands start: the first word after `argv[0]` that is not an option or
    the separate value of one in `takes_value`. The one home for skipping a command's leading
    options (a wrapper's, git's own before its subcommand)."""
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in takes_value else 1
    return i


def join(words: Sequence[str]) -> str:
    """An argv as one line `argv` turns back into the same argv, for showing and storing."""
    return shlex.join(words)


# -- the reading -----------------------------------------------------------------------------

_WORDISH = frozenset(
    {"word", "number", "string", "raw_string", "concatenation", "ansi_c_string"}
)
_NOT_ARGUMENTS = frozenset({"variable_assignment", "file_redirect", "herestring_redirect"})


class _Reading:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.plain = True
        self.writes_to: list[str] = []
        self.whole = True

    def line(self, text: str, *, depth: int) -> None:
        if depth > _MAX_DEPTH:
            self.plain = False
            self.whole = False
            return
        if not text.strip():
            return
        if len(text) > PLAIN_MAX_CHARS:
            self.plain = False
        root = _parse(text)
        if not _plain_tree(root):
            self.plain = False
        if root.has_error:
            self.whole = False
        for node in _walk(root):
            if node.type == "command":
                self._command_node(node, depth=depth)
            elif node.type == "redirected_statement":
                self._fed_to_a_shell(node, depth=depth)
            elif node.type == "file_redirect":
                self._redirect(node)

    def _redirect(self, node: Node) -> None:
        operator = next((_text(c) for c in node.children if not c.is_named), "")
        target = node.named_children[-1] if node.named_children else None
        if target is None or target.type == "file_descriptor" or ">" not in operator:
            return
        if operator.endswith("&") and (target.type == "number" or _text(target) == "-"):
            # `2>&1`, `>&2`, `>&-`: a descriptor, never a file; `>& out` is a file.
            return
        self.writes_to.append(_value(target)[0])

    def _command_node(self, node: Node, *, depth: int) -> None:
        argv = _argv_of(node)
        if not argv:
            return
        self.command(argv, depth=depth, scan_words=False)
        expanded = _brace_argv(node)
        if expanded != argv:
            # A brace the shell expands into more words (`find. -{delete,print}` runs
            # `find. -delete -print`): the line is not plain already, and is read both ways.
            self.command(expanded, depth=depth, scan_words=False)
        if _runs_a_shell(argv):
            for child in node.named_children:
                if child.type == "herestring_redirect" and child.named_children:
                    self.line(_value(child.named_children[-1])[0], depth=depth + 1)

    def _fed_to_a_shell(self, node: Node, *, depth: int) -> None:
        """A heredoc whose command is a shell, or a wrapper starting one, is a line that shell
        runs."""
        inner = node.named_children[0] if node.named_children else None
        if inner is None or inner.type != "command" or not _runs_a_shell(_argv_of(inner)):
            return
        for child in node.named_children[1:]:
            if child.type == "heredoc_redirect":
                for part in child.named_children:
                    if part.type == "heredoc_body":
                        self.line(_text(part), depth=depth + 1)

    def command(self, argv: tuple[str, ...], *, depth: int, scan_words: bool) -> None:
        """`argv` and every command it starts. `scan_words` reads a `$( )` in a word as a
        command, which only a word from an argv needs: in a parsed line the tree already
        holds every substitution as a command."""
        if not argv:
            return
        head = _head(argv)
        inner = _wrapped(head, argv)
        if inner is None or head != "env":
            own = argv if inner is None else argv[: len(argv) - len(inner)]
            self.commands.append(own)
        if inner:
            self.command(inner, depth=depth, scan_words=scan_words)
        if head in _RUNS_A_STRING or head in _RUNS_A_FILE or head in _STDIN_ARGUMENTS:
            # A string run as a line, a file nobody read, or arguments that arrive on stdin:
            # what runs is not the text.
            self.plain = False
        if head in _SHELLS and _reads_stdin(argv):
            # Its commands arrive on stdin (`… | sh`, `bash -s`): what runs is not the text. A
            # heredoc or here-string it is fed is read as a line where the tree holds it.
            self.plain = False
        if head == "env" and len(argv) - 1 != len(inner or ()):
            # An assignment or an option: the command runs in an environment or a directory
            # the text does not show (`_WRAPPERS`).
            self.plain = False
        for text in _lines_it_runs(head, argv):
            self.line(text, depth=depth + 1)
        if head == "find":
            self._find(argv, depth=depth, scan_words=scan_words)
        if scan_words:
            for word in argv:
                for found in _SUBSTITUTION.finditer(word):
                    self.line(found.group(1) or found.group(2) or "", depth=depth + 1)

    def _find(self, argv: tuple[str, ...], *, depth: int, scan_words: bool) -> None:
        if any(may_be(word, "-delete") for word in argv):
            # `-delete` is an `rm` of what `find` matched: a command the table names.
            self.commands.append(("find", "-delete"))
        # Every `-exec` body, each up to its own `;` or `+`: a second one is a command too.
        body: list[str] | None = None
        for word in argv[1:]:
            if body is None:
                if word in _FIND_RUNS:
                    body = []
            elif word in (";", "+"):
                self.command(tuple(body), depth=depth, scan_words=scan_words)
                body = None
            elif word != "{}":
                body.append(word)
        if body:
            self.command(tuple(body), depth=depth, scan_words=scan_words)


def _lines_it_runs(head: str, argv: tuple[str, ...]) -> Iterator[str]:
    """The lines a command hands a shell: an `eval` string, a shell's `-c` body, an `env -S`
    string (which `env` splits into a command), and a `git -c alias.<name>=!<line>` override,
    which runs `<line>` when the alias is used."""
    if head in _RUNS_A_STRING:
        yield " ".join(argv[1:])
    if head in _SHELLS:
        body = _shell_body(argv)
        if body is not None:
            yield body
    if head == "env":
        yield from _split_strings(argv)
    if head in _PROGRAM_OPTIONS:
        yield from _program_values(argv, _PROGRAM_OPTIONS[head])
    if head in _AWKS:
        for word in argv[1:]:
            for found in _AWK_RUNS.finditer(word):
                yield next(g for g in found.groups() if g is not None)
    if head == "git":
        yield from _git_runs(argv)
        for i, word in enumerate(argv[:-1]):
            if word == "-c":
                found = _GIT_SHELL_ALIAS.match(argv[i + 1]) or _GIT_COMMAND_KEY.match(
                    argv[i + 1]
                )
                if found is not None:
                    yield found.group(found.lastindex or 1)


def _shell_body(argv: tuple[str, ...]) -> str | None:
    """A shell's `-c` body: the first operand, when a `-c` is among its options, alone or in a
    cluster (`-c`, `-ec`, `-xc`); `None` when it runs a script or reads stdin. A `-c` word
    after a script's name is read as one too, though the shell hands it to the script: reading
    more only raises a tier, and `sh script.sh -c 'rm x'` was read so before."""
    runs_body = False
    i = 1
    while i < len(argv):
        word = argv[i]
        if word == "--":
            i += 1
            break
        if word in _SHELL_VALUE_OPTIONS:
            i += 2
        elif len(word) > 1 and word.startswith(("-", "+")):
            if word.startswith("-") and not word.startswith("--"):
                runs_body = runs_body or "c" in word[1:]
                # A cluster ending in `o` takes the next word as its value (`-eo pipefail`).
                i += 2 if word[-1] in "oO" else 1
            else:
                i += 1
        else:
            break
    if runs_body and i < len(argv):
        return argv[i]
    return argv[argv.index("-c") + 1] if "-c" in argv[:-1] else None


def _git_runs(argv: tuple[str, ...]) -> Iterator[str]:
    """The lines a git subcommand runs (`_GIT_PROGRAM_OPTIONS`, `_GIT_RUNS_THE_REST`)."""
    words = argv[first_operand(argv, GIT_GLOBAL_OPTIONS) :]
    if not words:
        return
    if tuple(words[:2]) in _GIT_RUNS_THE_REST:
        rest = (
            [w for w in words[2:] if not w.startswith("-")] if words[0] == "submodule" else []
        )
        # `submodule foreach` takes its command as words the shell joins; `bisect run` as argv.
        yield " ".join(rest) if words[0] == "submodule" else join(words[2:])
        return
    options = _GIT_PROGRAM_OPTIONS.get(words[0])
    if options is None:
        return
    yield from _program_values(words, options)
    shorts = {o for o in options if len(o) == 2}
    for i, word in enumerate(words[1:], start=1):
        if word in shorts and i + 1 < len(words):
            yield words[i + 1]
        elif len(word) > 2 and word[:2] in shorts:
            yield word[2:]


def _program_values(argv: tuple[str, ...], options: frozenset[str]) -> Iterator[str]:
    """The values of `options` in `argv`, as `--opt=value` or `--opt value`, an abbreviation
    of the option included."""
    for i, word in enumerate(argv[1:], start=1):
        name, eq, value = word.partition("=")
        if len(name) > 2 and name.startswith("--") and any(o.startswith(name) for o in options):
            if eq:
                yield value
            elif i + 1 < len(argv):
                yield argv[i + 1]


_QUOTED_BRACE: Final = str.maketrans({"{": "\x01", "}": "\x02", ",": "\x03"})
_UNQUOTED_BRACE: Final = str.maketrans({"\x01": "{", "\x02": "}", "\x03": ","})


def _brace_argv(node: Node) -> tuple[str, ...]:
    """A `command` node's words after brace expansion. Only an unquoted brace expands: inside
    quotes, `{`, `}` and `,` stand for themselves, so they are hidden while expanding."""
    words: list[str] = []
    for child in node.named_children:
        if child.type in _NOT_ARGUMENTS:
            continue
        target = child.named_children[0] if child.type == "command_name" else child
        parts = target.named_children if target.type == "concatenation" else [target]
        source = "".join(
            _value(part)[0]
            if part.type in ("word", "number")
            else _value(part)[0].translate(_QUOTED_BRACE)
            for part in parts
        )
        words += [w.translate(_UNQUOTED_BRACE) for w in _brace_expanded(source)]
    return tuple(words)


def _brace_expanded(word: str, limit: int = 64) -> list[str]:
    """`word` after brace expansion (`-{a,b}` is `-a -b`), at most `limit` words; a `${…}`
    is a parameter, not a brace."""
    found = _BRACE.search(word)
    if found is None:
        return [word]
    out: list[str] = []
    for alternative in found.group(1).split(","):
        out += _brace_expanded(word[: found.start()] + alternative + word[found.end() :], limit)
        if len(out) >= limit:
            break
    return out[:limit]


def may_be(word: str, literal: str) -> bool:
    """Whether the shell may pass `word` as `literal`: it is, or it is a glob that matches it
    (`-del*` names a file `-delete` where one exists). The one home of the question, read
    here and by the command table's option rules (`approval/commands.py`); the glob is read by
    the one matcher, `paths.glob_match`."""
    return word == literal or (
        bool(_GLOB & set(word)) and glob_match(literal, (word,)) is not None
    )


def _reads_stdin(argv: tuple[str, ...]) -> bool:
    """Whether a shell runs commands from stdin: no `-c` body and no script, or `-s`."""
    if _shell_body(argv) is not None:
        return False
    i = 1
    while i < len(argv) and len(argv[i]) > 1 and argv[i].startswith(("-", "+")):
        word = argv[i]
        if word.startswith("-") and not word.startswith("--") and "s" in word[1:]:
            return True
        i += 2 if word in _SHELL_VALUE_OPTIONS else 1
    return i >= len(argv)


def _split_strings(argv: tuple[str, ...]) -> Iterator[str]:
    """The strings `env -S` splits into the command it starts, from `env`'s own words."""
    inner = _wrapped("env", argv)
    own = argv if inner is None else argv[: len(argv) - len(inner)]
    for i, word in enumerate(own[1:], start=1):
        if word in ("-S", "--split-string") and i + 1 < len(own):
            yield own[i + 1]
        elif word.startswith("--split-string="):
            yield word.split("=", 1)[1]
        elif word.startswith("-S") and len(word) > 2:
            yield word[2:]


def runs_its_body(argv: Sequence[str]) -> bool:
    """Whether `argv` is a shell handed a `-c` body (`bash -c "cd x && git switch y"`): the
    body is read as a line of its own (`read_line`), so the shell's own part runs nothing the
    reading has not seen."""
    words = tuple(argv)
    return bool(words) and _head(words) in _SHELLS and _shell_body(words) is not None


def _runs_a_shell(argv: tuple[str, ...]) -> bool:
    """Whether `argv`, or the command its wrappers start, is a shell."""
    while argv:
        if _head(argv) in _SHELLS:
            return True
        inner = _wrapped(_head(argv), argv)
        if not inner:
            return False
        argv = inner
    return False


def _argv_of(node: Node) -> tuple[str, ...]:
    """A `command` node's words after quote removal, without its assignments and redirects."""
    words: list[str] = []
    for child in node.named_children:
        if child.type in _NOT_ARGUMENTS:
            continue
        target = child.named_children[0] if child.type == "command_name" else child
        words.append(_value(target)[0])
    return tuple(words)


def _head(argv: tuple[str, ...]) -> str:
    return argv[0].rsplit("/", 1)[-1] if argv else ""


def _walk(node: Node) -> Iterator[Node]:
    """Every node under `node`, in order, `node` first."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _text(node: Node) -> str:
    return (node.text or b"").decode("utf-8", "replace")


def _plain_tree(root: Node) -> bool:
    for node in _walk(root):
        if node.type == "ERROR" or node.is_missing:
            return False
        if node.is_named:
            if node.type not in _PLAIN_KINDS:
                return False
            if node.type in ("word", "number") and not _literal(node):
                return False
            if node.type == "string" and not _literal_string(node):
                return False
        elif node.type not in _PLAIN_TOKENS:
            return False
    return True


def _literal(node: Node) -> bool:
    """Codex's `is_literal_word_or_number`: no child node, no leading `=`, and none of the
    characters the shell would act on."""
    text = _text(node)
    return (
        not node.named_children and not text.startswith("=") and not (_NOT_LITERAL & set(text))
    )


def _literal_string(node: Node) -> bool:
    """A double-quoted string the shell passes as written: only text, and no escape it would
    undo (Codex's `parse_double_quoted_string`)."""
    if any(c.type != "string_content" for c in node.named_children):
        return False
    inner = _text(node)[1:-1]
    return not re.search(r"\\[$`\"\\\n]", inner)


def _value(node: Node) -> tuple[str, bool]:
    """A word's text as the shell would pass it after quote removal, and whether the shell
    would pass exactly that. A word the shell would compute is its text with the quotes
    removed, which is what the table reads to raise a tier: `--for\\ce` is read as `--force`."""
    text = _text(node)
    match node.type:
        case "raw_string":
            return text[1:-1], True
        case "string":
            return _undo_double_quotes(text[1:-1]), _literal_string(node)
        case "concatenation":
            parts = [_value(child) for child in node.named_children]
            return "".join(v for v, _ in parts), all(ok for _, ok in parts)
        case "word" | "number":
            return _undo_backslashes(text), _literal(node)
        case _:
            return text, False


def _undo_backslashes(text: str) -> str:
    """POSIX quote removal outside quotes: a backslash keeps the next character, and a
    backslash before a newline joins the lines."""
    return re.sub(
        r"\\(.)", lambda m: "" if m.group(1) == "\n" else m.group(1), text, flags=re.S
    )


def _undo_double_quotes(text: str) -> str:
    """Inside double quotes a backslash escapes only `$`, a backtick, `"`, `\\`, and a
    newline."""
    return re.sub(r"\\([$`\"\\\n])", lambda m: "" if m.group(1) == "\n" else m.group(1), text)


def _wrapped(head: str, argv: tuple[str, ...]) -> tuple[str, ...] | None:
    """The command a wrapper starts, or `None` when `head` is not a wrapper or starts
    nothing (`env` alone, `sudo -l`, `command -v rm`)."""
    takes_value = _WRAPPERS.get(head)
    if takes_value is None:
        return None
    queries = _QUERIES.get(head, frozenset())
    i = 1
    while i < len(argv):
        word = argv[i]
        if word in queries:
            return None
        if word in takes_value:
            i += 2
        elif word.startswith("-") or (head == "env" and "=" in word):
            i += 1
        elif head == "timeout" and i == first_operand(argv, takes_value):
            i += 1  # the duration
        else:
            return argv[i:]
    return None
