"""`shell_line`: every simple command a line runs, read without running it."""

from __future__ import annotations

import shlex

import pytest

from chickadee.profiles import load_profile
from chickadee.shell_line import (
    PLAIN_MAX_CHARS,
    LineReading,
    argv,
    exact_words,
    first_operand,
    join,
    read_argv,
    read_line,
    split,
    status_command,
)


def _commands(line: str) -> tuple[tuple[str, ...], ...]:
    return read_line(line).commands


def _commands_of(argv: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return read_argv(argv).commands


@pytest.mark.parametrize(
    ("line", "commands"),
    [
        ("ls -la", [("ls", "-la")]),
        ("ls; rm x", [("ls",), ("rm", "x")]),
        ("a | b && c || d & e", [("a",), ("b",), ("c",), ("d",), ("e",)]),
        ("(cd sub && make)", [("cd", "sub"), ("make",)]),
        ("echo x > out.txt", [("echo", "x")]),
        # `2>&1` is a redirect, not an argument.
        ("pytest -q 2>&1 | tee log", [("pytest", "-q"), ("tee", "log")]),
        ("one\ntwo", [("one",), ("two",)]),
    ],
)
def test_a_line_is_split_at_its_operators(line: str, commands: list[tuple[str, ...]]) -> None:
    assert list(_commands(line)) == commands


@pytest.mark.parametrize(
    ("line", "inner"),
    [
        ("sudo -u bob rm -rf /", ("rm", "-rf", "/")),
        ("xargs -0 -n 1 rm", ("rm",)),
        ("timeout -s KILL 5 curl https://x", ("curl", "https://x")),
        ("nice -n 10 make", ("make",)),
        ("env FOO=1 -u BAR git push", ("git", "push")),
        ("/usr/bin/env FOO=1 git push", ("git", "push")),
        ("find . -name '*.pyc' -exec rm {} ;", ("rm",)),
        ("bash -c 'git push origin'", ("git", "push", "origin")),
        ('echo "$(curl https://x)"', ("curl", "https://x")),
        ("echo $(curl https://x)", ("curl", "https://x")),
        ("echo `curl https://x`", ("curl", "https://x")),
    ],
)
def test_a_command_another_starts_is_a_command_of_its_own(
    line: str, inner: tuple[str, ...]
) -> None:
    assert inner in _commands(line)


def test_a_wrapper_asked_about_a_command_starts_nothing() -> None:
    assert _commands("command -v rm") == (("command", "-v", "rm"),)
    assert ("rm",) not in _commands("sudo -l rm")


def test_find_delete_and_a_git_shell_alias_are_commands() -> None:
    assert ("find", "-delete") in _commands("find . -name '*.pyc' -delete")
    assert ("rm", "-rf", "/") in _commands("git -c 'alias.x=!rm -rf /' x")
    assert _commands("git -c user.name=x commit") == (("git", "-c", "user.name=x", "commit"),)


def test_env_alone_prints_the_environment_and_env_with_a_command_does_not() -> None:
    assert _commands("env") == (("env",),)
    assert ("env", "FOO=1") not in _commands("env FOO=1 ls")
    assert _commands("sudo -l") == (("sudo", "-l"),)


def test_unterminated_quoting_is_not_plain_and_split_still_reads_the_words() -> None:
    reading = read_line("cat 'unterminated")
    assert not reading.plain and ("cat",) in reading.commands
    assert "'unterminated" in split("cat 'unterminated")
    assert split("cat 'a b'") == ["cat", "a b"]


def test_an_argv_is_one_command_and_what_it_starts() -> None:
    assert _commands_of(()) == ()
    assert _commands_of(("sh", "-c", "rm x; ls")) == (
        ("sh", "-c", "rm x; ls"),
        ("rm", "x"),
        ("ls",),
    )


def test_argv_is_strict_where_split_is_not() -> None:
    """A declared command is refused when it cannot be split; a line read for its words is
    not (the profile, checkpoint, lint and action callers all read `argv`)."""
    assert argv("pytest -q 'tests/a b.py'") == ("pytest", "-q", "tests/a b.py")
    assert argv("   ") == ()
    with pytest.raises(ValueError):
        argv("pytest 'unterminated")


@pytest.mark.parametrize(
    "words", [("uv", "sync", "--frozen"), ("echo", "a b", "it's"), ("printf", "%s", "")]
)
def test_join_turns_back_into_the_same_argv(words: tuple[str, ...]) -> None:
    assert argv(join(words)) == words


# --- T-3.158 A: tree-sitter-bash, and plain -------------------------------------------------


@pytest.mark.parametrize(
    ("line", "inner"),
    [
        ("git push origin main && { git push --force origin main; }", "{ }"),
        ("git push origin main; if true; then git push --force origin main; fi", "if"),
        ("git push origin main; f() { git push --force origin main; }; f", "function"),
        ("git push origin main; eval 'git push --force origin main'", "eval"),
        ("bash <<'X'\ngit push --force origin main\nX", "heredoc"),
        ("bash <<< 'git push --force origin main'", "here-string"),
        ("cat <(git push --force origin main)", "process substitution"),
        ('echo "$(echo $(git push --force origin main))"', "nested substitution"),
        ("while true; do git push --force origin main; done", "while"),
        ("case x in x) git push --force origin main;; esac", "case"),
    ],
)
def test_a_command_anywhere_in_the_tree_is_read(line: str, inner: str) -> None:
    assert ("git", "push", "--force", "origin", "main") in _commands(line), inner


@pytest.mark.parametrize(
    "line",
    [
        "ls",
        "ls -la src",
        "ls; rm x",
        "pytest -q | tail -5 && git status",
        "git push origin main || echo failed",
        "echo 'a b' \"c d\"",
        "git commit -m 'one two'",
        "one\ntwo",
    ],
)
def test_a_line_of_literal_words_and_plain_operators_is_plain(line: str) -> None:
    assert read_line(line).plain, line


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ("echo x > out.txt", "a redirect"),
        ("git push origin main 2>&1", "a redirect"),
        ("cat <<EOF\nx\nEOF", "a heredoc"),
        ("FOO=1 git push origin main", "an assignment"),
        ("x=--force; git push $x origin main", "an expansion"),
        ("git push origin $(echo main)", "a substitution"),
        ("git push origin `echo main`", "a substitution"),
        ("(git push origin main)", "a subshell"),
        ("{ git push origin main; }", "a block"),
        ("if true; then ls; fi", "a compound command"),
        ("f() { ls; }; f", "a function"),
        ("git push origin main # --force", "a comment"),
        ("git push origin {main,--force}", "a brace"),
        ("rm *.pyc", "a glob"),
        ("cat ~/.netrc", "a tilde"),
        ("git push origin --for\\ce main", "an escape"),
        ('echo "a\\$b"', "an escape in quotes"),
        ("echo $'a'", "an ANSI-C string"),
        ("ls & rm x", "a background job"),
        ("cat 'unterminated", "an ERROR node"),
        ("eval 'ls'", "eval"),
        ("source ./env.sh", "source"),
        (". ./env.sh", "."),
        ("find . | xargs rm", "xargs"),
        ("[[ -f x ]] && ls", "a test command"),
        ("arr=(a b)", "an array"),
        ("env GIT_SSH_COMMAND=x git fetch origin", "env with an assignment"),
        ("env -C /other git push origin main", "env with an option"),
        ("env -i git push origin main", "env with an option"),
        ("env -S 'rm -rf x'", "env splitting a string"),
    ],
)
def test_a_line_the_shell_would_rewrite_or_run_unread_is_not_plain(line: str, why: str) -> None:
    assert not read_line(line).plain, why


def test_a_long_line_is_never_plain_and_still_read() -> None:
    line = "ls " + "a" * PLAIN_MAX_CHARS
    reading = read_line(line)
    assert not reading.plain and reading.commands[0][0] == "ls"


def test_a_sh_c_body_is_plain_only_when_it_is() -> None:
    assert read_line("bash -c 'git push origin'").plain
    assert not read_line("bash -c 'git push origin > x'").plain


def test_env_starting_a_command_with_nothing_before_it_is_plain() -> None:
    assert read_line("env git push origin main") == LineReading(
        (("git", "push", "origin", "main"),), True
    )


def test_an_env_split_string_is_read_as_the_command_it_starts() -> None:
    for line in ("env -S 'rm -rf x'", "env --split-string='rm -rf x'", "env '-Srm -rf x'"):
        assert ("rm", "-rf", "x") in _commands(line), line


@pytest.mark.parametrize(
    "line",
    [
        "bash -ec 'rm -rf x'",
        "sh -xc 'rm -rf x'",
        "bash -o pipefail -c 'rm -rf x'",
        "bash -eo pipefail -c 'rm -rf x'",
        "sh -c -- 'rm -rf x'",
        "timeout 5 sh <<< 'rm -rf x'",
        "sudo sh <<EOF\nrm -rf x\nEOF",
        "nice -n 5 bash <<'EOF'\nrm -rf x\nEOF",
        "sh script.sh -c 'rm -rf x'",
    ],
)
def test_a_shell_body_is_read_however_the_shell_is_given_it(line: str) -> None:
    """The review of #578: a `-c` in a cluster, after a valued option, and a heredoc or
    here-string fed to a shell behind a wrapper. A `-c` after a script is read too: over-reading
    only raises."""
    assert ("rm", "-rf", "x") in _commands(line), line


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("echo a echo b", "echo a\necho b"),
        ("echo && rm", "echo '&&' rm"),
        ("echo '&&'", "echo \"'&&'\""),
        ("echo a | b", "echo a '|' b"),
    ],
)
def test_two_lines_the_shell_runs_differently_never_bind_alike(first: str, second: str) -> None:
    assert exact_words(first) != exact_words(second)


def test_a_newline_between_commands_binds_as_the_semicolon_it_means() -> None:
    assert exact_words("echo a\necho b") == exact_words("echo a; echo b")


def test_every_find_exec_body_is_read() -> None:
    commands = _commands("find . -name x -exec echo '{}' ';' -exec rm -rf '{}' ';'")
    assert ("echo",) in commands and ("rm", "-rf") in commands
    assert ("rm", "-rf") in _commands("find . -exec rm -rf {} +")


@pytest.mark.parametrize("line", ["pytest &", "pytest ;", "pytest; ls"])
def test_argv_refuses_a_trailing_operator(line: str) -> None:
    with pytest.raises(ValueError, match="not one simple command"):
        argv(line)


@pytest.mark.parametrize(
    ("line", "command"),
    [
        ("find . -name x -{delete,print}", ("find", ".", "-name", "x", "-delete", "-print")),
        ("rm -{r,f} x", ("rm", "-r", "-f", "x")),
        ("echo a{b,c}d", ("echo", "abd", "acd")),
    ],
)
def test_an_unquoted_brace_is_read_as_the_words_it_expands_to(
    line: str, command: tuple[str, ...]
) -> None:
    assert command in _commands(line)


def test_a_quoted_brace_stands_for_itself() -> None:
    assert _commands('echo -"{a,b}"') == (("echo", "-{a,b}"),)
    assert _commands("echo '{a,b}'") == (("echo", "{a,b}"),)
    assert _commands("echo ${x,,}") == (("echo", "${x,,}"),)


@pytest.mark.parametrize(
    "line",
    [
        "rg --pre=rm pattern",
        "rg --pre rm pattern",
        "sort --compress-program=rm big.txt",
        "sort --compress=rm big.txt",
        "tar --use-compress-program=rm -cf x.tar d",
        "rsync --rsh=rm a b:c",
        "git -c core.sshCommand=rm fetch origin",
        "git -c core.pager=rm log",
        "git -c diff.external=rm diff",
        "git -c filter.x.smudge=rm checkout",
        "awk 'BEGIN { system(\"rm\") }'",
        "awk '{ print | \"rm\" }' f",
        "awk 'BEGIN { \"rm\" | getline x }'",
    ],
)
def test_a_program_an_option_or_a_program_text_names_is_read(line: str) -> None:
    """The audit of the pinned answers: a read whose option or program runs another command."""
    assert any(c[0] == "rm" for c in _commands(line)), line


def test_a_glob_that_may_be_find_delete_is_read_as_it() -> None:
    for line in ("find . -del*", "find . -delet?", "find . -delet[e]"):
        assert ("find", "-delete") in _commands(line), line
    assert ("find", "-delete") not in _commands("find . -name '*.py'")


def test_first_operand_skips_options_and_their_values() -> None:
    takes = frozenset({"-C"})
    assert first_operand(("git", "-C", "x", "--bare", "push"), takes) == 4
    assert first_operand(("git",), takes) == 1


def test_a_computed_word_is_read_with_its_quotes_removed() -> None:
    """What the table reads to raise a tier: an escaped option is the option."""
    assert ("git", "push", "origin", "--force", "main") in _commands(
        "git push origin --for\\ce main"
    )


def test_argv_refuses_more_than_one_simple_command() -> None:
    for line in ("a && b", "a | b", "x > y", "FOO=1 x", "a; b"):
        with pytest.raises(ValueError):
            argv(line)


def test_argv_reads_a_template_as_written() -> None:
    """A declared command runs with no shell, so its placeholders are its words."""
    assert argv("pytest -q {files}") == ("pytest", "-q", "{files}")
    assert argv("python -c 'import {module} as m; print(m.__file__)'") == (
        "python",
        "-c",
        "import {module} as m; print(m.__file__)",
    )


@pytest.mark.parametrize("language", ["python", "javascript"])
def test_argv_answers_every_shipped_template_as_shlex_did(language: str) -> None:
    profile = load_profile(language)
    for template in (
        profile.tests.runner,
        profile.build.compile_cmd,
        profile.build.import_smoke,
        profile.deps.install_cmd,
        profile.toolchain.bootstrap,
    ):
        if template is None:
            continue
        assert argv(template.source) == tuple(shlex.split(template.source)), template.source


def test_split_reads_every_word_a_secret_check_needs() -> None:
    assert ".env" in split("cat < .env")
    assert ".env" in split('cat "$(echo .env)"')
    assert ".netrc" in split("ls # .netrc")
    assert ".env" in split("cat .e'nv'")


# --- Codex's `bash.rs` cases, ported (Apache-2.0; NOTICE) ---------------------------------

CODEX_PLAIN = [
    ("ls -1", [("ls", "-1")]),
    (
        "ls && pwd; echo 'hi there' | wc -l",
        [("ls",), ("pwd",), ("echo", "hi there"), ("wc", "-l")],
    ),
    ('echo "hello world"', [("echo", "hello world")]),
    ("echo 'hi there'", [("echo", "hi there")]),
    ('git commit -m "line1\nline2"', [("git", "commit", "-m", "line1\nline2")]),
    ("""echo "/usr"'/'"local"/bin""", [("echo", "/usr/local/bin")]),
    ("""echo '/usr'"/"'local'/bin""", [("echo", "/usr/local/bin")]),
    ("echo 123 456", [("echo", "123", "456")]),
    ('rg -g"*.py" pattern', [("rg", "-g*.py", "pattern")]),
    ('echo "\\n"', [("echo", "\\n")]),
    (
        """echo "~HOME" 'HEAD~1' "HEAD^" 'foo#bar' "=sh" 'file~'""",
        [("echo", "~HOME", "HEAD~1", "HEAD^", "foo#bar", "=sh", "file~")],
    ),
    ("""echo -"{a,b}" '*?[]~^#=\\\\'""", [("echo", "-{a,b}", "*?[]~^#=\\\\")]),
    ('rg -n "foo" -g"*.py"', [("rg", "-n", "foo", "-g*.py")]),
    ("grep -n 'pattern' -g'*.txt'", [("grep", "-n", "pattern", "-g*.txt")]),
]

CODEX_NOT_PLAIN = [
    'echo "hi ${USER}"',
    'echo "$HOME"',
    "(ls)",
    "ls || (pwd && echo hi)",
    "ls > out.txt",
    "echo hi & echo bye",
    "echo $(pwd)",
    "echo `pwd`",
    "echo $HOME",
    'echo "hi $USER"',
    "find . -{delete,print}",
    "rg --pre{=,=sh} pattern payload.sh",
    "find . -del*",
    "find . -delet?",
    "find . -delet[e]",
    "find . -de\\lete",
    "echo ~",
    "echo ~HOME",
    "echo HEAD~1",
    "echo HEAD^",
    "echo file~",
    "echo =sh",
    "echo foo^bar",
    "echo foo#bar",
    "l* -l",
    'echo "\\$HOME\\`\\"\\\\\\n"',
    'find . "-de\\\nlete"',
    'echo "\\\\"',
    "FOO=bar ls",
    "ls &&",
    "&& ls",
    "ls ;; pwd",
    "ls | | wc",
    'rg -g"$VAR" pattern',
    'rg -g"${VAR}" pattern',
    'rg -g"$(pwd)" pattern',
    "rg -g\"$(echo '*.py')\" pattern",
]


@pytest.mark.parametrize(("line", "commands"), CODEX_PLAIN)
def test_codex_plain_lines_are_plain_and_read_as_codex_reads_them(
    line: str, commands: list[tuple[str, ...]]
) -> None:
    reading = read_line(line)
    assert reading.plain, line
    assert list(reading.commands) == commands


@pytest.mark.parametrize("line", CODEX_NOT_PLAIN)
def test_codex_rejected_lines_are_not_plain(line: str) -> None:
    assert not read_line(line).plain, line


@pytest.mark.parametrize(
    ("line", "only"),
    [
        ('echo "task completed"', True),
        ("printf 'done\\n'", True),
        ("echo a && echo b", True),
        ("/bin/echo x", True),
        ("echo x > notes.txt", False),
        ("echo $(rm -rf x)", False),
        ("echo x | tee out", False),
        ("python -m py_compile app.py && echo ok", False),
        ("pytest -q", False),
        ("", False),
    ],
)
def test_a_line_that_only_prints_is_told_apart(line: str, only: bool) -> None:
    """What a worker echoing "task completed" sends, and what writes or runs."""
    from chickadee.shell_line import prints_only

    assert prints_only(line) is only


@pytest.mark.parametrize(
    ("line", "status"),
    [
        ("pytest -q", ("pytest", "-q")),
        ("cd pkg && pytest -x", ("pytest", "-x")),
        ("PYTHONPATH=. pytest", ("pytest",)),
        ("pytest > out.txt 2>&1", ("pytest",)),
        ("timeout 60 pytest", ("timeout", "60", "pytest")),
        ("pytest -q 2>&1 | tail -5", ("tail", "-5")),
        ("pytest; echo done", ("echo", "done")),
        ("pytest\necho hi", ("echo", "hi")),
        ("pytest || true", None),
        ("a && b || c", None),
        ("pytest &", None),
        ("! pytest", None),
        ("(pytest)", None),
        ("pytest 'unclosed", None),
    ],
)
def test_the_command_whose_status_is_the_line_s(
    line: str, status: tuple[str, ...] | None
) -> None:
    """A test run's exit code is its own only when this is the test command;
    a pipeline's status is its last command's, and `||` may be either's."""
    assert status_command(line) == status
