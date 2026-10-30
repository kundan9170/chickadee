"""The REPL's input: `/` completes every command `/help` lists,
filtered as typed, and their fixed words; `@` completes the server's paths fuzzily, inserting
the plain path; history survives a restart and keeps no line holding a secret; Enter sends and
Alt+Enter is a new line."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.keys import Keys

from chickadee_cli.commands import COMMANDS
from chickadee_cli.prompt import Files, InputCompleter, ScreenedHistory, _keys, history_path
from chickadee_serve.frames import ErrorBody, ErrorCode
from chickadee_wire.errors import WireError


def _gone() -> WireError:
    return WireError(503, ErrorBody(code=ErrorCode.NOT_FOUND, message="gone"))


PATHS = ("src/app.py", "src/retry.py", "tests/test_retry.py", "README.md")


def _complete(text: str, paths: tuple[str, ...] = PATHS) -> list[str]:
    completer = InputCompleter(lambda: paths)
    return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]


def test_slash_lists_every_command_and_filters_as_typed() -> None:
    assert _complete("/") == [f"/{name}" for name in COMMANDS]
    assert _complete("/re") == ["/rename", "/redo", "/resume", "/replay"]


def test_a_commands_fixed_words_complete() -> None:
    assert _complete("/verbose ") == ["q", "normal", "v"]
    assert _complete("/handoff s") == ["squash"]
    assert _complete("/journal ") == [], "a value is not a word to type"


def test_at_completes_paths_fuzzily_inserting_the_plain_path() -> None:
    assert _complete("look at @retry") == ["src/retry.py", "tests/test_retry.py"]
    assert _complete("@srcapp") == ["src/app.py"]
    assert _complete("no at sign here") == []


def test_files_are_asked_once_and_again_after_a_turn() -> None:
    asked: list[int] = []

    def fetch() -> tuple[str, ...]:
        asked.append(1)
        return PATHS

    files = Files(fetch)
    files.paths()
    files.paths()
    assert len(asked) == 1
    files.stale()
    files.paths()
    assert len(asked) == 2


def test_a_server_that_cannot_list_keeps_the_last_list() -> None:
    answers: list[tuple[str, ...] | WireError] = [PATHS]

    def fetch() -> tuple[str, ...]:
        answer = answers[0]
        if isinstance(answer, WireError):
            raise answer
        return answer

    files = Files(fetch)
    files.paths()
    files.stale()
    answers[0] = _gone()
    assert files.paths() == PATHS


def test_history_survives_a_restart_and_keeps_no_secret(tmp_path: Path) -> None:
    path = history_path(tmp_path / "state")
    history = ScreenedHistory(path, lambda line: "sk-" in line)
    history.append_string("fix the retry")
    history.append_string("use sk-abcdefghijklmnopqrstu")
    again = FileHistory(str(path))
    assert list(again.load_history_strings()) == ["fix the retry"]
    assert not path.is_relative_to(Path.cwd()) or "state" in path.parts


def test_history_writes_nothing_when_the_server_cannot_say(tmp_path: Path) -> None:
    def unsure(_line: str) -> bool:
        raise _gone()

    path = history_path(tmp_path)
    ScreenedHistory(path, unsure).append_string("hello")
    assert list(FileHistory(str(path)).load_history_strings()) == []


def test_enter_sends_and_alt_enter_is_a_new_line() -> None:
    bound = {tuple(b.keys) for b in _keys().bindings}
    assert (Keys.ControlM,) in bound
    assert (Keys.Escape, Keys.ControlM) in bound
