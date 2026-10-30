"""A session's name: a `title` line wins over D100's first line, the latest wins; the
`title` role names a session once, after its first turn, charged, never over the person's
name and never when a cap refuses its rung; a failure leaves the first line, unsaid."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from chickadee.client.request import ChatRequest
from chickadee.context.session import Session
from chickadee.context.vocabulary import PayloadError
from chickadee.naming import ASKED_CHARS, Namer, title_from
from chickadee.schema.adapter import Completion, StopReason, Usage
from chickadee.sessions import SESSION_TITLE_WIDTH, Sessions, SessionTitle, record_title

USAGE = Usage(prompt_tokens=120, completion_tokens=6)


@pytest.fixture
def sessions(tmp_path: Path) -> Iterator[Sessions]:
    held = Sessions(tmp_path / "sessions")
    yield held
    held.close_all()


class _Model:
    """A title model: its answers in turn, each request kept; an answer that is an exception
    is raised."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.asked: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.asked.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Completion(
            text=answer, usage=USAGE, latency_ms=5, stop_reason=StopReason.END_TURN
        )

    def close(self) -> None:
        pass


def _namer(
    session: Session, model: _Model | None, *, stopped: str | None = None
) -> tuple[Namer, list[Usage]]:
    charged: list[Usage] = []
    namer = Namer(
        session,
        model,
        None if model is None else object(),  # type: ignore[arg-type]
        "name it\n{asked}\n---\n{reply}",
        charge=charged.append,
        stopped=lambda: stopped,
    )
    return namer, charged


def _first_turn(session: Session) -> None:
    session.record_user("Can you look at the latest pr 670. Which is about ui.")
    session.record_assistant("PR #670 adds the question card and the background pane.")


# -- the rule --------------------------------------------------------------------------------


def test_a_title_line_wins_over_the_first_line_and_the_latest_wins(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    _first_turn(held)
    assert sessions.title("S-one") == "Can you look at the latest pr 670. Which is about ui."
    record_title(held, SessionTitle("PR 670 UI review", "model"))
    assert sessions.title("S-one") == "PR 670 UI review"
    record_title(held, SessionTitle("TUI panes", "person"))
    assert sessions.info("S-one").title == "TUI panes"
    sessions.close("S-one")
    assert Sessions(sessions.root).title("S-one") == "TUI panes", "read back from the log"


def test_a_title_is_one_line_of_words_within_the_width(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    for bad in ("", "   ", "two\nlines", "x" * (SESSION_TITLE_WIDTH + 1)):
        with pytest.raises(PayloadError):
            record_title(held, SessionTitle(bad, "person"))


@pytest.mark.parametrize(
    ("answer", "title"),
    [
        ("PR 670 UI review", "PR 670 UI review"),
        ('"PR 670 UI review."', "PR 670 UI review"),
        ("Title: **Retry the fetcher**\n\nbecause…", "Retry the fetcher"),
        ("“Curly quoted”", "Curly quoted"),
        ("   \n", None),
        ("w" * 80, "w" * (SESSION_TITLE_WIDTH - 1) + "…"),
    ],
)
def test_a_model_s_answer_is_cleaned_to_a_title(answer: str, title: str | None) -> None:
    assert title_from(answer) == title


# -- the call --------------------------------------------------------------------------------


def test_the_first_turn_is_named_once_charged_and_by_the_model(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    model = _Model("PR 670 UI review")
    namer, charged = _namer(held, model)
    namer()
    assert model.asked == [], "no turn yet: nothing to name it from"
    _first_turn(held)
    namer()
    namer()
    assert len(model.asked) == 1, "once"
    prompt = model.asked[0].messages[0].content
    assert "latest pr 670" in prompt and "question card" in prompt
    assert charged == [USAGE]
    assert sessions.title("S-one") == "PR 670 UI review"


def test_only_the_opening_is_shown_to_the_model(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    held.record_user("a" * (ASKED_CHARS * 3))
    model = _Model("Long paste")
    namer, _ = _namer(held, model)
    namer()
    assert str(model.asked[0].messages[0].content).count("a") <= ASKED_CHARS + 20


def test_a_person_s_name_is_never_overwritten(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    _first_turn(held)
    record_title(held, SessionTitle("My name", "person"))
    model = _Model("Model name")
    namer, charged = _namer(held, model)
    namer()
    assert model.asked == [] and charged == []
    assert sessions.title("S-one") == "My name"


@pytest.mark.parametrize("why", ["refused", "failed", "off", "empty"])
def test_a_refused_or_failed_call_leaves_the_first_line(sessions: Sessions, why: str) -> None:
    held = sessions.create("S-one")
    _first_turn(held)
    model = None if why == "off" else _Model(RuntimeError("down") if why == "failed" else "  ")
    stopped = "the total cap is reached" if why == "refused" else None
    namer, charged = _namer(held, model, stopped=stopped)
    namer()
    namer()
    assert sessions.title("S-one") == "Can you look at the latest pr 670. Which is about ui."
    if model is not None:
        assert len(model.asked) == (0 if why == "refused" else 1), "never retried"
    assert charged == ([USAGE] if why == "empty" else []), "an answer is charged, even empty"
