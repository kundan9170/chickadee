"""The per-call syntax check: at each edit call's answer every file
it changed is judged by the in-process parse (`box/syntax.SyntaxChecker.inproc`, gate A1's
check, moved with it); `reject` undoes a call that broke a file
that parsed before it, `report` keeps it and says so, `off` judges nothing. Nothing spawns."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from chickadee.box.syntax import SyntaxChecker
from chickadee.edits.draft import CandidateDraft, DraftRefused, EditCheck
from chickadee.edits.formats.base import Targets
from chickadee.edits.lines import Decoded, decode
from chickadee.edits.match import Region
from chickadee.profiles.loader import load_profile

FILES = {
    "app.py": "def go() -> int:\n    return 1\n",
    "lib.py": "x = 1\n",
    "crlf.py": "def go():\r\n    return 1\r\n",
    "broken.py": "def go(:\n    return 1\n",
    "README.md": "# app\n",
    "ui.js": "function go() {\n  return 1;\n}\n",
}
Mode = Literal["edit", "create", "delete"]
E: Mode = "edit"
C: Mode = "create"
D: Mode = "delete"


def _targets(*targets: tuple[str, Mode]) -> Targets:
    """The paths a draft is given: the writable ones in order, the deletes, and
    which are new files."""
    return Targets(
        writable=tuple(p for p, m in targets if m != D),
        deletes=tuple(p for p, m in targets if m == D),
        creates=frozenset(p for p, m in targets if m == C),
    )


def _read(path: str) -> Decoded | None:
    text = FILES.get(path)
    return None if text is None else decode(text)


def _draft(
    *targets: tuple[str, Mode],
    mode: EditCheck = "reject",
    language: str = "python",
) -> CandidateDraft:
    syntax = SyntaxChecker(Path("/nowhere"), (load_profile(language),))
    return CandidateDraft(
        _targets(*(targets or (("app.py", E),))),
        _read,
        lambda _: None,
        judge=syntax.inproc,
        edit_check=mode,
    )


def _call(draft: CandidateDraft, *edits: tuple[str, Region | None, str]) -> str | DraftRefused:
    with draft.call() as call:
        for path, region, content in edits:
            refused = call.apply(path, region, content)
            if refused is not None:
                return refused
        return call.answer()


DROP_COLON = ("app.py", Region(15, 16), "")  # "def go() -> int:" loses its colon


def test_a_call_that_drops_a_colon_is_undone_under_reject() -> None:
    draft = _draft()
    refused = _call(draft, DROP_COLON)
    assert isinstance(refused, DraftRefused) and refused.failure == "SYNTAX_FAIL"
    assert refused.reason.startswith("app.py:1: ") and "the call is undone" in refused.reason
    assert draft.text("app.py") == FILES["app.py"]


def test_a_good_call_stands_and_its_answer_says_so() -> None:
    draft = _draft()
    said = _call(draft, ("app.py", Region(28, 29), "2"))
    assert said == "ok: app.py\n    return 2\nsyntax ok: app.py"


def test_a_call_is_judged_once_and_undone_whole() -> None:
    draft = _draft(("app.py", E), ("lib.py", E))
    refused = _call(draft, ("lib.py", None, "x = 2\n"), DROP_COLON)
    assert isinstance(refused, DraftRefused) and refused.failure == "SYNTAX_FAIL"
    assert draft.text("lib.py") == FILES["lib.py"]  # the good half is undone too
    # An intermediate break mended later in the same call is never seen.
    said = _call(draft, DROP_COLON, ("app.py", Region(15, 15), ":"))
    assert isinstance(said, str) and said.endswith("syntax ok: app.py")


def test_a_file_broken_before_the_call_is_reported_not_refused() -> None:
    draft = _draft(("broken.py", E))
    said = _call(draft, ("broken.py", Region(20, 21), "2"))
    assert isinstance(said, str)
    assert "\nsyntax error: broken.py:1: " in said and said.endswith(", as before this call")
    assert draft.text("broken.py") == "def go(:\n    return 2\n"
    mended = _call(draft, ("broken.py", Region(7, 7), ")"))
    assert isinstance(mended, str) and mended.endswith("syntax ok: broken.py")


def test_a_new_file_written_broken_is_refused() -> None:
    draft = _draft(("new.py", C))
    refused = _call(draft, ("new.py", None, "def f(:\n"))
    assert isinstance(refused, DraftRefused) and refused.failure == "SYNTAX_FAIL"
    assert draft.absent("new.py")


def test_report_keeps_the_call_and_off_judges_nothing() -> None:
    reported = _draft(mode="report")
    said = _call(reported, DROP_COLON)
    assert isinstance(said, str) and "syntax error: app.py:1: " in said
    assert reported.text("app.py") == "def go() -> int\n    return 1\n"
    off = _draft(mode="off")
    said = _call(off, DROP_COLON)
    assert said == "ok: app.py\ndef go() -> int"


def test_a_crlf_file_is_judged_in_its_own_newlines() -> None:
    draft = _draft(("crlf.py", E))
    said = _call(draft, ("crlf.py", Region(20, 21), "2"))
    assert isinstance(said, str) and said.endswith("syntax ok: crlf.py")


def test_a_file_the_profile_does_not_own_is_unchecked() -> None:
    draft = _draft(("README.md", E))
    said = _call(draft, ("README.md", None, "# app :(\n"))
    assert said == "ok: README.md written, 1 lines\nsyntax unchecked: README.md"


def test_a_command_mode_profile_spawns_nothing() -> None:
    profile = load_profile("javascript")
    assert profile.syntax is not None and profile.syntax.mode == "cmd"
    # A checker given no spawn cannot reach its command path: unchecked proves it did not.
    draft = _draft(("ui.js", E), language="javascript")
    said = _call(draft, ("ui.js", None, "function go( {\n"))
    assert said == "ok: ui.js written, 1 lines\nsyntax unchecked: ui.js"


def test_inproc_never_needs_a_process() -> None:
    a1 = SyntaxChecker(Path("/nowhere"), (load_profile("python"),))
    assert a1.inproc("README.md", b"x") is None
    ok = a1.inproc("app.py", b"x = 1\n")
    assert ok is not None and ok.ok
    js = SyntaxChecker(Path("/nowhere"), (load_profile("javascript"),))
    assert js.inproc("ui.js", b"function (") is None
    # Gate A1, given no spawn, raised `GateSpawn` here; the checker that
    # replaced it judges nothing it would need a process for, and says so (None).
    assert js.check("ui.js", b"function (") is None


def test_source_too_complex_to_compile_fails_and_never_crashes() -> None:
    # T-3.134 review: compile() raised MemoryError and RecursionError past A1, out of the call.
    a1 = SyntaxChecker(Path("/nowhere"), (load_profile("python"),))
    for text in ("-" * 200_000 + "1\n", "a" + ".b" * 300_000 + "\n"):
        judged = a1.inproc("m.py", text.encode())
        assert judged is not None and not judged.ok
        assert judged.detail is not None and "too complex to compile" in judged.detail
    draft = _draft()
    refused = _call(draft, ("app.py", None, "x = " + "-" * 200_000 + "1\n"))
    assert isinstance(refused, DraftRefused) and refused.failure == "SYNTAX_FAIL"
    assert draft.text("app.py") == FILES["app.py"]
