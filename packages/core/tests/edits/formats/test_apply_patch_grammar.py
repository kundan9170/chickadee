"""The V4A grammar and its parser agree: every patch `apply_patch.lark`
admits, the parser takes; every envelope the grammar refuses, the parser refuses too, except the
leniencies only the parser has and the two refusals only it can make, each named here. And the
grammar is read the same by LLGuidance, the decoder OpenAI constrains with, as by Lark."""

from __future__ import annotations

import random

import pytest
from lark import Lark
from lark.exceptions import LarkError
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper, grammar_from

from chickadee.edits.formats import FORMATS, apply_patch, grammar
from chickadee.edits.formats.base import FileEdits, Targets

GRAMMAR = Lark(grammar("apply_patch"), parser="earley", lexer="dynamic")
NO_TARGETS = Targets(writable=(), deletes=())


def _admits(text: str) -> bool:
    try:
        GRAMMAR.parse(text)
    except LarkError:
        return False
    return True


def _takes(text: str) -> bool:
    parsed = apply_patch.parse(text, NO_TARGETS)
    return isinstance(parsed, FileEdits) and bool(parsed.edits)


def _patch(*lines: str) -> str:
    return "\n".join(lines) + "\n"


ADMITTED = {
    "update": _patch(
        "*** Begin Patch", "*** Update File: app.py", "-x = 1", "+x = 2", "*** End Patch"
    ),
    "context and scopes": _patch(
        "*** Begin Patch",
        "*** Update File: app.py",
        "@@ class Widget",
        "@@ def render",
        " def render(self):",
        "-    return 1",
        "+    return 2",
        "@@",
        "-y = 1",
        "+y = 2",
        "*** End of File",
        "*** End Patch",
    ),
    "a diff's range as scope": _patch(
        "*** Begin Patch",
        "*** Update File: a.py",
        "@@ -1,2 +1,2 @@",
        "-a",
        "+b",
        "*** End Patch",
    ),
    "add and update": _patch(
        "*** Begin Patch",
        "*** Add File: new.py",
        "+print('hi')",
        "+",
        "*** Update File: app.py",
        "+import new",
        "*** End Patch",
    ),
    "no final newline": "*** Begin Patch\n*** Add File: a.py\n+x\n*** End Patch",
    "an empty new file": _patch(
        "*** Begin Patch", "*** Add File: x/__init__.py", "*** End Patch"
    ),
    "a delete (D196 §10.3)": _patch(
        "*** Begin Patch", "*** Delete File: a.py", "*** End Patch"
    ),
    "a move with an edit": _patch(
        "*** Begin Patch",
        "*** Update File: a.py",
        "*** Move to: b.py",
        "-a",
        "+b",
        "*** End Patch",
    ),
    "a move alone": _patch(
        "*** Begin Patch", "*** Update File: a.py", "*** Move to: pkg/b.py", "*** End Patch"
    ),
    "markers quoted as text": _patch(
        "*** Begin Patch",
        "*** Update File: doc.md",
        " *** End Patch",
        "-*** Begin Patch",
        "+*** Update File: x",
        "*** End Patch",
    ),
}


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_what_the_grammar_admits_the_parser_takes(name: str) -> None:
    text = ADMITTED[name]
    assert _admits(text), name
    assert _takes(text), name


REFUSED_BY_BOTH = {
    "no end": _patch("*** Begin Patch", "*** Add File: a.py", "+x"),
    "no section": _patch("*** Begin Patch", "*** End Patch"),
    "a delete holding lines": _patch(
        "*** Begin Patch", "*** Delete File: a.py", "-a", "*** End Patch"
    ),
    "a move not first in its section": _patch(
        "*** Begin Patch",
        "*** Update File: a.py",
        "-a",
        "+b",
        "*** Move to: b.py",
        "*** End Patch",
    ),
    "a move after an add": _patch(
        "*** Begin Patch", "*** Add File: a.py", "*** Move to: b.py", "*** End Patch"
    ),
    "an add file's context line": _patch(
        "*** Begin Patch", "*** Add File: a.py", " x", "*** End Patch"
    ),
    "an update with no change": _patch(
        "*** Begin Patch", "*** Update File: a.py", " x", "*** End Patch"
    ),
    "a hunk of context alone": _patch(
        "*** Begin Patch",
        "*** Update File: a.py",
        "@@ one",
        " x",
        "@@ two",
        "-a",
        "+b",
        "*** End Patch",
    ),
    "a stray line": _patch(
        "*** Begin Patch", "*** Update File: a.py", "x = 1", "-a", "+b", "*** End Patch"
    ),
    "lines before a section": _patch("*** Begin Patch", "+x", "*** End Patch"),
}


@pytest.mark.parametrize("name", sorted(REFUSED_BY_BOTH))
def test_what_the_grammar_refuses_the_parser_refuses(name: str) -> None:
    text = REFUSED_BY_BOTH[name]
    assert not _admits(text), name
    assert not _takes(text), name


ONLY_THE_PARSER_TAKES = {
    "text around the envelope": "Here it is:\n"
    + _patch("*** Begin Patch", "*** Add File: a.py", "+x", "*** End Patch")
    + "Done.\n",
    "a blank line in a hunk": _patch(
        "*** Begin Patch", "*** Update File: a.py", "-a", "", "+b", "*** End Patch"
    ),
    "no space after the colon": _patch(
        "*** Begin Patch", "*** Add File:a.py", "+x", "*** End Patch"
    ),
    "trailing space on a marker": _patch(
        "*** Begin Patch  ", "*** Add File: a.py", "+x", "*** End Patch"
    ),
    "an end-of-file line anywhere": _patch(
        "*** Begin Patch", "*** End of File", "*** Add File: a.py", "+x", "*** End Patch"
    ),
}


@pytest.mark.parametrize("name", sorted(ONLY_THE_PARSER_TAKES))
def test_the_parsers_leniencies_are_named(name: str) -> None:
    # A reply needs them; a call decoded under the grammar never produces them.
    text = ONLY_THE_PARSER_TAKES[name]
    assert _takes(text) and not _admits(text), name


ONLY_THE_PARSER_REFUSES = {
    "one section per file": _patch(
        "*** Begin Patch",
        "*** Add File: a.py",
        "+x",
        "*** Add File: a.py",
        "+y",
        "*** End Patch",
    ),
    "a hunk that changes nothing": _patch(
        "*** Begin Patch", "*** Update File: a.py", "-x", "+x", "*** End Patch"
    ),
}


@pytest.mark.parametrize("name", sorted(ONLY_THE_PARSER_REFUSES))
def test_the_two_refusals_only_the_parser_can_make(name: str) -> None:
    # A grammar has no memory: it cannot see a repeated path, or a line put back as it was.
    text = ONLY_THE_PARSER_REFUSES[name]
    assert _admits(text) and not _takes(text), name


# -- generated: the two agree on every envelope, the named cases aside -------------------------

_BODY = (
    "*** Add File: a.py",
    "*** Add File: b.py",
    "*** Update File: a.py",
    "*** Update File: c.py",
    "*** Delete File: a.py",
    "*** Move to: d.py",
    "*** End of File",
    "@@",
    "@@ def go",
    "+x",
    "+y",
    "-x",
    "-z",
    " x",
    " ",
    "+",
    "x",
)


def _generated(rng: random.Random) -> str:
    body = [rng.choice(_BODY) for _ in range(rng.randint(0, 7))]
    if body and rng.random() < 0.7:
        # Most envelopes open a section, so the generator reaches the grammar's language.
        body[0] = rng.choice(_BODY[:5])
    return _patch("*** Begin Patch", *body, "*** End Patch")


def _dropped(text: str) -> str:
    """`text` without the lines the parser drops: an empty line or an end-of-file line
    anywhere, and a whitespace-only one before the first section (inside one, `" "` is a
    context line)."""
    kept: list[str] = []
    opened = False
    for ln in text.splitlines():
        opened = opened or ln.startswith(("*** Add File", "*** Update File", "*** Delete File"))
        if ln and ln != "*** End of File" and (opened or ln.strip()):
            kept.append(ln)
    return _patch(*kept)


def _parser_only_refusal(text: str) -> bool:
    """One of the two refusals no grammar can make: a path given twice, or a hunk whose
    removed and added lines are the same."""
    headers = [
        ln
        for ln in text.splitlines()
        if ln.startswith(("*** Add File", "*** Update File", "*** Delete File"))
    ]
    paths = [h.split(": ", 1)[1] for h in headers]
    if len(paths) != len(set(paths)):
        return True
    parsed = apply_patch.parse(text, NO_TARGETS)
    return not isinstance(parsed, FileEdits) and "changes nothing" in parsed.hint


def test_the_grammar_and_the_parser_agree_on_generated_envelopes() -> None:
    rng = random.Random(1729)
    admitted = 0
    for _ in range(4000):
        text = _generated(rng)
        admits, takes = _admits(text), _takes(text)
        if admits:
            admitted += 1
            assert takes or _parser_only_refusal(text), text
        elif takes:
            # Generated envelopes are canonical (no text around them, a space after each colon,
            # no trailing space), so the parser's only leniencies left are the lines it drops:
            # a blank line, and an end-of-file line away from a section's end.
            assert _admits(_dropped(text)), text
    assert admitted > 200  # the generator reaches the grammar's language, not only its edge


def test_the_grammar_is_the_one_format_that_keeps_one() -> None:
    assert grammar("apply_patch").startswith("// The V4A patch")
    with pytest.raises(KeyError):
        grammar("nonesuch")
    assert "apply_patch" in FORMATS


# -- LLGuidance, the decoder that constrains it ------------------------------------------------


class _Bytes:
    """A byte-level vocabulary, so LLGuidance can judge any text without a model's own."""

    eos_token_id = 256
    bos_token_id = None
    tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]

    def __call__(self, text: str | bytes) -> list[int]:
        return list(text.encode() if isinstance(text, str) else text)


_LLG = grammar_from("lark", grammar("apply_patch"))
_BYTES = LLTokenizer(TokenizerWrapper(_Bytes()))


def _decodable(text: str) -> bool:
    matcher = LLMatcher(_BYTES, _LLG, log_level=0)
    return all(matcher.consume_token(b) for b in text.encode()) and matcher.is_accepting()


def test_the_grammar_is_inside_llguidances_lark() -> None:
    # No lookarounds, priorities or imports: OpenAI's decoder would refuse the tool.
    is_error, said = LLMatcher.validate_grammar_with_warnings(_LLG)
    assert not is_error and not said, said


def test_llguidance_and_lark_admit_the_same_patches() -> None:
    cases = [
        *ADMITTED.values(),
        *REFUSED_BY_BOTH.values(),
        *ONLY_THE_PARSER_TAKES.values(),
        *ONLY_THE_PARSER_REFUSES.values(),
    ]
    rng = random.Random(1729)
    cases += [_generated(rng) for _ in range(500)]
    differ = [text for text in cases if _decodable(text) != _admits(text)]
    assert not differ, differ[:3]
