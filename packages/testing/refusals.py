"""A refusal as a test reads it: what it states, and what it tells the model to do.

Every failure a model reads is `<code and fields>. Do: <the next action>`. A pin on what a
refusal says compares its statement, and holds that an instruction follows, so a test written
in older versions keeps its meaning and gains the one check D221 added.
"""

from __future__ import annotations

DO = ". Do: "


LINE = "\nDo: "
"""A multi-line refusal (`INVALID_ARGUMENTS: tool` and its violations) gives it a line."""


def stated(text: str) -> str:
    """`text` less its instruction: the code, the fields and the reason, as in older
    versions."""
    if LINE in text:
        head, _, rest = text.partition(LINE)
        _, nl, tail = rest.partition("\n")
        return head + (nl + tail if nl else "")
    head, _, _ = text.partition(DO)
    return head


def instruction(text: str) -> str:
    """What `text` tells the model to do; fails when it tells it nothing."""
    if LINE in text:
        return text.partition(LINE)[2].partition("\n")[0]
    _, sep, do = text.partition(DO)
    assert sep, f"a refusal with no instruction (D221): {text!r}"
    return do


def bare[F](failure: F) -> F:
    """`failure` with its site's instruction set aside, to compare what it states: a typed
    failure that carries one (`InvalidArgument.do`) must have it."""
    import dataclasses

    do = getattr(failure, "do", None)
    assert do, f"a refusal with no instruction (D221): {failure!r}"
    return dataclasses.replace(failure, do=None)  # type: ignore[type-var]
