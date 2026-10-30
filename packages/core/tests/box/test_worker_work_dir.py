"""The worker's instructions name the configured `work.dir`, and no
shipped template names the default folder by its literal path."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.adapters.base import SHARED_TEMPLATES
from chickadee.box.worker import instructions
from chickadee.schema import sha256_hex
from chickadee.work import DEFAULT_DIR, WORK_DIR
from chickadee.zeus.prefix import TEMPLATES

RENDERED = {TEMPLATES / "zeus.md", SHARED_TEMPLATES / "worker.md"}
"""The templates `work.with_work_dir` fills: Zeus's prefix and the worker's instructions."""


def _templates() -> list[Path]:
    return [p for root in (TEMPLATES, SHARED_TEMPLATES) for p in Path(root).rglob("*.md")]


def test_the_worker_is_told_the_configured_folder() -> None:
    told = instructions(work_dir="notes/designs")
    assert "The work folder (`notes/designs/`) holds the design and the plan" in told
    assert "docs/chickadee" not in told and WORK_DIR not in told


@pytest.mark.parametrize("configured", ["notes/designs/", "./notes/designs"])
def test_a_folder_is_told_as_the_fence_reads_it(configured: str) -> None:
    assert "(`notes/designs/`)" in instructions(work_dir=configured)


def test_with_the_default_the_instructions_are_todays_bytes() -> None:
    """The pin of the default's bytes: moved once by D208, its `why` line removed."""
    told = instructions(work_dir=DEFAULT_DIR)
    assert (
        sha256_hex(told) == "e5b91ba3256caf7c25ae719136d49188318c21d9c55001af44f374bfb6aa2f6a"
    )


def test_no_shipped_template_names_the_default_folder() -> None:
    """A rendered template names `{work_dir}`: the folder is the config's, not the prompt's."""
    named = [str(p) for p in _templates() if "docs/chickadee" in p.read_text(encoding="utf-8")]
    assert named == []


def test_only_a_rendered_template_names_the_placeholder() -> None:
    """Any other template reaches its model as written: a `{work_dir}` there would be sent
    literally. A new one names the folder only once its loader fills it."""
    unrendered = [str(p) for p in _templates() if WORK_DIR in p.read_text(encoding="utf-8")]
    assert sorted(unrendered) == sorted(str(p) for p in RENDERED)


def test_a_long_work_dir_is_measured_where_the_request_sends_it() -> None:
    """The load-time fit check counts the prompt as rendered."""
    from chickadee.config import (
        _template_tokens,  # pyright: ignore[reportPrivateUsage]
    )

    short = _template_tokens("shared", "worker.md", DEFAULT_DIR)
    long = _template_tokens("shared", "worker.md", "a/" * 400 + "work")
    assert long > short
