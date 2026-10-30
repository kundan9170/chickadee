"""No refusal without an exit the harness owns.

D230's run of 2026-10-24 ended at refusals whose only exit was *the person edits config*:
`.DS_Store` in the work folder, the person's own files there, a dirty
checkout. Each is gone. This file keeps the next one from arriving unnoticed: every
text in the core that sends someone to edit their config is listed below with why it may, and
a new one fails until it is listed, or given an exit the harness owns.

The pairs D230 names, which together left no exit, are pinned where they now open: the work
folder's owner check against the shell's put-back of a file the operating system wrote
(`tests/tools/test_doc.py`, `tests/engine/test_declared_write.py`), and the owner check against
the person's own files (`tests/tools/test_doc.py`, `tests/engine/test_work_folder.py`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chickadee.work import owned_error

SRC = Path(__file__).resolve().parents[1] / "src" / "chickadee"

SENDS_TO_CONFIG = re.compile(
    r"(?:in|to) (?:your |the )engine config|config\.toml|set \[[a-z_.]+\]"
    r"|set [a-z_]+\.[a-z_]+ =",
    re.IGNORECASE,
)
"""A text that tells someone to change their configuration: something set, added or edited
in it, never a text that only says where a value comes from."""

ALLOWED: dict[str, str] = {
    "zeus/docs.py": (
        "the work folder unusable as a folder: `[work] dir` is a file, or git ignores it. "
        "Since T-3.215 (D230 Amendment 3) the harness first offers a free folder for this "
        "repository, kept for this clone; the config hint is said only after a no, an "
        "unattended run, or no free folder, as the person's own way forward"
    ),
    "work.py": (
        "`owned_error`'s not-a-folder answer, which `zeus/docs.py` turns into the refusal "
        "above; and B7's one notice that `work.commit = false` keeps designs out of git"
    ),
    "pycall.py": (
        "running the worker's Python unsandboxed is a security setting the person makes "
        "deliberately; no harness remedy should offer it in one step (D226 decision 11)"
    ),
    "isolation/container.py": (
        "the container backend is opt-in, said to the person setting up the sandbox, never "
        "a refusal a model reads"
    ),
    "engine.py": (
        "`switch_model` to a model with no price row: raised to the person's host, who "
        "chose the model, never to a model"
    ),
}
"""Every core file that may say it, and why. Configuration's own modules are left out: their
texts describe the configuration."""

SKIPPED_DIRS = ("state/", "init/")
SKIPPED_FILES = ("config.py",)


def _sayers() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in SKIPPED_FILES or rel.startswith(SKIPPED_DIRS):
            continue
        text = path.read_text(encoding="utf-8")
        for literal in re.findall(r"\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'", text):
            if SENDS_TO_CONFIG.search(literal):
                found.setdefault(rel, []).append(literal)
    return found


def test_no_new_text_sends_the_person_to_edit_their_config() -> None:
    unlisted = {rel: lines for rel, lines in _sayers().items() if rel not in ALLOWED}
    assert unlisted == {}, (
        "a text sends someone to edit their config: give it an exit the harness owns "
        f"(D230 decision 6), or list it in ALLOWED with why: {unlisted}"
    )


def test_the_list_holds_no_file_that_no_longer_says_it() -> None:
    """A ratchet: a file that lost its last such text leaves the list too."""
    stale = sorted(set(ALLOWED) - set(_sayers()))
    assert stale == [], f"ALLOWED names files that no longer send anyone to config: {stale}"


@pytest.mark.parametrize("theirs", ["notes.md", "README.md", ".DS_Store", "old/README.md"])
def test_the_work_folder_never_refuses_on_what_its_owner_check_used_to(
    tmp_path: Path, theirs: str
) -> None:
    """The first of D230's pairs, from the owner check's side: nothing the shell's put-back
    could keep the person from removing is a reason to refuse any more."""
    path = tmp_path / "docs" / "chickadee" / theirs
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n")
    assert owned_error(tmp_path, "docs/chickadee") is None
