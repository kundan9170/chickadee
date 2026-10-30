"""PR-7 — profiles freeze at run start; a mid-run edit does not take effect.

Without this, two runs replayed from the same journal could diverge for reasons the
journal does not record, which would make replay a claim rather than a property.
"""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from chickadee.profiles import ProfileSet, load_profile


def test_mid_run_edit_has_no_effect(builtin: pathlib.Path) -> None:
    loaded = ProfileSet.load(builtin_dir=builtin)
    before = loaded.by_name("python")
    assert before is not None and before.tests.runner is not None
    assert before.tests.runner.argv[0] == "pytest"

    (builtin / "python.toml").write_text(
        'name = "python"\nextensions = [".py"]\ntree_sitter_grammar = "python"\n'
        '[syntax]\nmode = "inproc"\n'
        '[tests]\nrunner = "nose {selector}"\n'
    )

    after = loaded.by_name("python")
    assert after is not None and after.tests.runner is not None
    assert after.tests.runner.argv[0] == "pytest", "the in-flight run keeps what it loaded"

    fresh = load_profile("python", builtin_dir=builtin)
    assert fresh.tests.runner is not None
    assert fresh.tests.runner.argv[0] == "nose", "a new run sees the edit"


def test_profile_objects_are_immutable(builtin: pathlib.Path) -> None:
    p = load_profile("python", builtin_dir=builtin)
    with pytest.raises(ValidationError):
        p.name = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        p.tests.timeout_s = 1  # type: ignore[misc]
