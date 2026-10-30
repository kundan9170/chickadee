"""The `local` layer: the person's choices for this clone.

`<repo-local>/work.toml`, beside `trusted.toml` and never committed, holds the work folder the
person chose for this repository when the configured one could not be used. It wins over the
engine config here alone, and may hold `[work] dir` and nothing else.
"""

from __future__ import annotations

import pathlib

import pytest
from engines import config_toml, raw_config
from repos import layer_roots

from chickadee.state.layers import LAYER_ORDER, load_layers
from chickadee.state.local import LOCAL_FILE, record_work_dir
from chickadee.state.merge import MergeError
from chickadee.state.resolve import resolve
from chickadee.state.roots import Roots


def _person(roots: Roots, work_dir: str | None = None) -> None:
    """A complete engine config, with `[work] dir` when given."""
    raw = raw_config()
    if work_dir is not None:
        raw["work"] = {"dir": work_dir}
    (roots.user / "config.toml").write_text(config_toml(raw))


def test_the_layer_sits_between_the_repository_and_the_environment() -> None:
    assert LAYER_ORDER == ("defaults", "user", "project", "local", "environment")


def test_a_recorded_folder_wins_over_the_engine_config_here(tmp_path: pathlib.Path) -> None:
    roots = layer_roots(tmp_path)
    _person(roots, "docs/designs")
    record_work_dir(roots.repo_local, "notes/chickadee")

    resolved = resolve(load_layers(roots, environ={}))

    assert resolved.engine.work.dir == "notes/chickadee"
    assert (roots.repo_local / LOCAL_FILE).read_text() == 'dir = "notes/chickadee"\n'


def test_without_the_file_the_engine_config_decides(tmp_path: pathlib.Path) -> None:
    roots = layer_roots(tmp_path)
    _person(roots, "docs/designs")

    assert resolve(load_layers(roots, environ={})).engine.work.dir == "docs/designs"


def test_recording_again_replaces_the_folder(tmp_path: pathlib.Path) -> None:
    roots = layer_roots(tmp_path)
    _person(roots)
    record_work_dir(roots.repo_local, "notes/chickadee")
    record_work_dir(roots.repo_local, "chickadee-work")

    assert resolve(load_layers(roots, environ={})).engine.work.dir == "chickadee-work"


@pytest.mark.parametrize(
    "body",
    ['dir = "x"\ncommit = false\n', "[spend]\nhard_cost_usd = 9.0\n", 'other = "y"\n'],
)
def test_anything_but_the_folder_refuses_naming_the_file(
    tmp_path: pathlib.Path, body: str
) -> None:
    """The file is the harness's record of one answer, never a second engine config."""
    roots = layer_roots(tmp_path)
    _person(roots)
    (roots.repo_local / LOCAL_FILE).write_text(body)

    with pytest.raises(MergeError) as refused:
        resolve(load_layers(roots, environ={}))
    assert LOCAL_FILE in str(refused.value) and "layer: local" in str(refused.value)
