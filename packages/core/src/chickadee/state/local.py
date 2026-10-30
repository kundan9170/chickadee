"""The person's choices for this clone: the `local` layer.

`<repo-local>/work.toml`, beside `trusted.toml`: never committed, one per clone, and plain text
so a person can read and delete what they chose. It holds one answer, the work folder the person
chose for this repository when the configured one could not be used (a file at its path, or a
folder git ignores). Read as the `local` configuration layer, between the repository's file and
the environment (`layers.load_layers`), and refused at load holding anything else
(`resolve.check_local`). The engine config, the person's for every repository, is never written
for it (`chickadee setup` never edits it).
"""

from __future__ import annotations

import json
import pathlib
from typing import Final

from chickadee.durable.files import write_atomic

__all__ = ["LOCAL_FILE", "record_work_dir"]

LOCAL_FILE: Final = "work.toml"
"""The file's name under the clone's repo-local directory."""


def record_work_dir(repo_local: pathlib.Path, work_dir: str) -> pathlib.Path:
    """Keep `work_dir` as this clone's work folder, replacing any kept before: the one writer of
    the file, on the person's yes. Crash-safe: old or new, never a partial file. A TOML basic
    string is written as JSON quotes it, which TOML reads the same for a path. Returns the
    file."""
    path = repo_local / LOCAL_FILE
    write_atomic(path, f"dir = {json.dumps(work_dir)}\n".encode())
    return path
