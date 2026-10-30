"""A box role's reader opens its own run's blobs: the box shell stores a long
command's output in the run journal and hands back a reference, which answered
`HANDLE_EXPIRED{absent}` while the reader had no run blob store. Another run's blobs stay
closed."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from box_helpers import FakeRunJournal

from chickadee.box.compose import _reader  # pyright: ignore[reportPrivateUsage]
from chickadee.box.report import stored_handle
from chickadee.context.results import ToolResult
from chickadee.context.terms import Vision
from chickadee.journal.errors import BlobMissing
from chickadee.journal.port import JournalPort
from chickadee.secrets.denylist import Denylist


class _RunBlobs(FakeRunJournal):
    def get_blob(self, sha: str) -> bytes:
        if sha not in self.blobs:
            raise BlobMissing(sha, run_id=self.run_id)
        return self.blobs[sha]


def test_a_shell_outputs_reference_opens_in_the_box_reader(tmp_path: Path) -> None:
    journal = _RunBlobs()
    output = "".join(f"line {n}\n" for n in range(500)).encode("utf-8")
    journal.put_blob(output)
    reader = _reader(tmp_path, Denylist(), Vision("worker"), cast(JournalPort, journal), None)
    got = reader.read(stored_handle("output", journal.run_id, output), lines="490-492")
    assert isinstance(got, ToolResult), got
    assert got.content is not None and "line 490" in got.content and "line 491" in got.content


def test_another_runs_blob_does_not_open(tmp_path: Path) -> None:
    journal, other = _RunBlobs(), b"another run's output\n"
    journal.put_blob(other)
    reader = _reader(tmp_path, Denylist(), Vision("worker"), cast(JournalPort, journal), None)
    got = reader.read(stored_handle("output", "run-2", other))
    assert not isinstance(got, ToolResult)
    assert "HANDLE_EXPIRED" in got.render()
