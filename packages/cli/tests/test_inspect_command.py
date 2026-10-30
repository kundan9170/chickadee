"""`chickadee inspect` serves the reader's JSON and the built page, read-only; outside a
repository it says so and exits 1."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chickadee.engine import inspect_runs
from chickadee_cli import inspector
from chickadee_cli.surface import USAGE


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    web = tmp_path / "web"
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), inspector.handler(inspect_runs(state=tmp_path / "state"), web)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", web
    server.shutdown()
    server.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_api_answers_and_an_unknown_run_is_not_found(served: tuple[str, Path]) -> None:
    base, _ = served
    status, body = _get(f"{base}/api/index")
    assert status == 200 and json.loads(body)["runs"] == []
    assert _get(f"{base}/api/runs/nope")[0] == 404
    assert _get(f"{base}/api/elsewhere")[0] == 404


def test_the_page_is_served_once_built(served: tuple[str, Path]) -> None:
    base, web = served
    status, body = _get(f"{base}/")
    assert status == 503 and b"npm run build" in body
    web.mkdir()
    (web / "index.html").write_text("<html>page</html>")
    assert _get(f"{base}/any/route") == (200, b"<html>page</html>")


def test_outside_a_repository_it_says_so(tmp_path: Path) -> None:
    said: list[str] = []
    code = inspector.run(
        tmp_path, repo=None, state=None, host="127.0.0.1", port=0, web=None, say=said.append
    )
    assert code == USAGE and "not a repository" in said[0]
