"""`chickadee inspect`: a local page showing what each model was sent and said.

Host-side, like `gc` and `replay`: `chickadee.engine.inspect_runs` reads the runs and sessions
on disk in this process, read-only, so it runs beside a session without touching it. This module
only serves: the page's built files (`ui/inspector/dist`, a React app) and a small JSON API the
page polls with `after=<seq>`, so only new lines travel.

    GET /api/index every session and run
    GET /api/runs/<id>?after=<seq> a run's lines, with messages, requests and blobs
    GET /api/sessions/<id>?after=<seq> a session's lines, with Zeus's prompt
    GET /api/zeus-template the template on disk, for a session from an older version
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urlparse

from chickadee.engine import Inspection, NotARepository, inspect_runs
from chickadee_cli.surface import DONE, USAGE

__all__ = ["WEB", "handler", "run"]

WEB: Final = Path(__file__).resolve().parents[4] / "ui" / "inspector" / "dist"
"""The built page in this checkout. The CLI is installed editable from the repository, so the
page sits beside it; `--web` names another build."""

_UNBUILT: Final = (
    "The page is not built yet. Run, in the chickadee checkout:\n\n"
    "  cd ui/inspector && npm install && npm run build\n"
)


def handler(inspection: Inspection, web: Path) -> type[BaseHTTPRequestHandler]:
    """The request handler over `inspection`, serving the page from `web`."""
    api: dict[str, Callable[[str, int], Any]] = {
        "runs": inspection.run,
        "sessions": inspection.session,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def do_GET(self) -> None:
            url = urlparse(self.path)
            after = parse_qs(url.query).get("after", ["0"])[0]
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts[:1] != ["api"]:
                    return self._static(url.path)
                if parts == ["api", "index"]:
                    return self._json(inspection.index())
                if parts == ["api", "zeus-template"]:
                    return self._json(inspection.zeus_template())
                if len(parts) == 3 and parts[1] in api:
                    return self._json(api[parts[1]](parts[2], int(after or 0)))
                self._json({"error": "no such route"}, HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, ValueError) as exc:
                self._json({"error": f"not found: {exc}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # a damaged record is said on the page, not a dead server
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _send(self, status: int, kind: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, body: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, "application/json", json.dumps(body).encode())

        def _static(self, path: str) -> None:
            if not (web / "index.html").is_file():
                return self._send(503, "text/plain; charset=utf-8", _UNBUILT.encode())
            target = (web / path.lstrip("/")).resolve()
            if not target.is_file() or web.resolve() not in target.parents:
                target = web / "index.html"
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, kind, target.read_bytes())

    return Handler


def run(
    cwd: Path,
    *,
    repo: Path | None,
    state: Path | None,
    host: str,
    port: int,
    web: Path | None,
    say: Callable[[str], None],
) -> int:
    """Serve until interrupted. `1` when there is nothing to read or the port is taken."""
    try:
        inspection = inspect_runs(cwd if repo is None else repo, state=state)
    except NotARepository as exc:
        say(f"inspect: not a repository: {exc}; pass --repo or --state")
        return USAGE
    page = web or WEB
    try:
        server = ThreadingHTTPServer((host, port), handler(inspection, page))
    except OSError as exc:
        say(f"inspect: cannot listen on {host}:{port}: {exc.strerror}; try --port")
        return USAGE
    say(f"chickadee inspect: reading {inspection.state}")
    if not (page / "index.html").is_file():
        say(f"the page is not built: {_UNBUILT.splitlines()[-1].strip()}")
    say(f"open http://{host}:{server.server_address[1]}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    server.server_close()
    return DONE
