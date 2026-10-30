"""`chickadee-serve` (and `python -m chickadee_serve`): one repository's engine, or a
recording, behind the wire. `run(argv)` is what the CLI mounts as
`chickadee serve`.

    chickadee-serve [--repo PATH] [--tcp PORT [--host HOST]]
    chickadee-serve --replay STATE_DIR --session ID [--speed N] [--tcp PORT [--host HOST]]

**Live** serves the repository containing `--repo` (the working directory by default) on a
Unix socket in its repo-local state, or on loopback TCP. **Replay** serves a recorded session
read-only, from its own scratch directory, which is where its socket and token file are. Either
writes its tokens owner-only to `serve.token` beside the socket and prints where both are,
never a token. SIGINT or SIGTERM stops it: the engine is closed, the socket and the token
file removed, and it exits 0.

Exit codes are the CLI's, which `chickadee serve` shares: 0 stopped cleanly; 1 a
usage error, a bind the server forbids or a config that does not load; 4 a path that is not a
repository; 5 a repository another server already serves (`refusals.REFUSAL_EXIT`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from chickadee.engine import PacketEngine
from chickadee.state.load import load_host_config
from chickadee_serve.auth import TokenStore
from chickadee_serve.bind import Bind, Tcp, loopback_tcp
from chickadee_serve.refusals import REFUSAL_EXIT
from chickadee_serve.server import (
    listen,
    live,
    refused,
    replay,
    running,
    scratch_dir,
    socket_in,
)

__all__ = ["USAGE", "main", "parser", "run"]

USAGE = 1
"""A command line that does not parse: `1`, not argparse's own `2`."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(USAGE, f"{self.prog}: error: {message}\n")


def parser() -> argparse.ArgumentParser:
    """The command line, which `chickadee serve` mounts as is."""
    parser = _Parser(
        prog="chickadee-serve",
        description="Serve one repository's engine, or a recorded session, over a socket.",
    )
    parser.add_argument("--repo", type=Path, default=None, help="the repository (default: .)")
    parser.add_argument("--tcp", type=int, metavar="PORT", help="loopback TCP, not a socket")
    parser.add_argument("--host", default="127.0.0.1", help="with --tcp: a loopback address")
    parser.add_argument("--replay", type=Path, metavar="STATE_DIR", help="a recorded state dir")
    parser.add_argument("--session", help="with --replay: the session to serve")
    parser.add_argument("--speed", type=float, default=1.0, help="with --replay: 0 is at once")
    return parser


def _say(message: str) -> None:
    print(f"chickadee-serve: {message}", file=sys.stderr, flush=True)


def _bind(args: argparse.Namespace, directory: Path) -> Bind:
    if args.tcp is not None:
        return loopback_tcp(args.host, args.tcp)
    return socket_in(directory)


def _announce(where: Bind, token_file: Path) -> None:
    address = where.url if isinstance(where, Tcp) else str(where.path)
    _say(f"serving on {address}; tokens in {token_file}")


def _serve_replay(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.session is None:
        parser.error("--replay needs --session")
    scratch = scratch_dir()
    try:
        store = TokenStore()
        state = scratch / "state"
        served, facade = replay(args.replay, args.session, state, store, speed=args.speed)
        with running(store, scratch, lambda: _bind(args, scratch)) as up:
            _announce(up.bind, up.token_file)
            facade.start()
            try:
                listen(served, up.bind)
            finally:
                facade.stop()
    finally:
        # The answers given to a replay are the replay's, not the recording's: they go with it.
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def _serve_live(args: argparse.Namespace) -> int:
    host = load_host_config(args.repo if args.repo is not None else Path.cwd())
    state = host.roots.repo_local
    store = TokenStore()
    # The server lock first: a second server is refused before an engine is built for it.
    with running(store, state, lambda: _bind(args, state)) as up:
        engine = PacketEngine(host.repo, host.engine, trust=host.trust)
        try:
            served = live(engine, store)
            _announce(up.bind, up.token_file)
            listen(served, up.bind)
        finally:
            engine.close()
    return 0


def run(argv: Sequence[str]) -> int:
    command = parser()
    args = command.parse_args(list(argv))
    try:
        if args.replay is not None:
            return _serve_replay(args, command)
        return _serve_live(args)
    except Exception as exc:
        why = refused(exc)
        if why is None:
            raise
        reason, message = why
        _say(message)
        return REFUSAL_EXIT[reason]


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
