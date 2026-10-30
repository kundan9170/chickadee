"""`chickadee`.

**Bare `chickadee` is a socket client of its own engine.** It hosts the repository's engine
on a thread of this process (`chickadee_wire.host.host_in_process`), connects to it, and runs
the REPL. `chickadee attach` is the same REPL against a server already running;
`chickadee serve` is the same hosting step without the REPL. `run`, `ask` and `resume` host
the same way and follow one turn without a REPL (`batch.py`).

**Confined.** Nothing here imports `chickadee.*`. What reaches the engine is imported
inside the command that needs it: hosting (`chickadee_wire.host`), `serve`
(`chickadee_serve.__main__`), and the commands that need no server, `init`, `setup`, `replay`,
`gc`, `trust` and `hooks`. So importing this module loads no engine (`test_repl_confined.py`).

**The surface is `surface.SURFACE`**, the exit-code table as values: each command here is a row
of it, and
leaves through `surface.leave`, which refuses a code the row does not hold. `main()` is the
entry point so that a command line that does not parse exits `1`, the usage code, rather
than Click's `2`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer

from chickadee_cli import __version__
from chickadee_cli.attach import NotStaged, Staging
from chickadee_cli.batch import BatchIO, follow, prepend
from chickadee_cli.commands import NO_ATTACHING
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.repl import Repl
from chickadee_cli.session import (
    NoSessionToContinue,
    ask_which,
    continuing_line,
    open_session,
)
from chickadee_cli.surface import (
    BARE,
    DONE,
    NOT_A_TTY,
    PRECONDITION,
    STOPPED,
    USAGE,
    VERSION,
    leave,
)
from chickadee_serve.bind import Tcp
from chickadee_serve.frames import TokenClass, TurnWorkspace
from chickadee_serve.refusals import REFUSAL_EXIT, Refusal
from chickadee_wire.client import Client
from chickadee_wire.errors import HostRefused, NotServing, WireError

if TYPE_CHECKING:
    from chickadee_wire.host import Hosted

__all__ = ["app", "main", "repl_over"]


_REMEDY: Final[dict[Refusal, str]] = {  # an unbindable path's message names --tcp itself
    "held": "; talk to it with `chickadee attach`",
    "config": (
        "\n  `chickadee setup` writes the engine config's template if there is none; name a "
        "model in it, the rungs that use it, and its price"
    ),
}

BATCH_HINT: Final = (
    "For a script, say the goal as a command:\n"
    '  chickadee run --yes "<goal>"     or     chickadee ask "<question>"'
)
"""What a terminal-only command says to a pipe: the batch commands, which a script can run."""
BATCH_EQUIVALENT: Final = f"chickadee: not a terminal, so no REPL. {BATCH_HINT}"
TUI_EQUIVALENT: Final = f"chickadee tui: not a terminal, so no TUI. {BATCH_HINT}"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Chickadee: a conversation with your repository.",
)
mcp = typer.Typer(help="The MCP servers this repository may call .")
app.add_typer(mcp, name="mcp")

Continue = Annotated[
    bool, typer.Option("--continue", help="continue the most recently updated session")
]
SessionOpt = Annotated[str | None, typer.Option("--session", help="continue this session")]
New = Annotated[
    bool,
    typer.Option("--new", help="start a fresh session, continuing none of the open work"),
]
WorktreeOpt = Annotated[
    bool,
    typer.Option("--worktree", help="run this session's work in a worktree, not this checkout"),
]
InPlaceOpt = Annotated[
    bool, typer.Option("--in-place", help="run this session's work in this checkout")
]
TcpOpt = Annotated[
    int | None,
    typer.Option("--tcp", help="a loopback port, in place of the repository's socket"),
]
ServerOpt = Annotated[
    Path | None,
    typer.Option(
        "--server",
        help="attach to the server whose socket and token file are in this directory, as "
        "`chickadee serve --replay` announces them",
    ),
]
Yes = Annotated[
    bool,
    typer.Option("--yes", help="answer every question with its default, as automation"),
]
Json = Annotated[
    bool, typer.Option("--json", help="the session's records on stdout, one per line")
]
Files = Annotated[
    list[Path] | None,
    typer.Option("--file", help="an image, a PDF or a text file sent with the turn"),
]
SecretFile = Annotated[
    bool,
    typer.Option(
        "--allow-secret-file",
        help="send a --file whose text holds a secret value; refused without it",
    ),
]


_UsageError: Final[type[Exception]] = next(
    c for c in typer.BadParameter.__mro__ if c.__name__ == "UsageError"
)
"""Click's usage error, which Typer vendors without exporting: found through the one public
subclass, so this holds whichever Click Typer carries."""


def main() -> None:
    """The `chickadee` entry point: the app, with a command line that does not parse exiting
    `1` instead of Click's `2`."""
    try:
        code = app(standalone_mode=False)
    except _UsageError as exc:
        exc.show()  # pyright: ignore[reportAttributeAccessIssue]
        sys.exit(USAGE)
    except typer.Abort:
        print("chickadee: aborted", file=sys.stderr)
        sys.exit(USAGE)
    sys.exit(code if isinstance(code, int) else 0)


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _workspace(command: str, worktree: bool, in_place: bool) -> TurnWorkspace | None:
    """Where the person said the session works: sent with the first turn,
    and the kind of open work a new session continues. None leaves it to the session."""
    if worktree and in_place:
        _err(f"chickadee {command}: --worktree and --in-place name two places; pick one")
        leave(command, USAGE)
    if worktree:
        return "worktree"
    return "in_place" if in_place else None


def _out(text: str) -> None:
    print(text, flush=True)


@app.callback(invoke_without_command=True)
def bare(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="print the version")] = False,
    resume_latest: Continue = False,
    session: SessionOpt = None,
    tcp: TcpOpt = None,
    new: New = False,
    worktree: WorktreeOpt = False,
    in_place: InPlaceOpt = False,
) -> None:
    """Bare `chickadee`: host this repository's engine and open the REPL."""
    if version:
        print(__version__)
        leave(VERSION, DONE)
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        _err(BATCH_EQUIVALENT)
        leave(BARE, NOT_A_TTY)
    where = _workspace(BARE, worktree, in_place)
    with _hosted(BARE, tcp) as hosted, hosted.client() as client:
        code = repl_over(
            client,
            hosted.state_dir,
            session=session,
            resume_latest=resume_latest,
            fresh=new,
            workspace=where,
        )
    leave(BARE, code)


@app.command()
def attach(
    tcp: TcpOpt = None,
    resume_latest: Continue = False,
    session: SessionOpt = None,
    new: New = False,
    worktree: WorktreeOpt = False,
    in_place: InPlaceOpt = False,
) -> None:
    """The REPL against a server already running on this repository (`chickadee serve`)."""
    from chickadee_wire.host import state_dir_of

    where = _workspace("attach", worktree, in_place)
    try:
        state_dir = state_dir_of(Path.cwd())
    except HostRefused as exc:
        _err(f"chickadee: {exc}")
        leave("attach", PRECONDITION)
    try:
        client = Client.attach(state_dir, tcp=_loopback(tcp))
    except NotServing as exc:
        _err(f"chickadee: {exc}; start one with `chickadee serve`")
        leave("attach", USAGE)
    with client:
        code = repl_over(
            client,
            state_dir,
            session=session,
            resume_latest=resume_latest,
            fresh=new,
            workspace=where,
        )
    leave("attach", code)


@app.command()
def tui(
    resume_latest: Continue = False,
    session: SessionOpt = None,
    tcp: TcpOpt = None,
    server: ServerOpt = None,
) -> None:
    """The TUI: Zeus, Athena and the worker on one screen, over an engine
    hosted in this process, or over the server in `--server`'s directory (a replay's, say)."""
    if not _screen():
        _err(TUI_EQUIVALENT)
        leave("tui", NOT_A_TTY)
    if server is not None:
        try:
            client = Client.attach(server, tcp=_loopback(tcp))
        except NotServing as exc:
            _err(f"chickadee tui: {exc}")
            leave("tui", USAGE)
        with client:
            leave("tui", _tui_over(client, session, resume_latest, repo=None))
    with _hosted("tui", tcp) as hosted, hosted.client() as client:
        code = _tui_over(client, session, resume_latest, repo=hosted.repo.name)
    leave("tui", code)


def _screen() -> bool:
    """Whether a full screen can run here: input from a terminal and output to one."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _tui_over(
    client: Client,
    session: str | None,
    resume_latest: bool,
    *,
    repo: str | None,
) -> int:
    """The TUI over `client`, on its session; the code. Textual is imported here, so a
    command that opens no screen never loads it.

    Leaving with work not handed off asks what the REPL asks as it leaves:
    the screen's handoff overlay, before it closes. A replay has
    nothing to hand off."""
    from chickadee_cli import tui_prefs
    from chickadee_cli.tui.app import run as run_tui

    try:
        info = open_session(client, session_id=session, resume_latest=resume_latest)
    except (NoSessionToContinue, WireError) as exc:
        _err(f"chickadee tui: {exc}")
        return USAGE
    path = tui_prefs.prefs_path()
    return run_tui(
        client,
        info.session_id,
        repo=repo,
        resume_paused=session is not None or resume_latest,
        full_screen=path is not None and tui_prefs.load(path).full_screen,
        remember=None
        if path is None
        else lambda on: tui_prefs.save(path, tui_prefs.TuiPrefs(full_screen=on)),
    )


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="what to do")],
    yes: Yes = False,
    resume_latest: Continue = False,
    session: SessionOpt = None,
    in_place: InPlaceOpt = False,
    file: Files = None,
    allow_secret_file: SecretFile = False,
    json: Json = False,
    new: New = False,
    worktree: WorktreeOpt = False,
) -> None:
    """One turn toward `goal`, followed until the session is idle. A new
    session continues the newest open work, unless `--new`. With `--yes`
    the work runs in a worktree unless `--in-place`."""
    asked = _workspace("run", worktree, in_place)
    _turn(
        "run",
        goal,
        yes=yes,
        latest=resume_latest,
        session=session,
        file=file,
        allow_secret_file=allow_secret_file,
        json=json,
        fresh=new,
        workspace=asked or ("worktree" if yes else None),
        only=asked,
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="what to ask")],
    yes: Yes = False,
    resume_latest: Continue = False,
    session: SessionOpt = None,
    file: Files = None,
    allow_secret_file: SecretFile = False,
    json: Json = False,
) -> None:
    """A question, answered in one turn."""
    _turn(
        "ask",
        question,
        yes=yes,
        latest=resume_latest,
        session=session,
        file=file,
        allow_secret_file=allow_secret_file,
        json=json,
        # A question needs no open work: `ask` never continues one.
        fresh=True,
    )


@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(metavar="SESSION", help="the session to resume")],
    yes: Yes = False,
    json: Json = False,
) -> None:
    """Resume a session whose process died, and follow its unfinished run."""
    with _hosted("resume", None) as hosted, _clients(hosted, yes=yes) as (client, answerer):
        try:
            info = open_session(client, session_id=session_id, resume_latest=False)
            resumed = client.resume(info.session_id)
        except WireError as exc:
            _err(f"chickadee resume: {exc}")
            leave("resume", USAGE)
        how = _batch(json=json, answerer=answerer)
        for run_id in resumed.reconciled:
            how.say(f"reconciled run {run_id}")
        if resumed.in_flight is None:
            how.say(f"session {info.session_id}: nothing was in flight")
            leave("resume", DONE)
        how.say(f"session {info.session_id}: continuing run {resumed.in_flight}")
        code = _follow("resume", client, info.session_id, how, after=info.last_seq, send=None)
    leave("resume", code)


@app.command()
def fork(
    session_id: Annotated[str, typer.Argument(metavar="SESSION", help="the session to fork")],
    at: Annotated[
        int | None, typer.Option("--at", help="the seq to fork at; the last by default")
    ] = None,
    reason: Annotated[str | None, typer.Option("--reason", help="why, for the child")] = None,
) -> None:
    """A new session from another's history."""
    from chickadee_cli import fork as forking

    def say(text: str, tone: Tone) -> None:
        for line in style.lines(text, tone):
            _out(line)

    style = Style(tty=sys.stdout.isatty())
    with _hosted("fork", None) as hosted, hosted.client() as client:
        code = forking.run(
            client,
            session_id,
            at=at,
            reason=reason,
            source=_TerminalLine() if sys.stdin.isatty() else None,
            say=say,
            style=style,
        )
    leave("fork", code)


@app.command()
def replay(run_id: Annotated[str, typer.Argument(metavar="RUN", help="the run")]) -> None:
    """Fold a run's journal and say whether it replays identically."""
    from chickadee_cli import replay as replaying

    code, text = replaying.run(Path.cwd(), run_id)
    _out(text)
    leave("replay", code)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def serve(ctx: typer.Context) -> None:
    """Serve this repository's engine on its socket, without a REPL."""
    from chickadee_serve.__main__ import run as serving

    leave("serve", serving(list(ctx.args)))


@app.command(name="trust")
def trust_repository() -> None:
    """Show what this repository declares that can run, and accept it for this clone."""
    from chickadee_cli import trust as trusting

    ask = _TerminalLine().read if sys.stdin.isatty() else None
    leave("trust", trusting.run(Path.cwd(), ask=ask, say=_out))


@app.command(name="hooks")
def hook_points() -> None:
    """Show the hook points and what this repository's config binds at each."""
    from chickadee_cli import hooks as hooking

    leave("hooks", hooking.run(Path.cwd(), say=_out))


@mcp.command()
def trust(
    server: Annotated[str, typer.Argument(help="the MCP server")],
    tools: Annotated[list[str] | None, typer.Argument(help="its tools")] = None,
) -> None:
    """Record the digests of a server's tools as trusted."""
    del server, tools
    _err("chickadee mcp trust: not built yet; it arrives with the MCP client")
    leave("mcp trust", USAGE)


@app.command()
def gc(
    older_than: Annotated[
        int | None,
        typer.Option("--older-than", min=0, help="days since a run was last written"),
    ] = None,
    blob_class: Annotated[
        list[str] | None, typer.Option("--class", help="a bulk class to prune (repeatable)")
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run", help="only this run")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="say, remove nothing")] = False,
) -> None:
    """Remove old finished runs' bulk blobs, logging each run."""
    from chickadee_cli import gc as collecting

    code, text = collecting.run(
        Path.cwd(),
        older_than_days=older_than if older_than is not None else collecting.RETAIN_DAYS,
        classes=blob_class or (),
        run_id=run_id,
        dry_run=dry_run,
    )
    (_out if code == 0 else _err)(text)
    leave("gc", code)


@app.command()
def inspect(
    repo: Annotated[
        Path | None,
        typer.Option("--repo", help="the project chickadee runs in (default: here)"),
    ] = None,
    state: Annotated[
        Path | None,
        typer.Option("--state", help="a state directory holding runs/ and sessions/, instead"),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="the address to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=0, help="the port to listen on")] = 8765,
    web: Annotated[
        Path | None,
        typer.Option("--web", help="a built page to serve instead of this checkout's"),
    ] = None,
) -> None:
    """Show what each model was sent and said, in a local web page (read-only)."""
    from chickadee_cli import inspector

    leave(
        "inspect",
        inspector.run(
            Path.cwd(), repo=repo, state=state, host=host, port=port, web=web, say=_err
        ),
    )


@app.command()
def init(
    force: Annotated[bool, typer.Option("--force", help="re-initialise")] = False,
) -> None:
    """Prepare this repository."""
    from chickadee_cli import init as initialise

    code, text = initialise.run(Path.cwd(), force=force, say=_out)
    (_out if code == 0 else _err)(text)
    leave("init", code)


@app.command()
def setup(
    install_servers: Annotated[
        bool,
        typer.Option(
            "--install-servers",
            help="install every missing language server that has an installer, unasked",
        ),
    ] = False,
) -> None:
    """Write the engine config when absent, probe the sandbox, and list each
    language's servers, offering to install the missing ones."""
    from chickadee_cli import setup as configure

    def ask(need: configure.ServerNeed) -> bool:
        if install_servers:
            return True
        if not sys.stdin.isatty():
            return False
        return typer.confirm(
            f"{need.server} ({', '.join(need.languages)}) is missing; install it now? "
            f"This runs: {' '.join(need.install or ())}",
            default=False,
        )

    code, text = configure.run(ask=ask)
    _out(text)
    leave("setup", code)


# -- shared steps -----------------------------------------------------------------------------


def _loopback(port: int | None) -> Tcp | None:
    return Tcp("127.0.0.1", port) if port is not None else None


@contextmanager
def _hosted(command: str, tcp: int | None) -> Iterator[Hosted]:
    """This repository's engine, hosted in this process; a refusal leaves `command` with the
    refusal's exit code."""
    from chickadee_wire.host import host_in_process

    _prepared(command)
    try:
        hosted = host_in_process(Path.cwd(), tcp=_loopback(tcp))
    except HostRefused as exc:
        _err(f"chickadee: {exc}{_REMEDY.get(exc.reason, '')}")
        leave(command, REFUSAL_EXIT[exc.reason])
    if hosted.fallback is not None and isinstance(hosted.address, Tcp):
        _err(
            f"chickadee: {hosted.fallback}; serving on {hosted.address.url} instead "
            f"(from another terminal: `chickadee attach --tcp {hosted.address.port}`)"
        )
    with hosted:
        yield hosted


def _prepared(command: str) -> None:
    """The folder made ready before an engine starts in it: a git repository made
    where there is none, and `init` run, the person told each step while they wait, rather
    than a refusal that names a command to run. Leaves with `init`'s code when it cannot."""
    from chickadee_cli import init as initialise

    failed = initialise.prepare(Path.cwd(), say=lambda line: _err(f"chickadee: {line}"))
    if failed is not None:
        code, text = failed
        _err(text)
        leave(command, code)


@contextmanager
def _clients(hosted: Hosted, *, yes: bool) -> Iterator[tuple[Client, Client | None]]:
    """The person's client, and with `--yes` an automation one to answer by default."""
    with hosted.client() as client:
        if not yes:
            yield client, None
            return
        with hosted.client(TokenClass.AUTOMATION) as answerer:
            yield client, answerer


class _TerminalLine:
    """A question's answer, read from the terminal."""

    def read(self, prompt_text: str, default: str = "") -> str:
        del default  # `input` edits no line
        return input(prompt_text)


def _batch(*, json: bool, answerer: Client | None) -> BatchIO:
    asks = answerer is None and sys.stdin.isatty()
    return BatchIO(
        json=json,
        out=_out,
        err=_err,
        style=Style(tty=sys.stdout.isatty() and not json),
        answerer=answerer,
        source=_TerminalLine() if asks else None,
    )


def _turn(
    command: str,
    goal: str,
    *,
    yes: bool,
    latest: bool,
    session: str | None,
    file: list[Path] | None,
    json: bool,
    fresh: bool = False,
    workspace: TurnWorkspace | None = None,
    only: TurnWorkspace | None = None,
    allow_secret_file: bool = False,
) -> None:
    """`workspace` goes with the turn; `only`, the flag the person typed, is the kind of open
    work a new session continues. Each `--file` is checked by the server
    before the turn is sent; one it refuses, or one holding a secret without
    `--allow-secret-file`, ends the command `USAGE` with nothing sent."""
    text = goal if sys.stdin.isatty() else prepend(sys.stdin.read(), goal)
    with _hosted(command, None) as hosted, _clients(hosted, yes=yes) as (client, answerer):
        try:
            info = open_session(
                client, session_id=session, resume_latest=latest, fresh=fresh, only=only
            )
        except (NoSessionToContinue, WireError) as exc:
            _err(f"chickadee {command}: {exc}")
            leave(command, USAGE)
        files = _files(command, client, info.session_id, file or (), allow_secret_file)
        how = _batch(json=json, answerer=answerer)
        how.say(f"session {info.session_id}")
        continuing = continuing_line(info, repl=False)
        if continuing is not None:
            how.say(continuing)
        code = _follow(
            command,
            client,
            info.session_id,
            how,
            after=info.last_seq,
            send=text,
            workspace=workspace,
            attachments=files,
        )
    leave(command, code)


def _files(
    command: str, client: Client, sid: str, paths: Sequence[Path], allow_secret: bool
) -> tuple[tuple[str, bytes], ...]:
    """`--file`'s files as the turn sends them, each checked by the server first (the
    REPL's `/attach` path): a refusal, or a secret not allowed, ends the command `USAGE`."""
    if not paths:
        return ()
    staging = Staging(client, sid)
    if not staging.offered:
        _err(f"chickadee {command}: --file: {NO_ATTACHING}")
        leave(command, USAGE)
    for path in paths:
        staged = staging.stage_file(path.expanduser())
        if isinstance(staged, NotStaged):
            _err(f"chickadee {command}: --file {path}: {staged.why}")
            leave(command, USAGE)
    held = staging.secrets()
    if held and not allow_secret:
        _err(
            f"chickadee {command}: --file {', '.join(held)} holds a secret value; "
            "pass --allow-secret-file to send it"
        )
        leave(command, USAGE)
    over = staging.over()
    if over is not None:
        _err(f"chickadee {command}: --file: {over}")
        leave(command, USAGE)
    return Staging.files(staging.take())


def _follow(
    command: str,
    client: Client,
    sid: str,
    how: BatchIO,
    *,
    after: int,
    send: str | None,
    workspace: TurnWorkspace | None = None,
    attachments: Sequence[tuple[str, bytes]] = (),
) -> int:
    try:
        return follow(
            client,
            sid,
            how,
            after=after,
            send=send,
            workspace=workspace,
            attachments=attachments,
        )
    except WireError as exc:
        _err(f"chickadee {command}: {exc}")
        return STOPPED


def repl_over(
    client: Client,
    state_dir: Path,
    *,
    session: str | None,
    resume_latest: bool,
    fresh: bool = False,
    workspace: TurnWorkspace | None = None,
) -> int:
    """The REPL over `client`, on a terminal: a prompt the session's events print above. A new
    session continues the repository's open work, asking once when there are several, unless
    `fresh`. `workspace` (`--worktree`, `--in-place`) goes with the first
    turn, and a new session continues open work of that kind alone."""
    from prompt_toolkit.patch_stdout import patch_stdout

    from chickadee_cli.prompt import Files, history_path, prompt_session

    try:
        info = open_session(
            client,
            session_id=session,
            resume_latest=resume_latest,
            fresh=fresh,
            choose=ask_which,
            only=workspace,
        )
    except (NoSessionToContinue, WireError) as exc:
        _err(f"chickadee: {exc}")
        return USAGE
    except KeyboardInterrupt:
        # Ctrl-C at the open-work question: the person stopped before any session began.
        print(flush=True)
        return STOPPED
    continuing = continuing_line(info)
    if continuing is not None:
        print(continuing, flush=True)
    files = Files(lambda: client.files(info.session_id))
    shown: list[Repl] = []
    """The REPL, once built: the toolbar is asked for before it is, and then reads it."""
    prompt = prompt_session(
        toolbar=lambda: shown[0].toolbar() if shown else "",
        files=files,
        history=history_path(state_dir),
        secret=client.holds_secret,
    )

    class _Prompt:
        def read(self, prompt_text: str, default: str = "") -> str:
            # Asked at every redraw: a question that opens while a line is typed turns `> `
            # into `? ` at once.
            return prompt.prompt(
                lambda: shown[0].prompt_mark() if shown else prompt_text, default=default
            )

    with patch_stdout(raw=True):
        repl = Repl(
            client,
            info.session_id,
            source=_Prompt(),
            write=print,
            style=Style(tty=True),
            state_dir=state_dir,
            # An existing session opened again: a run left paused in it continues.
            resume_paused=session is not None or resume_latest,
            turn_ended=files.stale,
            workspace=workspace,
        )
        shown.append(repl)
        return repl.run()
