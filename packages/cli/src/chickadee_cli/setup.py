"""`chickadee setup` — the person's engine config, and whether a sandbox runs here.

Thin by construction, as `init` is: it calls the facade and renders what comes back.
Writing only when absent is core's; the probe is core's; this file prints. Its registration in
the Typer app is `main.py`'s, which builds every command row.

Exit codes: `0` when the engine config was written or was already there, `1` when it could
not be written. A sandbox that does not run is reported, not a failure: the person decides.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from chickadee.engine import (
    Installed,
    Probe,
    SeedReport,
    ServerNeed,
    install_server,
    probe_native,
    seed_engine_config,
    server_needs,
)
from chickadee_cli.surface import DONE, USAGE


def run(
    path: Path | None = None,
    *,
    probe: Callable[[], Probe] = probe_native,
    servers: Callable[[], tuple[ServerNeed, ...]] = server_needs,
    ask: Callable[[ServerNeed], bool] = lambda _need: False,
    installer: Callable[[ServerNeed], Installed] = install_server,
) -> tuple[int, str]:
    """Seed the engine config at `path` (the default location when None), probe the sandbox,
    list each language's servers, install each missing one the person says yes to (`ask`),
    and return the exit code and what to print. A failed install is reported, never a
    failed setup: the server's verdicts are unavailable until it is installed."""
    try:
        report = seed_engine_config(path)
    except OSError as exc:
        where = path if path is not None else "the engine config"
        return USAGE, f"could not write {where}: {exc}"
    needs = servers()
    installed = {n.server: installer(n) for n in needs if n.installable and ask(n)}
    return DONE, "\n".join((render(report, probe()), *server_lines(needs, installed)))


def server_lines(needs: tuple[ServerNeed, ...], installed: dict[str, Installed]) -> list[str]:
    """Each language's servers on this machine, and what became of each install asked for."""
    lines = ["  servers"]
    for need in needs:
        head = f"    {'/'.join(need.languages):<22} {need.server:<28}"
        done = installed.get(need.server)
        if done is not None:
            said = done.said if done.ok else f"not installed: {done.said}"
            lines.append(f"{head} {said}")
        elif need.found is not None:
            lines.append(f"{head} found at {need.found}")
        elif need.install is None:
            lines.append(f"{head} missing; install it yourself")
        elif need.needs is not None:
            lines.append(f"{head} missing; installing it needs {need.needs}")
        else:
            lines.append(f"{head} missing; setup installs it with: {' '.join(need.install)}")
    return lines


def render(report: SeedReport, sandbox: Probe) -> str:
    if report.written:
        lines = [f"wrote       {report.path}"]
    else:
        lines = [f"unchanged   {report.path} already exists; setup never edits it"]
    cap = report.hard_cost_cap_usd
    # A `$None` would read as a number to anyone skimming; the note says what is missing.
    lines.append(f"  hard cap    {'none in force (see note)' if cap is None else _usd(cap)}")
    if report.note is not None:
        lines.append(f"  note        {report.note}")
    level = sandbox.level.value
    if sandbox.ran:
        lines.append(f"  sandbox     {level}, runs")
    else:
        # The reason and the fix are printed, never applied.
        lines.append(f"  sandbox     {level}, does not run")
        lines.append(f"    reason    {sandbox.reason}")
        if sandbox.fix:
            lines.append(f"    fix       {sandbox.fix}")
    return "\n".join(lines)


def _usd(amount: Decimal) -> str:
    """Dollars with at least two decimals, and never fewer digits than the person wrote:
    `5.0` reads `$5.00`, and `12.345` stays `$12.345`."""
    exponent = amount.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) else 0
    return f"${amount.quantize(Decimal('0.01')) if places < 2 else amount}"
