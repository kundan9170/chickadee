"""`[[commands]]` rows from both configs into one table.

What these pin: a row whose first token names a directory is refused at load; each layer's
rows are kept with their layer, and a row that names its own layer is refused; unaccepted, the
repository's rows are inert and the person's stand alone; accepted, the repository's row wins
a prefix both name; no row lowers a shipped row; the repository's rows are in the trust digest;
the table the engine builds is what a shell reads, through its executor. The DISCOVERY
pre-filter and the scout read the same table (`scout/test_escapes.py`), and the run header
names the inert rows (`engine/test_run_header.py`).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from engines import raw_config
from repos import init_repo
from sandboxes import Enforcing, Scripted
from tool_args import checked

from chickadee.approval.commands import CommandClass, CommandTableError, parse_row
from chickadee.config import EngineConfig
from chickadee.context.homes import Scratch
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.terms import Vision
from chickadee.errors import ConfigError
from chickadee.executor.executor import Executor
from chickadee.schema.authorization import Tier
from chickadee.shell import Shell, ShellClaim
from chickadee.state.layers import Layer
from chickadee.state.resolve import resolve
from chickadee.state.trust import executable_subset


def _resolved(user: list[dict[str, Any]], project: list[dict[str, Any]]) -> EngineConfig:
    layers = [
        Layer("user", None, {**raw_config(), "commands": user}),
        Layer("project", None, {"commands": project}),
    ]
    return resolve(layers).engine


def _cls(config: EngineConfig, line: str, *, accepted: bool) -> CommandClass | None:
    found = config.command_table(accepted=accepted).classify_line(line)
    return None if found is None else found.cls


def test_a_row_naming_a_directory_is_refused() -> None:
    with pytest.raises(CommandTableError, match=r"names a directory.*\['deploy\.sh'\]"):
        parse_row({"prefix": ["./deploy.sh"], "class": "push"})


def test_each_layers_rows_keep_their_layer() -> None:
    config = _resolved(
        [{"prefix": ["reset-db.sh"], "class": "destructive"}],
        [{"prefix": ["deploy.sh"], "class": "push"}],
    )
    assert [(c.prefix, c.layer) for c in config.commands] == [
        (("reset-db.sh",), "user"),
        (("deploy.sh",), "project"),
    ]


def test_a_row_naming_its_own_layer_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"may not set `layer`"):
        _resolved([], [{"prefix": ["deploy.sh"], "class": "push", "layer": "user"}])


def test_unaccepted_the_repositorys_rows_are_inert_and_the_persons_stand() -> None:
    config = _resolved(
        [{"prefix": ["reset-db.sh"], "class": "destructive"}],
        [{"prefix": ["deploy.sh"], "class": "push"}],
    )
    assert _cls(config, "./reset-db.sh", accepted=False) is CommandClass.DESTRUCTIVE
    assert _cls(config, "scripts/deploy.sh", accepted=False) is None
    assert _cls(config, "scripts/deploy.sh", accepted=True) is CommandClass.PUSH


def test_accepted_the_repositorys_row_wins_a_prefix_both_name() -> None:
    config = _resolved(
        [{"prefix": ["deploy.sh"], "class": "destructive"}],
        [{"prefix": ["deploy.sh"], "class": "push"}],
    )
    assert _cls(config, "deploy.sh", accepted=False) is CommandClass.DESTRUCTIVE
    assert _cls(config, "deploy.sh", accepted=True) is CommandClass.PUSH


def test_a_row_that_lowers_a_shipped_row_is_refused_at_load() -> None:
    with pytest.raises(ConfigError, match=r"never lower"):
        _resolved([], [{"prefix": ["git", "push"], "class": "read_only"}])


def test_the_repositorys_rows_are_in_its_trust_digest() -> None:
    """Moved: an edited row is a new digest, so accepting a row once never accepts its
    replacement."""
    plain = executable_subset({})
    one = executable_subset({"commands": [{"prefix": ["deploy.sh"], "class": "push"}]})
    two = executable_subset({"commands": [{"prefix": ["deploy.sh"], "class": "build"}]})
    assert plain.is_empty and not one.is_empty
    assert len({plain.digest, one.digest, two.digest}) == 3


class _Asked:
    def __init__(self) -> None:
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return True


def test_a_shell_reads_the_engines_table_through_its_executor(tmp_path: Path) -> None:
    """`deploy.sh` is unknown to the shipped table, so it ran unrecognised; with the person's
    row it is a push, asked at the class's floor, as the engine's executor hands the table
    on."""
    config = _resolved([{"prefix": ["deploy.sh"], "class": "push"}], [])
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    own = Scratch()
    reader = Reader(own, HandleResolver(root, own.blob), vision=Vision("z"))
    asked = _Asked()
    shell = Shell(
        Executor(
            Enforcing(),
            supervisor=Scripted(b"ok\n"),
            commands=config.command_table(accepted=False),
        ),
        root,
        lambda data: reader.store("output", data),
        authority=asked,
    )
    shell.run(
        checked("shell", {"command": "./deploy.sh --prod"}),
        operation_id="op-1",
    )
    [claim] = asked.claims
    assert claim.tier is not Tier.T0 and claim.classification.startswith("push")
