"""Every surface that reads a repository file asks the one read rule.

One tree, every surface. The tree holds a secret by name (`.env`), a secret by a
directory-shaped entry nested below the root (`sub/.docker/config.json`), a source file a
project's own `add` names (`settings_prod.py`), a link to each, and one plain file. A surface
that reads, or whose answer depends on the file, asks `read_denied`: the name, then where it
lands. A surface that judges a name it will not open asks `name_denied`. Each refuses in its
own typed way, never with an empty answer that reads as "no such file".

This replaces `test_denylist.py::test_the_same_answer_on_every_surface`, which named five
surfaces and ran none of them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chickadee.context.citations import Citation, Verdict, check, workspace_lines
from chickadee.context.homes import Scratch
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import SecretPath, ToolResult
from chickadee.context.terms import Vision
from chickadee.executor.policy import secret_path_denial
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.graph.indexer import Indexer
from chickadee.scout.findings import Checker
from chickadee.scout.search import Search
from chickadee.secrets.denylist import Denylist, resolve

LISTED = Denylist.configure(add=("settings_prod.py",))
"""The built-ins, plus a project's own source file holding production settings."""

PLAIN = "app.py"
SECRETS = (".env", "sub/.docker/config.json", "settings_prod.py")
LINKS = ("env_link.txt", "docker_link.json", "settings_link.py")
"""Each links to the secret at the same index. Its own name is plain."""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    files = {
        PLAIN: "MARK = 1\n",
        ".env": "MARK=sk-live\n",
        "sub/.docker/config.json": '{"MARK": "auth"}\n',
        "settings_prod.py": "MARK = 'prod-password'\n\n\ndef prod_secret():\n    pass\n",
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    for link, target in zip(LINKS, SECRETS, strict=True):
        os.symlink(repo / target, repo / link)
    return repo


# -- the Reader: read_denied, before and after the open -----------------------------


@pytest.mark.parametrize("rel", SECRETS + LINKS)
def test_the_reader_refuses_a_secret_by_name_and_by_landing(root: Path, rel: str) -> None:
    home = Scratch()
    reader = Reader(
        home, HandleResolver(root, home.blob), vision=Vision("zeus"), denylist=LISTED
    )
    assert isinstance(reader.read(rel), SecretPath)
    assert isinstance(reader.read(PLAIN), ToolResult)


# -- scout search: read_denied on every hit, both engines --------------------------------------


def _hits(out: str) -> list[str]:
    return sorted({line.split(":", 1)[0] for line in out.splitlines()[1:] if line.strip()})


@pytest.mark.parametrize("engine", ["rg", "fallback"])
def test_scout_search_never_returns_a_line_of_a_secret(root: Path, engine: str) -> None:
    """`rg` printed `sub/.docker/config.json:1:...`, and the hit was judged by the list's
    root-relative match, which a nested directory-shaped entry does not meet."""
    search = Search(root=root, denylist=LISTED)
    if engine == "fallback":
        search.rg = None
    elif search.rg is None:
        pytest.skip("rg is not installed")
    out = search("MARK")
    assert _hits(out) == [PLAIN]
    assert "sk-live" not in out and "auth" not in out and "prod-password" not in out


def test_scout_path_search_never_names_a_secret(root: Path) -> None:
    out = Search(root=root, denylist=LISTED)("*", kind="path")
    assert _hits(out) == [PLAIN]


# -- citations: name_denied for the verdict, read_denied in the line source ------------------


@pytest.mark.parametrize("rel", SECRETS + LINKS)
def test_a_citation_of_a_secret_is_secret_never_ok_or_no_file(root: Path, rel: str) -> None:
    """`sub/.docker/config.json:1` was checked as a file and passed; `env_link.txt:1` was
    `NO_FILE`. Both are `SECRET`: the first by its name, the second by where it lands."""
    read = workspace_lines(root, denylist=LISTED)
    got = check(Citation(rel, 1, 1, f"{rel}:1"), read=read, denylist=LISTED)
    assert got.verdict is Verdict.SECRET, got.render()
    plain = check(Citation(PLAIN, 1, 1, f"{PLAIN}:1"), read=read, denylist=LISTED)
    assert plain.verdict is Verdict.OK


# -- the executor's pre-filter: name_denied on each argv token --------------------------------


@pytest.mark.parametrize(
    "token",
    [".env", "sub/.docker/config.json", "settings_prod.py", "~/.docker/config.json"],
)
def test_the_executor_refuses_a_secret_named_in_argv(
    root: Path, tmp_path: Path, token: str
) -> None:
    """`cat sub/.docker/config.json` and `cat ~/.docker/config.json` passed the pre-filter,
    while `read` refused both."""
    request = ExecRequest(("cat", token), cwd=root, policy=PolicyClass.DISCOVERY, timeout_s=5)
    denial = secret_path_denial(request, LISTED, home=tmp_path / "home")
    assert denial is not None and denial.reason.startswith("SECRET_PATH:")


@pytest.mark.parametrize("token", [PLAIN, *LINKS])
def test_the_executor_judges_names_only(root: Path, token: str) -> None:
    """A link's own name is plain. The process the argv runs opens the target, which the
    sandbox masks where it sits (`resolve`, below); the pre-filter judges words, not files."""
    request = ExecRequest(("cat", token), cwd=root, policy=PolicyClass.DISCOVERY, timeout_s=5)
    assert secret_path_denial(request, LISTED) is None


# -- the scout's pointer check: read_denied before it reads ------------------------------------


@pytest.mark.parametrize("rel", SECRETS + LINKS)
def test_the_scouts_pointer_check_does_not_read_a_secret(root: Path, rel: str) -> None:
    """A pointer into a secret fails, as a pointer into a missing file does."""
    checker = Checker(root, None, denylist=LISTED)
    assert checker.lines(rel) is None
    assert checker.lines(PLAIN) == ["MARK = 1"]


# -- the graph's walk: read_denied, a Gap ----------------------------------------------------


def test_the_graph_does_not_index_a_secret_source_file(root: Path) -> None:
    """`settings_prod.py` is Python the profile parses: its symbols reached the planner. It,
    and a link to it, are each a `Gap` naming the denial. `.env` and the JSON are not source,
    and stay `unprofiled language` gaps, as before."""
    indexer = Indexer(root, denylist=LISTED)
    try:
        report = indexer.full_index()
        gaps = {g.path: g.reason for g in report.gaps}
        assert gaps["settings_prod.py"].startswith("denied: secret-bearing path")
        assert gaps["settings_link.py"].startswith("denied: secret-bearing path")
        assert gaps[".env"] == "unprofiled language"
        assert report.files_indexed == 1
        names = {r[0] for r in indexer.conn.execute("SELECT name FROM symbols")}
        assert "prod_secret" not in names
    finally:
        indexer.close()


# -- the sandbox mask: name_denied on every walked entry --------------------------------------


def test_the_sandbox_masks_a_nested_directory_shaped_secret(root: Path) -> None:
    """`resolve` matched root-relative names only, so `sub/.docker/config.json` was refused by
    `read` and left unmasked for a process. A link is masked by its own name, which is plain;
    its target is masked where it sits."""
    masked = {p.relative_to(root).as_posix() for p in resolve(LISTED, [root])}
    assert masked == set(SECRETS)


def test_the_sandbox_masks_a_nested_secret_directory_as_one(tmp_path: Path) -> None:
    """`sub/.ssh/` was walked into and masked key by key, so a key written there during a
    session was unmasked. It is one mask now, as a root-level `.ssh/` is."""
    keys = tmp_path / "repo" / "sub" / ".ssh"
    keys.mkdir(parents=True)
    (keys / "id_ed25519").write_text("k\n", encoding="utf-8")
    (keys / "known_hosts").write_text("h\n", encoding="utf-8")
    masked = resolve(LISTED, [tmp_path / "repo"])
    assert masked == (keys,)
