"""Every containment site's answer, pinned before the move to `paths.within`.

D154 counted twelve places in ten files asking whether a path is inside a root, in five
spellings; the guard found one more (`verifier/done_when.py`), and the move one more again
(`graph/lsp.py`). Each is driven here through the smallest callable that holds it, over one
tree, for the six cases decision 5 names:

- `dotdot`: a path whose `..` climbs out of the root;
- `absolute`: an absolute path outside the root;
- `link_out`: a link inside the root that lands outside it;
- `link_in`: a link inside the root that lands inside it;
- `root`: the root itself;
- `via_link`: an ordinary file, under a root reached through a link.

A site that refuses a case by its spelling before it asks where the path lands is pinned all
the same: the answer is what the site gives. These were written against today's sites and
passed on them. When the sites moved to `paths.within`, no expected value changed. A site
deleted by the move (`scout/tools._target_readable`) is pinned through the caller that held it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo

from chickadee.context.citations import workspace_lines
from chickadee.context.results import InvalidArgument
from chickadee.errors import InfraError
from chickadee.executor.executor import Executor
from chickadee.graph.conversation import rel_of
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import _declared_version  # pyright: ignore[reportPrivateUsage]
from chickadee.isolation.none import Passthrough
from chickadee.paths import contained
from chickadee.scout.findings import Checker
from chickadee.scout.search import Search
from chickadee.shell import Shell
from chickadee.tools.validate import Arguments
from chickadee.workspace import restore_operation, snapshot_operation

CASES = ("dotdot", "absolute", "link_out", "link_in", "root", "via_link")


def _write(base: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def build_tree(tmp_path: Path) -> dict[str, Path]:
    """`real/` is the root, a git repository with a Python manifest; `outside/` is beside it;
    `rootlink` is a link to `real/`. Inside the root: `a.py`, `sub/b.py`, links to a file and
    a directory inside it, and links to a file and a directory outside it."""
    real = tmp_path / "real"
    outside = tmp_path / "outside"
    manifest = '[project]\nname = "p"\ndependencies = ["requests==2.0"]\n'
    _write(real, {"a.py": "MARK = 1\n", "sub/b.py": "MARK = 2\n", "pyproject.toml": manifest})
    _write(outside, {"c.py": "MARK = 3\n", "pyproject.toml": manifest.replace("2.0", "9.9")})
    os.symlink(real / "a.py", real / "link_in.py")
    os.symlink(real / "sub", real / "link_in_dir")
    os.symlink(outside / "c.py", real / "link_out.py")
    os.symlink(outside, real / "link_out_dir")
    init_repo(real, commit="init")
    link = tmp_path / "rootlink"
    os.symlink(real, link)
    return {"real": real, "outside": outside, "link": link}


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    return build_tree(tmp_path)


def _file_case(tree: dict[str, Path], case: str) -> tuple[Path, str]:
    """(root, path) for a case asked of a file."""
    real, outside = tree["real"], tree["outside"]
    return {
        "dotdot": (real, "sub/../../outside/c.py"),
        "absolute": (real, (outside / "c.py").as_posix()),
        "link_out": (real, "link_out.py"),
        "link_in": (real, "link_in.py"),
        "root": (real, "."),
        "via_link": (tree["link"], "a.py"),
        "nul": (real, "a\x00b"),
    }[case]


def _dir_case(tree: dict[str, Path], case: str) -> tuple[Path, str]:
    """(root, path) for a case asked of a directory (the shell's `cwd`)."""
    real, outside = tree["real"], tree["outside"]
    return {
        "dotdot": (real, "sub/../../outside"),
        "absolute": (real, outside.as_posix()),
        "link_out": (real, "link_out_dir"),
        "link_in": (real, "link_in_dir"),
        "root": (real, "."),
        "via_link": (tree["link"], "sub"),
        "nul": (real, "a\x00b"),
    }[case]


# -- each site's answer, as a short string ---------------------------------------------


def contained_in(tree: dict[str, Path], case: str) -> str:
    """`paths.contained`, the containment rule's one home: the Applier's gate 1 was
    this site, and wrapped its words in a `Reject`; the box's edit port answers them to the
    model."""
    root, path = _file_case(tree, case)
    why = contained(path, root)
    return "ok" if why is None else why.split(":")[0]


def shell(tree: dict[str, Path], case: str) -> str:
    root, cwd = _dir_case(tree, case)
    # Parsing never spawns; the executor is real so the shell reads its table as in use.
    sh = Shell(Executor(Passthrough()), root, lambda data: None)  # type: ignore[arg-type]
    got = sh.parse(Arguments("shell", {"command": "true", "cwd": cwd}))
    return got.why.split(" is ", 1)[-1] if isinstance(got, InvalidArgument) else "ok"


def workspace(tree: dict[str, Path], case: str) -> str:
    root, path = _file_case(tree, case)
    snapshot_operation(tree["real"], "op-1")
    try:
        restore_operation(root, "op-1", (path,))
    except InfraError as exc:
        return "outside" if "outside the worktree" in str(exc) else "root"
    return "ok"


def citations(tree: dict[str, Path], case: str) -> str:
    root, path = _file_case(tree, case)
    return "lines" if workspace_lines(root)(path) is not None else "none"


def findings(tree: dict[str, Path], case: str) -> str:
    root, path = _file_case(tree, case)
    checker = Checker(root, None)
    return f"lines={checker.lines(path) is not None} exists={checker.exists(path)}"


def manifest(tree: dict[str, Path], case: str) -> str:
    root, path = _file_case(tree, case)
    cache: dict[Any, Any] = {}
    return str(_declared_version(root, path, ("pyproject.toml",), "requests", cache))


def indexer(tree: dict[str, Path], case: str) -> str:
    root, path = _file_case(tree, case)
    ix = Indexer(root)
    try:
        ix.full_index()
        given = Path(path) if Path(path).is_absolute() else root / path
        return str(ix.reindex([given]).files_indexed)
    finally:
        ix.close()


def lsp(tree: dict[str, Path], case: str) -> str:
    root, path = _file_case(tree, case)
    given = Path(path) if Path(path).is_absolute() else root / path
    return str(rel_of(root, given.absolute().as_uri()))


SITES = {
    "contained": contained_in,
    "shell": shell,
    "workspace": workspace,
    "citations": citations,
    "findings": findings,
    "manifest": manifest,
    "indexer": indexer,
    "lsp": lsp,
}
"""The packer's `WorldState.is_file` and the verifier's `import_provenance` were sites too;
they retired with the packet loop, and their pins with them."""


def search_hits(root: Path) -> list[str]:
    """The scout's in-process search over `root`: the paths of its hits for `MARK`. Its
    fallback reads through links, so a link landing outside the root is refused there
    (`scout/tools._target_readable` before the move)."""
    out = Search(root=root, rg=None)("MARK")
    return sorted(line.split(":", 1)[0] for line in out.splitlines()[1:] if ":" in line)


# -- the pins ------------------------------------------------------------------------------

OUT = "resolves outside the worktree"
EXPECTED: dict[str, dict[str, str]] = {
    "contained": {
        "dotdot": "path escapes with '..'",
        "absolute": "absolute path",
        "link_out": OUT,
        "link_in": "ok",
        "root": OUT,
        "via_link": "ok",
    },
    "shell": {
        "dotdot": "outside the workspace",
        "absolute": "outside the workspace",
        "link_out": "outside the workspace",
        "link_in": "ok",
        "root": "ok",
        "via_link": "ok",
    },
    "workspace": {
        "dotdot": "outside",
        "absolute": "outside",
        "link_out": "outside",
        "link_in": "ok",
        "root": "root",
        "via_link": "ok",
    },
    "citations": {
        "dotdot": "none",
        "absolute": "none",
        "link_out": "none",
        "link_in": "lines",
        "root": "none",
        "via_link": "lines",
    },
    "findings": {
        "dotdot": "lines=False exists=False",
        "absolute": "lines=False exists=False",
        "link_out": "lines=False exists=False",
        "link_in": "lines=True exists=True",
        "root": "lines=False exists=False",
        "via_link": "lines=True exists=True",
    },
    "manifest": {
        "dotdot": "None",
        "absolute": "None",
        "link_out": "None",
        "link_in": "==2.0",
        "root": "None",
        "via_link": "==2.0",
    },
    "indexer": {
        "dotdot": "0",
        "absolute": "0",
        "link_out": "0",
        "link_in": "1",
        "root": "0",
        "via_link": "1",
    },
    "lsp": {
        "dotdot": "None",
        "absolute": "None",
        "link_out": "None",
        "link_in": "a.py",
        "root": ".",
        "via_link": "a.py",
    },
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("site", sorted(SITES))
def test_each_site_answers_as_it_did(tree: dict[str, Path], site: str, case: str) -> None:
    assert SITES[site](tree, case) == EXPECTED[site][case]


def test_the_scouts_fallback_search_refuses_a_link_out_of_the_root(
    tree: dict[str, Path],
) -> None:
    """`link_out.py` holds `MARK` through a link to `outside/c.py`: never a hit. `link_in.py`
    lands inside the root and is one. The same through a root reached by a link."""
    expected = ["a.py", "link_in.py", "sub/b.py"]
    assert search_hits(tree["real"]) == expected
    assert search_hits(tree["link"]) == expected


# -- what the move changed: a path no filesystem can name ---------------------------------

NUL_REFUSED = {
    "contained": OUT,
    "shell": "outside the workspace",
    "citations": "none",
    "findings": "lines=False exists=False",
}
"""These four raised `ValueError` on a NUL byte before the move (`Path.resolve` and
`os.path.realpath` both do), out of a gate or a tool argument a model supplies. `within`
answers None for a path it cannot resolve, so each now refuses it as it
refuses any path outside the root. The other sites answered as before and are not listed."""


@pytest.mark.parametrize("site", sorted(NUL_REFUSED))
def test_a_path_holding_a_nul_byte_is_refused_never_raised(
    tree: dict[str, Path], site: str
) -> None:
    assert SITES[site](tree, "nul") == NUL_REFUSED[site]
