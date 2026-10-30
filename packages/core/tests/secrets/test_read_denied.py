"""The read rule (SE-1 layer 1), pinned before it moves.

The Reader judged reads through a rule private to it: every tail of the name asked for, then
every tail of where the path lands, spelled relative to the workspace inside it and absolutely
outside. These pins were written against that private rule and passed on it. When the rule
moved to `secrets.read_denied`, only `before_open` and `after_open` below changed (setup);
every expected value is the one the private rule gave.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import chickadee.context.opener as opener_module
from chickadee.context.files import open_nonblocking
from chickadee.context.homes import Scratch
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import InvalidArgument, SecretPath, ToolResult
from chickadee.context.terms import Vision
from chickadee.paths import name_under
from chickadee.secrets.denylist import Denial, Denylist, read_denied

# -- the rule, as the Reader asks it ---------------------------------------------------------


def _reader(ws: Path, denylist: Denylist | None = None) -> Reader:
    home = Scratch()
    return Reader(home, HandleResolver(ws, home.blob), vision=Vision("zeus"), denylist=denylist)


def _asked(ws: Path, asked: str) -> tuple[Path, str]:
    """The path and the name the Reader gives what was asked for (its `_locate`)."""
    path, key, _ = _reader(ws)._opener._locate(asked)  # pyright: ignore[reportPrivateUsage]
    return path, key


def _workspace(ws: Path) -> Path:
    """The workspace as the Reader hands it to the rule: its real path."""
    return Path(os.path.realpath(os.path.normpath(os.path.abspath(ws))))


def before_open(ws: Path, asked: str, denylist: Denylist | None = None) -> Denial | None:
    """The rule before the open: the name asked for, then where it lands."""
    path, key = _asked(ws, asked)
    return read_denied(key, path, denylist or Denylist(), workspace=_workspace(ws))


def after_open(
    ws: Path, asked: str, real: Path, denylist: Denylist | None = None
) -> Denial | None:
    """The rule after the open, about the file the descriptor holds (`real`)."""
    path, key = _asked(ws, asked)
    return read_denied(key, path, denylist or Denylist(), workspace=_workspace(ws), real=real)


# -- the tree ----------------------------------------------------------------------------------


def _write(base: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A directory outside the workspace holding secrets of its own."""
    path = tmp_path / "home"
    _write(path, {".ssh/config": "Host *\n", ".docker/config.json": "{}\n", "plain.txt": "x\n"})
    return path


@pytest.fixture
def ws(tmp_path: Path, home: Path) -> Iterator[Path]:
    """The workspace: secrets, plain files, and links into it and out of it."""
    path = tmp_path / "ws"
    _write(
        path,
        {
            "a.py": "x = 1\n",
            ".env": "TOKEN=hunter2\n",
            "keys/server.pem": "-----BEGIN-----\n",
            "sub/.docker/config.json": "{}\n",
            "pkg/mod.py": "y = 2\n",
        },
    )
    os.symlink(path / ".env", path / "notes.txt")
    os.symlink(path / "a.py", path / "alias.py")
    os.symlink(home / ".ssh" / "config", path / "ssh.cfg")
    os.symlink(home / "plain.txt", path / "outside.txt")
    yield path


# -- refusal by the name asked for -------------------------------------------------------------


@pytest.mark.parametrize(
    ("asked", "denied"),
    [
        pytest.param(".env", Denial(".env", ".env"), id="a-name-pattern"),
        pytest.param(
            "keys/server.pem", Denial("keys/server.pem", "*.pem"), id="a-format-anywhere"
        ),
        pytest.param("./keys/../.env", Denial(".env", ".env"), id="spelled-the-long-way"),
        pytest.param(
            "sub/.docker/config.json",
            Denial(".docker/config.json", ".docker/config.json"),
            id="a-directory-pattern-nested",
        ),
        pytest.param("a.py", None, id="a-plain-file"),
        pytest.param(
            "missing/.env", Denial("missing/.env", ".env"), id="a-name-with-nothing-there"
        ),
    ],
)
def test_a_read_is_refused_by_the_name_asked_for(
    ws: Path, asked: str, denied: Denial | None
) -> None:
    assert before_open(ws, asked) == denied


# -- refusal by where a link lands -------------------------------------------------------------


def test_a_link_inside_the_workspace_is_judged_by_where_it_lands(ws: Path) -> None:
    assert before_open(ws, "notes.txt") == Denial(".env", ".env")
    assert before_open(ws, "alias.py") is None


def test_a_link_out_of_the_workspace_is_judged_by_its_absolute_landing(
    ws: Path, home: Path
) -> None:
    # The denial names the tail the entry matched, never the home directory above it.
    assert before_open(ws, "ssh.cfg") == Denial(".ssh/config", ".ssh/**")
    assert before_open(ws, "outside.txt") is None


# -- refusal by every tail of an absolute path -------------------------------------------------


def test_every_tail_of_an_absolute_path_is_judged(ws: Path, home: Path) -> None:
    """A directory-shaped entry names the file wherever it sits, outside the workspace too."""
    asked = (home / ".docker" / "config.json").as_posix()
    assert before_open(ws, asked) == Denial(".docker/config.json", ".docker/config.json")
    ssh = (home / ".ssh" / "config").as_posix()
    assert before_open(ws, ssh) == Denial(".ssh/config", ".ssh/**")
    assert before_open(ws, (home / "plain.txt").as_posix()) is None


# -- after the open, of the file opened --------------------------------------------------------


def test_after_the_open_the_file_opened_is_judged(ws: Path, home: Path) -> None:
    """The name passed before the open; the descriptor holds another file. That file decides."""
    assert after_open(ws, "pkg/mod.py", ws / ".env") == Denial(".env", ".env")
    landing = (home / ".ssh" / "config").resolve()
    assert after_open(ws, "pkg/mod.py", landing) == Denial(".ssh/config", ".ssh/**")
    assert after_open(ws, "pkg/mod.py", ws / "a.py") is None
    assert after_open(ws, "pkg/mod.py", ws / "pkg" / "mod.py") is None, "it is the file asked"


# -- the switch --------------------------------------------------------------------------------


def test_allow_secret_read_turns_the_rule_off(ws: Path, home: Path) -> None:
    off = Denylist(allow_secret_read=True)
    for asked in (
        ".env",
        "notes.txt",
        "ssh.cfg",
        (home / ".docker" / "config.json").as_posix(),
    ):
        assert before_open(ws, asked, off) is None
    assert after_open(ws, "pkg/mod.py", ws / ".env", off) is None


# -- a path swapped between the check and the open ---------------------------------------------


def _swap_on_open(
    monkeypatch: pytest.MonkeyPatch, target: Path, secret: Path, *, before: bool
) -> None:
    opening = open_nonblocking

    def swap() -> None:
        if not target.is_symlink():
            target.unlink()
            os.symlink(secret, target)

    def open_(path: Path) -> int:
        if path == target and before:
            swap()
        fd = opening(path)
        if path == target and not before:
            swap()
        return fd

    monkeypatch.setattr(opener_module, "open_nonblocking", open_)


@pytest.mark.parametrize(
    ("before", "refused"),
    [
        pytest.param(True, SecretPath("pkg/mod.py", ".env"), id="just-before-the-open"),
        pytest.param(
            False,
            InvalidArgument(
                "path",
                "pkg/mod.py cannot be read",
                do="do not read it again; go on without it, or ask the person about it",
            ),
            id="just-after-the-open",
        ),
    ],
)
def test_a_path_swapped_between_the_check_and_the_open_is_not_read(
    monkeypatch: pytest.MonkeyPatch, ws: Path, before: bool, refused: object
) -> None:
    """Before the open, the link reaches the secret and the rule refuses what was opened.
    After it, the name no longer reaches the open file, so nothing says what was opened."""
    _swap_on_open(monkeypatch, ws / "pkg" / "mod.py", ws / ".env", before=before)
    assert _reader(ws).read("pkg/mod.py") == refused


# -- the Reader asks the rule (review of #357) -------------------------------------------------


@pytest.mark.parametrize(
    "asked",
    [
        ".env",
        "keys/server.pem",
        "./keys/../.env",
        "sub/.docker/config.json",
        "a.py",
        "notes.txt",
        "alias.py",
        "ssh.cfg",
        "outside.txt",
        "{home}/.docker/config.json",
        "{home}/plain.txt",
    ],
)
def test_the_reader_refuses_exactly_what_the_rule_refuses(
    ws: Path, home: Path, asked: str
) -> None:
    """The pins above judge the rule itself. This one drives `Reader.read` over the same
    cases, so a Reader that stopped asking the rule, or asked it with another workspace,
    fails here: each refusal is the rule's pattern under the name the Reader gives the path,
    and each file the rule passes is read."""
    asked = asked.format(home=home)
    _, key = _asked(ws, asked)
    denial = before_open(ws, asked)
    result = _reader(ws).read(asked)
    if denial is not None:
        assert result == SecretPath(key, denial.pattern)
    else:
        assert isinstance(result, ToolResult), result


# -- the rule's inputs (review of #357) --------------------------------------------------------


def test_a_relative_path_lands_under_the_workspace_never_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ws: Path
) -> None:
    """`notes.txt` is a link to `.env`. Asked from another working directory, the landing is
    still judged under the workspace, so the link does not launder the key."""
    monkeypatch.chdir(tmp_path)
    workspace = _workspace(ws)
    assert read_denied("notes.txt", Path("notes.txt"), Denylist(), workspace=workspace) == (
        Denial(".env", ".env")
    )


def test_a_name_under_a_root_never_climbs_out_of_it(ws: Path) -> None:
    assert name_under(ws, ws) == "."
    assert name_under(ws, ws / "pkg" / "mod.py") == "pkg/mod.py"
    assert name_under(ws, ws.parent / "home") is None
    assert name_under(ws, ws / ".." / "home" / ".ssh") is None, (
        "a `..` below the root leaves it"
    )


# -- T-3.102: the rule never raises, and a linked workspace is judged by its real path ---------


def test_the_reader_refuses_a_target_holding_a_nul_byte(ws: Path) -> None:
    """Before T-3.102 this raised `ValueError` out of the rule (`realpath`), and `ReadTools.run`
    catches nothing, so a model's `\\u0000` failed the turn."""
    result = _reader(ws).read("a\x00b")
    assert isinstance(result, InvalidArgument), result
    assert result.field == "path"


def test_the_rule_does_not_raise_on_a_path_it_cannot_resolve(ws: Path) -> None:
    """`realpath` cannot resolve a NUL byte. The name's verdict stands: a plain name passes,
    a secret's name is still refused."""
    workspace = _workspace(ws)
    plain = read_denied("a\x00b", workspace / "a\x00b", Denylist(), workspace=workspace)
    assert plain is None
    secret = read_denied(
        "a\x00b/.env", workspace / "a\x00b/.env", Denylist(), workspace=workspace
    )
    assert secret == Denial("a\x00b/.env", ".env")


@pytest.fixture
def linked(tmp_path: Path) -> Path:
    """A workspace reached through a symlink whose real path sits under `.aws/`: a plain file,
    a link to it, a `.env`, and a link to a secret outside the workspace."""
    real = tmp_path / "home" / ".aws" / "proj"
    _write(real, {"a.py": "x = 1\n", ".env": "TOKEN=hunter2\n"})
    _write(tmp_path / "home" / ".aws", {"credentials": "[default]\n"})
    outside = tmp_path / "keys"
    _write(outside, {"id_rsa": "-----BEGIN-----\n"})
    os.symlink(real / "a.py", real / "alias.py")
    os.symlink(outside / "id_rsa", real / "key.txt")
    # A sibling of the real workspace, caught only by its absolute spelling (`.aws/**`).
    os.symlink(tmp_path / "home" / ".aws" / "credentials", real / "creds.txt")
    link = tmp_path / "ws-link"
    os.symlink(real, link)
    return link


def test_a_workspace_reached_through_a_link_reads_its_own_files(linked: Path) -> None:
    """Before T-3.102 both were `SecretPath(..., '.aws/**')`: the landing was spelled absolutely
    because the workspace was not resolved, and the workspace's own ancestor matched."""
    reader = _reader(linked)
    assert isinstance(reader.read("a.py"), ToolResult)
    assert isinstance(reader.read("alias.py"), ToolResult)


def test_a_workspace_reached_through_a_link_still_refuses_its_secrets(linked: Path) -> None:
    reader = _reader(linked)
    assert reader.read(".env") == SecretPath(".env", ".env")
    assert reader.read("key.txt") == SecretPath("key.txt", "id_rsa*")
    assert reader.read("creds.txt") == SecretPath("creds.txt", ".aws/**"), (
        "a landing outside the real workspace is still spelled absolutely (review of #362)"
    )


def test_a_landing_already_known_is_judged_even_when_the_workspace_cannot_be_resolved(
    ws: Path,
) -> None:
    """After the open the caller knows the file (`real=`). A workspace `realpath` cannot take
    leaves that landing spelled absolutely and judged, never dropped as allowed (review of
    #362)."""
    unresolvable = Path("/w\x00orkspace")
    denial = read_denied(
        "pkg/mod.py",
        unresolvable / "pkg/mod.py",
        Denylist(),
        workspace=unresolvable,
        real=ws / ".env",
    )
    assert denial is not None and denial.pattern == ".env"


def test_a_linked_workspace_names_a_file_one_way_however_it_is_spelled(linked: Path) -> None:
    """The file's real absolute path is under the workspace's real path, so it is named as
    the relative read names it and read, not refused by the workspace's own ancestors
    (review of #362)."""
    reader = _reader(linked)
    by_name = reader.read("a.py")
    by_real_path = reader.read((linked.resolve() / "a.py").as_posix())
    assert isinstance(by_name, ToolResult) and isinstance(by_real_path, ToolResult)
    assert by_real_path.handle == by_name.handle


def test_a_name_holding_a_nul_byte_is_judged_as_spelled_up_to_it(ws: Path) -> None:
    """Anything reading the name as a C string opens `.env` for `.env\\x00x`, so the rule
    refuses it as `.env`, though no landing can be resolved (review of #362)."""
    workspace = _workspace(ws)
    denial = read_denied(".env\x00x", workspace / ".env\x00x", Denylist(), workspace=workspace)
    assert denial == Denial(".env", ".env")
