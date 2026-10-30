"""`denylist.resolve` with its walk memo (`secrets/walk_memo.py`): an unchanged
directory is not walked again, and the mask is never other than a whole walk's.

What these pin: the memo's answer is the whole walk's; a secret written into a directory the
memo kept is masked; one removed is no longer listed; a directory changed within `RACY_NS` is
never kept; one holding a symlink is never kept; another list is another memo; a memo that
cannot be read is no memo.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from chickadee.secrets.denylist import Denylist, resolve
from chickadee.secrets.walk_memo import RACY_NS, WalkMemo

_OLD = time.time() - 3600


def _tree(root: Path) -> Path:
    for rel, text in {
        "src/app.py": "x = 1\n",
        "src/deep/.env": "TOKEN=1\n",
        "keys/id.pem": "-----\n",
        "notes.txt": "hi\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id_ed25519").write_text("k\n", encoding="utf-8")
    return root


def _age(root: Path) -> None:
    """Every directory under `root` last changed an hour ago: past `RACY_NS`, so kept."""
    for here, dirs, _files in os.walk(root):
        for name in dirs:
            os.utime(Path(here) / name, (_OLD, _OLD))
    os.utime(root, (_OLD, _OLD))


def _again(root: Path, memo: WalkMemo) -> tuple[Path, ...]:
    return resolve(Denylist(), [root], memo=memo)


def test_the_memo_answers_as_the_whole_walk(tmp_path: Path) -> None:
    root = _tree(tmp_path / "home")
    _age(root)
    memo = WalkMemo(tmp_path / "memo")
    whole = resolve(Denylist(), [root])
    assert {p.relative_to(root).as_posix() for p in whole} >= {"src/deep/.env", ".ssh"}
    assert _again(root, memo) == whole
    assert _again(root, memo) == whole, "from the memo"
    assert _again(root, WalkMemo(tmp_path / "memo")) == whole, "from disk, another process"


def test_a_secret_written_into_a_kept_directory_is_masked(tmp_path: Path) -> None:
    root = _tree(tmp_path / "home")
    _age(root)
    memo = WalkMemo(tmp_path / "memo")
    _again(root, memo)
    (root / "src" / "server.key").write_text("k\n", encoding="utf-8")
    assert root / "src" / "server.key" in _again(root, memo)


def test_a_secret_removed_is_no_longer_listed(tmp_path: Path) -> None:
    root = _tree(tmp_path / "home")
    _age(root)
    memo = WalkMemo(tmp_path / "memo")
    assert root / "src" / "deep" / ".env" in _again(root, memo)
    (root / "src" / "deep" / ".env").unlink()
    assert root / "src" / "deep" / ".env" not in _again(root, memo)


def test_a_directory_changed_just_now_is_never_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coarse clock can leave a second change inside the same `mtime`: a directory younger
    than `RACY_NS` is walked every time."""
    root = _tree(tmp_path / "home")
    _age(root)
    os.utime(root / "src", None)  # changed now
    memo = WalkMemo(tmp_path / "memo")
    _again(root, memo)
    kept = next(iter(memo._held.values()))  # pyright: ignore[reportPrivateUsage]
    assert str(root / "src") not in kept and str(root / "keys") in kept
    assert RACY_NS >= 2_000_000_000


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_a_directory_holding_a_symlink_is_never_kept(tmp_path: Path) -> None:
    """A link's verdict reads its target's kind, which can change with no move here."""
    root = _tree(tmp_path / "home")
    (root / "linked").symlink_to(root / "keys")
    _age(root)
    memo = WalkMemo(tmp_path / "memo")
    _again(root, memo)
    kept = next(iter(memo._held.values()))  # pyright: ignore[reportPrivateUsage]
    assert str(root) not in kept and str(root / "src") in kept


def test_another_list_is_another_memo(tmp_path: Path) -> None:
    root = _tree(tmp_path / "home")
    _age(root)
    memo = WalkMemo(tmp_path / "memo")
    _again(root, memo)
    stricter = Denylist(patterns=(*Denylist().patterns, "*.txt"))
    assert root / "notes.txt" in resolve(stricter, [root], memo=memo)


def test_a_memo_that_cannot_be_read_is_no_memo(tmp_path: Path) -> None:
    root = _tree(tmp_path / "home")
    _age(root)
    _again(root, WalkMemo(tmp_path / "memo"))
    for kept in (tmp_path / "memo").glob("*.json"):
        kept.write_text('{"version": 1, "dirs": {"x": [1]}}', encoding="utf-8")
    assert _again(root, WalkMemo(tmp_path / "memo")) == resolve(Denylist(), [root])


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_a_secret_added_under_a_linked_root_is_masked(tmp_path: Path) -> None:
    """A linked `$HOME`: the root's verdict is kept by the directory it names, never by the
    link's own `mtime`, which a file added inside does not move."""
    real = _tree(tmp_path / "real")
    _age(real)
    linked = tmp_path / "home"
    linked.symlink_to(real)
    os.utime(linked, (_OLD, _OLD), follow_symlinks=False)
    memo = WalkMemo(tmp_path / "memo")
    _again(linked, memo)
    (real / ".env").write_text("TOKEN=2\n", encoding="utf-8")
    assert linked / ".env" in _again(linked, memo)
