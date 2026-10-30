"""`--file` on `run` and `ask`: each file checked by the
server before the turn is sent, through the REPL's `/attach` path; a refusal, or a secret
without `--allow-secret-file`, ends the command `USAGE` with nothing sent."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from engines import answer
from files import pdf, png
from hosting import SID, hosted

from chickadee_cli.batch import BatchIO, follow
from chickadee_cli.main import _files
from chickadee_cli.render.style import Style
from chickadee_cli.surface import DONE, USAGE

SECRET = b"OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789\n"


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_each_kind_is_sent_with_the_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shot = _write(tmp_path / "shot.png", png(40, 30))
    spec = _write(tmp_path / "spec.pdf", pdf("section four: the settings page, in blue please"))
    notes = _write(tmp_path / "notes.txt", b"two, not one\n")
    out: list[str] = []
    how = BatchIO(json=False, out=out.append, err=out.append, style=Style(tty=False))
    with hosted(monkeypatch, [answer("seen all three")]) as host, host.client() as client:
        files = _files("run", client, SID, [shot, spec, notes], allow_secret=False)
        assert [name for name, _ in files] == ["shot.png", "spec.pdf", "notes.txt"]
        after = client.session_info(SID).last_seq
        assert follow(client, SID, how, after=after, send="look", attachments=files) == DONE
    assert any("attached A-1 shot.png · image · 40x30" in line for line in out)
    assert any("attached A-2 spec.pdf · pdf · 1 page" in line for line in out)
    assert any("attached A-3 notes.txt · text · 1 line" in line for line in out)


@pytest.mark.parametrize(
    ("name", "data", "said"),
    [
        ("blob.bin", bytes(range(256)), "not an image, a PDF or a text file"),
        (".env", SECRET, "secret"),
    ],
)
def test_a_refused_file_ends_the_command_with_nothing_sent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    data: bytes,
    said: str,
) -> None:
    path = _write(tmp_path / name, data)
    with hosted(monkeypatch, []) as host, host.client() as client:
        with pytest.raises(typer.Exit) as ended:
            _files("run", client, SID, [path], allow_secret=False)
        assert ended.value.exit_code == USAGE
        assert client.session_info(SID).last_seq == 0
    assert said in capsys.readouterr().err


def test_a_secret_is_sent_when_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _write(
        tmp_path / ".env", b"OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789\n"
    )
    with hosted(monkeypatch, []) as host, host.client() as client:
        assert [n for n, _ in _files("run", client, SID, [path], allow_secret=True)] == [".env"]
