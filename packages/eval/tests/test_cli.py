"""`chickadee-eval`: run a subset into an archive, list the archives, diff two."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from engines import config_toml, raw_config, with_prices
from scripted_arm import delegates_and_doubles, opener, write_problem

from chickadee_eval import archive, cli
from chickadee_eval.arms.chickadee import ARM_NAME, ChickadeeStats
from chickadee_eval.manifest import Language, Problem
from chickadee_eval.runner import ProblemRecord


def record(pid: str, solved: bool, cost: float) -> ProblemRecord:
    return ProblemRecord(
        problem_id=pid,
        language="python",
        arm=ARM_NAME,
        outcome="pass" if solved else "fail",
        solved=solved,
        solved_claim=solved,
        exit_code=0 if solved else 1,
        attempts=1,
        cost_usd=cost,
        wall_ms=10,
        sandbox="none",
        network_verified=False,
    )


def archived(home: Path, label: str, rows: list[ProblemRecord], refused: int) -> Path:
    out = archive.create("2026-10-23", {"label": label, "subset": "e1-smoke"}, root=home)
    (out / archive.RESULTS_NAME).write_text(
        "".join(json.dumps(asdict(r)) + "\n" for r in rows), encoding="utf-8"
    )
    stats = [ChickadeeStats(r.problem_id, turns=1, refused_calls=refused) for r in rows]
    (out / cli.STATS_NAME).write_text(
        "".join(json.dumps(s.as_json()) + "\n" for s in stats), encoding="utf-8"
    )
    return out


def test_the_manifest_holds_the_arm_and_the_default_subset() -> None:
    manifest = cli.load_manifest()
    assert cli.pick_subset(manifest, cli.DEFAULT_SUBSET).count == 8
    assert any(a.name == ARM_NAME for a in manifest.arms)


def test_an_unknown_subset_says_which_exist() -> None:
    with pytest.raises(cli.CommandError, match="e1-dev"):
        cli.pick_subset(cli.load_manifest(), "e1-nope")


def test_a_missing_benchmark_says_how_to_get_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(cli.POLYGLOT_ENV, raising=False)
    code = cli.main(["--results-root", str(tmp_path), "run", "--polyglot-root", str(tmp_path)])
    assert code == 1
    assert "git clone https://github.com/Aider-AI/polyglot-benchmark" in capsys.readouterr().err


def test_the_commands_variables_are_not_read_as_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `CHICKADEE_*` variable is a config key, and an unknown one refuses the load: named
    so, the benchmark's own variables stopped every run before it started (2026-10-23)."""
    user = tmp_path / "config" / "chickadee"
    user.mkdir(parents=True)
    (user / "config.toml").write_text(config_toml(with_prices(raw_config())), encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv(cli.POLYGLOT_ENV, str(tmp_path))
    monkeypatch.setenv(cli.HOME_ENV, str(tmp_path))
    assert cli.engine_identity()


def test_a_config_that_does_not_load_is_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CHICKADEE_NO_SUCH_KEY", "1")
    write_problem(tmp_path / "bench" / "python" / "exercises" / "practice" / "forth")
    code = cli.main(
        [
            "--results-root",
            str(tmp_path / "home"),
            "run",
            "--problems",
            "forth",
            "--polyglot-root",
            str(tmp_path / "bench"),
            "--allow-unsandboxed",
        ]
    )
    assert code == 1
    assert "unknown key `no_such_key`" in capsys.readouterr().err


def test_problems_are_chosen_by_id_or_by_name_in_the_subset_order() -> None:
    problems = [
        Problem(id="python/x/forth", language=Language.PYTHON),
        Problem(id="javascript/x/forth", language=Language.JAVASCRIPT),
        Problem(id="python/x/react", language=Language.PYTHON),
    ]
    chosen = cli.select(problems, "react,javascript/x/forth", None)
    assert [p.id for p in chosen] == ["javascript/x/forth", "python/x/react"]
    assert len(cli.select(problems, None, 2)) == 2
    with pytest.raises(cli.CommandError, match="names no problem"):
        cli.select(problems, "nope", None)


def test_run_archives_the_results_the_stats_and_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One problem of the smoke subset, end to end: the real runner and arm, the real engine
    with a scripted model, and the hidden tests run for real."""
    pid = "python/exercises/practice/forth"
    write_problem(tmp_path / "bench" / pid)
    (tmp_path / "bench" / "python" / "exercises" / "practice").mkdir(exist_ok=True)
    monkeypatch.setattr(cli, "open_host_engine", opener(monkeypatch, delegates_and_doubles()))
    monkeypatch.setattr(cli, "engine_identity", lambda: {"rungs": "scripted"})
    home = tmp_path / "home"

    code = cli.main(
        [
            "--results-root",
            str(home),
            "run",
            "--problems",
            "forth",
            "--label",
            "smoke-1",
            "--polyglot-root",
            str(tmp_path / "bench"),
            "--allow-unsandboxed",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0, out
    (found,) = archive.find(home)
    rows = [
        json.loads(line) for line in (found / archive.RESULTS_NAME).read_text().splitlines()
    ]
    assert [(r["problem_id"], r["solved"]) for r in rows] == [(pid, True)]
    stats = json.loads((found / cli.STATS_NAME).read_text())
    assert stats["problem_id"] == pid and stats["tasks"] == 1
    assert (found / archive.REPORT_NAME).is_file()
    kept = found / cli.PROBLEMS_NAME / pid
    assert "== turn 1: pass ==" in (kept / "tests.txt").read_text()
    assert "+    return x * 2" in (kept / "solution.diff").read_text()
    assert list((kept / "journals" / "runs").glob("*/journal.jsonl"))
    config = json.loads((found / archive.CONFIG_NAME).read_text())
    assert config["label"] == "smoke-1" and config["problems"] == [pid]
    assert "solved 1/1" in out


def test_list_shows_each_archive_with_its_label_and_headline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archived(tmp_path, "before", [record("a", True, 0.5), record("b", False, 0.5)], refused=3)
    assert cli.main(["--results-root", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "before" in out and "solved 1/2" in out and "refused calls 6" in out


def test_diff_finds_archives_by_label_and_names_what_flipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archived(tmp_path, "before", [record("a", True, 0.5), record("b", False, 0.5)], refused=3)
    archived(tmp_path, "after", [record("a", True, 0.4), record("b", True, 0.4)], refused=1)

    assert cli.main(["--results-root", str(tmp_path), "diff", "before", "after"]) == 0
    out = capsys.readouterr().out
    assert "fixed: **1**" in out and "`b`" in out
    assert "| refused_calls | 6 | 2 |" in out


def test_diff_says_how_to_find_a_label_it_does_not_know(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--results-root", str(tmp_path), "diff", "x", "y"]) == 1
    assert "chickadee-eval list" in capsys.readouterr().err
