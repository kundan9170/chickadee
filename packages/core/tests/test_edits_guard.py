"""The edits guard fails a second text matcher, line renderer or splitter.

Run over fixture trees, and once over the live one, which must be exactly its baseline.
Loaded by file path, as the other guard tests load the runner.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chickadee.edits import lines

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDS = REPO_ROOT / "scripts" / "guards"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_edits_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
edits: Any = importlib.import_module("guards.edits")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(edits, "REPO_ROOT", root)
    monkeypatch.setattr(edits, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [(v.path.removeprefix(CORE), v.line, v.text[:2]) for v in edits.find(files)]


# -- E1, a second format parser -------------------------------------------------------------


def test_a_format_marker_outside_the_formats_fails(tree: Path) -> None:
    _write(
        tree,
        "adapters/base.py",
        "import re\n"
        'MARKERS = ("<<<<<<< SEARCH", ">>>>>>> REPLACE")\n'
        'BEGIN = "*** Begin Patch"\n'
        'HUNK = re.compile(r"^@@ -(\\d+)")\n'
        "def f(line):\n"
        '    return line.startswith("*** Update File: ") or line.startswith("@@")\n',
    )
    assert _found(tree) == [
        ("adapters/base.py", 2, "E1"),
        ("adapters/base.py", 2, "E1"),
        ("adapters/base.py", 3, "E1"),
        ("adapters/base.py", 4, "E1"),
        ("adapters/base.py", 6, "E1"),
        ("adapters/base.py", 6, "E1"),
    ]


def test_look_alikes_prose_and_the_formats_pass(tree: Path) -> None:
    _write(
        tree,
        "renderer/text.py",
        '"""A reply may hold `<<<<<<< SEARCH` blocks or a `*** Begin Patch` envelope."""\n'
        'RULE = "======="\n'
        'BOLD = "**bold** @@ not a header"\n'
        'SHIFT = "a << b >> c"\n',
    )
    _write(
        tree,
        "edits/formats/apply_patch.py",
        'BEGIN = "*** Begin Patch"\nSEARCH = "<<<<<<< SEARCH"\nHUNK = "@@"\n',
    )
    assert _found(tree) == []


# -- E8, a second protocol --------------------------------------------------------------------


def test_reading_a_call_or_branching_on_a_protocol_outside_the_protocols_fails(
    tree: Path,
) -> None:
    _write(
        tree,
        "box/coder.py",
        "import re\n"
        'TOOL_LINE = re.compile(r"^TOOL\\s+(\\S+)")\n'
        'ALIAS = "NEED:"\n'
        'PREFIX = "^TOOL "\n'
        "def f(policy):\n"
        '    if policy.protocol == "native":\n'
        "        return 1\n"
        '    return policy.protocol in ("prompt_simple", "prompt_schema")\n'
        'LOOSE = r"(?im)^\\s*tool\\s+"\n'
        'OPENING = r"(?m)^\\s*TOOL "\n'
        "def g(p, form):\n"
        "    match p.tool_protocol:\n"
        '        case "native":\n'
        "            return 1\n"
        '    return form == "prompt_schema"\n'
        "def h(text):\n"
        '    return text.split()[0] == "TOOL" or text.startswith("NEED:")\n',
    )
    assert _found(tree) == [
        ("box/coder.py", 2, "E8"),
        ("box/coder.py", 3, "E8"),
        ("box/coder.py", 4, "E8"),
        ("box/coder.py", 6, "E8"),
        ("box/coder.py", 8, "E8"),
        ("box/coder.py", 9, "E8"),
        ("box/coder.py", 10, "E8"),
        ("box/coder.py", 12, "E8"),
        ("box/coder.py", 15, "E8"),
        ("box/coder.py", 17, "E8"),
        ("box/coder.py", 17, "E8"),
        ("box/coder.py", 17, "E8"),
    ]


def test_prose_other_backends_and_the_protocols_pass(tree: Path) -> None:
    _write(
        tree,
        "isolation/container.py",
        '"""A worker writes a `TOOL` line or `NEED:`; a backend may be "native"."""\n'
        '__all__ = ["TOOL"]\n'
        "def f(backend, cfg, line):\n"
        '    note = "the TOOL it named"\n'
        '    if cfg.sandbox.backend == "native" or line.form != "native":\n'
        "        return 1\n"
        '    if line.role == "tool" or note.startswith("TOOLS"):\n'
        "        return 2\n"
        '    return backend == "container"\n',
    )
    _write(
        tree,
        "adapters/protocols/forms.py",
        'TOOL = r"^TOOL\\s+"\nNEED = "NEED:"\ndef f(p):\n    return p == "native"\n',
    )
    assert _found(tree) == []


# -- E7, draft text changed outside `CandidateDraft.apply` -----------------------------------


def test_a_draft_written_outside_its_one_write_fails(tree: Path) -> None:
    _write(
        tree,
        "tools/builtin/sneak.py",
        "def run(draft, path):\n"
        "    draft._texts[path] = 'x'\n"
        "    draft._texts = {}\n"
        "    draft._texts.update({path: 'y'})\n"
        "    del draft._texts[path]\n",
    )
    _write(
        tree,
        "edits/draft.py",
        "class CandidateDraft:\n"
        "    def answer(self, path):\n"
        "        self._texts[path] += '!'\n"
        "        self._texts[path] = '?'\n"
        "        self._texts = {}\n"
        "        self._texts.update({})\n"
        "        del self._texts[path]\n",
    )
    _write(
        tree,
        "tools/copy.py",
        "class CandidateDraft:\n"
        "    def apply(self, path, text):\n"
        "        self._texts[path] = text\n",
    )
    assert _found(tree) == [
        ("edits/draft.py", 3, "E7"),
        ("edits/draft.py", 4, "E7"),
        ("edits/draft.py", 5, "E7"),
        ("edits/draft.py", 6, "E7"),
        ("edits/draft.py", 7, "E7"),
        ("tools/builtin/sneak.py", 2, "E7"),
        ("tools/builtin/sneak.py", 3, "E7"),
        ("tools/builtin/sneak.py", 4, "E7"),
        ("tools/builtin/sneak.py", 5, "E7"),
        ("tools/copy.py", 3, "E7"),
    ]


def test_the_drafts_own_write_its_making_and_its_undoing_and_its_reads_pass(tree: Path) -> None:
    _write(
        tree,
        "edits/draft.py",
        "class CandidateDraft:\n"
        "    def __init__(self):\n"
        "        self._texts = {}\n"
        "    def apply(self, path, text):\n"
        "        self._texts[path] = text\n"
        "    def _restore(self, kept):\n"
        "        self._texts = kept\n"
        "    def text(self, path):\n"
        "        return self._texts.get(path)\n",
    )
    assert _found(tree) == []


def test_the_drafts_text_reached_from_outside_fails_a_read_and_an_alias_too(tree: Path) -> None:
    _write(
        tree,
        "tools/builtin/look.py",
        "def run(d, p):\n"
        "    held = d._texts\n"
        "    held[p] = 'x'\n"
        "    setattr(d, '_texts', {})\n"
        "    return getattr(d, 'texts', None)\n",
    )
    assert _found(tree) == [
        ("tools/builtin/look.py", 2, "E7"),
        ("tools/builtin/look.py", 4, "E7"),
    ]


# -- E4, a tool's arguments in a template ------------------------------------------------------


def _templates(root: Path, rel: str, text: str) -> list[tuple[str, int, str]]:
    path = root / SRC / "adapters" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    files = sorted((root / SRC).rglob("*.md"))
    return [(v.path.removeprefix(CORE), v.line, v.text[:2]) for v in edits.find(files)]


def test_a_schema_or_typed_arguments_in_a_template_fail(tree: Path) -> None:
    found = _templates(
        tree,
        "qwen_coder/templates/tools/edit.md",
        'Edit a file.\nTOOL edit path=<string> old=<block>\n```json\n{"properties": {}}\n```\n',
    )
    assert found == [
        ("adapters/qwen_coder/templates/tools/edit.md", 2, "E4"),
        ("adapters/qwen_coder/templates/tools/edit.md", 4, "E4"),
    ]


def test_an_example_line_names_values_and_passes(tree: Path) -> None:
    assert (
        _templates(tree, "qwen_coder/templates/examples/edit.md", "TOOL {name} path=src/a.py\n")
        == []
    )
    assert (
        _templates(tree, "qwen_coder/templates/formats/x.md", "TOOL read path=src/foo.py\n")
        == []
    )


def test_a_type_in_an_example_line_or_spelled_in_words_fails(tree: Path) -> None:
    """An example names values: a type in one is a second argument list, wherever it is."""
    found = _templates(
        tree,
        "qwen_coder/templates/examples/edit.md",
        "TOOL {name} path=<string>\nTOOL edit path=<a path>\n",
    )
    assert found == [
        ("adapters/qwen_coder/templates/examples/edit.md", 1, "E4"),
        ("adapters/qwen_coder/templates/examples/edit.md", 2, "E4"),
    ]


# -- E2, a second text matcher ---------------------------------------------------------------


def test_an_anchor_searched_for_outside_the_matcher_fails(tree: Path) -> None:
    _write(
        tree,
        "box/draft.py",
        "def apply(text, hunk, old_string, anchor):\n"
        "    n = text.count(hunk.old)\n"
        "    text = text.replace(old_string, hunk.new, 1)\n"
        "    at = text.find(anchor)\n"
        "    head, _, tail = text.partition(hunk.old_text)\n"
        "    return text.index(old)\n",
    )
    assert _found(tree) == [
        ("box/draft.py", 2, "E2"),
        ("box/draft.py", 3, "E2"),
        ("box/draft.py", 4, "E2"),
        ("box/draft.py", 5, "E2"),
        ("box/draft.py", 6, "E2"),
    ]


def test_searches_for_what_is_not_an_anchor_pass(tree: Path) -> None:
    _write(
        tree,
        "adapters/base.py",
        "def f(text, needle, old, by_id, ids, old_id, holder, x):\n"
        "    at = text.find(needle, 3)\n"
        '    text = text.replace("\\r\\n", "\\n")\n'
        "    if old in by_id:\n"
        "        return ids.index(old_id)\n"
        "    return holder.count(x) + old.count(x)\n",
    )
    assert _found(tree) == []


def test_the_matcher_passes(tree: Path) -> None:
    _write(
        tree,
        "edits/match.py",
        "def resolve(current, old, new):\n"
        "    n = current.count(old)\n"
        "    return current.replace(old, new, 1)\n",
    )
    assert _found(tree) == []


# -- E3, a diff as a diff --------------------------------------------------------------------


def test_the_diff_mode_outside_schema_fails(tree: Path) -> None:
    _write(
        tree,
        "applier/applier.py",
        "from chickadee.schema import patch\n"
        "def f(p):\n"
        "    if p.mode is ApplyMode.DIFF:\n"
        "        return 1\n"
        "    return p.mode == patch.ApplyMode.DIFF\n",
    )
    assert _found(tree) == [("applier/applier.py", 3, "E3"), ("applier/applier.py", 5, "E3")]


def test_other_modes_and_the_schema_pass(tree: Path) -> None:
    _write(
        tree,
        "applier/applier.py",
        "def f(p, r):\n"
        "    if p.kind is ChangeKind.DIFF:\n"
        "        return 0\n"
        "    return p.mode is ApplyMode.SEARCH_REPLACE or r is ApplyReason.DIFF_REJECT\n",
    )
    _write(tree, "schema/patch.py", "class ApplyMode:\n    DIFF = 'diff'\nX = ApplyMode.DIFF\n")
    assert _found(tree) == []


# -- E6, a grammar outside its format ---------------------------------------------------------


def _found_all(root: Path) -> list[tuple[str, int, str]]:
    files = sorted(p for p in (root / SRC).rglob("*") if p.suffix in (".py", ".lark", ".gbnf"))
    return [(v.path.removeprefix(CORE), v.line, v.text[:2]) for v in edits.find(files)]


def test_a_grammar_outside_the_formats_or_beside_no_parser_fails(tree: Path) -> None:
    _write(tree, "client/patch.lark", "start: line+\n")
    _write(tree, "edits/formats/orphan.lark", "start: line+\n")
    _write(tree, "adapters/local.gbnf", "root ::= line+\n")
    _write(
        tree,
        "client/inline.py",
        'G = """\nstart: begin section+ end\n"""\nH = "%import common.LF"\n',
    )
    assert _found_all(tree) == [
        ("adapters/local.gbnf", 0, "E6"),
        ("client/inline.py", 1, "E6"),
        ("client/inline.py", 4, "E6"),
        ("client/patch.lark", 0, "E6"),
        ("edits/formats/orphan.lark", 0, "E6"),
    ]


def test_a_grammar_beside_its_parser_and_look_alikes_pass(tree: Path) -> None:
    _write(tree, "edits/formats/apply_patch.lark", "start: begin\n%import common.LF\n")
    _write(tree, "edits/formats/apply_patch.py", 'RULE = "start: begin"\n')
    _write(
        tree,
        "client/notes.py",
        '"""start: a docstring may quote a rule"""\n'
        'A = "the start: of a sentence"\n'
        'B = "restart: x"\n'
        'C = "import common"\n',
    )
    assert _found_all(tree) == []


# -- E5, a second renderer or splitter -------------------------------------------------------


def test_the_bar_outside_lines_fails_but_prose_passes(tree: Path) -> None:
    _write(
        tree,
        "context/view.py",
        '"""Rows are `N│text`."""\n'
        "import re\n"
        'ROW = re.compile(r"(\\d+)│(.*)")\n'
        '"""A row read back: `7│x`."""\n'
        "def row(n, t):\n"
        '    """`n│t`."""\n'
        '    return f"{n}│{t}"\n'
        'BAR = "\\u2502"\n',
    )
    assert _found(tree) == [
        ("context/view.py", 3, "E5"),
        ("context/view.py", 7, "E5"),
        ("context/view.py", 8, "E5"),
    ]


def test_a_path_line_hit_fails(tree: Path) -> None:
    _write(
        tree,
        "boxctx/worker.py",
        "def f(path, number, line, h, rel_path, start_line):\n"
        '    a = f"{path}:{number}:{line}"\n'
        '    b = f"{h.path}:{h.line}: {h.text}"\n'
        '    return a + b + f"{rel_path}:{start_line}:"\n',
    )
    assert _found(tree) == [
        ("boxctx/worker.py", 2, "E5"),
        ("boxctx/worker.py", 3, "E5"),
        ("boxctx/worker.py", 4, "E5"),
    ]


def test_ids_mounts_citations_and_tool_diagnostics_are_not_hits(tree: Path) -> None:
    _write(
        tree,
        "ids.py",
        "def f(kind, packet_id, attempt, n, src, dst, path, line, end, row, loc, target, s):\n"
        "    a = f'{kind}:{packet_id}:{attempt}:{n}'\n"
        '    b = f"{src}:{dst}:ro"\n'
        '    c = f"{path}:{line}" + f"{path}:{line}-{end}" + f"{kind}:{row}:x"\n'
        "    d = f\"{row.get('filename')}:{loc.get('row', '?')}: x\"\n"
        "    return a + b + c + d + f\"{target}:{s.get('line', '?')}: x\"\n",
    )
    assert _found(tree) == []


def test_lines_numbered_by_another_splitter_fail(tree: Path) -> None:
    _write(
        tree,
        "scout/search.py",
        "def f(text, n, a, b):\n"
        "    for i, line in enumerate(text.splitlines(), start=1):\n"
        "        pass\n"
        '    for i, line in enumerate(text.split("\\n")):\n'
        "        pass\n"
        "    x = text.splitlines()[n - 1]\n"
        '    return text.split("\\n")[a:b]\n',
    )
    assert _found(tree) == [
        ("scout/search.py", 2, "E5"),
        ("scout/search.py", 4, "E5"),
        ("scout/search.py", 6, "E5"),
        ("scout/search.py", 7, "E5"),
    ]


def test_a_split_that_numbers_nothing_passes(tree: Path) -> None:
    _write(
        tree,
        "handoff.py",
        "def f(out, text, xs):\n"
        "    rows = [ln for ln in out.splitlines() if ln]\n"
        "    n = len(text.splitlines())\n"
        "    first = text.splitlines()[0]\n"
        "    last = text.splitlines()[-1]\n"
        "    rest = text.splitlines()[1:]\n"
        '    cells = enumerate(text.split("\\t"))\n'
        "    return enumerate(xs)\n",
    )
    assert _found(tree) == []


def test_any_splitlines_in_a_worker_view_fails_once(tree: Path) -> None:
    _write(
        tree,
        "tools/catalogue.py",
        "def capped(text):\n"
        "    lines = text.splitlines(keepends=True)\n"
        "    return dict(enumerate(text.splitlines()))\n",
    )
    _write(
        tree, "boxctx/worker.py", "def f(text):\n    return [x for x in text.splitlines()]\n"
    )
    _write(tree, "tools/args.py", 'def f(text):\n    return text.split("\\t")\n')
    _write(tree, "tools/hits.py", 'def f(text):\n    return text.split("\\n")\n')
    assert _found(tree) == [
        ("boxctx/worker.py", 2, "E5"),
        ("tools/catalogue.py", 2, "E5"),
        ("tools/catalogue.py", 3, "E5"),
        ("tools/hits.py", 2, "E5"),
    ]


def test_a_line_tag_cut_outside_lines_fails(tree: Path) -> None:
    _write(
        tree,
        "boxctx/worker.py",
        "import hashlib\n"
        "def tag(line):\n"
        "    return hashlib.sha256(line.encode()).hexdigest()[:3]\n"
        "def tag4(line, h):\n"
        "    return h.hexdigest()[0:4]\n"
        "def bound(h, lines):\n"
        "    return h.hexdigest()[: lines.TAG_DIGITS]\n",
    )
    assert _found(tree) == [
        ("boxctx/worker.py", 3, "E5"),
        ("boxctx/worker.py", 5, "E5"),
        ("boxctx/worker.py", 7, "E5"),
    ]


def test_an_id_or_digest_cut_longer_passes(tree: Path) -> None:
    _write(
        tree,
        "secrets/redact.py",
        "def f(h, n):\n"
        "    a = h.hexdigest()[:8]\n"
        "    b = h.hexdigest()[:16]\n"
        "    c = h.hexdigest()[3:]\n"
        "    d = h.digest()[:3]\n"
        "    return h.hexdigest()[:n]\n",
    )
    assert _found(tree) == []


def test_the_home_passes(tree: Path) -> None:
    _write(
        tree,
        "edits/lines.py",
        'BAR = "│"\n'
        "def number(path, n, text):\n"
        '    rows = [f"{n}{BAR}{t}" for n, t in enumerate(text.splitlines())]\n'
        '    return f"{path}:{n}:{text}"\n'
        "def tag(h):\n"
        "    return h.hexdigest()[:TAG_DIGITS]\n",
    )
    assert _found(tree) == []


def test_the_guards_bar_is_the_renderers() -> None:
    # A guard imports only the standard library, so it restates the bar; this keeps it true.
    assert edits.BAR == lines.BAR


# -- exemptions, baseline, the live tree -----------------------------------------------------


def test_exemptions_are_by_scope(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = f"{CORE}graph/indexer.py"
    monkeypatch.setattr(edits, "EXEMPT", {(rel, "git_recency"): "process output"})
    _write(
        tree,
        "graph/indexer.py",
        "def git_recency(out):\n"
        '    return dict(enumerate(out.split("\\n")))\n'
        "def other(out):\n"
        '    return dict(enumerate(out.split("\\n")))\n',
    )
    assert _found(tree) == [("graph/indexer.py", 4, "E5")]


def test_a_stale_exemption_fails(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edits, "EXEMPT", {(f"{SRC}/gone.py", "f"): "was a copy"})
    _write(tree, "ok.py", "x = 1\n")
    files = sorted((tree / SRC).rglob("*.py"))
    assert [v.text for v in edits.find(files)] == ["stale exemption: f"]


def test_the_live_tree_is_exactly_its_baseline() -> None:
    outcome = runner.evaluate(edits.GUARD, REPO_ROOT)
    assert not outcome.failed, runner.failures_of(outcome)
    assert dict(outcome.counts) == edits.BASELINE


# The copies T-3.127 removed, as they were: each is a site, so the guard would have failed
# them, and fails the next.
_OLD_COPIES = {
    "applier/hunks.py": (
        "def resolve_hunks(text, hunks):\n"
        "    for i, hunk in enumerate(hunks):\n"
        "        n = text.count(hunk.old)\n"
        "        text = text.replace(hunk.old, hunk.new, 1)\n"
        "    return text\n"
    ),
    "context/fit.py": (
        "def number_lines(lines, start, width):\n"
        '    return [f"{n:>{width}}│{t}" for n, t in enumerate(lines, start=start)]\n'
    ),
    "scout/findings.py": (
        "def lines(text):\n    return dict(enumerate(text.splitlines(), start=1))\n"
    ),
}


def test_the_guard_finds_the_copies_removed(tree: Path) -> None:
    for rel, body in _OLD_COPIES.items():
        _write(tree, rel, body)
    assert {rel for rel, _, _ in _found(tree)} == set(_OLD_COPIES)
