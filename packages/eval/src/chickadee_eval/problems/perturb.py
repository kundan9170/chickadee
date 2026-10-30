"""D23 contamination control: seeded identifier renames and spec tweaks.

Polyglot problems are Exercism exercises. They have been public for years, they are in every
crawl, and a model that has memorised `affine_cipher.encode` does not have to solve anything
to emit it. §5.3 states the consequence plainly: *"if the perturbed delta is much smaller
than the original-set delta, memorisation is doing the work and the headline claim must be
restated."* That comparison only means something if the perturbation actually removes the
memorisation handle while leaving the problem intact.

So this generator has exactly two obligations, and they pull against each other:

**Remove the handle.** Every token a memorised answer could key on — the module name, the
function and class names the tests import, the canonical phrasing of the instructions — is
rewritten. A suffix would not do: `affine_cipher_x7` still contains `affine_cipher`, and the
retrieval it enables is the thing being controlled for. Names are replaced outright.

**Keep it solvable.** A perturbation that broke the problem would show up as a lower
solve rate, be read as evidence against memorisation, and be exactly wrong. So renames are
applied to the stub, the tests and the instructions *together*, and `test_perturb.py` proves
solvability the only way that counts: it perturbs a real benchmark problem along with its
reference solution and runs the perturbed tests against it. If the perturbation broke the
problem, the reference solution stops passing and the test fails.

**Why the spec tweak is prose-only.** "Altered specs" is deliberately read as
*rephrased*, not *changed*. A generator that altered what a problem asks for would have to
regenerate its tests to match, and a generator that silently disagreed with its own oracle
would manufacture unsolvable problems — the failure mode above, at scale. Substitutions come
from a closed table of English phrasings with no technical content, and never apply inside
backticks or fenced blocks, so identifiers and worked examples are untouched.

**Everything is derived from `(seed, problem_id)`.** Same inputs, same output, on any
machine, forever. A perturbation nobody can regenerate is an anecdote, and the comparison
would rest on a set that no longer exists.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from chickadee_eval.manifest import Language

PERTURBED_SUFFIX = "-perturbed"
"""Appended to the source id. The perturbed problem is a *different problem* with its own
row in the archive — reusing the original id would make the two indistinguishable in
`results.jsonl`, which is where the delta is computed."""


class PerturbError(RuntimeError):
    """The problem could not be perturbed. Raised rather than returning it unchanged.

    An unperturbed problem in the perturbed set is worse than a missing one: it reports a
    memorisation delta of zero for a problem that was never perturbed, and reads as evidence
    that memorisation is not happening.
    """


# Declarations whose names the tests import. Deliberately anchored at column zero: a nested
# helper is an implementation detail the tests never name, and renaming it would edit the
# candidate's internals for no contamination benefit.
DECL_PATTERNS: dict[Language, tuple[re.Pattern[str], ...]] = {
    Language.PYTHON: (
        re.compile(r"^(?:def|class)\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^([A-Za-z_]\w*)\s*=\s*", re.M),
    ),
    Language.JAVASCRIPT: (
        re.compile(
            r"^export\s+(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M
        ),
        re.compile(r"^export\s+class\s+([A-Za-z_$][\w$]*)", re.M),
        re.compile(r"^export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)", re.M),
    ),
}

# Names that are structural rather than the problem's own vocabulary. Renaming `__init__`
# would not remove a memorisation handle; it would break the language.
PROTECTED = frozenset(
    {
        "main",
        "self",
        "cls",
        "default",
        "constructor",
        "module",
        "exports",
        "require",
        "prototype",
    }
)

# Pseudo-words, assembled deterministically. A generated name has to be pronounceable enough
# to read in a traceback and carry none of the original's meaning — `kelvar_dosune` tells a
# memorising model nothing, and tells a human debugging a failed run that they are looking
# at a perturbed problem.
_ONSETS = ("b", "d", "f", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "tr", "kl")
_NUCLEI = ("a", "e", "i", "o", "u", "ai", "ea", "ou")
_CODAS = ("", "", "l", "n", "r", "s", "m", "k")

# Closed substitution table: English phrasing with no technical content. Ordered longest
# first so a longer phrase wins before a shorter one inside it can match. Each entry must be
# a pure rephrasing — if a substitution could change what the problem asks for, it does not
# belong here, because nothing downstream would catch it.
PROSE_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("You are given", "You receive"),
    ("you are given", "you receive"),
    ("Write a function", "Implement a routine"),
    ("write a function", "implement a routine"),
    ("Your task is to", "You must"),
    ("your task is to", "you must"),
    ("For example", "As an illustration"),
    ("for example", "as an illustration"),
    ("In this exercise", "In this problem"),
    ("in this exercise", "in this problem"),
    ("Note that", "Observe that"),
    ("note that", "observe that"),
    ("You should", "You ought to"),
    ("you should", "you ought to"),
)


@dataclass(frozen=True)
class Rename:
    old: str
    new: str


@dataclass(frozen=True)
class Perturbation:
    """What was done, recorded so a run can be explained without regenerating it."""

    source_id: str
    perturbed_id: str
    language: Language
    seed: int
    renames: tuple[Rename, ...]
    module_rename: Rename | None
    prose_edits: int

    @property
    def is_empty(self) -> bool:
        """No handle was removed. `generate` refuses this rather than shipping it."""
        return not self.renames and self.module_rename is None and self.prose_edits == 0


def perturbed_id(source_id: str) -> str:
    return f"{source_id}{PERTURBED_SUFFIX}"


def _rng_int(seed: int, *parts: str) -> int:
    """A stable integer from the seed and some strings.

    `hashlib` rather than `random.Random`, because `Random`'s stream is a CPython
    implementation detail. A perturbed set that regenerated differently on another
    interpreter would fail the one property D23 actually needs.
    """
    material = f"{seed}\x00" + "\x00".join(parts)
    return int.from_bytes(hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest())


def coined_name(seed: int, problem_id: str, old: str, *, syllables: int = 2) -> str:
    """A deterministic pseudo-word to replace `old`.

    Keyed on the original name as well as the problem, so two identifiers in one problem get
    different replacements and the same identifier always gets the same one.
    """
    n = _rng_int(seed, problem_id, old)
    out: list[str] = []
    for i in range(syllables):
        n, o = divmod(n, len(_ONSETS))
        n, v = divmod(n, len(_NUCLEI))
        n, c = divmod(n, len(_CODAS))
        out.append(_ONSETS[o] + _NUCLEI[v] + (_CODAS[c] if i == syllables - 1 else ""))
    return "".join(out)


def _snake_or_camel(old: str, coined: str) -> str:
    """Match the original's shape so the perturbed source still reads as idiomatic.

    A `PascalCase` class replaced by a snake_case name would be a second, unintended signal
    in the diff — and one a model could learn to spot as "this is the perturbed set".
    """
    if old[:1].isupper():
        return coined[:1].upper() + coined[1:]
    return coined


def find_identifiers(source: str, language: Language) -> tuple[str, ...]:
    """Public declarations in a stub, in first-seen order.

    Order matters: it is what makes the rename list stable across runs, and the list is
    recorded in the manifest.
    """
    seen: dict[str, None] = {}
    for pattern in DECL_PATTERNS[language]:
        for name in pattern.findall(source):
            if name in PROTECTED or name.startswith("__"):
                continue
            seen.setdefault(name, None)
    return tuple(seen)


def rename_in(text: str, renames: tuple[Rename, ...]) -> str:
    """Apply renames in one pass, treating `_` as a boundary.

    One pass, not one per rename: sequential passes let an earlier replacement be matched
    and rewritten again by a later one, so `a -> b` followed by `b -> c` silently turns
    every original `a` into `c`. A single alternation cannot chain.

    `\\b` is the obvious boundary and the wrong one. Under it `decode` inside
    `test_decode_exercism` is not a match, because `_` is a word character — so the name
    survives in every test method that mentions it. That matters here specifically:
    `PromptView.test_names` carries test *names* into prompts by design,
    so an unrenamed `test_decode_exercism` walks the original identifier straight back into
    the thing D23 is removing it from.

    Underscore-as-boundary renames the component and leaves genuine extensions alone:
    `test_decode_exercism` is rewritten, `decoder` and `decoded` are not.
    """
    if not renames:
        return text
    table = {r.old: r.new for r in renames}
    alternation = "|".join(re.escape(r.old) for r in sorted(renames, key=lambda r: -len(r.old)))
    pattern = re.compile(rf"(?<![A-Za-z0-9])({alternation})(?![A-Za-z0-9])")
    return pattern.sub(lambda m: table[m.group(1)], text)


def _split_code_spans(text: str) -> list[tuple[str, bool]]:
    """Split markdown into (chunk, is_code) runs.

    Prose substitution must not reach inside code. Fenced blocks and inline backticks carry
    identifiers and worked examples, and rewriting an example's text would alter what the
    problem asks for — the one thing this generator promises not to do.
    """
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in re.finditer(r"```.*?```|`[^`\n]*`", text, re.S):
        if m.start() > pos:
            out.append((text[pos : m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def tweak_prose(text: str) -> tuple[str, int]:
    """Rephrase instruction prose from the closed table. Returns (text, edits)."""
    edits = 0
    chunks: list[str] = []
    for chunk, is_code in _split_code_spans(text):
        if is_code:
            chunks.append(chunk)
            continue
        for old, new in PROSE_SUBSTITUTIONS:
            if old in chunk:
                edits += chunk.count(old)
                chunk = chunk.replace(old, new)
        chunks.append(chunk)
    return "".join(chunks), edits


def _text_files(root: Path) -> list[Path]:
    """Files whose contents are rewritten. Binary and metadata files are copied untouched."""
    keep = {".py", ".js", ".md", ".txt", ".json", ".toml", ".jsx", ".mjs", ".cjs"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in keep)


def plan(source: Path, problem_id: str, language: Language, seed: int) -> Perturbation:
    """Decide what to rename, without writing anything.

    Separated from `apply` so a manifest can record the plan, and so a review can read what
    a perturbed set did without regenerating 20 directories.
    """
    if not source.is_dir():
        raise PerturbError(f"{problem_id}: no directory at {source}")

    stub = _find_stub(source, language, problem_id)
    identifiers = find_identifiers(stub.read_text(encoding="utf-8"), language)
    renames = tuple(
        Rename(old=name, new=_snake_or_camel(name, coined_name(seed, problem_id, name)))
        for name in identifiers
    )

    stem = stub.stem
    module_rename = (
        Rename(old=stem, new=coined_name(seed, problem_id, f"module:{stem}"))
        if stem not in PROTECTED
        else None
    )

    prose_edits = 0
    for doc in source.rglob("*.md"):
        _, n = tweak_prose(doc.read_text(encoding="utf-8"))
        prose_edits += n

    return Perturbation(
        source_id=problem_id,
        perturbed_id=perturbed_id(problem_id),
        language=language,
        seed=seed,
        renames=renames,
        module_rename=module_rename,
        prose_edits=prose_edits,
    )


def _find_stub(source: Path, language: Language, problem_id: str) -> Path:
    """The candidate-facing stub, which is what declares the names the tests import.

    `.meta/` is excluded explicitly. It holds the reference solution — `example.py`,
    `proof.ci.js` — and picking it here would mean planning renames against the answer key
    rather than the stub. See issue #60, where the loader does exactly that.
    """
    suffix = ".py" if language is Language.PYTHON else ".js"
    test_markers = ("_test.py", ".spec.js", ".test.js")
    candidates = [
        p
        for p in sorted(source.rglob(f"*{suffix}"))
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(source).parts[:-1])
        and not any(p.name.endswith(m) for m in test_markers)
        and p.name not in ("babel.config.js", "jest.config.js")
    ]
    if not candidates:
        raise PerturbError(f"{problem_id}: no {suffix} stub outside .meta/ under {source}")
    return candidates[0]


def apply(source: Path, dest: Path, perturbation: Perturbation) -> None:
    """Materialize the perturbed problem at `dest`.

    The source layout is preserved, `.meta/` included: the reference solution is what
    `test_perturb.py` uses to prove the problem is still solvable, and a loader that must
    exclude `.meta/` anyway is the right place for that exclusion — not here,
    where dropping it would remove the only evidence the perturbation is safe.
    """
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    renames = perturbation.renames
    if perturbation.module_rename:
        renames = (*renames, perturbation.module_rename)

    for path in _text_files(dest):
        text = path.read_text(encoding="utf-8")
        text = rename_in(text, renames)
        if path.suffix == ".md":
            text, _ = tweak_prose(text)
        path.write_text(text, encoding="utf-8")

    # Renamed last: rewriting contents first means every import already names the new module,
    # and the file moves to meet them.
    if perturbation.module_rename:
        old, new = perturbation.module_rename.old, perturbation.module_rename.new
        for path in sorted(dest.rglob("*")):
            if path.is_file() and path.stem.startswith(old):
                path.rename(path.with_name(new + path.name[len(old) :]))


def generate(
    source_root: Path, dest_root: Path, problem_id: str, language: Language, seed: int
) -> Perturbation:
    """Plan and materialize one perturbed problem. Returns what was done."""
    perturbation = plan(source_root / problem_id, problem_id, language, seed)
    if perturbation.is_empty:
        raise PerturbError(
            f"{problem_id}: nothing was perturbed — no identifiers, no module rename and no "
            "prose edits. An unperturbed problem in the perturbed set reports a memorisation "
            "delta of zero for a problem that was never perturbed (D23)."
        )
    apply(source_root / problem_id, dest_root / perturbation.perturbed_id, perturbation)
    return perturbation
