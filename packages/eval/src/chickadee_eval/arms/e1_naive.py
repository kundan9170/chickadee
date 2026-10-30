"""The E1 naive control — the arm G1 exists to scrutinise.

Definition from §2: *full stub file + problem text, whole-file replace, bounded retry,
failure output verbatim*. That is what a competent person writes when they have a model and
an afternoon, and it is deliberately not crippled — G1's question is whether the control is
*"an obvious strawman relative to stock harness"*, so every weakening here would be a
finding against the harness's own numbers rather than a saving.

Three consequences of taking the anti-strawman duty seriously:

**It gets the whole stub, not an excerpt.** No packing, no context selection — those are the
harness's levers, and lending them to the control would measure a blurred pair.

**It retries with real failure text.** A developer reruns and reads the traceback. A control
that retried blind would fail for a reason no real workflow shares, and its solve rate would
understate what a naive approach achieves.

**It uses the same model and the same retry ladder as the arm it is compared against**
(temperature same or higher on retry, never lower). Any difference here is a difference in the
experiment rather than in the approach.

**Where §2 and §5.1 collide, and how.** §2 asks for failure output *verbatim*; §5.1 forbids
test bodies from entering any prompt. Raw `pytest` output prints the failing assertion's
source, so feeding it satisfies neither:

    > assert f(2) == 4, "doubling two has to give four"

`redact_oracle()` removes exactly the lines that appear in the hidden test file and leaves
everything else untouched — exception type, computed values, the `where 0 = f(2)` line, the
traceback location. "Verbatim" is read as *not summarised* rather than *including source
echoes*, and the redaction is checkable rather than hoped at: after it, `polyglot.leaks()`
must return empty, which is a test. Both arms are fed through the same function, so parity
holds and the comparison stays fair.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from chickadee.engine import ModelClient, to_request
from chickadee.schema import Completion, Prompt, RetryParams
from chickadee_eval.problems import polyglot
from chickadee_eval.problems.polyglot import LoadedProblem, PromptView
from chickadee_eval.runner import Attempt

REDACTED = "<redacted: hidden test source (D7)>"

INSTRUCTIONS = """\
You are given a programming problem and a stub file. Replace the stub file's entire
contents with a working implementation.

Reply with the complete file and nothing else. Do not explain. Do not wrap the file in
commentary. A fenced code block is acceptable; its contents are taken as the whole file.
"""


def redact_oracle(output: str, problem: LoadedProblem) -> tuple[str, int]:
    """Remove hidden-test source lines from runner output. Returns (text, removed).

    Line-for-line so positions are preserved — a traceback whose line numbers shifted would
    be less useful than one with a hole in it, and the hole is visible.

    An output line is removed when it **contains** a line `polyglot.oracle_lines` names, the
    comparison `polyglot.leaks()` makes, so what survives here is what that check accepts
    whatever the runner wraps a source echo in: pytest's `>` and `E`, or jest's code frame
    (`> 11 | expect(solve(puzzle)).toEqual(expected);`). Comparing whole lines only, as
    this first did, let jest's frames through, and a JavaScript retry was refused as a leak
    and the run discarded (2026-10-24). A line equal to any long line of the hidden file is
    removed too. Short lines are left alone for the same reason the leak scan ignores them:
    `from lib import f` is a real line of the hidden file and also the least informative
    thing in the output.
    """
    body = {
        line.strip()
        for line in problem.test_source().splitlines()
        if len(line.strip()) >= polyglot.MIN_MARKER
    }
    watched = [line for _, line in polyglot.oracle_lines(problem)]
    out: list[str] = []
    removed = 0
    for raw in output.splitlines():
        # pytest prefixes source echoes with '>' or 'E' and pads them, so the bare text has
        # to be compared as well — otherwise the marker alone defeats the match and the
        # redaction silently does nothing.
        bare = raw.lstrip("> E\t")
        if bare.strip() in body or raw.strip() in body or any(w in raw for w in watched):
            out.append(REDACTED)
            removed += 1
        else:
            out.append(raw)
    return "\n".join(out), removed


@dataclass(frozen=True)
class NaiveConfig:
    """Everything that changes this arm's behaviour, named so a manifest can pin it."""

    model_id: str
    max_attempts: int = 3
    temperature_ladder: tuple[float, ...] = (0.0, 0.2, 0.4)
    """Same or higher on retry, never lower. Validated, because a ladder that descends
    would make retries quieter rather than more varied and the arm would look worse for a
    reason unrelated to the approach."""
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if len(self.temperature_ladder) < self.max_attempts:
            raise ValueError(
                f"temperature_ladder has {len(self.temperature_ladder)} entries for "
                f"{self.max_attempts} attempts; every attempt needs a declared temperature"
            )
        ladder = self.temperature_ladder[: self.max_attempts]
        if any(b < a for a, b in pairwise(ladder)):
            raise ValueError(f"temperature ladder must not descend (D5), got {ladder}")


def extract_file(reply: str) -> str:
    """The whole-file replacement in a model reply.

    A fenced block wins when present, since that is what the instruction asks for and what
    a model reliably produces. Otherwise the reply is taken as the file — deliberately
    permissive: a control that rejected an unfenced but perfectly good file would fail for a
    formatting reason, which is a harness concern rather than a model one.
    """
    if "```" not in reply:
        return reply.strip() + "\n"
    parts = reply.split("```")
    if len(parts) < 3:
        return reply.strip() + "\n"
    block = parts[1]
    first, _, rest = block.partition("\n")
    # Drop a language tag, but only if the first line is one rather than code.
    body = rest if first.strip().isalpha() and len(first.strip()) <= 12 else block
    return body.strip() + "\n"


@dataclass
class NaiveArm:
    """Whole-file replace with bounded retry. Satisfies `runner.ArmDriver`."""

    client: ModelClient
    config: NaiveConfig
    oracles: Mapping[str, LoadedProblem]
    """Every problem this arm may be asked for, by id, held so a failure can be redacted
    against **its own** oracle. Keyed rather than singular because `runner.run_arm` builds
    one driver and hands it every problem in turn: a single `LoadedProblem` fixed at
    construction meant that from the second problem onward `redact_oracle` matched against
    the wrong file, so the real oracle survived into the retry prompt. `run_problem` scans
    against the right problem and would have raised `TestBodyLeak`, discarding the run — a
    loud failure rather than a silent leak, but the arm was unusable past one problem
    either way.

    The arm never renders anything from these — `attempt()` gets a `PromptView` for that,
    and `redact_oracle` is the only reader of `test_source()` on this path."""
    score: Callable[[Path], tuple[bool, str]]
    """Runs the hidden tests and returns (passed, output). Injected so the arm cannot
    reach the oracle itself and so the loop is testable without a toolchain."""
    prices: ArmPrices | None = None
    """The model's declared rates. `None` reports zero cost *and says so* — see `price()`."""
    name: str = "e1_naive"
    prompts: list[str] = field(default_factory=list[str])
    """The prompts rendered for the **most recent** problem, reset on every `attempt()`.
    Per-problem because `Attempt` is per-problem: accumulating across problems put unrelated
    prompts in each record, re-scanned every earlier problem's prompts against every later
    oracle, and grew the archive quadratically."""
    tokens_in: int = 0
    tokens_out: int = 0
    """Also per-problem, and for the same reason: a cumulative count in a per-problem
    record is not the number the row claims to hold."""

    def _render(self, view: PromptView, failure: str, attempt: int) -> str:
        head = (
            f"{INSTRUCTIONS}\n"
            f"# Problem\n\n{view.instructions}\n\n"
            f"# File: {view.stub_path}\n\n{view.stub_source}\n"
        )
        if not failure:
            return head
        return (
            f"{head}\n"
            f"# Attempt {attempt - 1} failed. Test output:\n\n{failure}\n\n"
            "Reply with the corrected complete file.\n"
        )

    def attempt(self, problem: PromptView, workspace: Path) -> Attempt:
        oracle = self.oracles.get(problem.id)
        if oracle is None:
            # Refused rather than redacted against whatever is to hand: an arm that cannot
            # identify the oracle cannot promise anything about D7, and guessing is how the
            # wrong-oracle bug reached a prompt in the first place.
            raise KeyError(
                f"{problem.id!r} has no oracle registered with this arm; without it the "
                "retry feedback cannot be redacted (D7, §5.1)"
            )
        # Per-problem state, reset here rather than at construction — one driver serves the
        # whole run.
        self.prompts = []
        self.tokens_in = 0
        self.tokens_out = 0

        target = workspace / problem.stub_path
        failure = ""
        cost = 0.0
        passed = False
        n = 0  # `max_attempts >= 1` is validated, so the loop always rebinds this.

        for n in range(1, self.config.max_attempts + 1):
            rendered = self._render(problem, failure, n)
            self.prompts.append(rendered)
            # The coder path's rendering: the instructions as the system message, the
            # rest of the prompt as the user message. `complete()` built exactly this until
            # D155 retired it.
            completion = self.client.send(
                to_request(
                    Prompt(prefix=INSTRUCTIONS, volatile=rendered[len(INSTRUCTIONS) :]),
                    RetryParams(
                        temperature=self.config.temperature_ladder[n - 1],
                        max_tokens=self.config.max_output_tokens,
                    ),
                )
            )
            self.tokens_in += completion.usage.prompt_tokens
            self.tokens_out += completion.usage.completion_tokens
            if self.prices is not None:
                cost += price(completion, self.prices)
            target.write_text(extract_file(completion.text), encoding="utf-8")

            passed, raw = self.score(workspace)
            if passed:
                break
            # §5.1 before §2: the output is fed back, minus what D7 forbids.
            failure, _removed = redact_oracle(raw, oracle)

        notes = [] if passed else ["exhausted attempts"]
        if self.prices is None:
            notes.append(
                f"unpriced: cost_usd=0 is not a measurement "
                f"(tokens in={self.tokens_in} out={self.tokens_out})"
            )
        return Attempt(
            solved_claim=passed,
            cost_usd=cost,
            attempts=n,
            prompts=tuple(self.prompts),
            detail="; ".join(notes),
        )


@dataclass(frozen=True)
class ArmPrices:
    """USD per token, declared by the eval and not read from the harness.

    The harness has a price table of its own in `chickadee.config`, and importing it here
    would be the exact coupling `test_isolation` forbids: a price edit
    inside the thing being measured would move the cost column with nothing in the eval's
    diff to show it. the comparison is priced from the preregistered protocol, so the
    rates live on the arm.
    """

    usd_per_input_token: float
    usd_per_output_token: float
    usd_per_cached_input_token: float | None = None
    """`None` = the provider does not price cached input separately, so it is charged at the
    full input rate. Pessimistic when the discount is unknown — the direction that cannot
    flatter the arm being measured."""

    def __post_init__(self) -> None:
        rates = (self.usd_per_input_token, self.usd_per_output_token)
        if any(r < 0 for r in rates) or (
            self.usd_per_cached_input_token is not None and self.usd_per_cached_input_token < 0
        ):
            raise ValueError("token prices must not be negative")


def price(completion: Completion, prices: ArmPrices) -> float:
    """USD for one completion, from reported tokens and the arm's declared rates.

    `Completion.usage` carries *tokens*, not money — nothing on the adapter path knows a
    price. So cost is computed here, and an arm without declared rates is priced at zero and
    says so in `detail` rather than reporting a number it did not derive. §5.9 compares arms
    on cost; the failure that matters is a zero that reads as "cheap" rather than as
    "unpriced".
    """
    u = completion.usage
    cached = u.cached_prompt_tokens or 0
    fresh = max(u.prompt_tokens - cached, 0)
    cached_rate = (
        prices.usd_per_cached_input_token
        if prices.usd_per_cached_input_token is not None
        else prices.usd_per_input_token
    )
    return (
        fresh * prices.usd_per_input_token
        + cached * cached_rate
        + u.completion_tokens * prices.usd_per_output_token
    )


def arm_config(config: NaiveConfig) -> Mapping[str, object]:
    """The arm's configuration as a flat mapping, for `validate_arms_differ_in_one_variable`.

    Flat because the check compares field by field: a nested structure would compare as
    one opaque value and two arms differing in three nested fields would read as differing
    in one.
    """
    return {
        "arm": "e1_naive",
        "model_id": config.model_id,
        "max_attempts": config.max_attempts,
        "temperature_ladder": config.temperature_ladder,
        "max_output_tokens": config.max_output_tokens,
    }


def ladder_is_non_descending(ladder: Sequence[float]) -> bool:
    """D5 as a predicate, exported so a manifest can be checked before a run starts."""
    return all(b >= a for a, b in pairwise(ladder))
