"""Pinned stock-harness driver and usage normalization.

One module, **two uses that must never be confused**, which is why `Role` is required rather
than inferred:

| Role | §2 row | Model | What it may support |
|---|---|---|---|
| `E1_DIAGNOSTIC` | E1 external diagnostic | the same cheap coder | G1's anti-strawman check |
| `E2_BASELINE` | E2 baseline | the pinned frontier model | the headline cost comparison |

§2 states the boundary outright — the E1 row is *"anti-strawman check only; not an E2
headline"* — and §2 adds that E2 *"does not use the naive runner as its baseline"*. A driver
that let one run stand in for the other would make that boundary a matter of who remembered
it, so `Result.headline_eligible` is computed from the role and the accounting rather than
being a flag a caller sets.

**Refusal is the specified behaviour, not a failure mode.** *"No candidate baseline can
use the exact pinned frontier model with auditable API accounting → E2 is blocked; do not
replace it with an in-house runner or label a weaker comparison headline."* So a harness that
cannot report token counts raises for `E2_BASELINE` instead of returning an estimate. An
estimated cost in a cost comparison is the whole result.

Tokens and dollars are refused separately, because they are separate facts. A harness can
report counts and no price; `cost_usd or 0.0` then turns *unknown* into *free*, and that
zero does not stay put — it reaches the run archive, `RunSummary.spent_usd`, and the
`hard_cost_usd` cap, so a baseline could spend its entire budget with the cap never seeing
a dollar. A timeout is the same hazard wearing a different hat: the harness billed for
everything it did before the clock ran out, so that path parses whatever partial accounting
it printed and blocks E2 rather than recording an abandoned paid run as $0.

**Pinning is recorded, never inferred.** Version and flags come from the manifest and are
echoed into every record. A harness that reports a different version than the one
pinned is a mismatch the run must not absorb.

This module builds the driver; `T-6.1` pins the actual harness and model for E2.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from chickadee_eval.problems.polyglot import PromptView
from chickadee_eval.runner import Attempt


class Role(StrEnum):
    """Which §2 row this invocation is. Required — the two are not interchangeable."""

    E1_DIAGNOSTIC = "e1_diagnostic"
    E2_BASELINE = "e2_baseline"


class BaselineBlocked(RuntimeError):
    """The baseline cannot meet E2's requirements, so E2 is blocked.

    Raised rather than downgraded. The alternative — continuing with a weaker comparison and
    labelling it headline — is the specific outcome §7 forbids by name.
    """


class AccountingUnavailable(BaselineBlocked):
    """The harness ran but reported no auditable token or cost accounting."""


@dataclass(frozen=True)
class HarnessSpec:
    """A pinned external harness. Every field is recorded in the run archive."""

    name: str
    version: str
    argv: tuple[str, ...]
    """`{file}` and `{prompt_file}` are substituted per token. An argv list, never a shell
    string: a problem statement is benchmark-supplied text and must stay one argument."""
    model_id: str
    flags: tuple[str, ...] = ()
    version_argv: tuple[str, ...] = ()
    """How to ask the harness its version, so the pin can be verified rather than trusted."""

    def rendered(self, *, file: str, prompt_file: str) -> list[str]:
        subs = {"{file}": file, "{prompt_file}": prompt_file}
        out = [*self.argv, *self.flags]
        return [subs.get(tok, tok) for tok in out]


@dataclass(frozen=True)
class Usage:
    """Normalized accounting. `None` means the harness did not report it — never zero.

    Zero and unknown are different facts and only one of them can be summed. Collapsing them
    would make a harness that reports nothing look like the cheapest arm in the comparison.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def auditable(self) -> bool:
        """Token accounting is present. Necessary for E2, and not sufficient — see `priced`."""
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def priced(self) -> bool:
        """A monetary figure was actually reported.

        Separate from `auditable` because tokens and dollars are separate facts and only
        one of them is the cost column. A harness can print counts and no price, and
        `cost_usd or 0.0` then turns "unknown" into "free" — which flows into the run
        archive, into `spent_usd`, and into the `hard_cost_usd` cap, so an E2 baseline
        could burn its whole budget without the cap ever seeing a dollar.
        """
        return self.cost_usd is not None


# Harnesses print usage in their own shapes. Patterns are tried in order and the first that
# matches wins; an unmatched output yields an unauditable `Usage`, which for E2 is a refusal
# rather than a zero.
#
# These prose patterns are **unverified against a real harness** — T-6.1 pins one, and until
# then they are informed guesses at aider- and opencode-shaped output. That is tolerable
# only because of which way the failure falls: a pattern that does not match yields
# `auditable=False`, and for `E2_BASELINE` that raises. A wrong guess blocks a run; it
# cannot quietly produce a number. Abbreviated counts ("1.2k sent") are deliberately not
# parsed — a rounded token count is not auditable accounting, so it should refuse.
USAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"tokens:\s*(?P<prompt>[\d,_]+)\s*sent.*?(?P<completion>[\d,_]+)\s*received",
        re.I | re.S,
    ),
    re.compile(
        r"prompt[_ ]tokens[\"'\s:=]+(?P<prompt>[\d,_]+)"
        r".*?completion[_ ]tokens[\"'\s:=]+(?P<completion>[\d,_]+)",
        re.I | re.S,
    ),
    re.compile(r"input[:=\s]+(?P<prompt>[\d,_]+)\s+output[:=\s]+(?P<completion>[\d,_]+)", re.I),
)
COST_PATTERN = re.compile(r"\$\s*(?P<cost>\d+\.\d+)")


def normalize_usage(output: str) -> Usage:
    """Pull token counts and cost out of a harness's own reporting.

    JSON is preferred when the harness emits it, because a regex over prose is a guess and a
    guess in the cost column is what §5.9 is protecting.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                blob = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(blob, dict) and "usage" in blob:
                u = blob["usage"]
                if isinstance(u, dict):
                    return Usage(
                        prompt_tokens=_int_or_none(u.get("prompt_tokens")),
                        completion_tokens=_int_or_none(u.get("completion_tokens")),
                        cost_usd=_float_or_none(u.get("cost_usd") or u.get("cost")),
                    )

    for pattern in USAGE_PATTERNS:
        m = pattern.search(output)
        if m:
            cost = COST_PATTERN.search(output)
            return Usage(
                prompt_tokens=_digits(m.group("prompt")),
                completion_tokens=_digits(m.group("completion")),
                cost_usd=float(cost.group("cost")) if cost else None,
            )
    return Usage()


def _captured(exc: subprocess.TimeoutExpired) -> str:
    """Whatever the harness printed before the clock ran out.

    `TimeoutExpired` carries the output captured so far, as `bytes` or `str` depending on
    how the process was started. Decoded leniently: this is a last look at a run that has
    already gone wrong, and an encoding error here would replace a partial answer with none.
    """
    parts: list[str] = []
    for buf in (exc.output, exc.stderr):
        if isinstance(buf, bytes):
            parts.append(buf.decode("utf-8", errors="replace"))
        elif isinstance(buf, str):
            parts.append(buf)
    return "".join(parts)


def _digits(raw: str) -> int:
    """`1,200` and `1_200` are the same count. Separators are display, not data."""
    return int(raw.replace(",", "").replace("_", ""))


def _int_or_none(v: object) -> int | None:
    return v if isinstance(v, int) else None


def _float_or_none(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def verify_pin(spec: HarnessSpec) -> str:
    """Confirm the installed harness is the pinned one. Returns what it reported.

    §6 requires every external harness version to be pinned and recorded, and a pin nobody
    checks is a note. A mismatch raises: absorbing it would silently compare against a
    different harness than the archive claims.
    """
    if shutil.which(spec.argv[0]) is None:
        raise BaselineBlocked(f"{spec.name}: {spec.argv[0]!r} is not installed")
    if not spec.version_argv:
        return "unverified — no version_argv declared"
    try:
        proc = subprocess.run(
            list(spec.version_argv), capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BaselineBlocked(f"{spec.name}: version probe failed: {exc}") from exc
    reported = (proc.stdout + proc.stderr).strip()
    if spec.version not in reported:
        raise BaselineBlocked(
            f"{spec.name}: pinned {spec.version!r} but the installed harness reports "
            f"{reported!r}. A run absorbing this compares against a harness the archive "
            "does not describe."
        )
    return reported


@dataclass
class ApiBaselineArm:
    """Drives a pinned external harness. Satisfies `runner.ArmDriver`."""

    spec: HarnessSpec
    role: Role
    timeout_s: int = 600
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    prompts: list[str] = field(default_factory=list[str])
    last_usage: Usage = field(default_factory=Usage)

    @property
    def name(self) -> str:
        return f"{self.spec.name}@{self.spec.version}:{self.role}"

    @property
    def headline_eligible(self) -> bool:
        """Whether a result from this arm may be reported as an E2 headline.

        Computed, never set. `E1_DIAGNOSTIC` is an anti-strawman check by definition, and a
        role that could be relabelled after the fact is not a boundary.

        Requires a price as well as token counts: the headline claim is about cost, and an
        arm whose cost column is a defaulted zero cannot carry one.
        """
        return (
            self.role is Role.E2_BASELINE
            and self.last_usage.auditable
            and self.last_usage.priced
        )

    def attempt(self, problem: PromptView, workspace: Path) -> Attempt:
        # Per-problem, reset here: `run_arm` builds one driver and hands it every problem,
        # so an accumulating list puts earlier problems' prompts into this problem's record
        # and re-scans them against this problem's oracle.
        self.prompts = []
        self.last_usage = Usage()

        prompt = (
            f"{problem.instructions}\n\n"
            f"Implement {problem.stub_path}. Reply with the complete file.\n"
        )
        self.prompts.append(prompt)
        prompt_file = workspace / ".eval-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self.spec.rendered(file=problem.stub_path, prompt_file=prompt_file.name)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=workspace,
                env=dict(self.env),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # A timeout is not a free run: the harness billed for everything it did before
            # the clock ran out. Whatever it managed to print is parsed, because partial
            # accounting beats an invented zero.
            self.last_usage = normalize_usage(_captured(exc))
            detail = f"{self.spec.name} exceeded {self.timeout_s}s"
            if self.role is Role.E2_BASELINE:
                raise AccountingUnavailable(
                    f"{detail}, with spend {'partially ' if self.last_usage.auditable else ''}"
                    "unaccounted. §7: E2 is blocked rather than downgraded. Recording an "
                    "abandoned paid run as $0 would understate the baseline's cost, which "
                    "is the one number the comparison is about."
                ) from exc
            return self._attempt_from(
                solved_claim=False, detail=f"{detail}; cost after timeout is unknown"
            )
        finally:
            prompt_file.unlink(missing_ok=True)

        self.last_usage = normalize_usage(proc.stdout + proc.stderr)
        if self.role is Role.E2_BASELINE and not self.last_usage.auditable:
            raise AccountingUnavailable(
                f"{self.spec.name}@{self.spec.version} reported no auditable token "
                "accounting. §7: E2 is blocked rather than downgraded — an estimated cost "
                "in a cost comparison is the whole result."
            )
        if self.role is Role.E2_BASELINE and not self.last_usage.priced:
            raise AccountingUnavailable(
                f"{self.spec.name}@{self.spec.version} reported tokens "
                f"({self.last_usage.prompt_tokens} in, {self.last_usage.completion_tokens} "
                "out) but no cost. Defaulting that to $0 would put an unknown spend into "
                "the archive, the run total and the hard-cost cap as if it were free, so "
                "§7 blocks instead. Pinning rates for this harness is `T-6.1`'s job."
            )
        return self._attempt_from(
            solved_claim=proc.returncode == 0,
            detail=f"exit={proc.returncode} wall_ms={int((time.monotonic() - started) * 1000)}",
        )

    def _attempt_from(self, *, solved_claim: bool, detail: str) -> Attempt:
        """Build the `Attempt`, labelling an unpriced zero as unpriced.

        Only `E1_DIAGNOSTIC` reaches here without a price — E2 has already refused — and a
        diagnostic's cost column is not a headline number. It still must not read as
        "free": zero and unknown are different facts, and the token counts stay in `detail`
        so the run can be priced from the archive once rates exist.
        """
        u = self.last_usage
        if not u.priced:
            detail = (
                f"{detail}; unpriced: cost_usd=0 is not a measurement "
                f"(tokens in={u.prompt_tokens} out={u.completion_tokens})"
            )
        return Attempt(
            solved_claim=solved_claim,
            cost_usd=u.cost_usd or 0.0,
            prompts=tuple(self.prompts),
            detail=detail,
        )


def arm_config(spec: HarnessSpec, role: Role) -> Mapping[str, object]:
    """Flat configuration for the one-variable check. See `e1_naive.arm_config`."""
    return {
        "arm": "api_baseline",
        "role": role,
        "harness": spec.name,
        "harness_version": spec.version,
        "model_id": spec.model_id,
        "flags": spec.flags,
    }


def assert_not_headline(arms: Sequence[ApiBaselineArm]) -> None:
    """Refuse a headline claim built on diagnostic arms.

    Called before a report labels a comparison headline. The E1 external diagnostic exists
    to check that the naive control is not a strawman; reading it as a conventional-harness
    baseline is the mislabelling §2 warns about, and it is easiest to commit months later
    when only the numbers remain.
    """
    offenders = [a.name for a in arms if not a.headline_eligible]
    if offenders:
        raise BaselineBlocked(
            f"not headline-eligible: {offenders}. An E1 diagnostic arm is an anti-strawman "
            "check (§2), and an arm without auditable accounting cannot carry a cost claim."
        )
