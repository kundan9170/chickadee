"""The limits register: `Limit`, `LIMITS`, `resolve()`, `check_fit()`.

Every per-role read, hold and call bound lives here, once. Each used to be a number
typed where it was first needed — 8 worker tool calls in `roles.py`, 8,000 packed tokens in
`packer/types.py`, 4,096 inline bytes in `zeus/tools/spec.py` — and the copies disagreed with
each other and with the docs. The limits table is a *rendering* of `LIMITS`;
`test_limits_register.py` fails when the two differ.

A token limit is a fraction of the rung's window, because one config serves 128K and 256K
rungs; it becomes a number once per config load (`resolve()`), never per call, so every call
in a session reads the same numbers. `check_fit()` then refuses, at load, a role that cannot
fit one request on its rung — a config that can never work fails when it is read, not on the
first request of an afternoon's session.

Like `roles.py`, this module imports nothing from `zeus/`, `orchestrator/` or `packer/`:
they read it, and it must not read them. And `roles.py` reads it: each role's caps are rows of
this register, so this module takes the cast only inside the functions that resolve
against it, never at import — the dependency runs one way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal

from chickadee.errors import ConfigError, HarnessBug

if TYPE_CHECKING:
    from chickadee.config import RungEndpoint
    from chickadee.roles import Role

__all__ = [
    "CHARS_PER_TOKEN",
    "HARNESS_OWNERS",
    "LIMITS",
    "Limit",
    "LimitPer",
    "LimitUnit",
    "ResolvedLimit",
    "chars_for_tokens",
    "check_fit",
    "harness_limits",
    "limit",
    "override_keys",
    "render_limits",
    "resolve",
    "resolved_value",
    "role_limits",
    "tokens_est",
    "tokens_for_chars",
    "window",
]

LimitUnit = Literal[
    "window_fraction",
    "tokens",
    "calls",
    "chars",
    "bytes",
    "lines",
    "count",
    "seconds",
    "percent",
    "cents",
]
LimitPer = Literal[
    "attempt",
    "packet",
    "request",
    "call",
    "response",
    "turn",
    "task",
    "lineage",
    "agent",
    "design",
    "result",
    "note",
    "constraint",
    "reset",
    "process",
    "server",
]

HARNESS_OWNERS: Final = frozenset({"agents", "hooks", "lsp", "skills", "turn", "web"})
"""Owners of rows that belong to no role, one number for the whole harness:
`agents`, how many agents Zeus runs at once; `hooks`, the bounds on what hooks may steer; `lsp`,
the index's language servers; `skills`, the skills index and `find_skill`;
`turn`, what the person may attach to one turn; `web`, what one web call
may take and return. `resolve` leaves
them out of the per-role table, and `harness_limits` resolves them."""

CHARS_PER_TOKEN: Final = 3.5
"""The token estimator: its one home. Here rather than in the packer
because the config loader reads this module and the packer pulls in the graph; the packer,
the client's no-usage fallback and the dispatch line all import it."""


def tokens_for_chars(chars: int) -> int:
    """`ceil(chars / 3.5)`, no tokenizer. Never negative."""
    return math.ceil(max(chars, 0) / CHARS_PER_TOKEN)


def chars_for_tokens(tokens: int) -> int:
    """The most characters the estimator counts as at most `tokens`: a token limit as a model
    can measure
    it, since its own tokenizer is not the estimator. Never negative."""
    return math.floor(max(tokens, 0) * CHARS_PER_TOKEN)


def tokens_est(text: str) -> int:
    """The estimate for a string: `tokens_for_chars(len(text))`. An estimate, never reported as
    a
    count."""
    return tokens_for_chars(len(text))


@dataclass(frozen=True)
class Limit:
    name: str
    """The config key under `limits.`, e.g. `worker.packed_context`."""
    value: float | int
    """The default; a fraction when `unit == "window_fraction"`."""
    unit: LimitUnit
    per: LimitPer
    """What one allowance covers."""
    applies_to: str
    """A `ROLES` name, or `all`."""
    why: str
    """One sentence: what goes wrong without it."""
    measured_by: str
    """The measurement row that recalibrates it."""


@dataclass(frozen=True)
class ResolvedLimit:
    name: str
    role: str
    """The role it was resolved for; `all` expands to one row per role."""
    rung: str
    """The rung that role runs on."""
    tokens_or_value: int
    """A fraction resolved to tokens; any other unit, the value itself."""
    source: Literal["default", "config"]


def _row(
    name: str,
    value: float | int,
    unit: LimitUnit,
    per: LimitPer,
    why: str,
    measured_by: str,
) -> Limit:
    # The row's owner is its name's first segment; a row can only apply to the role it is
    # named for, so the column is derived rather than typed twice.
    return Limit(name, value, unit, per, name.split(".", 1)[0], why, measured_by)


LIMITS: tuple[Limit, ...] = (
    _row(
        "worker.design_layer",
        0.15,
        "window_fraction",
        "packet",
        "records of every packet already finished in the design",
        "W2",
    ),
    _row(
        "worker.max_tool_calls",
        200,
        "calls",
        "task",
        "a ceiling over a whole task, every round of it, for work still moving: the worker "
        "finds, edits, runs and re-runs, and a stall is the progress watchdog's to hand up",
        "W3",
    ),
    _row(
        "worker.repeat",
        2,
        "count",
        "task",
        "the same call twice in a row is noted; the fourth ends the round, since a model "
        "repeating a refused call rarely changes its mind",
        "W3",
    ),
    _row(
        "worker.print_only",
        1,
        "count",
        "task",
        "a shell call that only prints is answered and not run; past the row the round ends: a "
        'worker echoing "task completed" in a loop never replied',
        "W3",
    ),
    _row(
        "worker.max_cumulative_result_tokens",
        0.30,
        "window_fraction",
        "attempt",
        "what is left of the window after the packed context, design layer, prefix and "
        "output reserve",
        "W2",
    ),
    _row(
        "worker.task_seconds",
        1800,
        "seconds",
        "task",
        "a worker round that runs past it is stopped as `limit`, told apart from a person's "
        "stop, so one task cannot hold the episode forever",
        "W3",
    ),
    _row(
        "worker.keep_recent",
        4,
        "count",
        "request",
        "the newest tool exchanges a fit never evicts: the ones the worker is acting on",
        "W3",
    ),
    _row(
        "worker.result_max",
        8000,
        "tokens",
        "result",
        "a worker reads to write and needs whole functions; the rest stays behind a handle",
        "W3",
    ),
    _row(
        "worker.whole_file_bytes",
        8192,
        "bytes",
        "result",
        "a plain `read(path)` at or below it returns the whole file, above it the outline; "
        "apart from the read cap, so a file worth reading whole is not cut",
        "W3",
    ),
    _row(
        "worker.read_bytes",
        16384,
        "bytes",
        "result",
        'the most one read shows of lines, a symbol or `depth: "full"`: the worker\'s own, '
        "never Zeus's 4 KB, which showed a small model 96 lines a read",
        "W3",
    ),
    _row(
        "worker.read_lines",
        400,
        "lines",
        "result",
        "the line bound beside `worker.read_bytes`, wide enough that the byte cap binds first "
        "on code",
        "W3",
    ),
    _row(
        "worker.raise_share",
        25,
        "percent",
        "call",
        "a raised read takes at most this share of the context still free, whatever the "
        "ceiling",
        "W3",
    ),
    _row(
        "worker.raise_strikes",
        2,
        "count",
        "task",
        "restarts past the read cap on one file before its cap is raised for the task: a model "
        "that pages never strikes, one that loops gets the pointer",
        "W3",
    ),
    _row(
        "worker.held_turns",
        6,
        "count",
        "turn",
        "a view older than this many model turns is shown again when re-read, since attention "
        "on it may have faded",
        "W3",
    ),
    _row(
        "worker.held_tokens",
        16000,
        "tokens",
        "call",
        "a view with more than this much conversation after it is shown again when re-read",
        "W3",
    ),
    _row(
        "worker.insist_showings",
        1,
        "count",
        "call",
        "a recent view asked again right after its pointer is shown this many times, then "
        "only the repetition bound answers",
        "W3",
    ),
    _row(
        "worker.output_floor",
        16000,
        "tokens",
        "call",
        "the coder rung's `max_output_tokens` must allow a whole file to be written",
        "W2",
    ),
    _row(
        "athena.max_tool_calls",
        100,
        "calls",
        "request",
        "one elaboration or repair; she explores the map the worker is handed",
        "W3",
    ),
    _row(
        "athena.max_cumulative_result_tokens",
        0.30,
        "window_fraction",
        "request",
        "the rest of her window holds design, plan, records and history",
        "W4",
    ),
    _row(
        "athena.repeat",
        2,
        "count",
        "request",
        "the same call twice in a row is noted, the fourth refused; never her round's end, so "
        'an open task is never stranded (she ran `find "test"` three times)',
        "W3",
    ),
    _row(
        "athena.result_max",
        8000,
        "tokens",
        "result",
        "the most one of her reads may be raised to: the worker's ceiling, "
        "since she reviews the files it writes",
        "W3",
    ),
    _row(
        "athena.whole_file_bytes",
        8192,
        "bytes",
        "result",
        "a plain `read(path)` at or below it returns the whole file, above it the outline; "
        "apart from the read cap, so a file worth reading whole is not cut",
        "W3",
    ),
    _row(
        "athena.read_bytes",
        16384,
        "bytes",
        "result",
        'the most one read shows of lines, a symbol or `depth: "full"`: her\'s own, never '
        "Zeus's 4 KB, which showed a small model 96 lines a read",
        "W3",
    ),
    _row(
        "athena.read_lines",
        400,
        "lines",
        "result",
        "the line bound beside `athena.read_bytes`, wide enough that the byte cap binds first "
        "on code",
        "W3",
    ),
    _row(
        "athena.raise_share",
        25,
        "percent",
        "call",
        "a raised read takes at most this share of the context still free, whatever the "
        "ceiling",
        "W3",
    ),
    _row(
        "athena.raise_strikes",
        2,
        "count",
        "task",
        "restarts past the read cap on one file before its cap is raised for the task: a model "
        "that pages never strikes, one that loops gets the pointer",
        "W3",
    ),
    _row(
        "athena.held_turns",
        6,
        "count",
        "turn",
        "a view older than this many model turns is shown again when re-read, since attention "
        "on it may have faded",
        "W3",
    ),
    _row(
        "athena.held_tokens",
        16000,
        "tokens",
        "call",
        "a view with more than this much conversation after it is shown again when re-read",
        "W3",
    ),
    _row(
        "athena.insist_showings",
        1,
        "count",
        "call",
        "a recent view asked again right after its pointer is shown this many times, then "
        "only the repetition bound answers",
        "W3",
    ),
    _row(
        "athena.note_max",
        200,
        "tokens",
        "note",
        "a note is a reminder, not a document",
        "W4",
    ),
    _row(
        "athena.resets",
        1,
        "count",
        "request",
        "compaction step 5 at most once per request, or a reset loops",
        "W4",
    ),
    _row(
        "athena.max_recut_depth",
        2,
        "count",
        "lineage",
        "re-cuts of re-cuts: past it she may give feedback, take the task over or drop it, "
        "never re-cut it again",
        "W3",
    ),
    _row(
        "athena.one_task_max_tasks",
        3,
        "count",
        "lineage",
        "tasks a one-task plan may delegate, its own, re-cut and dropped ones counted: at it "
        "`delegate` is refused, and she finishes, takes over or ends OUTGROWN",
        "W3",
    ),
    _row(
        "athena.keep_recent",
        4,
        "count",
        "request",
        "the newest tool exchanges a fit of her task-loop request never evicts: the ones "
        "she is acting on",
        "W3",
    ),
    _row(
        "athena.design_summary_max",
        250,
        "tokens",
        "design",
        "the design summary",
        "W5",
    ),
    _row(
        "athena.reset_summary_max",
        2000,
        "tokens",
        "reset",
        "the reset summary is the only story she keeps through a reset",
        "W5",
    ),
    _row(
        "critic.max_tool_calls",
        50,
        "calls",
        "call",
        "reads a design or a diff and the code around it",
        "W3",
    ),
    _row(
        "critic.max_cumulative_result_tokens",
        0.40,
        "window_fraction",
        "call",
        "small template, no history",
        "W3",
    ),
    _row(
        "critic.turn_preview_max",
        2000,
        "chars",
        "turn",
        "a cited turn longer than this arrives as a preview and a handle",
        "W1",
    ),
    _row(
        "locate.max_tool_calls",
        40,
        "calls",
        "call",
        "was 15; a search needing more should come back `truncated`",
        "W3",
    ),
    _row(
        "locate.max_cumulative_result_tokens",
        0.40,
        "window_fraction",
        "call",
        "locate returns pointers, but reads to find them",
        "W3",
    ),
    _row(
        "locate.snapshot_tokens",
        4096,
        "tokens",
        "call",
        "the scout's map of the repository; a snapshot over it is cut to fit, never sent whole",
        "W8",
    ),
    _row(
        "zeus.max_tool_calls",
        64,
        "calls",
        "turn",
        "a calibration; unchanged",
        "W3",
    ),
    _row(
        "zeus.max_cumulative_result_tokens",
        0.50,
        "window_fraction",
        "turn",
        "a turn that reads without deciding; compaction handles the rest",
        "W3",
    ),
    _row(
        "zeus.constraint_text_max",
        500,
        "chars",
        "constraint",
        "one technical statement per constraint",
        "W1",
    ),
    _row(
        "zeus.inline_bytes",
        4096,
        "bytes",
        "result",
        "`INLINE_BYTES_DEFAULT`; Zeus reads to decide (invariant 13) — unchanged",
        "W3",
    ),
    _row(
        "zeus.shell_inline_bytes",
        8192,
        "bytes",
        "result",
        "`SHELL_INLINE_BYTES_DEFAULT` — unchanged",
        "W3",
    ),
    _row(
        "zeus.result_lines",
        200,
        "lines",
        "result",
        "`RESULT_LINES_DEFAULT` — unchanged",
        "W3",
    ),
    _row(
        "zeus.held_turns",
        6,
        "count",
        "turn",
        "a view older than this many model turns is shown again when re-read, since attention "
        "on it may have faded",
        "W3",
    ),
    _row(
        "zeus.held_tokens",
        16000,
        "tokens",
        "call",
        "a view with more than this much conversation after it is shown again when re-read",
        "W3",
    ),
    _row(
        "zeus.insist_showings",
        1,
        "count",
        "call",
        "a recent view asked again right after its pointer is shown this many times, then "
        "only the repetition bound answers",
        "W3",
    ),
    _row(
        "zeus.shell_yield_s",
        30,
        "seconds",
        "call",
        "a shell command still running after this becomes a job and the call answers; 0 turns "
        "auto-backgrounding off: a command blocks up to the shell's timeout",
        "W3",
    ),
    _row(
        "zeus.max_jobs",
        4,
        "count",
        "process",
        "jobs running at once in a session; past it a command cannot become a job and "
        "`background` is `JOBS_FULL`",
        "W3",
    ),
    _row(
        "zeus.max_wakes",
        3,
        "count",
        "turn",
        "turns a job's exit may open since the person last spoke, so a job that starts a job "
        "cannot wake Zeus forever",
        "W3",
    ),
    _row(
        "zeus.job_max_runtime_s",
        1800,
        "seconds",
        "process",
        "a task's runtime before its group is reaped and its exit recorded `timed_out`; a "
        "server has none",
        "W3",
    ),
    _row(
        "zeus.job_max_wait_s",
        600,
        "seconds",
        "call",
        "the longest one `job wait` blocks: past it the call answers `still running`",
        "W3",
    ),
    _row(
        "zeus.job_max_log_bytes",
        64 * 1024 * 1024,
        "bytes",
        "process",
        "a job's output over both files, checked once a second; past it the job is killed, "
        "not captured",
        "W3",
    ),
    _row(
        "zeus.tell_text_max",
        4000,
        "chars",
        "call",
        "a tell's text: a message to the team, never a design; past it `tell` is refused, "
        "and a change that needs more is an amendment",
        "W3",
    ),
    _row(
        "lsp.servers_max",
        8,
        "count",
        "process",
        "language servers alive at once across every episode, a language's linter counted as "
        "one (8: a language and its linter both fit for one to four languages); "
        "at the cap a pass starts its own session, nothing waits",
        "W9",
    ),
    _row(
        "lsp.idle_s",
        600,
        "seconds",
        "server",
        "an episode's server asked nothing this long is stopped and started again on its next "
        "use, so a paused episode holds no memory",
        "W9",
    ),
    _row(
        "reader.max_tool_calls",
        30,
        "calls",
        "request",
        "tool calls in a read-only agent's round; past it the round's next turn is "
        "its report, `partial`",
        "W10",
    ),
    _row(
        "reader.max_cumulative_result_tokens",
        0.40,
        "window_fraction",
        "request",
        "the tool results one reader round may read, of its own model's window",
        "W10",
    ),
    _row(
        "reader.seconds",
        300,
        "seconds",
        "request",
        "a reader round's wall time; past it no tool call runs and its next turn is its report",
        "W10",
    ),
    _row(
        "reader.max_cost_cents",
        50,
        "cents",
        "request",
        "a reader round's spend, inside the session's cap; past it no tool call runs and its "
        "next turn is its report",
        "W10",
    ),
    _row(
        "reader.report_reasks",
        1,
        "count",
        "request",
        "re-asks for a reader's report that misses a section or does not parse; then `failed`",
        "W10",
    ),
    _row(
        "reviewer.max_tool_calls",
        40,
        "calls",
        "request",
        "tool calls in a review agent's round: reads, `tests_for`, T0 commands; past "
        "it the round's next turn is its report, `partial`",
        "W10",
    ),
    _row(
        "reviewer.max_cumulative_result_tokens",
        0.40,
        "window_fraction",
        "request",
        "the tool results one reviewer round may read, of its own model's window",
        "W10",
    ),
    _row(
        "reviewer.seconds",
        600,
        "seconds",
        "request",
        "a reviewer round's wall time; past it no tool call runs and its next turn is its "
        "report",
        "W10",
    ),
    _row(
        "reviewer.max_cost_cents",
        50,
        "cents",
        "request",
        "a reviewer round's spend, inside the session's cap; past it no tool call runs and its "
        "next turn is its report",
        "W10",
    ),
    _row(
        "reviewer.report_reasks",
        1,
        "count",
        "request",
        "re-asks for a reviewer's report that misses a section or does not parse; then "
        "`failed`",
        "W10",
    ),
    _row(
        "researcher.max_tool_calls",
        30,
        "calls",
        "request",
        "tool calls in a researcher's round: searches and fetches; past it the round's "
        "next turn is its report, `partial`",
        "W10",
    ),
    _row(
        "researcher.max_cumulative_result_tokens",
        0.40,
        "window_fraction",
        "request",
        "the pages and results one researcher round may read, of its own model's window",
        "W10",
    ),
    _row(
        "researcher.seconds",
        600,
        "seconds",
        "request",
        "a researcher round's wall time, longer than a reader's because the web is slower "
        "than a disk; past it no tool call runs and its next turn is its report",
        "W10",
    ),
    _row(
        "researcher.max_cost_cents",
        50,
        "cents",
        "request",
        "a researcher round's spend, inside the session's cap; past it no tool call runs and "
        "its next turn is its report",
        "W10",
    ),
    _row(
        "researcher.report_reasks",
        1,
        "count",
        "request",
        "re-asks for a researcher's report that misses a section or does not parse; then "
        "`failed`",
        "W10",
    ),
    _row(
        "skills.index_max_rows",
        24,
        "count",
        "request",
        "skills listed in Zeus's resident index; past it the index says how many were "
        "withheld and names `find_skill`",
        "W10",
    ),
    _row(
        "skills.index_desc_chars",
        280,
        "chars",
        "result",
        "a description's length in the resident index, cut visibly past it, so the index's "
        "cost is bounded by the two rows",
        "W10",
    ),
    _row(
        "skills.find_results",
        5,
        "count",
        "call",
        "skills one `find_skill` returns, best first; more would read like the index it "
        "replaces",
        "W10",
    ),
    _row(
        "agents.max_parallel",
        3,
        "calls",
        "response",
        "`spawn_agent` calls one response may make, each on a thread of its own; past it, "
        "refused and not counted",
        "W10",
    ),
    _row(
        "agents.max_follow_ups",
        3,
        "count",
        "agent",
        "`follow_up` rounds one agent takes after its spawn; past it, refused, and a fresh "
        "agent is spawned with what it needs",
        "W10",
    ),
    _row(
        "hooks.max_continues",
        1,
        "count",
        "request",
        "a hook's `Continue` at one stop of a worker's round, Athena's request or Zeus's turn; "
        "a total, never a streak, so a busy role is still bounded",
        "W3",
    ),
    _row(
        "hooks.max_retries",
        1,
        "count",
        "task",
        "a hook's `RetryTask` of one task before Athena reviews it; past it the report goes to "
        "her as it would with no hook",
        "W3",
    ),
    _row(
        "hooks.max_steering",
        3,
        "count",
        "design",
        "every level-2 verdict a hook took in one episode, retries and Athena's continues "
        "together: what keeps a run from steering itself in a circle",
        "W3",
    ),
    _row(
        "hooks.recheck_wait_s",
        30,
        "seconds",
        "request",
        "how long a worker's stop waits for the background re-check before it compares the "
        "task's files with their start; usually nothing is queued and it waits not at all",
        "W3",
    ),
    _row(
        "web.fetch_max_chars",
        40000,
        "chars",
        "call",
        "a page's text one `web_fetch` returns; past it the answer says how much is left and "
        "the offset that reads on",
        "W3",
    ),
    _row(
        "web.fetch_connect_s",
        8,
        "seconds",
        "call",
        "connecting to a host for `web_fetch`; past it, `FETCH_FAILED{reason: timeout}`",
        "W3",
    ),
    _row(
        "web.fetch_timeout_s",
        15,
        "seconds",
        "call",
        "one GET of `web_fetch`, connection to last byte; a redirect followed is another",
        "W3",
    ),
    _row(
        "web.search_results",
        5,
        "count",
        "call",
        "results one `web_search` returns, every source asked together; more reads like a "
        "page of links, not an answer",
        "W3",
    ),
    _row(
        "web.search_snippet_chars",
        300,
        "chars",
        "result",
        "a search result's snippet, cut visibly past it: a snippet points at a page, and the "
        "page is `web_fetch`'s",
        "W3",
    ),
    _row(
        "web.fetch_page_s",
        20,
        "seconds",
        "call",
        "a page's own scripts under `lightpanda`, past which the browser ends them and dumps "
        "what loaded",
        "W3",
    ),
    _row(
        "web.url_max_chars",
        2000,
        "chars",
        "call",
        "a URL `web_fetch` will send; a longer one is refused before anything is sent, since "
        "length is how a URL carries out what it read",
        "W3",
    ),
    _row(
        "turn.attachments_max",
        8,
        "count",
        "turn",
        "files the person attaches to one turn; more reads as a folder handed over, which "
        "is a path to say, not bytes to send",
        "W3",
    ),
    _row(
        "turn.attachment_bytes_max",
        24 * 1024 * 1024,
        "bytes",
        "turn",
        "one turn's attachments together, decoded: the request body stays under a vendor's "
        "32 MB once base64 adds its third",
        "W3",
    ),
    _row(
        "turn.attachment_file_max_bytes",
        25 * 1024 * 1024,
        "bytes",
        "call",
        "one file before anything decodes it, so a huge file is refused unread",
        "W3",
    ),
    _row(
        "turn.pdf_file_max_bytes",
        20 * 1024 * 1024,
        "bytes",
        "call",
        "one PDF: its base64 stays under both vendors' request caps",
        "W3",
    ),
    _row(
        "turn.pdf_pages_max",
        100,
        "count",
        "call",
        "one PDF's pages, the page cap a vendor reads a PDF whole under",
        "W3",
    ),
    _row(
        "turn.pdf_text_min_chars",
        32,
        "chars",
        "call",
        "non-space characters a PDF's pages hold on average before it counts as having a "
        "text layer; below it is a scan, and is said to be one",
        "W3",
    ),
    _row(
        "turn.text_file_max_bytes",
        1024 * 1024,
        "bytes",
        "call",
        "one text file the person attaches; a larger one is a path Zeus can read in ranges",
        "W3",
    ),
    _row(
        "turn.text_inline_bytes",
        32 * 1024,
        "bytes",
        "result",
        "a text attachment shown whole in the turn; a longer one is shown as its outline "
        "and read by its handle",
        "W3",
    ),
    _row(
        "all.compact_trigger",
        0.80,
        "window_fraction",
        "call",
        "`COMPACT_TRIGGER_FRACTION`; overridable per role as `limits.<role>.compact_trigger`",
        "W4",
    ),
    _row(
        "all.compact_target",
        0.50,
        "window_fraction",
        "call",
        "where a box compaction stops; a cut that frees too little fires again at once",
        "W4",
    ),
)
"""The limits, in order. The one list."""

_BY_NAME = {lim.name: lim for lim in LIMITS}


def limit(name: str) -> Limit:
    """A row by name. An unknown name is a config error naming the register — the same
    answer an unknown role gets, never a `KeyError` from wherever the name was typed."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ConfigError(
            f"no limit named {name!r} in the limits register; the rows are {sorted(_BY_NAME)}"
        ) from None


def window(rung: RungEndpoint) -> int:
    """The same choice `Ctx.window` makes under `window_basis`.
    With `context_window` as the basis the output reserve is subtracted by `fits()`,
    not here."""
    return rung.input_limit if rung.input_limit is not None else rung.context_window


def _suffix(name: str) -> str:
    return name.split(".", 1)[1]


def _rows_by_key(limits: tuple[Limit, ...], roles: tuple[Role, ...]) -> dict[str, Limit]:
    keys = {f"limits.{lim.name}": lim for lim in limits}
    for lim in limits:
        if lim.applies_to == "all":
            keys |= {f"limits.{r.name}.{_suffix(lim.name)}": lim for r in roles}
    return keys


def override_keys(
    limits: tuple[Limit, ...] = LIMITS, roles: tuple[Role, ...] | None = None
) -> frozenset[str]:
    """Every key a config may set under `limits.`: one per row, plus one per role for each
    `all.` row (`limits.<role>.compact_trigger`). `roles` defaults to the cast, read here
    rather than at import (the module docstring says why)."""
    if roles is None:
        from chickadee.roles import ROLES

        roles = ROLES
    return frozenset(_rows_by_key(limits, roles))


def _validated(lim: Limit, key: str, value: object) -> float | int:
    # bool is an int to Python and never a limit to a person; `true` for a cap is a typo.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key} = {value!r}: a limit is a number")
    if lim.unit == "window_fraction":
        if not 0 < value <= 1:
            raise ConfigError(f"{key} = {value!r}: a window fraction is in (0, 1]")
        return value
    if lim.unit == "percent" and (not isinstance(value, int) or not 0 < value <= 100):
        raise ConfigError(f"{key} = {value!r}: a percent is a whole number in (0, 100]")
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} = {value!r}: a {lim.unit} limit is a positive integer")
    return value


def resolve(
    limits: tuple[Limit, ...],
    roles: tuple[Role, ...],
    rungs: Mapping[str, RungEndpoint],
    overrides: Mapping[str, float | int],
    *,
    assignment: Mapping[str, str],
) -> tuple[ResolvedLimit, ...]:
    """Every row, for every role it applies to, as a number on that role's rung.

    `overrides` is keyed `limits.<name>` (and `limits.<role>.<suffix>` for an `all` row). A
    key that names no row is refused, listing the valid ones: a typo in a limit's name is
    the silent failure of a bound that was meant to change and did not. `assignment` is the
    config's role → rung answer (`EngineConfig.assignment`), naming every role.
    """
    rows = _rows_by_key(limits, roles)
    unknown = sorted(set(overrides) - set(rows))
    if unknown:
        raise ConfigError(
            f"{unknown} name no row of the limits register; valid keys are {sorted(rows)}"
        )
    # Every override is checked, including a shared `limits.all.*` key that per-role keys
    # shadow for every role: a value that is refused is refused whether or not it is read.
    checked = {key: _validated(rows[key], key, value) for key, value in overrides.items()}
    by_name = {r.name: r for r in roles}
    out: list[ResolvedLimit] = []
    for lim in limits:
        if lim.applies_to in HARNESS_OWNERS:
            continue
        if lim.applies_to == "all":
            targets = roles
        elif lim.applies_to in by_name:
            targets = (by_name[lim.applies_to],)
        else:
            from chickadee.roles import role

            targets = (role(lim.applies_to),)
        for r in targets:
            keys = [f"limits.{lim.name}"]
            if lim.applies_to == "all":
                # A per-role key wins over the shared one.
                keys.insert(0, f"limits.{r.name}.{_suffix(lim.name)}")
            hit = next((k for k in keys if k in checked), None)
            value = checked[hit] if hit else lim.value
            rung_name = assignment[r.name]
            if lim.unit == "window_fraction":
                # Exact decimal arithmetic: 0.29 * 100,000 in binary floats is
                # 28,999.999…, and the journaled bound would be one token short of the
                # number a person wrote.
                n = math.floor(Fraction(str(value)) * window(rungs[rung_name]))
            else:
                n = int(value)
            out.append(
                ResolvedLimit(lim.name, r.name, rung_name, n, "config" if hit else "default")
            )
    resolved = tuple(out)
    _check_compaction_band(resolved)
    return resolved


def harness_limits(
    limits: tuple[Limit, ...], overrides: Mapping[str, float | int]
) -> dict[str, int]:
    """Each harness-wide row (`HARNESS_OWNERS`) as a number: its override, checked as every
    row's is, or its default. `overrides` is `resolve`'s, keyed `limits.<name>`."""
    out: dict[str, int] = {}
    for lim in limits:
        if lim.applies_to not in HARNESS_OWNERS:
            continue
        key = f"limits.{lim.name}"
        value = _validated(lim, key, overrides[key]) if key in overrides else lim.value
        out[lim.name] = int(value)
    return out


def _check_compaction_band(resolved: tuple[ResolvedLimit, ...]) -> None:
    """A compaction that stops at or above its trigger fires again on the next call
    (`all.compact_target`'s own reason) — a config that can never settle, refused at load."""
    by_role: dict[str, dict[str, int]] = {}
    for rl in resolved:
        if rl.name in ("all.compact_trigger", "all.compact_target"):
            by_role.setdefault(rl.role, {})[rl.name] = rl.tokens_or_value
    for role_name, pair in by_role.items():
        trigger = pair.get("all.compact_trigger")
        target = pair.get("all.compact_target")
        if trigger is not None and target is not None and target >= trigger:
            raise ConfigError(
                f"role {role_name!r}: compact_target ({target} tokens) is not below "
                f"compact_trigger ({trigger}); every compaction would fire again at once "
            )


def resolved_value(resolved: Sequence[ResolvedLimit], role_name: str, name: str) -> int:
    """One resolved row's value for one role: tokens for a fraction, else the value."""
    for rl in resolved:
        if rl.role == role_name and rl.name == name:
            return rl.tokens_or_value
    # `resolve()` produces every row for every role it applies to; a missing one is a caller
    # passing a partial table, not an operator's config.
    raise HarnessBug(f"limit {name!r} was not resolved for role {role_name!r}")


def role_limits(resolved: Sequence[ResolvedLimit], role_name: str) -> dict[str, int]:
    """Every resolved row of one role, by its name: what that role's dispatcher rules read
    (`tools/rules.py`), as a box role is handed them."""
    return {rl.name: rl.tokens_or_value for rl in resolved if rl.role == role_name}


NEVER_EVICTED: Mapping[str, tuple[str, ...]] = {
    "worker": ("worker.design_layer",),
    # The scout's map of the repository is in its first request, always.
    "locate": ("locate.snapshot_tokens",),
}
"""The rows each role holds in every request whatever else is evicted, by role. The
design, constraints and plan are measured per call by `fits()`; they are not known at load.
A role not named holds only its template."""


def _never_evicted_rows(r: Role) -> tuple[str, ...]:
    return NEVER_EVICTED.get(r.name, ())


def _never_evicted(r: Role, resolved: tuple[ResolvedLimit, ...]) -> int:
    return sum(resolved_value(resolved, r.name, n) for n in _never_evicted_rows(r))


def _one_result(r: Role, resolved: tuple[ResolvedLimit, ...]) -> int:
    """The largest single tool result the role may take, in tokens."""
    if r.name == "worker":
        return resolved_value(resolved, "worker", "worker.result_max")
    if r.tools == "none":
        return 0
    inline = resolved_value(resolved, "zeus", "zeus.inline_bytes")
    if r.tools in ("zeus", "athena"):
        # Both can take a command's output — Zeus from her shell, Athena from `request_exec`
        # — and the shell's cap is the larger of the two byte caps.
        shell = resolved_value(resolved, "zeus", "zeus.shell_inline_bytes")
        largest = tokens_for_chars(max(inline, shell))
        if r.name == "athena":
            # Her reads have their own ceiling, above both byte caps.
            return max(largest, resolved_value(resolved, "athena", "athena.result_max"))
        return largest
    # The scout surface has no shell, so its byte-bounded reads are the largest result a
    # byte cap bounds. Its line-bounded ones (`find`, `grep_in`: `zeus.result_lines`) have
    # no size in tokens until a line has a width, so load time cannot count them; `fits()`
    # measures them per call.
    return tokens_for_chars(inline)


def check_fit(
    role: Role,
    rung: RungEndpoint,
    resolved: tuple[ResolvedLimit, ...],
    template_tokens: int,
    *,
    rung_name: str,
) -> None:
    """Refuse, at load, a role that cannot fit one request on its rung, `rung_name` being
    the rung the config assigned it, which the refusal names.

    Load-time checking catches the configuration that can never work; `fits()`
    still gates each call, catching the request that does not work this time.
    """
    name = rung_name
    w = window(rung)
    held = _never_evicted(role, resolved)
    result = _one_result(role, resolved)
    floors = template_tokens + held + result + rung.max_output_tokens
    if floors > w:
        raise ConfigError(
            f"role {role.name!r} cannot fit one request on rung {name!r}: its floors "
            f"({floors} tokens: template {template_tokens}, never-evicted {held}"
            f"{_rows_named(role)}, one "
            f"result {result}, output {rung.max_output_tokens}) exceed the window ({w}) — "
            "lower its limits under `limits.` or give it a larger rung"
        )
    if role.name == "worker":
        floor = resolved_value(resolved, "worker", "worker.output_floor")
        if rung.max_output_tokens < floor:
            raise ConfigError(
                f"role 'worker' on rung {name!r}: max_output_tokens "
                f"({rung.max_output_tokens}) is below worker.output_floor ({floor}); a "
                "whole file could not be written"
            )


def _rows_named(r: Role) -> str:
    """The rows behind the never-evicted figure, so a refusal says which row to lower."""
    rows = _never_evicted_rows(r)
    return f" ({', '.join(rows)})" if rows else ""


HEADER = "| Name | Default | Unit | Per | Applies to | Why | Measured by |"


def _default_text(lim: Limit) -> str:
    # Fractions keep two places so the table reads as a column (0.30, not 0.3), and more
    # only when the value has them: a rendering that rounded 0.125 to 0.12 would publish a
    # number the code does not apply, and the drift test could not tell.
    if lim.unit != "window_fraction":
        return str(lim.value)
    two = f"{lim.value:.2f}"
    return two if float(two) == lim.value else repr(float(lim.value))


def render_limits(limits: tuple[Limit, ...] = LIMITS) -> str:
    """What the limits table contains — the same rows, rendered."""
    lines = [HEADER, "|---|---|---|---|---|---|---|"]
    for lim in limits:
        lines.append(
            f"| `{lim.name}` | {_default_text(lim)} | {lim.unit} | {lim.per} | "
            f"`{lim.applies_to}` | {lim.why} | {lim.measured_by} |"
        )
    return "\n".join(lines)
