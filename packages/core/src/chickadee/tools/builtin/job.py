"""`job`, a built-in: the jobs Zeus's shell commands became.

`list`, `tail`, `wait` and `kill`, over the `jobs` port (`zeus/jobs.JobTable`). A job's exit is
told without being asked for, so `wait` is for a result needed now, never a poll.

A thin run over the `jobs` port, with its output capped as the shell caps one, behind a
handle from the `handles` port.
"""

from __future__ import annotations

from typing import Final

from chickadee.context.results import CallCancelled, InvalidArgument, ToolResult, UnknownJob
from chickadee.edits.lines import split_lines
from chickadee.executor.types import joined_output
from chickadee.process.supervisor import Cancelled
from chickadee.tools.catalogue import capped
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "job")

TAIL_LINES: Final = 50
"""`tail`'s default: the end of a log, which is where a failure or a server's port is."""


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    jobs = required(ctx.jobs, "jobs")
    action = args.text("action")
    if action == "list":
        return Said(jobs.listing())
    job_id = args.text_or_none("id")
    if job_id is None:
        return InvalidArgument(
            "id",
            f"{action} names a job, and no id was sent",
            do="call job(action='list') for the ids, then send this with id",
        )
    if jobs.is_run(job_id):
        # The team's run sits beside the jobs in `list`, and is no job: its tail is
        # its task board, and it is never killed or waited on here.
        if action == "tail":
            return Said(jobs.run_board(job_id))
        return InvalidArgument(
            "id",
            f"{job_id} is the team's run, not a job: it cannot be {action}ed here",
            do="pause stops it, the user's /drop ends it, and you are told how it ends",
        )
    if not jobs.known(job_id):
        return UnknownJob(job_id)
    if action == "kill":
        ended = jobs.kill(job_id, "zeus")
        return Said(f"{job_id} killed" if ended else f"{job_id} is not running")
    if action == "wait":
        timeout = args.integer("timeout_s", jobs.max_wait_s)
        try:
            exit_ = jobs.await_exit(job_id, timeout, ctx.cancel)
        except Cancelled:
            return CallCancelled(SPEC.name)
        return Said(exit_ if exit_ is not None else f"{job_id} still running after {timeout} s")
    return _tail(call, args, ctx, job_id)


def _tail(call: Invocation, args: Arguments, ctx: ToolContext, job_id: str) -> Answer:
    jobs = required(ctx.jobs, "jobs")
    handles = required(ctx.handles, "handles")
    read = jobs.output(job_id)
    # An ended job's output is its record, both streams joined: there is no stream to pick.
    stream = "both" if read.kept else args.text_or_none("stream") or "both"
    text = (
        read.out
        if stream == "out"
        else read.err
        if stream == "err"
        else joined_output(read.out, read.err)
    )
    want = args.integer("lines", TAIL_LINES)
    shown = "".join(split_lines(text, keepends=True)[-want:])
    handle = handles.store("output", text.encode("utf-8"))
    content, truncation = capped(shown, handle, limit=jobs.inline_bytes, unit="bytes")
    # A log past one read's bound is held from its end, and the header says the handle is not
    # the whole of it.
    held = (
        f"{read.skipped} bytes before this unread"
        if read.skipped and not read.kept
        else f"kept at its exit, {read.skipped} bytes from its middle dropped"
        if read.skipped
        else "whole output"
    )
    return ToolResult(
        handle=handle,
        content=content,
        header=f"{job_id} · last {want} lines of {stream} · {held}: {handle.id}",
        truncation=truncation,
        notes=() if text else ("[no output yet]",),
    )


TOOL = Tool(SPEC, frozenset({"jobs", "handles"}), run)
