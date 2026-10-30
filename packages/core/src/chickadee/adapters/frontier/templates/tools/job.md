List, tail, wait on or kill the session's jobs: the shell commands still running. `action` is required.

`list` names every job of the session, running or ended. `tail` shows the last `lines` of a
job's output (`stream`: `out`, `err` or `both`), capped as a shell result, with a handle for the
whole of it. Once a job has ended, its output is what was kept: the start and end of each stream,
joined, with the size of any gap between them marked. `kill` ends a job; its exit is recorded and you are not told of it again. `wait`
blocks until the job exits or `timeout_s` passes (at most `zeus.job_max_wait_s`), and answers
with the exit; an exit a `wait` returned is not told again.

You are told when a job exits without asking, so do not poll with `tail` or `wait`: wait only
when you need the result before anything else. A job's exit is information, not verification:
when you report it, say it came from a job and what it ran. `UNRESOLVED` means no job of that id
in this session; `job(action='list')` names them. Serialised: one at a time.
