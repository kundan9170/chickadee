Run a command in the workspace under the sandbox: git, tests, builds. `command` is required.

The command runs in a sandbox over the workspace: the worktree is writable, the network off,
credentials stripped, secret files unreadable, and `/tmp` and a package cache are kept for the
session. Every command that is not a pure read is recorded, and what it changed can be undone by
its operation id; in the session's worktree it is committed on its own. A source file it changes
or creates is put back (a new one removed), unless the command is git's own operation (a switch,
a stash, a merge): source goes through `coder`.

Some commands ask the user first, as the user's settings say: an install, a fetch, a push, a
server, anything destructive. Approved, a command runs outside the sandbox, as the user would
run it: the network on, their environment, anywhere they can write. Set `network: true` on a
command that needs the network and does not say so itself, such as a project script; set
`sandbox: false` on a server (below) and on a command that failed because of the sandbox (a
write outside the worktree, a file it could not read). Either asks the user. With no sandbox
nothing is sealed, so a command may wait for the user to confirm it.

Install a project's dependencies into the worktree, where they need no approval:
`python3 -m venv .venv && .venv/bin/pip install …`, `npm install`, `uv sync`. A pipe reports
its last command's exit code: `pip install x | tail -5` says `exit 0` when pip failed, so leave
the pipe off, or start with `set -o pipefail`, when the exit code matters.

**A server needs `sandbox: false`.** Inside the sandbox nothing can reach a server: not the
user's browser, and not a check such as `curl localhost:8000`. Send `sandbox: false` with every
command that starts a server (`npm run dev`, `python3 -m http.server`, a `node server.js` or
`python app.py` that serves) and with every check against one, and `kind: "server"` with a
server whose command does not say it is one, such as a script. The user is asked; approved, it
runs outside the sandbox and starts as a job at once. Sent without it, a server the harness
knows is asked anyway, and one it does not know runs sealed: it fails, or runs where nobody can
open it.

A command still running after `zeus.shell_yield_s` becomes a job: the call answers with its id
(`j1`) and what it printed so far, the command runs on, and you are told when it exits, so do
not poll. `background: true` makes it a job at once, as a server is by default; a `server` job
is told only of an exit nobody asked for, a crash. What a job writes after it becomes one is not
recorded, put back or undone, so a command that writes source is for `coder`, not a job.
`JOBS_FULL` names the running jobs when no more may run.

Output under `shell_inline_cap` returns untouched; over it, the head that fits plus a
`Truncation` naming the total and a handle for the rest. Commands serialise by default.
`REFUSED_ABOVE_T0` names the missing grant: obtain authority, never retry the same call.
`SANDBOX_REQUIRED`: nothing can run here as configured; do not retry, tell the user.
